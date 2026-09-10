"""人工校正 API 闭环：编辑/新建/继承/作废/锁定与掩码复用边界。"""

import json
from pathlib import Path

from conftest import (
    CannedTranscriber,
    erase_to_completion,
    make_client,
    make_video,
    run_to_analysis,
    wait_for,
)

ERASED_MIX = Path("05-mask-erase") / "erased_mix.wav"


def timeline(client, task_id):
    resp = client.get(f"/api/tasks/{task_id}/diarization")
    assert resp.status_code == 200, resp.text
    return resp.json()


def segment_by_index(data, index):
    for speaker in data["speakers"]:
        for segment in speaker["segments"]:
            if segment["index"] == index:
                return speaker, segment
    raise AssertionError(f"没有时段索引 {index}")


def reassign(client, task_id, index, target):
    return client.put(
        f"/api/tasks/{task_id}/segments/{index}/speaker",
        json={"target": target},
    )


def transcript_speakers(client, task_id):
    data = client.get(f"/api/tasks/{task_id}/transcripts").json()
    return {s["speaker"] for s in data["samples"]}


def test_改派立即保存_展示和擦除校验读取生效时段(client, tmp_path):
    video = make_video(tmp_path)
    task = run_to_analysis(client, video)
    resp = reassign(client, task["id"], 0, "SPEAKER_01")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["correction"]["state"] == "active"
    assert segment_by_index(body, 0)[0]["label"] == "SPEAKER_01"
    assert segment_by_index(body, 1)[0]["label"] == "SPEAKER_01"
    assert segment_by_index(body, 2)[0]["label"] == "SPEAKER_00"
    assert client.get(
        f"/api/tasks/{task['id']}/audio-segment?start=0.2&end=0.9"
    ).status_code == 200
    erase = client.post(
        f"/api/tasks/{task['id']}/erase", json={"speakers": ["SPEAKER_01"]}
    )
    assert erase.status_code == 200, erase.text


def test_新建说话人可命名试听和擦除但没有样本转写(client, tmp_path):
    video = make_video(tmp_path)
    task = run_to_analysis(client, video)
    resp = reassign(client, task["id"], 1, "__new__")
    assert resp.status_code == 200, resp.text
    label = resp.json()["speakers"][1]["label"]
    assert label == "SPEAKER_02"
    rename = client.put(
        f"/api/tasks/{task['id']}/speakers/{label}/name", json={"name": "新角色"}
    )
    assert rename.status_code == 200, rename.text
    data = timeline(client, task["id"])
    new_speaker = next(s for s in data["speakers"] if s["label"] == label)
    assert new_speaker["name"] == "新角色"
    assert new_speaker["segments"] == [{"start": 0.95, "end": 1.25, "index": 1}]
    assert label not in transcript_speakers(client, task["id"])
    done = erase_to_completion(client, task["id"], [label])
    assert "新角色" in Path(done["output_path"]).name


def test_校正快照随工作区继承_放弃全部校正回自动结果(tmp_path):
    video = make_video(tmp_path)
    with make_client(tmp_path) as client:
        first = run_to_analysis(client, video)
        resp = reassign(client, first["id"], 0, "SPEAKER_01")
        assert resp.status_code == 200, resp.text
        second = run_to_analysis(client, video)
        data = timeline(client, second["id"])
        assert data["correction"]["state"] == "active"
        assert segment_by_index(data, 0)[0]["label"] == "SPEAKER_01"

        discarded = client.delete(f"/api/tasks/{first['id']}/corrections")
        assert discarded.status_code == 200, discarded.text
        data = timeline(client, first["id"])
        assert data["correction"]["state"] == "none"
        assert segment_by_index(data, 0)[0]["label"] == "SPEAKER_00"


def test_重新说话人分离后校正作废并给出提示(tmp_path):
    video = make_video(tmp_path)
    with make_client(tmp_path) as client:
        first = run_to_analysis(client, video)
        assert reassign(client, first["id"], 0, "SPEAKER_01").status_code == 200
        second = run_to_analysis(client, video, num_speakers=4)
        data = timeline(client, second["id"])
        assert data["correction"]["state"] == "invalidated"
        assert "已作废" in data["correction"]["message"]
        assert segment_by_index(data, 0)[0]["label"] == "SPEAKER_00"


def test_工作区任务运行中校正写操作锁定(tmp_path):
    video = make_video(tmp_path)
    transcriber = CannedTranscriber(gate=__import__("threading").Event())
    with make_client(tmp_path, transcriber=transcriber) as client:
        task = client.post("/api/tasks", json={"video_path": str(video)}).json()
        wait_for(
            lambda: next(
                s for s in client.get(f"/api/tasks/{task['id']}").json()["stages"]
                if s["id"] == "transcribe"
            )["status"] == "running"
        )
        data = timeline(client, task["id"])
        assert data["correction"]["editable"] is False
        assert "排队中或运行中" in data["correction"]["locked_reason"]
        resp = reassign(client, task["id"], 0, "SPEAKER_01")
        assert resp.status_code == 409, resp.text
        assert "排队中或运行中" in resp.json()["detail"]
        split_resp = client.put(
            f"/api/tasks/{task['id']}/segments/0/split",
            json={
                "results": [
                    {"start": 0.2, "end": 0.5, "target": "SPEAKER_00"},
                    {"start": 0.5, "end": 0.9, "target": "SPEAKER_01"},
                ]
            },
        )
        assert split_resp.status_code == 409, split_resp.text
        assert "排队中或运行中" in split_resp.json()["detail"]
        transcriber.gate.set()
        wait_for(
            lambda: client.get(f"/api/tasks/{task['id']}").json()["state"] == "completed"
        )
        assert reassign(client, task["id"], 0, "SPEAKER_01").status_code == 200


def test_校正变化后同目标重擦不复用旧混音(client, tmp_path):
    video = make_video(tmp_path)
    task = run_to_analysis(client, video)
    first = erase_to_completion(client, task["id"], ["SPEAKER_00"])
    mix = Path(first["workspace_dir"]) / ERASED_MIX
    before = mix.stat().st_mtime_ns

    assert reassign(client, task["id"], 0, "SPEAKER_01").status_code == 200
    second = erase_to_completion(client, task["id"], ["SPEAKER_00"])
    assert mix.stat().st_mtime_ns != before
    manifest = json.loads((Path(task["workspace_dir"]) / "manifest.json").read_text())
    assert manifest["mask_erase"]["correction_revision"] == 1
    assert Path(second["output_path"]).is_file()


def test_边界移动API不可用_改派切分与放弃校正不回归(client, tmp_path):
    video = make_video(tmp_path)
    task = run_to_analysis(client, video)

    # 旧边界移动端点已移除：PUT 边界不再可用（404 路由不存在 / 405 方法不允许）。
    gone = client.put(
        f"/api/tasks/{task['id']}/segments/0/boundary",
        json={"side": "end", "new_time": 0.93},
    )
    assert gone.status_code in (404, 405), gone.text

    # 改派仍正常。
    resp = reassign(client, task["id"], 0, "SPEAKER_01")
    assert resp.status_code == 200, resp.text
    assert segment_by_index(resp.json(), 0)[0]["label"] == "SPEAKER_01"

    # 切分仍正常：把时段 0 拆成两段，第二段改派给 SPEAKER_01。
    split_resp = client.put(
        f"/api/tasks/{task['id']}/segments/0/split",
        json={
            "results": [
                {"start": 0.2, "end": 0.5, "target": "SPEAKER_00"},
                {"start": 0.5, "end": 0.9, "target": "SPEAKER_01"},
            ]
        },
    )
    assert split_resp.status_code == 200, split_resp.text
    body = split_resp.json()
    assert body["correction"]["state"] == "active"
    assert segment_by_index(body, 0)[0]["label"] == "SPEAKER_00"
    assert segment_by_index(body, 1)[0]["label"] == "SPEAKER_01"
    assert segment_by_index(body, 1)[1]["start"] == 0.5

    # 放弃全部校正仍正常：回到自动分析结果。
    discarded = client.delete(f"/api/tasks/{task['id']}/corrections")
    assert discarded.status_code == 200, discarded.text
    data = timeline(client, task["id"])
    assert data["correction"]["state"] == "none"
    assert segment_by_index(data, 0)[0]["label"] == "SPEAKER_00"

