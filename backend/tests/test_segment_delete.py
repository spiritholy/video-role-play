"""删除说话人时段 API 闭环：时间轴移出/留白语义/放弃恢复/锁定/索引稳定。"""

from pathlib import Path

from app.inference import SpeakerSegment

from conftest import (
    CannedDiarizer,
    CannedTranscriber,
    erase_to_completion,
    make_client,
    make_video,
    run_to_analysis,
    wait_for,
)


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


def delete_segment(client, task_id, index):
    return client.delete(f"/api/tasks/{task_id}/segments/{index}")


def test_删除时段_时间轴移除_区间不再归属任何人(client, tmp_path):
    video = make_video(tmp_path)
    task = run_to_analysis(client, video)
    resp = delete_segment(client, task["id"], 1)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["correction"]["state"] == "active"
    assert body["correction"]["revision"] == 1
    # 原 [0.95, 1.25] 的 SPEAKER_01 时段消失，其余时段保持相对顺序重新编号
    assert len(body["speakers"]) == 1
    assert segment_by_index(body, 0)[0]["label"] == "SPEAKER_00"
    assert segment_by_index(body, 0)[1] == {"start": 0.2, "end": 0.9, "index": 0}
    assert segment_by_index(body, 1)[0]["label"] == "SPEAKER_00"
    assert segment_by_index(body, 1)[1] == {"start": 1.3, "end": 1.8, "index": 1}
    # 删除区间留白：试听与擦除仍可作用于剩余生效时段
    assert client.get(
        f"/api/tasks/{task['id']}/audio-segment?start=0.2&end=0.9"
    ).status_code == 200
    done = erase_to_completion(client, task["id"], ["SPEAKER_00"])
    assert Path(done["output_path"]).is_file()


def test_删除自动时段创建快照_放弃全部校正恢复(client, tmp_path):
    video = make_video(tmp_path)
    task = run_to_analysis(client, video)
    resp = delete_segment(client, task["id"], 0)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # 原有的 SPEAKER_00 [0.2,0.9] 被删除，下一个时段顺位到索引 0
    assert segment_by_index(body, 0)[0]["label"] == "SPEAKER_01"
    assert segment_by_index(body, 0)[1] == {"start": 0.95, "end": 1.25, "index": 0}
    assert (Path(task["workspace_dir"]) / "corrections.json").is_file()

    discarded = client.delete(f"/api/tasks/{task['id']}/corrections")
    assert discarded.status_code == 200, discarded.text
    data = timeline(client, task["id"])
    assert data["correction"]["state"] == "none"
    assert segment_by_index(data, 0)[0]["label"] == "SPEAKER_00"
    assert segment_by_index(data, 0)[1] == {"start": 0.2, "end": 0.9, "index": 0}


def test_删除后再切分另一时段_仍按新生效索引工作(client, tmp_path):
    video = make_video(tmp_path)
    task = run_to_analysis(client, video)
    assert delete_segment(client, task["id"], 1).status_code == 200
    resp = client.put(
        f"/api/tasks/{task['id']}/segments/0/split",
        json={
            "results": [
                {"start": 0.2, "end": 0.55, "target": "SPEAKER_00"},
                {"start": 0.55, "end": 0.9, "target": "__new__", "slot": "另一人"},
            ]
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["correction"]["revision"] == 2
    # 删除后再切分仍按新生效索引工作：原 SPEAKER_01 已不存在，顺延标签回收
    assert segment_by_index(body, 1)[0]["label"] == "SPEAKER_01"
    assert segment_by_index(body, 1)[1] == {"start": 0.55, "end": 0.9, "index": 1}
    assert segment_by_index(body, 2)[0]["label"] == "SPEAKER_00"
    assert segment_by_index(body, 2)[1] == {"start": 1.3, "end": 1.8, "index": 2}


def test_删除不存在的时段返回404(client, tmp_path):
    video = make_video(tmp_path)
    task = run_to_analysis(client, video)
    resp = delete_segment(client, task["id"], 99)
    assert resp.status_code == 404, resp.text
    assert "说话人时段不存在" in resp.json()["detail"]


def test_工作区任务运行中删除锁定(tmp_path):
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
        resp = delete_segment(client, task["id"], 0)
        assert resp.status_code == 409, resp.text
        assert "排队中或运行中" in resp.json()["detail"]
        transcriber.gate.set()
        wait_for(
            lambda: client.get(f"/api/tasks/{task['id']}").json()["state"] == "completed"
        )
        assert delete_segment(client, task["id"], 0).status_code == 200


def test_删除时段_基础数据含重叠仍生成有效快照(tmp_path):
    """生产 pyannote 输出可能含相邻/嵌套重叠：删除后快照必须立即生效。"""
    video = make_video(tmp_path)
    turns = (
        SpeakerSegment("SPEAKER_00", 0.2, 0.9),
        SpeakerSegment("SPEAKER_01", 0.95, 1.25),
        SpeakerSegment("SPEAKER_00", 1.3, 1.8),
        # 嵌套重叠：SPEAKER_01 长段内含 SPEAKER_00 短段（与真实生产数据同构）
        SpeakerSegment("SPEAKER_01", 2.0, 4.0),
        SpeakerSegment("SPEAKER_00", 2.5, 2.8),
    )
    with make_client(tmp_path, diarizer=CannedDiarizer(turns=turns)) as client:
        task = run_to_analysis(client, video)
        data = timeline(client, task["id"])
        assert data["correction"]["editable"] is True

        resp = delete_segment(client, task["id"], 1)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["correction"]["state"] == "active"
        assert body["correction"]["editable"] is True
        # 被删时段消失；其余时段（含重叠对）保留并重排索引
        segs = sorted(
            (s["start"], s["end"])
            for sp in body["speakers"]
            for s in sp["segments"]
        )
        assert segs == [(0.2, 0.9), (1.3, 1.8), (2.0, 4.0), (2.5, 2.8)]
        indexes = sorted(
            s["index"] for sp in body["speakers"] for s in sp["segments"]
        )
        assert indexes == [0, 1, 2, 3]

        # 删除后再删另一时段仍正常：快照不会因重叠作废、编辑不锁定
        second = delete_segment(client, task["id"], 0)
        assert second.status_code == 200, second.text
        assert second.json()["correction"]["state"] == "active"
        assert second.json()["correction"]["editable"] is True
