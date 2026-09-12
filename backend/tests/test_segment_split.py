"""时段切分 API 闭环：合法切分/新建槽位复用/单区间拒绝/非法草稿/锁定/作废/继承/生效读取。"""

import json
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


def split(client, task_id, index, results):
    return client.put(
        f"/api/tasks/{task_id}/segments/{index}/split",
        json={"results": results},
    )


def snapshot_file(ws: Path) -> Path:
    return ws / "corrections.json"


def test_合法切分一次替换原时段_时间轴刷新且生效时段读取切分结果(client, tmp_path):
    video = make_video(tmp_path)
    task = run_to_analysis(client, video)
    resp = split(
        client,
        task["id"],
        0,
        [
            {"start": 0.2, "end": 0.55, "target": "SPEAKER_00"},
            {"start": 0.55, "end": 0.9, "target": "SPEAKER_01"},
        ],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["correction"]["state"] == "active"
    assert body["correction"]["revision"] == 1
    # 原时段被两个结果区间替换，其余时段保持位置
    assert segment_by_index(body, 0)[0]["label"] == "SPEAKER_00"
    assert segment_by_index(body, 0)[1] == {"start": 0.2, "end": 0.55, "index": 0}
    assert segment_by_index(body, 1)[0]["label"] == "SPEAKER_01"
    assert segment_by_index(body, 1)[1] == {"start": 0.55, "end": 0.9, "index": 1}
    assert segment_by_index(body, 2)[0]["label"] == "SPEAKER_01"
    assert segment_by_index(body, 3)[0]["label"] == "SPEAKER_00"
    # 校验失败不产生部分保存
    assert snapshot_file(Path(task["workspace_dir"])).is_file()
    # 试听与擦除读取切分后的生效时段
    assert client.get(
        f"/api/tasks/{task['id']}/audio-segment?start=0.2&end=0.9"
    ).status_code == 200
    done = erase_to_completion(client, task["id"], ["SPEAKER_00"])
    assert Path(done["output_path"]).is_file()


def test_新建槽位复用_多个区间映射同一个顺延标签(client, tmp_path):
    video = make_video(tmp_path)
    task = run_to_analysis(client, video)
    resp = split(
        client,
        task["id"],
        0,
        [
            {"start": 0.2, "end": 0.4, "target": "__new__", "slot": "新角色"},
            {"start": 0.4, "end": 0.7, "target": "SPEAKER_00"},
            {"start": 0.7, "end": 0.9, "target": "__new__", "slot": "新角色"},
        ],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # 同一个槽位映射为同一个顺延 SPEAKER_02
    assert segment_by_index(body, 0)[0]["label"] == "SPEAKER_02"
    assert segment_by_index(body, 2)[0]["label"] == "SPEAKER_02"
    assert segment_by_index(body, 1)[0]["label"] == "SPEAKER_00"
    # 再切一次，第二个新建槽位顺延为 SPEAKER_03
    resp = split(
        client,
        task["id"],
        0,
        [
            {"start": 0.2, "end": 0.3, "target": "__new__", "slot": "另一个"},
            {"start": 0.3, "end": 0.4, "target": "SPEAKER_00"},
        ],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert segment_by_index(body, 0)[0]["label"] == "SPEAKER_03"


def test_切分非法草稿被拒绝且不产生部分保存(client, tmp_path):
    video = make_video(tmp_path)
    task = run_to_analysis(client, video)
    ws = Path(task["workspace_dir"])
    cases = [
        (
            [
                {"start": 0.2, "end": 0.25, "target": "SPEAKER_00"},
                {"start": 0.25, "end": 0.9, "target": "SPEAKER_01"},
            ],
            "每个结果区间至少 0.1 秒",
        ),
        (
            [
                {"start": 0.2, "end": 0.55, "target": "SPEAKER_00"},
                {"start": 0.6, "end": 0.9, "target": "SPEAKER_01"},
            ],
            "结果区间必须按时间连续且精确覆盖原时段",
        ),
        (
            [
                {"start": 0.2, "end": 0.6, "target": "SPEAKER_00"},
                {"start": 0.5, "end": 0.9, "target": "SPEAKER_01"},
            ],
            "结果区间必须按时间连续且互不重叠",
        ),
        (
            [
                {"start": 0.25, "end": 0.55, "target": "SPEAKER_00"},
                {"start": 0.55, "end": 0.9, "target": "SPEAKER_01"},
            ],
            "结果区间必须精确覆盖原时段",
        ),
        (
            [
                {"start": 0.2, "end": 0.55, "target": "SPEAKER_00"},
                {"start": 0.55, "end": 0.95, "target": "SPEAKER_01"},
            ],
            "结果区间必须精确覆盖原时段",
        ),
        (
            [
                {"start": 0.2, "end": 0.55, "target": "SPEAKER_99"},
                {"start": 0.55, "end": 0.9, "target": "SPEAKER_01"},
            ],
            "说话人不存在: SPEAKER_99",
        ),
        (
            [
                {"start": 0.2, "end": 0.55, "target": "SPEAKER_00"},
                {"start": 0.55, "end": 0.9, "target": "SPEAKER_00"},
            ],
            "至少一个结果区间不是原说话人才允许保存",
        ),
    ]
    for results, expected in cases:
        resp = split(client, task["id"], 0, results)
        assert resp.status_code == 400, (results, resp.text)
        assert resp.json()["detail"] == expected, (results, resp.text)
        # 校验失败不产生部分保存：校正快照仍不存在，时间轴仍是自动结果
        assert not snapshot_file(ws).is_file()
        data = timeline(client, task["id"])
        assert data["correction"]["state"] == "none"
        assert segment_by_index(data, 0)[1] == {"start": 0.2, "end": 0.9, "index": 0}


def test_切分校验不存在的时段(client, tmp_path):
    video = make_video(tmp_path)
    task = run_to_analysis(client, video)
    resp = split(
        client,
        task["id"],
        99,
        [
            {"start": 0.2, "end": 0.55, "target": "SPEAKER_00"},
            {"start": 0.55, "end": 0.9, "target": "SPEAKER_01"},
        ],
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "说话人时段不存在"


def test_切分请求结构非法由协议层拒绝(client, tmp_path):
    video = make_video(tmp_path)
    task = run_to_analysis(client, video)
    # 未指定 target
    resp = split(
        client,
        task["id"],
        0,
        [
            {"start": 0.2, "end": 0.55},
            {"start": 0.55, "end": 0.9, "target": "SPEAKER_01"},
        ],
    )
    assert resp.status_code == 422
    # 时间倒置由后端校验拒绝
    resp = split(
        client,
        task["id"],
        0,
        [
            {"start": 0.5, "end": 0.2, "target": "SPEAKER_00"},
            {"start": 0.55, "end": 0.9, "target": "SPEAKER_01"},
        ],
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "结果区间时间非法"
    # 只有一个结果区间：协议层直接拒绝（切分至少需要两个结果区间）
    resp = split(
        client,
        task["id"],
        0,
        [{"start": 0.2, "end": 0.9, "target": "SPEAKER_00"}],
    )
    assert resp.status_code == 422


def test_单区间切分被协议层拒绝_整段改说话人走改派唯一路径(client, tmp_path):
    video = make_video(tmp_path)
    task = run_to_analysis(client, video)
    # 单区间切分：协议层拒绝（至少需要两个结果区间），不产生快照
    resp = split(
        client,
        task["id"],
        0,
        [{"start": 0.2, "end": 0.9, "target": "SPEAKER_01"}],
    )
    assert resp.status_code == 422, resp.text
    assert not snapshot_file(Path(task["workspace_dir"])).is_file()
    data = timeline(client, task["id"])
    assert data["correction"]["state"] == "none"
    assert segment_by_index(data, 0)[0]["label"] == "SPEAKER_00"
    # 整段改说话人只走改派唯一路径：直接保存且其余时段不变
    resp = client.put(
        f"/api/tasks/{task['id']}/segments/0/speaker",
        json={"target": "SPEAKER_01"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["correction"]["state"] == "active"
    assert body["correction"]["revision"] == 1
    assert segment_by_index(body, 0)[0]["label"] == "SPEAKER_01"
    assert segment_by_index(body, 0)[1] == {"start": 0.2, "end": 0.9, "index": 0}
    assert segment_by_index(body, 1)[0]["label"] == "SPEAKER_01"
    assert segment_by_index(body, 2)[0]["label"] == "SPEAKER_00"
    assert snapshot_file(Path(task["workspace_dir"])).is_file()


def test_单区间新建说话人_切分被拒绝_改派路径承接(client, tmp_path):
    video = make_video(tmp_path)
    task = run_to_analysis(client, video)
    # 单区间切分（含新建说话人）被协议层拒绝
    resp = split(
        client,
        task["id"],
        0,
        [{"start": 0.2, "end": 0.9, "target": "__new__", "slot": "新角色"}],
    )
    assert resp.status_code == 422, resp.text
    # 新建说话人整段改派走改派唯一路径：顺延标签直接保存
    resp = client.put(
        f"/api/tasks/{task['id']}/segments/0/speaker",
        json={"target": "__new__"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert segment_by_index(body, 0)[0]["label"] == "SPEAKER_02"
    assert body["correction"]["revision"] == 1


def test_不足零点一秒的时段经改派路径改说话人(tmp_path):
    """移除单区间切分后，短时段整段改说话人的能力由改派对话框承接。"""
    video = make_video(tmp_path)
    turns = (
        SpeakerSegment("SPEAKER_00", 0.2, 0.25),
        SpeakerSegment("SPEAKER_01", 0.3, 0.6),
    )
    with make_client(tmp_path, diarizer=CannedDiarizer(turns=turns)) as client:
        task = run_to_analysis(client, video)
        # 单区间切分被协议层拒绝
        resp = split(
            client,
            task["id"],
            0,
            [{"start": 0.2, "end": 0.25, "target": "SPEAKER_01"}],
        )
        assert resp.status_code == 422, resp.text
        # 不足 0.1 秒的时段仍可通过改派对话框改说话人
        resp = client.put(
            f"/api/tasks/{task['id']}/segments/0/speaker",
            json={"target": "SPEAKER_01"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert segment_by_index(body, 0)[0]["label"] == "SPEAKER_01"
        assert segment_by_index(body, 0)[1] == {"start": 0.2, "end": 0.25, "index": 0}


def test_工作区任务运行中切分写操作锁定(tmp_path):
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
        resp = split(
            client,
            task["id"],
            0,
            [
                {"start": 0.2, "end": 0.55, "target": "SPEAKER_00"},
                {"start": 0.55, "end": 0.9, "target": "SPEAKER_01"},
            ],
        )
        assert resp.status_code == 409, resp.text
        assert "排队中或运行中" in resp.json()["detail"]
        transcriber.gate.set()
        wait_for(
            lambda: client.get(f"/api/tasks/{task['id']}").json()["state"] == "completed"
        )
        assert split(
            client,
            task["id"],
            0,
            [
                {"start": 0.2, "end": 0.55, "target": "SPEAKER_00"},
                {"start": 0.55, "end": 0.9, "target": "SPEAKER_01"},
            ],
        ).status_code == 200


def test_已保存切分随工作区继承_重新分离后作废并锁定(tmp_path):
    video = make_video(tmp_path)
    with make_client(tmp_path) as client:
        first = run_to_analysis(client, video)
        resp = split(
            client,
            first["id"],
            0,
            [
                {"start": 0.2, "end": 0.55, "target": "SPEAKER_00"},
                {"start": 0.55, "end": 0.9, "target": "SPEAKER_01"},
            ],
        )
        assert resp.status_code == 200, resp.text

        second = run_to_analysis(client, video)
        data = timeline(client, second["id"])
        assert data["correction"]["state"] == "active"
        assert segment_by_index(data, 1)[0]["label"] == "SPEAKER_01"
        assert segment_by_index(data, 1)[1] == {"start": 0.55, "end": 0.9, "index": 1}

        third = run_to_analysis(client, video, num_speakers=4)
        data = timeline(client, third["id"])
        assert data["correction"]["state"] == "invalidated"
        locked = split(
            client,
            third["id"],
            0,
            [
                {"start": 0.2, "end": 0.55, "target": "SPEAKER_00"},
                {"start": 0.55, "end": 0.9, "target": "SPEAKER_01"},
            ],
        )
        assert locked.status_code == 409, locked.text
        assert "已作废" in locked.json()["detail"]


def test_切分后掩码合成读取新生效时段_同目标重擦不复用旧混音(client, tmp_path):
    video = make_video(tmp_path)
    task = run_to_analysis(client, video)
    first = erase_to_completion(client, task["id"], ["SPEAKER_00"])
    mix = Path(first["workspace_dir"]) / "05-mask-erase" / "erased_mix.wav"
    before = mix.stat().st_mtime_ns

    resp = split(
        client,
        task["id"],
        0,
        [
            {"start": 0.2, "end": 0.55, "target": "SPEAKER_00"},
            {"start": 0.55, "end": 0.9, "target": "SPEAKER_01"},
        ],
    )
    assert resp.status_code == 200, resp.text
    second = erase_to_completion(client, task["id"], ["SPEAKER_00"])
    assert mix.stat().st_mtime_ns != before
    manifest = json.loads((Path(task["workspace_dir"]) / "manifest.json").read_text())
    assert manifest["mask_erase"]["correction_revision"] == 1
    assert Path(second["output_path"]).is_file()
