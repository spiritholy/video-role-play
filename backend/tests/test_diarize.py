"""说话人分离阶段：产物落盘、时间轴端点、时段试听端点、失败/取消边界。

说话人分离经全项目唯一注入边界（ML 推理接口）注入 canned 说话人时段，
不触碰真实模型；真实 ffmpeg 只负责生成切片音频内容。
"""

import json
import threading

from app.media import TaskError
from conftest import (
    CannedDiarizer,
    CannedSeparator,
    get_task,
    make_client,
    make_video,
    probe,
    submit_task,
    wait_for,
)

DIARIZATION = "03-diarize/diarization.json"


def diarize_stage(task: dict) -> dict:
    return next(s for s in task["stages"] if s["id"] == "diarize")


def test_完成后工作区存在说话人时段产物且清单更新(client, tmp_path):
    video = make_video(tmp_path, "认人.mp4")
    task = submit_task(client, video)
    wait_for(lambda: get_task(client, task["id"])["state"] == "completed")
    done = get_task(client, task["id"])

    stage = diarize_stage(done)
    assert stage["status"] == "done"
    assert stage["outputs"] == {"diarization": DIARIZATION}

    ws = tmp_path / "workspaces" / "认人"
    assert (ws / DIARIZATION).is_file(), "说话人时段产物应落盘"
    data = json.loads((ws / DIARIZATION).read_text(encoding="utf-8"))
    assert data["model"] == "canned-diarizer"
    assert data["num_speakers"] == 2
    assert {"speaker": "SPEAKER_00", "start": 1.3, "end": 1.8} in data["segments"]

    manifest = json.loads((ws / "manifest.json").read_text(encoding="utf-8"))
    assert [s["status"] for s in manifest["stages"]] == ["done"] * 5
    assert manifest["diarization"] == {
        "model": "canned-diarizer",
        "num_speakers": 2,
        "segments": DIARIZATION,
    }
    for entry in manifest["stages"]:
        for artifact in entry["artifacts"]:
            assert (ws / artifact).is_file(), artifact


def test_时间轴端点返回按说话人分组的时段(client, tmp_path):
    video = make_video(tmp_path, "时间轴.mp4")
    task = submit_task(client, video)
    wait_for(lambda: get_task(client, task["id"])["state"] == "completed")

    resp = client.get(f"/api/tasks/{task['id']}/diarization")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "canned-diarizer"
    assert data["num_speakers"] == 2
    assert data["duration"] == 1.8
    assert [s["label"] for s in data["speakers"]] == ["SPEAKER_00", "SPEAKER_01"]
    sp0 = data["speakers"][0]
    assert sp0["speech_time"] == 1.2, "SPEAKER_00 总说话时长应为 0.7 + 0.5"
    assert sp0["segments"] == [{"start": 0.2, "end": 0.9}, {"start": 1.3, "end": 1.8}]
    assert data["speakers"][1]["segments"] == [{"start": 0.95, "end": 1.25}]


def test_时间轴端点拒绝未知任务与未完成任务(tmp_path):
    with make_client(tmp_path) as client:
        assert client.get("/api/tasks/不存在/diarization").status_code == 404

    sep_gate = threading.Event()
    # separate 停驻在进度回调处，保证 diarize 阶段尚未开始
    with make_client(tmp_path, CannedSeparator(gate=sep_gate)) as client:
        video = make_video(tmp_path, "未完成.mp4")
        task = submit_task(client, video)
        wait_for(lambda: diarize_before(get_task(client, task["id"])))
        resp = client.get(f"/api/tasks/{task['id']}/diarization")
        assert resp.status_code == 404
        sep_gate.set()


def diarize_before(task: dict) -> bool:
    sep = next(s for s in task["stages"] if s["id"] == "separate")
    return sep["status"] == "running"


def test_时段试听端点返回指定区间音频(client, tmp_path):
    video = make_video(tmp_path, "切片.mp4")
    task = submit_task(client, video)
    wait_for(lambda: get_task(client, task["id"])["state"] == "completed")

    resp = client.get(f"/api/tasks/{task['id']}/audio-segment?start=0.2&end=0.9")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/")
    assert resp.content[:4] == b"RIFF", "时段试听端点应返回完整 WAV 音频"


def test_时段试听端点切片时长与区间一致(client, tmp_path):
    video = make_video(tmp_path, "时长.mp4", duration=2)
    task = submit_task(client, video)
    wait_for(lambda: get_task(client, task["id"])["state"] == "completed")

    resp = client.get(f"/api/tasks/{task['id']}/audio-segment?start=0.3&end=1.1")
    assert resp.status_code == 200
    out = tmp_path / "切片结果.wav"
    out.write_bytes(resp.content)
    duration = float(probe(out)["format"]["duration"])
    assert 0.6 <= duration <= 0.9, f"切片应约 0.8s，实测 {duration}"


def test_时段试听端点拒绝非法区间与未知任务(client, tmp_path):
    video = make_video(tmp_path, "校验.mp4")
    task = submit_task(client, video)
    wait_for(lambda: get_task(client, task["id"])["state"] == "completed")
    base = f"/api/tasks/{task['id']}/audio-segment"

    assert client.get(f"{base}?start=0.5&end=0.5").status_code == 400, "空区间应拒绝"
    assert client.get(f"{base}?start=0.9&end=0.2").status_code == 400, "倒序区间应拒绝"
    assert client.get(f"{base}?start=-1&end=0.5").status_code == 400, "负起点应拒绝"
    assert client.get("/api/tasks/不存在/audio-segment?start=0&end=1").status_code == 404


def test_超长时段试听截断到上限而非拒绝(client, tmp_path):
    video = make_video(tmp_path, "截断.mp4", duration=150)
    task = submit_task(client, video)
    wait_for(lambda: get_task(client, task["id"])["state"] == "completed")

    resp = client.get(f"/api/tasks/{task['id']}/audio-segment?start=0&end=100000")
    assert resp.status_code == 200, "点击任意时段都应可试听，超长只截断"
    out = tmp_path / "截断结果.wav"
    out.write_bytes(resp.content)
    duration = float(probe(out)["format"]["duration"])
    assert 115 <= duration <= 125, f"切片应截断到约 120s，实测 {duration}"


def test_零说话人时任务干净失败并显示原因(tmp_path):
    with make_client(tmp_path, diarizer=CannedDiarizer(turns=[])) as client:
        video = make_video(tmp_path, "无人.mp4")
        task = submit_task(client, video)
        wait_for(lambda: get_task(client, task["id"])["state"] == "failed")
        done = get_task(client, task["id"])

        assert "说话人" in done["error"]
        assert done["output_path"] is None
        assert diarize_stage(done)["status"] == "failed"

        # 已完成阶段产物与清单保留
        ws = tmp_path / "workspaces" / "无人"
        assert (ws / "02-separate" / "vocals.wav").is_file()
        manifest = json.loads((ws / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["state"] == "failed"
        assert manifest["stages"][1]["status"] == "done"
        assert manifest["stages"][2]["status"] == "failed"


def test_token缺失时任务失败并显示配置指引(tmp_path):
    message = TaskError(
        "HF token 未配置：pyannote 为门控模型，需设置环境变量 HF_TOKEN"
    )
    with make_client(tmp_path, diarizer=CannedDiarizer(fail=message)) as client:
        video = make_video(tmp_path, "无token.mp4")
        task = submit_task(client, video)
        wait_for(lambda: get_task(client, task["id"])["state"] == "failed")
        done = get_task(client, task["id"])

        assert "HF_TOKEN" in done["error"]
        assert "门控" in done["error"]
        assert diarize_stage(done)["status"] == "failed"
        assert diarize_stage(done)["error"] is not None


def test_说话人分离进度可见且随阶段推进(tmp_path):
    gate = threading.Event()
    with make_client(tmp_path, diarizer=CannedDiarizer(gate=gate)) as client:
        video = make_video(tmp_path, "分离进度.mp4")
        task = submit_task(client, video)
        wait_for(
            lambda: diarize_stage(get_task(client, task["id"]))["status"] == "running"
        )

        def progress_seen():
            stage = diarize_stage(get_task(client, task["id"]))
            return stage["status"] == "running" and (stage["progress"] or 0) >= 0.5

        wait_for(progress_seen, timeout=15)

        gate.set()
        wait_for(lambda: get_task(client, task["id"])["state"] == "completed")
        assert diarize_stage(get_task(client, task["id"]))["progress"] == 1.0


def test_取消运行中的说话人分离阶段(tmp_path):
    gate = threading.Event()
    with make_client(tmp_path, diarizer=CannedDiarizer(gate=gate)) as client:
        video = make_video(tmp_path, "取消分离.mp4")
        task = submit_task(client, video)
        wait_for(
            lambda: diarize_stage(get_task(client, task["id"]))["status"] == "running"
        )

        resp = client.post(f"/api/tasks/{task['id']}/cancel")
        assert resp.status_code == 200
        gate.set()  # diarizer 在进度回调处观察到取消并中止

        wait_for(lambda: get_task(client, task["id"])["state"] == "cancelled")
        done = get_task(client, task["id"])
        assert diarize_stage(done)["status"] == "cancelled"
        assert done["output_path"] is None
