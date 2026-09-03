"""样本转写阶段：产物落盘、转写端点、范围限定、进度/取消/失败边界。

转写经全项目唯一注入边界（ML 推理接口）注入 canned 转写文本，并记录
收到的样本段以断言"转写范围严格限定样本段、不覆盖全片"；真实 Whisper
与权重下载不进测试回路。
"""

import json
import threading

from app.media import TaskError
from conftest import (
    CannedSeparator,
    CannedTranscriber,
    get_task,
    make_client,
    make_video,
    submit_task,
    wait_for,
)

TRANSCRIPTS = "04-transcribe/transcripts.json"


def transcribe_stage(task: dict) -> dict:
    return next(s for s in task["stages"] if s["id"] == "transcribe")


def separate_running(task: dict) -> bool:
    sep = next(s for s in task["stages"] if s["id"] == "separate")
    return sep["status"] == "running"


def test_完成后工作区存在转写产物且清单更新(client, tmp_path):
    video = make_video(tmp_path, "认台词.mp4")
    task = submit_task(client, video)
    wait_for(lambda: get_task(client, task["id"])["state"] == "completed")
    done = get_task(client, task["id"])

    stage = transcribe_stage(done)
    assert stage["status"] == "done"
    assert stage["outputs"] == {"transcripts": TRANSCRIPTS}

    ws = tmp_path / "workspaces" / "认台词"
    assert (ws / TRANSCRIPTS).is_file(), "转写产物应落盘"
    data = json.loads((ws / TRANSCRIPTS).read_text(encoding="utf-8"))
    assert data["model"] == "canned-whisper"
    assert [
        (s["speaker"], s["start"], s["end"], s["text"]) for s in data["samples"]
    ] == [
        ("SPEAKER_00", 0.2, 0.9, "台词 0（SPEAKER_00）"),
        ("SPEAKER_00", 1.3, 1.8, "台词 1（SPEAKER_00）"),
        ("SPEAKER_01", 0.95, 1.25, "台词 2（SPEAKER_01）"),
    ]

    manifest = json.loads((ws / "manifest.json").read_text(encoding="utf-8"))
    assert [s["status"] for s in manifest["stages"]] == ["done"] * 5
    assert manifest["transcription"] == {
        "model": "canned-whisper",
        "samples": TRANSCRIPTS,
    }
    for entry in manifest["stages"]:
        for artifact in entry["artifacts"]:
            assert (ws / artifact).is_file(), artifact


def test_转写范围严格限定在样本段不覆盖全片(tmp_path):
    transcriber = CannedTranscriber()
    with make_client(tmp_path, transcriber=transcriber) as client:
        video = make_video(tmp_path, "长片.mp4", duration=60)
        task = submit_task(client, video)
        wait_for(lambda: get_task(client, task["id"])["state"] == "completed")

        assert len(transcriber.calls) == 1
        wav, samples = transcriber.calls[0]
        assert wav.name.startswith("track"), "转写应作用于提取出的源音频"
        # canned 说话人时段：SPEAKER_00 0.2–0.9 / 1.3–1.8，SPEAKER_01 0.95–1.25
        assert [
            (s.speaker, s.start, s.end)
            for s in sorted(samples, key=lambda s: (s.speaker, s.start))
        ] == [
            ("SPEAKER_00", 0.2, 0.9),
            ("SPEAKER_00", 1.3, 1.8),
            ("SPEAKER_01", 0.95, 1.25),
        ], "收到的样本段应为说话人时段的子集（较长且分散）"
        total = sum(s.end - s.start for s in samples)
        assert total * 2 < 60, "转写范围不得接近全片时长"


def test_转写端点返回每个说话人的样本台词(client, tmp_path):
    video = make_video(tmp_path, "台词端点.mp4")
    task = submit_task(client, video)
    wait_for(lambda: get_task(client, task["id"])["state"] == "completed")

    resp = client.get(f"/api/tasks/{task['id']}/transcripts")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "canned-whisper"
    assert [s["speaker"] for s in data["samples"]] == [
        "SPEAKER_00", "SPEAKER_00", "SPEAKER_01",
    ], "样本应按说话人与时间排序"
    assert all(s["text"] for s in data["samples"]), "每个样本段都应有文字台词"


def test_转写端点拒绝未知任务与未完成转写(tmp_path):
    with make_client(tmp_path) as client:
        assert client.get("/api/tasks/不存在/transcripts").status_code == 404

    sep_gate = threading.Event()
    # separate 停驻在进度回调处，保证 transcribe 阶段尚未开始
    with make_client(tmp_path, separator=CannedSeparator(gate=sep_gate)) as client:
        video = make_video(tmp_path, "未完成.mp4")
        task = submit_task(client, video)
        wait_for(lambda: separate_running(get_task(client, task["id"])))
        resp = client.get(f"/api/tasks/{task['id']}/transcripts")
        assert resp.status_code == 404
        sep_gate.set()


def test_转写阶段进度可见且随阶段推进(tmp_path):
    gate = threading.Event()
    with make_client(tmp_path, transcriber=CannedTranscriber(gate=gate)) as client:
        video = make_video(tmp_path, "转写进度.mp4")
        task = submit_task(client, video)
        wait_for(
            lambda: transcribe_stage(get_task(client, task["id"]))["status"]
            == "running"
        )

        def progress_seen():
            stage = transcribe_stage(get_task(client, task["id"]))
            return stage["status"] == "running" and (stage["progress"] or 0) >= 0.5

        wait_for(progress_seen, timeout=15)

        gate.set()
        wait_for(lambda: get_task(client, task["id"])["state"] == "completed")
        assert transcribe_stage(get_task(client, task["id"]))["progress"] == 1.0


def test_取消运行中的转写阶段(tmp_path):
    gate = threading.Event()
    with make_client(tmp_path, transcriber=CannedTranscriber(gate=gate)) as client:
        video = make_video(tmp_path, "取消转写.mp4")
        task = submit_task(client, video)
        wait_for(
            lambda: transcribe_stage(get_task(client, task["id"]))["status"]
            == "running"
        )

        resp = client.post(f"/api/tasks/{task['id']}/cancel")
        assert resp.status_code == 200
        gate.set()  # transcriber 在进度回调处观察到取消并中止

        wait_for(lambda: get_task(client, task["id"])["state"] == "cancelled")
        done = get_task(client, task["id"])
        assert transcribe_stage(done)["status"] == "cancelled"
        assert done["output_path"] is None


def test_转写失败时任务干净失败并显示原因(tmp_path):
    message = TaskError("Whisper 模型权重获取失败：网络不可达")
    with make_client(tmp_path, transcriber=CannedTranscriber(fail=message)) as client:
        video = make_video(tmp_path, "转写失败.mp4")
        task = submit_task(client, video)
        wait_for(lambda: get_task(client, task["id"])["state"] == "failed")
        done = get_task(client, task["id"])

        assert "Whisper" in done["error"]
        assert transcribe_stage(done)["status"] == "failed"
        assert transcribe_stage(done)["error"] is not None
        assert done["output_path"] is None

        # 已完成阶段产物与清单保留
        ws = tmp_path / "workspaces" / "转写失败"
        assert (ws / "03-diarize" / "diarization.json").is_file()
        manifest = json.loads((ws / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["state"] == "failed"
        statuses = {s["id"]: s["status"] for s in manifest["stages"]}
        assert statuses["diarize"] == "done"
        assert statuses["transcribe"] == "failed"
