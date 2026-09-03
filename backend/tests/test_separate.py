"""人声/伴奏分离阶段：产物落盘、试听端点、失败/取消状态、阶段进度。

分离经全项目唯一注入边界（ML 推理接口）注入 canned stems，不触碰真实
模型、不下载权重；真实 ffmpeg 只负责生成可区分的 canned 音频内容。
"""

import json
import threading

from app.media import TaskError
from conftest import (
    dominant_freq,
    get_task,
    make_video,
    mean_volume,
    submit_task,
    wait_for,
)

VOCAL = "02-separate/vocals.wav"
ACCOMP = "02-separate/accompaniment.wav"


def separate_stage(task: dict) -> dict:
    return next(s for s in task["stages"] if s["id"] == "separate")


def test_完成后工作区存在人声轨与伴奏轨且清单更新(client, tmp_path):
    video = make_video(tmp_path, "电影.mp4")
    task = submit_task(client, video)
    wait_for(lambda: get_task(client, task["id"])["state"] == "completed")
    done = get_task(client, task["id"])

    sep = separate_stage(done)
    assert sep["status"] == "done"
    assert sep["outputs"] == {"vocal": VOCAL, "accompaniment": ACCOMP}

    ws = tmp_path / "workspaces" / "电影"
    vocal = ws / VOCAL
    accomp = ws / ACCOMP
    assert vocal.is_file(), "人声轨产物应落盘"
    assert accomp.is_file(), "伴奏轨产物应落盘"

    # canned stems 内容可区分：人声轨滤掉 440Hz（近无声），伴奏轨保留 440Hz
    assert 400 < dominant_freq(accomp) < 480
    assert mean_volume(vocal) < mean_volume(accomp) - 18, "人声/伴奏应可区分"

    manifest = json.loads((ws / "manifest.json").read_text(encoding="utf-8"))
    assert [s["status"] for s in manifest["stages"]] == [
        "done", "done", "done", "done",
    ]
    assert manifest["separation"]["vocal"] == VOCAL
    assert manifest["separation"]["accompaniment"] == ACCOMP
    for entry in manifest["stages"]:
        for artifact in entry["artifacts"]:
            assert (ws / artifact).is_file(), artifact


def test_试听端点返回人声与伴奏音频(client, tmp_path):
    video = make_video(tmp_path, "试听.mp4")
    task = submit_task(client, video)
    wait_for(lambda: get_task(client, task["id"])["state"] == "completed")
    outputs = separate_stage(get_task(client, task["id"]))["outputs"]

    for artifact in (outputs["vocal"], outputs["accompaniment"]):
        resp = client.get(f"/api/tasks/{task['id']}/audio/{artifact}")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("audio/")
        assert resp.content[:4] == b"RIFF", "试听端点应返回完整 WAV 音频"


def test_试听端点拒绝未知任务与未知产物(client, tmp_path):
    video = make_video(tmp_path, "安全.mp4")
    task = submit_task(client, video)
    wait_for(lambda: get_task(client, task["id"])["state"] == "completed")

    assert client.get("/api/tasks/不存在/audio/" + VOCAL).status_code == 404
    base = f"/api/tasks/{task['id']}/audio"
    assert client.get(f"{base}/02-separate/不存在.wav").status_code == 404
    assert client.get(f"{base}/../../etc/passwd").status_code in (400, 404)
    # 未被列为阶段产物的文件即使存在于工作区也不可访问
    ws = tmp_path / "workspaces" / "安全"
    (ws / "01-audio-extract" / "track0.wav").write_bytes(b"RIFF....")
    assert client.get(f"{base}/01-audio-extract/track0.wav").status_code == 404


def test_分离阶段失败时任务失败且原因可见(tmp_path):
    from conftest import CannedSeparator, make_client

    with make_client(
        tmp_path,
        CannedSeparator(fail=TaskError("Demucs 模型权重下载失败: 连接超时")),
    ) as client:
        video = make_video(tmp_path, "失败.mp4")
        task = submit_task(client, video)
        wait_for(lambda: get_task(client, task["id"])["state"] == "failed")
        done = get_task(client, task["id"])

        assert "下载失败" in done["error"]
        assert done["output_path"] is None
        sep = separate_stage(done)
        assert sep["status"] == "failed"
        assert "下载失败" in sep["error"]

        # 已完成阶段的产物与清单保留
        ws = tmp_path / "workspaces" / "失败"
        assert (ws / "01-audio-extract").is_dir()
        manifest = json.loads((ws / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["state"] == "failed"
        assert manifest["stages"][0]["status"] == "done"
        assert manifest["stages"][1]["status"] == "failed"


def test_分离阶段进度可见且随阶段推进(tmp_path):
    from conftest import CannedSeparator, make_client

    gate = threading.Event()
    with make_client(tmp_path, CannedSeparator(gate=gate)) as client:
        video = make_video(tmp_path, "进度.mp4")
        task = submit_task(client, video)
        wait_for(lambda: get_task(client, task["id"])["state"] == "running")

        def progress_seen():
            sep = separate_stage(get_task(client, task["id"]))
            return sep["status"] == "running" and (sep["progress"] or 0) >= 0.5

        wait_for(progress_seen, timeout=15)

        gate.set()
        wait_for(lambda: get_task(client, task["id"])["state"] == "completed")
        sep = separate_stage(get_task(client, task["id"]))
        assert sep["progress"] == 1.0


def test_取消运行中的分离阶段(tmp_path):
    from conftest import CannedSeparator, make_client

    gate = threading.Event()
    with make_client(tmp_path, CannedSeparator(gate=gate)) as client:
        video = make_video(tmp_path, "取消分离.mp4")
        task = submit_task(client, video)
        wait_for(lambda: get_task(client, task["id"])["state"] == "running")
        wait_for(lambda: separate_stage(get_task(client, task["id"]))["status"] == "running")

        resp = client.post(f"/api/tasks/{task['id']}/cancel")
        assert resp.status_code == 200
        gate.set()  # 分离器在进度回调处观察到取消并中止

        wait_for(lambda: get_task(client, task["id"])["state"] == "cancelled")
        done = get_task(client, task["id"])
        sep = separate_stage(done)
        assert sep["status"] == "cancelled"
        assert done["output_path"] is None
