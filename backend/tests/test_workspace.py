"""工作区：以源视频命名，内含阶段产物与阶段清单。"""

import json

from conftest import (
    erase_to_completion,
    get_task,
    make_corrupt_video,
    make_video,
    submit_task,
    wait_for,
)


def test_完成任务的阶段产物与清单落盘(client, tmp_path):
    video = make_video(tmp_path, "电影.mp4")
    task = submit_task(client, video)
    wait_for(lambda: get_task(client, task["id"])["state"] == "completed")
    task = get_task(client, task["id"])

    ws = tmp_path / "workspaces" / "电影"
    assert ws.is_dir(), "工作区目录应以源视频名（不含扩展名）创建"
    manifest_path = ws / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["source"]["path"] == str(video)
    assert manifest["state"] == "completed"
    assert manifest["output"] is None, "分析完成不产成品"
    assert manifest["audio_track"]["index"] == 0
    assert [s["status"] for s in manifest["stages"]] == (
        ["done"] * 5 + ["pending"] * 2  # 分析完成；掩码合成/编码封装待擦除触发
    )

    stage_dirs = [p.name for p in ws.iterdir() if p.is_dir()]
    assert "01-audio-extract" in stage_dirs
    pcm = list((ws / "01-audio-extract").glob("*.wav"))
    assert len(pcm) == 1, "阶段一应产出解码降混后的 PCM wav"

    for entry in manifest["stages"]:
        for artifact in entry["artifacts"]:
            assert (ws / artifact).is_file(), artifact


def test_擦除后清单记录成品路径且与任务一致(client, tmp_path):
    video = make_video(tmp_path, "电影.mp4")
    task = submit_task(client, video)
    wait_for(lambda: get_task(client, task["id"])["state"] == "completed")

    done = erase_to_completion(client, task["id"], ["SPEAKER_00"])
    ws = tmp_path / "workspaces" / "电影"
    manifest = json.loads((ws / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "completed"
    assert manifest["output"]["path"] == done["output_path"]
    assert manifest["erase"]["speakers"] == ["SPEAKER_00"]
    assert [s["status"] for s in manifest["stages"]] == ["done"] * 7
    mix = list((ws / "05-mask-erase").glob("*.wav"))
    assert len(mix) == 1, "掩码合成阶段应产出擦除后的混音 wav"


def test_失败任务保留阶段清单与原因(client, tmp_path):
    video = make_corrupt_video(tmp_path)
    task = submit_task(client, video)
    wait_for(lambda: get_task(client, task["id"])["state"] == "failed")

    manifest = json.loads(
        (tmp_path / "workspaces" / "损坏" / "manifest.json").read_text("utf-8")
    )
    assert manifest["state"] == "failed"
    assert manifest["error"]
