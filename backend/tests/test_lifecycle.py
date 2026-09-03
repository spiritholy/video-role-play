"""任务生命周期：提交 → 运行 → 完成（tracer bullet 第一片）。"""

from pathlib import Path


def test_提交源视频创建任务并运行到完成(client, tmp_path):
    from conftest import get_task, make_video, submit_task, wait_for

    video = make_video(tmp_path, "样本.mp4")

    task = submit_task(client, video)
    assert task["state"] in ("queued", "running")
    assert task["video_path"] == str(video)
    assert task["error"] is None

    wait_for(lambda: get_task(client, task["id"])["state"] == "completed")
    done = get_task(client, task["id"])
    assert done["error"] is None
    assert done["output_path"] is not None
    assert Path(done["output_path"]).exists()

    # 状态与阶段轨迹完整
    stages = {s["id"]: s["status"] for s in done["stages"]}
    assert stages == {
        "audio_extract": "done",
        "separate": "done",
        "encode_mux": "done",
    }
