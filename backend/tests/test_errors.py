"""错误边界：路径不存在、无音轨、解码失败 → 任务干净失败并显示原因。"""

from pathlib import Path

from conftest import get_task, make_corrupt_video, make_video, submit_task, wait_for


def test_路径不存在时任务失败并给出原因(client):
    task = submit_task(client, Path("/nonexistent/不存在的视频.mp4"))
    wait_for(lambda: get_task(client, task["id"])["state"] == "failed")
    done = get_task(client, task["id"])
    assert "路径不存在" in done["error"]
    assert done["output_path"] is None


def test_无音轨时任务失败并给出原因(client, tmp_path):
    video = make_video(tmp_path, "无声.mp4", with_audio=False)
    task = submit_task(client, video)
    wait_for(lambda: get_task(client, task["id"])["state"] == "failed")
    done = get_task(client, task["id"])
    assert "没有音轨" in done["error"]


def test_解码失败时任务失败并给出原因(client, tmp_path):
    video = make_corrupt_video(tmp_path)
    task = submit_task(client, video)
    wait_for(lambda: get_task(client, task["id"])["state"] == "failed")
    done = get_task(client, task["id"])
    assert done["error"], "失败任务必须带原因"
    assert done["output_path"] is None
