"""多音轨：探测列出全部音轨、默认处理第一条、界面可切换。"""

from conftest import (
    dominant_freq,
    get_task,
    make_video,
    submit_task,
    wait_for,
)


def test_探测接口列出全部音轨(client, tmp_path):
    video = make_video(tmp_path, "多音轨.mp4", extra_tracks=1)
    resp = client.post("/api/probe", json={"video_path": str(video)})
    assert resp.status_code == 200
    info = resp.json()
    assert info["container"] == ".mp4"
    assert len(info["tracks"]) == 2
    codecs = {t["codec"] for t in info["tracks"]}
    assert codecs == {"aac"}


def test_默认处理第一条音轨(client, tmp_path):
    video = make_video(tmp_path, "多音轨.mp4", extra_tracks=1, audio_freq=440)
    task = submit_task(client, video)
    wait_for(lambda: get_task(client, task["id"])["state"] == "completed")
    freq = dominant_freq(get_task(client, task["id"])["output_path"])
    assert 400 < freq < 480, f"成品应来自 440Hz 的第一条音轨，实测 {freq:.0f}Hz"


def test_提交时切换音轨(client, tmp_path):
    video = make_video(tmp_path, "多音轨.mp4", extra_tracks=1, audio_freq=440)
    task = submit_task(client, video, audio_track=1)
    wait_for(lambda: get_task(client, task["id"])["state"] == "completed")
    freq = dominant_freq(get_task(client, task["id"])["output_path"])
    assert 800 < freq < 960, f"成品应来自 880Hz 的第二条音轨，实测 {freq:.0f}Hz"
