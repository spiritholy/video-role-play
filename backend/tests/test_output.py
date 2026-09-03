"""成品视频属性：视频流复制未重编码、AAC 256k 立体声、容器跟随源。"""

from conftest import (
    audio_bitrate,
    get_task,
    make_video,
    probe,
    submit_task,
    wait_for,
)


def run_to_completion(client, video):
    task = submit_task(client, video)
    wait_for(lambda: get_task(client, task["id"])["state"] == "completed")
    return get_task(client, task["id"])


def assert_audio_aac_256k_stereo(out_path):
    info = probe(out_path)
    audio = [s for s in info["streams"] if s["codec_type"] == "audio"]
    assert len(audio) == 1
    a = audio[0]
    assert a["codec_name"] == "aac"
    assert a["channels"] == 2
    assert a["sample_rate"] == "48000"
    bitrate = audio_bitrate(out_path)
    assert 230_000 <= bitrate <= 265_000, f"AAC 码率应约 256k，实测 {bitrate}"


def test_mp4源_视频复制_音频AAC256k立体声(client, tmp_path):
    video = make_video(tmp_path, "样片.mp4", audio_noise=True)
    source = probe(video)

    done = run_to_completion(client, video)
    out = done["output_path"]
    assert out == str(video.with_name("样片__baseline.mp4"))

    info = probe(out)
    v_src = next(s for s in source["streams"] if s["codec_type"] == "video")
    v_out = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert v_out["codec_name"] == v_src["codec_name"] == "h264"
    assert v_out["width"] == v_src["width"]
    assert v_out["height"] == v_src["height"]
    assert v_out["nb_frames"] == v_src["nb_frames"], "视频流应整段复制不重编码"
    assert "mp4" in info["format"]["format_name"].split(",")
    assert_audio_aac_256k_stereo(out)


def test_mkv源_容器跟随出mkv(client, tmp_path):
    video = make_video(tmp_path, "样片.mkv", audio_noise=True)
    done = run_to_completion(client, video)
    assert done["output_path"] == str(video.with_name("样片__baseline.mkv"))
    info = probe(done["output_path"])
    assert "matroska" in info["format"]["format_name"]
    assert_audio_aac_256k_stereo(done["output_path"])


def test_webm源_官方不支持AAC_落mkv(client, tmp_path):
    video = make_video(
        tmp_path, "样片.webm",
        video_codec="libvpx", audio_codec="libopus", audio_noise=True,
    )
    done = run_to_completion(client, video)
    assert done["output_path"] == str(video.with_name("样片__baseline.mkv"))
    info = probe(done["output_path"])
    assert "matroska" in info["format"]["format_name"]
    v_out = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert v_out["codec_name"] == "vp8", "WebM 源的视频流同样应复制不重编码"
    assert_audio_aac_256k_stereo(done["output_path"])
