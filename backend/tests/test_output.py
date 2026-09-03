"""成品视频属性：视频流复制未重编码、AAC 256k 立体声、容器跟随源。

成品由擦除动作生成（掩码合成 + 编码封装）。码率断言注入粉噪声 stems
（高熵内容编码器才吃满目标码率）；容器/流复制断言与具体 stem 内容无关。
"""

import subprocess
from pathlib import Path

from conftest import (
    NoiseStemsSeparator,
    audio_bitrate,
    erase_to_completion,
    get_task,
    make_client,
    make_video,
    probe,
    run_to_analysis,
    wait_for,
)


def make_subtitled_video(directory: Path, name: str = "字幕.mkv") -> Path:
    """带 SRT 字幕轨的 MKV：验证其他流在容器允许时一并复制。"""
    video = make_video(directory, "字幕.mkv", duration=2)
    srt = directory / "字幕.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nHello\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nWorld\n",
        encoding="utf-8",
    )
    tmp = directory / "__sub.mkv"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-i", str(srt),
         "-map", "0", "-map", "1", "-c", "copy", str(tmp)],
        check=True, capture_output=True,
    )
    tmp.replace(video)
    srt.unlink()
    return video


def assert_audio_aac_256k_stereo(out_path):
    info = probe(out_path)
    audio = [s for s in info["streams"] if s["codec_type"] == "audio"]
    assert len(audio) == 1, "成品应只有一条擦除混音音轨"
    a = audio[0]
    assert a["codec_name"] == "aac"
    assert a["channels"] == 2
    assert a["sample_rate"] == "48000"
    bitrate = audio_bitrate(out_path)
    assert 230_000 <= bitrate <= 265_000, f"AAC 码率应约 256k，实测 {bitrate}"


def erase_default_speaker(tmp_path, video):
    with make_client(tmp_path, separator=NoiseStemsSeparator()) as client:
        done = run_to_analysis(client, video)
        return erase_to_completion(client, done["id"], ["SPEAKER_00"])


def test_mp4源_视频复制_音频AAC256k立体声(client, tmp_path):
    video = make_video(tmp_path, "样片.mp4")
    source = probe(video)

    done = erase_default_speaker(tmp_path, video)
    out = done["output_path"]
    assert out == str(video.with_name("样片__erased-SPEAKER_00.mp4"))

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
    video = make_video(tmp_path, "样片.mkv")
    done = erase_default_speaker(tmp_path, video)
    assert done["output_path"] == str(video.with_name("样片__erased-SPEAKER_00.mkv"))
    info = probe(done["output_path"])
    assert "matroska" in info["format"]["format_name"]
    assert_audio_aac_256k_stereo(done["output_path"])


def test_webm源_官方不支持AAC_落mkv(client, tmp_path):
    video = make_video(tmp_path, "样片.webm", video_codec="libvpx", audio_codec="libopus")
    done = erase_default_speaker(tmp_path, video)
    assert done["output_path"] == str(video.with_name("样片__erased-SPEAKER_00.mkv"))
    info = probe(done["output_path"])
    assert "matroska" in info["format"]["format_name"]
    v_out = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert v_out["codec_name"] == "vp8", "WebM 源的视频流同样应复制不重编码"
    assert_audio_aac_256k_stereo(done["output_path"])


def test_mkv源的字幕轨在容器允许时一并复制(client, tmp_path):
    video = make_subtitled_video(tmp_path)
    done = erase_default_speaker(tmp_path, video)
    info = probe(done["output_path"])
    subtitles = [s for s in info["streams"] if s["codec_type"] == "subtitle"]
    assert subtitles, "容器允许时字幕流应随视频流一并复制"
    assert_audio_aac_256k_stereo(done["output_path"])
