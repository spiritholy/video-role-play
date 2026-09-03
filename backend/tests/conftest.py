"""共享 fixture：真实 ffmpeg 合成秒级视频 + 走 HTTP API 的测试客户端。"""

import json
import shutil
import struct
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

# 无条件断言工具存在，避免环境问题被误报为行为缺陷
assert FFMPEG and FFPROBE


def run_ffmpeg(args: list[str]) -> None:
    subprocess.run(
        [FFMPEG, "-v", "error", "-y", *args], check=True, capture_output=True
    )


def make_video(
    directory: Path,
    name: str = "源视频.mp4",
    *,
    duration: int = 2,
    audio_freq: int = 440,
    extra_tracks: int = 0,
    with_audio: bool = True,
    video_codec: str = "libx264",
    audio_codec: str = "aac",
    audio_noise: bool = False,
) -> Path:
    """合成一个秒级测试视频；extra_tracks 控制附加音轨数量（频率依次翻倍）。

    audio_noise=True 时用粉噪声代替正弦音（编码器会真正吃满目标码率，
    供成品码率断言使用）。
    """
    path = directory / name
    args = ["-f", "lavfi", "-i", f"testsrc2=size=128x72:rate=10:duration={duration}"]
    if with_audio:
        def audio_source(index: int) -> str:
            if audio_noise:
                return f"anoisesrc=color=pink:duration={duration}"
            return f"sine=frequency={audio_freq * 2 ** index}:duration={duration}"

        args += ["-f", "lavfi", "-i", audio_source(0)]
        for i in range(extra_tracks):
            args += ["-f", "lavfi", "-i", audio_source(i + 1)]
    args += ["-c:v", video_codec, "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if with_audio:
        args += ["-map", "0:v", "-map", "1:a"]
        for i in range(extra_tracks):
            args += ["-map", f"{i + 2}:a"]
        args += ["-c:a", audio_codec, "-shortest"]
    args += [str(path)]
    run_ffmpeg(args)
    return path


def make_corrupt_video(directory: Path, name: str = "损坏.mp4") -> Path:
    """合法 mp4 截断前几百字节，制造解码失败。"""
    good = make_video(directory, "__good.mp4")
    bad = directory / name
    bad.write_bytes(good.read_bytes()[:256])
    good.unlink()
    return bad


def probe(path: Path) -> dict:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(out.stdout)


def wait_for(condition, timeout: float = 30.0, interval: float = 0.05) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(interval)
    raise AssertionError("等待条件超时")


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    app = create_app(workspace_root=tmp_path / "workspaces")
    with TestClient(app) as c:
        yield c


def submit_task(client: TestClient, video: Path, audio_track: int | None = None) -> dict:
    body = {"video_path": str(video)}
    if audio_track is not None:
        body["audio_track"] = audio_track
    resp = client.post("/api/tasks", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def get_task(client: TestClient, task_id: str) -> dict:
    resp = client.get(f"/api/tasks/{task_id}")
    assert resp.status_code == 200, resp.text
    return resp.json()


def audio_bitrate(path: Path) -> int:
    """成品音频的实际码率（bps）。流级码率缺失（如 MKV）时汇总音频包估算。"""
    info = probe(path)
    a = next(s for s in info["streams"] if s["codec_type"] == "audio")
    if a.get("bit_rate"):
        return int(a["bit_rate"])
    index = a["index"]
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", str(index),
         "-show_entries", "packet=size", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    )
    total = sum(int(x) for x in out.stdout.split() if x)
    duration = float(a.get("duration") or info["format"]["duration"])
    assert duration > 0
    return int(total * 8 / duration)


def dominant_freq(path: Path) -> float:
    """解码媒体音频为 8kHz 单声道 PCM，用零交叉估算主频率。"""
    proc = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(path), "-f", "s16le",
         "-ac", "1", "-ar", "8000", "-"],
        check=True, capture_output=True,
    )
    data = proc.stdout
    n = len(data) // 2
    samples = struct.unpack(f"<{n}h", data[: n * 2])
    crossings = sum(
        1 for a, b in zip(samples, samples[1:]) if (a < 0) != (b < 0)
    )
    duration = n / 8000
    return crossings / 2 / duration
