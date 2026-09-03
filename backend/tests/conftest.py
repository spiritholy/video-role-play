"""共享 fixture：真实 ffmpeg 合成秒级视频 + 走 HTTP API 的测试客户端。

ML 推理走全项目唯一注入边界：默认 client 注入 CannedSeparator（canned
stems），真实模型与权重下载永不进入测试回路。
"""

import json
import shutil
import struct
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.inference import (
    ACCOMPANIMENT_FILENAME,
    VOCALS_FILENAME,
    Inference,
    SeparationStems,
)
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
    with make_client(tmp_path) as c:
        yield c


@contextmanager
def make_client(tmp_path: Path, separator=None):
    """按需注入自定义 separator（不传时用默认 canned）的测试客户端工厂。"""
    app = create_app(
        workspace_root=tmp_path / "workspaces",
        inference=Inference(separator=separator or CannedSeparator()),
    )
    with TestClient(app) as c:
        yield c


class CannedSeparator:
    """注入边界的替身：用 ffmpeg 高/低通生成可区分的人声/伴奏 wav。

    fail 时抛出的异常模拟真实模型加载/推理失败；gate 用于把分离器
    停驻在进度回调处，确定性观察进度与取消。
    """

    name = "canned-highpass"

    def __init__(self, fail: Exception | None = None, gate: threading.Event | None = None):
        self.fail = fail
        self.gate = gate
        self.calls: list[Path] = []

    def separate(self, wav: Path, out_dir: Path, on_progress) -> SeparationStems:
        self.calls.append(Path(wav))
        if self.fail is not None:
            raise self.fail
        out_dir.mkdir(parents=True, exist_ok=True)
        vocal = out_dir / VOCALS_FILENAME
        accomp = out_dir / ACCOMPANIMENT_FILENAME
        run_ffmpeg(["-i", str(wav), "-af", "highpass=f=4000", str(vocal)])
        run_ffmpeg(["-i", str(wav), "-af", "lowpass=f=200", str(accomp)])
        on_progress(0.5)
        if self.gate is not None:
            self.gate.wait(timeout=60)
        on_progress(1.0)
        return SeparationStems(vocal=vocal, accompaniment=accomp)


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


def mean_volume(path: Path) -> float:
    """平均响度（dB，ffmpeg volumedetect）；越小越安静。"""
    proc = subprocess.run(
        [FFMPEG, "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        check=True, capture_output=True, text=True,
    )
    for line in proc.stderr.splitlines():
        if "mean_volume:" in line:
            return float(line.split("mean_volume:")[1].replace("dB", "").strip())
    raise AssertionError(f"volumedetect 未输出 mean_volume: {proc.stderr[-400:]}")
