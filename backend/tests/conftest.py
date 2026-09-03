"""共享 fixture：真实 ffmpeg 合成秒级视频 + 走 HTTP API 的测试客户端。

ML 推理走全项目唯一注入边界：默认 client 注入 CannedSeparator（canned
stems）、CannedDiarizer（canned 说话人时段）与 CannedTranscriber（canned
转写文本），真实模型与权重下载永不进入测试回路。
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
    DiarizationResult,
    Inference,
    SeparationStems,
    SpeakerSegment,
    Transcription,
    TranscriptionResult,
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
def make_client(tmp_path: Path, separator=None, diarizer=None, transcriber=None):
    """按需注入自定义 separator/diarizer/transcriber（不传时用默认 canned）
    的测试客户端工厂。"""
    app = create_app(
        workspace_root=tmp_path / "workspaces",
        inference=Inference(
            separator=separator or CannedSeparator(),
            diarizer=diarizer or CannedDiarizer(),
            transcriber=transcriber or CannedTranscriber(),
        ),
    )
    with TestClient(app) as c:
        yield c


class CannedTranscriber:
    """样本转写替身：对每个样本段回一个 canned 台词，不触碰真实模型。

    记录收到的样本段供"转写范围严格限定"断言；fail 模拟生产 transcriber
    已映射完成的干净错误；gate 用于把转写停驻在进度回调处，确定性观察
    进度与取消。
    """

    name = "canned-whisper"

    def __init__(self, fail: Exception | None = None,
                 gate: threading.Event | None = None):
        self.fail = fail
        self.gate = gate
        self.calls: list[tuple[Path, tuple[SpeakerSegment, ...]]] = []

    def transcribe(self, wav: Path, samples, on_progress) -> TranscriptionResult:
        self.calls.append((Path(wav), tuple(samples)))
        if self.fail is not None:
            raise self.fail
        transcripts = tuple(
            Transcription(sample=s, text=f"台词 {i}（{s.speaker}）")
            for i, s in enumerate(samples)
        )
        on_progress(0.5)
        if self.gate is not None:
            self.gate.wait(timeout=60)
        on_progress(1.0)
        return TranscriptionResult(transcripts=transcripts)


class CannedDiarizer:
    """说话人分离替身：返回固定说话人时段，不触碰真实模型。

    fail 模拟生产 diarizer 已映射完成的干净错误；gate 用于把 diarize
    停驻在进度回调处，确定性观察进度与取消。
    """

    name = "canned-diarizer"

    DEFAULT_TURNS = (
        SpeakerSegment("SPEAKER_00", 0.2, 0.9),
        SpeakerSegment("SPEAKER_01", 0.95, 1.25),
        SpeakerSegment("SPEAKER_00", 1.3, 1.8),
    )

    def __init__(self, turns=DEFAULT_TURNS, fail: Exception | None = None,
                 gate: threading.Event | None = None):
        self.turns = tuple(turns)
        self.fail = fail
        self.gate = gate
        self.calls: list[Path] = []

    def diarize(self, wav: Path, out_dir: Path, on_progress) -> DiarizationResult:
        self.calls.append(Path(wav))
        if self.fail is not None:
            raise self.fail
        on_progress(0.5)
        if self.gate is not None:
            self.gate.wait(timeout=60)
        on_progress(1.0)
        return DiarizationResult(segments=self.turns)


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


class NoiseStemsSeparator:
    """粉噪声 stems 替身：人声/伴奏各一路独立粉噪声。

    高熵内容让 AAC 编码器吃满目标码率，供成品码率断言使用（正弦类
    canned 内容编码码率天然低于目标）。
    """

    name = "canned-noise"

    def separate(self, wav: Path, out_dir: Path, on_progress) -> SeparationStems:
        duration = 2
        out_dir.mkdir(parents=True, exist_ok=True)
        vocal = out_dir / VOCALS_FILENAME
        accomp = out_dir / ACCOMPANIMENT_FILENAME
        for path, seed in ((vocal, 1), (accomp, 2)):
            run_ffmpeg([
                "-f", "lavfi", "-i", f"anoisesrc=color=pink:duration={duration}:seed={seed}",
                "-ar", "48000", "-ac", "2", str(path),
            ])
        on_progress(1.0)
        return SeparationStems(vocal=vocal, accompaniment=accomp)


class ToneStemsSeparator:
    """固定音调 stems 替身：人声轨/伴奏轨分别为 3000Hz / 200Hz 全程正弦。

    人声与伴奏内容可区分（分段主频率与分频能量可测量），供掩码语义断言；
    振幅压到 0.4，避免混音求和削波改变波形。
    """

    name = "canned-tones"
    VOCAL_HZ = 3000
    ACCOMP_HZ = 200

    def __init__(self, duration: float = 4.0):
        self.duration = duration

    def separate(self, wav: Path, out_dir: Path, on_progress) -> SeparationStems:
        out_dir.mkdir(parents=True, exist_ok=True)
        vocal = out_dir / VOCALS_FILENAME
        accomp = out_dir / ACCOMPANIMENT_FILENAME
        run_ffmpeg([
            "-f", "lavfi", "-i", f"sine=frequency={self.VOCAL_HZ}:duration={self.duration}",
            "-af", "volume=0.4", "-ar", "48000", str(vocal),
        ])
        run_ffmpeg([
            "-f", "lavfi", "-i", f"sine=frequency={self.ACCOMP_HZ}:duration={self.duration}",
            "-af", "volume=0.4", "-ar", "48000", str(accomp),
        ])
        on_progress(1.0)
        return SeparationStems(vocal=vocal, accompaniment=accomp)


def submit_task(client: TestClient, video: Path, audio_track: int | None = None) -> dict:
    body = {"video_path": str(video)}
    if audio_track is not None:
        body["audio_track"] = audio_track
    resp = client.post("/api/tasks", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def erase(client: TestClient, task_id: str, speakers: list[str],
          output_dir: str | None = None):
    body: dict = {"speakers": speakers}
    if output_dir is not None:
        body["output_dir"] = output_dir
    return client.post(f"/api/tasks/{task_id}/erase", json=body)


def run_to_analysis(client: TestClient, video: Path, audio_track: int | None = None) -> dict:
    """提交任务并等待分析阶段（音轨提取→样本转写）完成。"""
    task = submit_task(client, video, audio_track)
    wait_for(lambda: get_task(client, task["id"])["state"] == "completed")
    return get_task(client, task["id"])


def erase_to_completion(client: TestClient, task_id: str, speakers: list[str],
                        output_dir: str | None = None) -> dict:
    """发起擦除并等待掩码合成与编码封装完成，返回任务快照。"""
    resp = erase(client, task_id, speakers, output_dir)
    assert resp.status_code == 200, resp.text
    wait_for(lambda: get_task(client, task_id)["state"] == "completed")
    return get_task(client, task_id)


def band_mean_volume(path: Path, start: float, end: float, prefilter: str) -> float:
    """指定时间段经预滤波后的平均响度（dB，volumedetect）；比较频带能量用。"""
    proc = subprocess.run(
        [FFMPEG, "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(path),
         "-af", f"{prefilter},volumedetect", "-f", "null", "-"],
        check=True, capture_output=True, text=True,
    )
    for line in proc.stderr.splitlines():
        if "mean_volume:" in line:
            return float(line.split("mean_volume:")[1].replace("dB", "").strip())
    raise AssertionError(f"volumedetect 未输出 mean_volume: {proc.stderr[-400:]}")


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


def dominant_freq(path: Path, start: float = 0.0, end: float | None = None,
                  rate: int = 8000) -> float:
    """解码媒体音频为单声道 PCM，用零交叉估算主频率。

    rate 需显著高于待测频率（零交叉法在接近奈奎斯特频率时失真）；
    start/end 限定测量时间段（掩码语义按分段断言）。
    """
    args = ["-i", str(path)]
    if end is not None:
        args = ["-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", *args]
    proc = subprocess.run(
        [FFMPEG, "-v", "error", *args, "-f", "s16le",
         "-ac", "1", "-ar", str(rate), "-"],
        check=True, capture_output=True,
    )
    data = proc.stdout
    n = len(data) // 2
    samples = struct.unpack(f"<{n}h", data[: n * 2])
    crossings = sum(
        1 for a, b in zip(samples, samples[1:]) if (a < 0) != (b < 0)
    )
    duration = n / rate
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
