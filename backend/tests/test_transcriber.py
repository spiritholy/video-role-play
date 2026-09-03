"""WhisperTranscriber：切片严格限定样本段、文本映射、进度与错误指引。

单元接缝注入模型构造与音频解码（真实 faster-whisper 与权重下载不进
测试回路）；转写行为本身由 API 级测试经统一注入边界（canned 转写文本）
覆盖。
"""

from pathlib import Path

import pytest

from app.inference import (
    WHISPER_SAMPLE_RATE,
    SpeakerSegment,
    Transcription,
    WhisperTranscriber,
)
from app.media import TaskError

SR = WHISPER_SAMPLE_RATE


class _FakeSeg:
    def __init__(self, text: str):
        self.text = text


class _FakeModel:
    def __init__(self, texts=(" Hello world. ",)):
        self.texts = texts
        self.chunks = []

    def transcribe(self, chunk):
        self.chunks.append(chunk)
        return iter([_FakeSeg(t) for t in self.texts]), None


def make_transcriber(model, audio_seconds: float) -> WhisperTranscriber:
    audio = [0.0] * int(audio_seconds * SR)
    return WhisperTranscriber(
        model_factory=lambda: model,
        audio_loader=lambda wav: audio,
    )


def test_只切片样本段不覆盖全片():
    model = _FakeModel()
    # 10s 音频，样本段仅 1.0–2.5 与 8.0–8.5
    t = make_transcriber(model, audio_seconds=10)
    result = t.transcribe(
        Path("任意.wav"),
        [SpeakerSegment("S1", 1.0, 2.5), SpeakerSegment("S1", 8.0, 8.5)],
        lambda f: None,
    )
    assert [len(c) for c in model.chunks] == [int(1.5 * SR), int(0.5 * SR)], (
        "送入模型的采样点应与样本段窗口一致"
    )
    assert result.transcripts == (
        Transcription(SpeakerSegment("S1", 1.0, 2.5), "Hello world."),
        Transcription(SpeakerSegment("S1", 8.0, 8.5), "Hello world."),
    )


def test_多段文本拼接并去除首尾空白():
    model = _FakeModel(texts=(" Hello,", " world "))
    t = make_transcriber(model, audio_seconds=2)
    result = t.transcribe(
        Path("任意.wav"), [SpeakerSegment("S1", 0.0, 1.0)], lambda f: None,
    )
    assert result.transcripts[0].text == "Hello, world"


def test_进度按样本段推进():
    reported = []
    t = make_transcriber(_FakeModel(), audio_seconds=2)
    t.transcribe(
        Path("任意.wav"),
        [SpeakerSegment("S1", 0.0, 0.5), SpeakerSegment("S1", 1.0, 1.5)],
        reported.append,
    )
    assert reported == [pytest.approx(0.5), pytest.approx(1.0)]


def test_权重下载失败映射为明确错误():
    def broken_factory():
        raise RuntimeError("网络不可达")

    with pytest.raises(TaskError) as ei:
        WhisperTranscriber(model_factory=broken_factory).load_model()
    message = str(ei.value)
    assert "权重获取失败" in message
    assert "无需账号" in message


def test_工厂抛出的干净错误原样透传不被二次包装():
    sentinel = TaskError("HF token 未配置")

    def failing_factory():
        raise sentinel

    with pytest.raises(TaskError) as ei:
        WhisperTranscriber(model_factory=failing_factory).load_model()
    assert ei.value is sentinel


def test_加载成功则缓存模型只构造一次():
    sentinel = object()
    calls = []

    def factory():
        calls.append(1)
        return sentinel

    t = WhisperTranscriber(model_factory=factory)
    assert t.load_model() is sentinel
    assert t.load_model() is sentinel
    assert len(calls) == 1, "模型应只构造一次（权重只下载一次）"
