"""PyannoteEmbedder 装配回归：窗口必须以 pyannote.core.Segment 传给 crop。

Inference.crop 内部访问 chunk.start/.end；传 [start, end] 裸列表会在推理期
报 'float' object has no attribute 'start'——这正是命名入库静默失败的根因
（在界面可见化之前，该错误曾被吞掉）。
"""

from pathlib import Path

from pyannote.core import Segment

from app.inference import PyannoteEmbedder


class FakeInference:
    """Inference 替身：记录收到的 chunk，回固定向量。"""

    def __init__(self):
        self.chunks = []

    def crop(self, wav, chunk):
        self.chunks.append((Path(wav), chunk))
        return [1.0, 0.0, 0.0, 0.0]


def test_窗口以Segment传给crop(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    fake = FakeInference()
    embedder = PyannoteEmbedder(inference_factory=lambda _token: fake)

    vectors = embedder.embed(
        tmp_path / "vocals.wav",
        [(0.2, 2.0), (5.0, 9.0)],
        lambda _fraction: None,
    )

    assert vectors == ([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    assert len(fake.chunks) == 2
    for (wav, chunk), (start, end) in zip(fake.chunks, [(0.2, 2.0), (5.0, 9.0)]):
        assert wav == tmp_path / "vocals.wav"
        assert isinstance(chunk, Segment), "crop 需要 Segment，而不是 [start, end] 列表"
        assert (chunk.start, chunk.end) == (start, end)
