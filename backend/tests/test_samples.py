"""样本段挑选策略：注入 canned 说话人时段与固定音频时长，断言所选
样本为"较长且分散"的时段，并验证截断上限、名额上限与排序。

纯函数验证（spec #1 测试决策）：不触碰管线与真实模型。
"""

from app.inference import SpeakerSegment
from app.samples import (
    MAX_SAMPLE_SECONDS,
    MAX_SAMPLES_PER_SPEAKER,
    select_samples,
)


def seg(speaker: str, start: float, end: float) -> SpeakerSegment:
    return SpeakerSegment(speaker, start, end)


def test_优先挑选较长的时段():
    turns = [seg("S1", 10, 12), seg("S1", 0, 8), seg("S1", 20, 23)]
    samples = select_samples(turns, duration=100)
    assert samples[0] == seg("S1", 0, 8), "最长的 8s 时段应入选"


def test_分散约束让远处时段胜过扎堆的更长时段():
    # duration=600 → 分散下限 60s：100–200 与 120–240 扎堆互斥，
    # 500–560 虽更短但因分散入选
    turns = [seg("S1", 100, 200), seg("S1", 120, 240), seg("S1", 500, 560)]
    samples = select_samples(turns, duration=600)
    assert samples == [seg("S1", 120, 240), seg("S1", 500, 560)]
    assert seg("S1", 100, 200) not in samples, "扎堆时段不应重复覆盖"


def test_分散放不满时退回最长补足且互不重叠():
    # duration=600 → 分散下限 60s：110–200 与 0–100 仅隔 10s 不满足
    # 分散，但补足回退后仍应入选（不重叠）；20–60 与 0–100 重叠被跳过
    turns = [seg("S1", 0, 100), seg("S1", 110, 200), seg("S1", 20, 60)]
    samples = select_samples(turns, duration=600)
    assert samples == [seg("S1", 0, 100), seg("S1", 110, 200)]


def test_样本之间互不重叠():
    turns = [seg("S1", 100, 200), seg("S1", 120, 240), seg("S1", 150, 300)]
    samples = select_samples(turns, duration=600)
    for i, a in enumerate(samples):
        for b in samples[i + 1:]:
            assert a.end <= b.start or b.end <= a.start, "样本段不应重叠"


def test_超长时段截断到上限():
    turns = [seg("S1", 30, 330)]
    samples = select_samples(turns, duration=600)
    assert samples == [seg("S1", 30, 30 + MAX_SAMPLE_SECONDS)]


def test_每个说话人最多三个样本():
    turns = [
        seg("S1", 0, 100),
        seg("S1", 1000, 1100),
        seg("S1", 2000, 2100),
        seg("S1", 3000, 3100),
        seg("S1", 4000, 4100),
    ]
    samples = select_samples(turns, duration=6000)
    assert len(samples) == MAX_SAMPLES_PER_SPEAKER
    assert samples[0] == seg("S1", 0, 100), "同长时段按时间先后取最早的"
    assert samples[-1] == seg("S1", 2000, 2100)


def test_多说话人各自独立挑选并按说话人与时间排序():
    turns = [
        seg("S2", 1000, 1010),
        seg("S1", 500, 560),
        seg("S2", 0, 20),
        seg("S1", 0, 100),
    ]
    samples = select_samples(turns, duration=2000)
    speakers = [s.speaker for s in samples]
    assert speakers == sorted(speakers)
    for speaker in set(speakers):
        spans = [s.start for s in samples if s.speaker == speaker]
        assert spans == sorted(spans)
    assert seg("S1", 0, 100) in samples
    assert seg("S2", 1000, 1010) in samples
