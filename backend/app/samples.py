"""样本段挑选：为每个说话人自动选出用于认人的代表性时段。

样本段取自说话人时段，默认偏好时长较长且在全片分散分布的时段；
样本转写阶段只转写这些样本段，刻意不做全片转写（spec #1）。
"""

from __future__ import annotations

from .inference import SpeakerSegment

# 每个说话人的样本段数量上限：认人两三段足够，也控制转写耗时
MAX_SAMPLES_PER_SPEAKER = 3
# 单个样本段的时长上限（秒）：与试听切片上限保持一致，保证台词文字与
# 试听音频对应同一区间；超长时段从段首截断
MAX_SAMPLE_SECONDS = 120.0
# 分散下限：样本段之间的最小间隔，按全片时长的比例取
MIN_GAP_RATIO = 0.1


def select_samples(
    segments: list[SpeakerSegment], duration: float
) -> list[SpeakerSegment]:
    """从说话人时段中为每个说话人挑选样本段（纯函数，可独立验证）。

    每个说话人按时长从长到短考虑：先在"与已选样本间隔不小于
    MIN_GAP_RATIO × 全片时长"的分散约束下挑选；放不满名额时退回
    "不与已选样本重叠"补足。返回按说话人与时间排序的样本段。
    """
    min_gap = MIN_GAP_RATIO * max(float(duration), 0.0)
    samples: list[SpeakerSegment] = []
    for speaker in sorted({s.speaker for s in segments}):
        turns = [s for s in segments if s.speaker == speaker]
        candidates = sorted(
            (_cap(s) for s in turns),
            key=lambda s: (-(s.end - s.start), s.start),
        )
        chosen: list[SpeakerSegment] = []
        for required_gap in (min_gap, 0.0):
            for seg in candidates:
                if len(chosen) >= MAX_SAMPLES_PER_SPEAKER:
                    break
                if seg not in chosen and all(
                    _interval_gap(seg, c) >= required_gap for c in chosen
                ):
                    chosen.append(seg)
        samples.extend(sorted(chosen, key=lambda s: s.start))
    return samples


def _cap(seg: SpeakerSegment) -> SpeakerSegment:
    if seg.end - seg.start <= MAX_SAMPLE_SECONDS:
        return seg
    return SpeakerSegment(seg.speaker, seg.start, seg.start + MAX_SAMPLE_SECONDS)


def _interval_gap(a: SpeakerSegment, b: SpeakerSegment) -> float:
    """两个时间区间的间隔；重叠时为负。"""
    if a.start >= b.end:
        return a.start - b.end
    if b.start >= a.end:
        return b.start - a.end
    return -1.0
