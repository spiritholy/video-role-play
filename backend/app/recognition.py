"""认人/纠错：用声纹库给说话人自动命名、合并同人簇、纠正标错段。

纯函数模块：不碰 ML 推理与文件系统。管线负责抽声纹（经推理注入容器的
embedder，一律使用 人声轨 避开背景音乐污染），本模块只做向量比较与时段
修正。算法（spec #9）：每簇取最多 5 个最长时段抽声纹求质心，与库做余弦
相似度 ≥ 匹配阈值才命名；同名多簇合并到说话时长最长的规范标签；对 ≥ 2
秒的时段做逐段验证，若某库名字相似度 ≥ 验证阈值且领先当前归属超过
margin，则重指派回该名字（该名字本集无标签时新建规范标签）。
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .inference import SpeakerSegment

# 每簇参与匹配的最长时段数（spec：最多 5 个）
MAX_SEGMENTS_PER_CLUSTER = 5
# 抽声纹窗口：时段中间 4 秒（时段不足则整段，spec #9）
EMBED_WINDOW_SECONDS = 4.0
# 逐段验证只针对长时段：短段易混、收益低（spec：≥ 2 秒才验证）
VERIFY_MIN_SECONDS = 2.0
# 匹配阈值：簇质心与库名字相似度达到才自动命名
MATCH_THRESHOLD = 0.80
# 验证阈值：单段与库名字相似度达到才考虑重指派
VERIFY_THRESHOLD = 0.85
# 重指派领先幅度：候选名字须领先当前归属超过该 margin，防抖动
REASSIGN_MARGIN = 0.10


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度；全零向量或长度不一致按 0 处理（无匹配信号）。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def centroid(vectors: Sequence[Sequence[float]]) -> list[float]:
    """逐元素平均；空输入返回空列表。"""
    if not vectors:
        return []
    dim = min(len(v) for v in vectors)
    if dim == 0:
        return []
    n = len(vectors)
    return [sum(v[i] for v in vectors) / n for i in range(dim)]


def middle_window(
    seg: SpeakerSegment, seconds: float = EMBED_WINDOW_SECONDS
) -> tuple[float, float]:
    """时段中间 `seconds` 秒的窗口；时段不足则整段。"""
    duration = seg.end - seg.start
    if duration <= seconds:
        return (seg.start, seg.end)
    offset = (duration - seconds) / 2
    return (seg.start + offset, seg.start + offset + seconds)


@dataclass(frozen=True)
class RecognitionPlan:
    """抽声纹窗口规划：簇级 + 逐段验证（含各自命名的用途）。"""

    cluster_windows: tuple[tuple[str, tuple[float, float]], ...] = ()
    verify_windows: tuple[tuple[int, tuple[float, float]], ...] = ()


def plan_embeddings(segments: Sequence[SpeakerSegment]) -> RecognitionPlan:
    """规划抽声纹窗口：每簇最多 5 个最长时段取中间窗口 + 全片 ≥2 秒时段。

    验证窗口按输入时段的下标记录，调用方（管线）负责把窗口向量按原窗口
    键映射回来；窗口去重是调用方的事，纯函数不关心重复。
    """
    by_speaker: dict[str, list[SpeakerSegment]] = defaultdict(list)
    for s in segments:
        by_speaker[s.speaker].append(s)
    cluster: list[tuple[str, tuple[float, float]]] = []
    for label in sorted(by_speaker):
        turns = sorted(
            by_speaker[label], key=lambda s: -(s.end - s.start)
        )[:MAX_SEGMENTS_PER_CLUSTER]
        cluster.extend((label, middle_window(s)) for s in turns)
    verify: list[tuple[int, tuple[float, float]]] = []
    for i, s in enumerate(segments):
        if s.end - s.start >= VERIFY_MIN_SECONDS:
            verify.append((i, middle_window(s)))
    return RecognitionPlan(
        cluster_windows=tuple(cluster), verify_windows=tuple(verify)
    )


@dataclass(frozen=True)
class RecognitionResult:
    """认人/纠错结果：自动命名映射 + 修正后时段。"""

    mapping: dict[str, str]
    segments: tuple[SpeakerSegment, ...]
    library_revision: int
    used_library: bool


def apply_recognition(
    segments: Sequence[SpeakerSegment],
    plan: RecognitionPlan,
    vectors_by_window: dict[tuple[float, float], Sequence[float]],
    name_vectors: dict[str, Sequence[Sequence[float]]],
    library_revision: int,
) -> RecognitionResult:
    """核心算法（纯函数）：簇匹配 → 同名合并 → 逐段验证重指派。

    name_vectors：库中每个名字的声纹向量集（已按当前 embedder 模型过滤，
    由调用方负责）；vectors_by_window：窗口 → 向量（窗口来自 plan）。
    有效库为空时返回恒等（与现状完全一致）。
    """
    name_centroids = {
        name: centroid(vs) for name, vs in name_vectors.items() if vs
    }
    if not name_centroids:
        return RecognitionResult(
            mapping={},
            segments=tuple(segments),
            library_revision=library_revision,
            used_library=False,
        )

    # 簇质心：每簇在 plan 里的窗口向量求平均
    cluster_pool: dict[str, list[list[float]]] = defaultdict(list)
    for label, window in plan.cluster_windows:
        v = vectors_by_window.get(window)
        if v is not None and len(v):
            cluster_pool[label].append(list(v))
    cluster_centroids = {
        label: centroid(vs) for label, vs in cluster_pool.items()
    }

    # 第 1 步：每簇匹配库名字（≥ 匹配阈值才命名；匹配不上保持未知名）
    matched: dict[str, tuple[str, float]] = {}
    for label, cv in cluster_centroids.items():
        best = _best_match(cv, name_centroids)
        if best is not None and best[1] >= MATCH_THRESHOLD:
            matched[label] = best

    # 第 2 步：同名多簇合并到说话时长最长的规范标签
    speech_time: dict[str, float] = defaultdict(float)
    for s in segments:
        speech_time[s.speaker] += s.end - s.start
    name_groups: dict[str, list[str]] = defaultdict(list)
    for label, (name, _sim) in matched.items():
        name_groups[name].append(label)
    canonical_of: dict[str, str] = {}
    for name, labels in name_groups.items():
        canonical_of[name] = max(labels, key=lambda label: speech_time[label])

    effective_of: dict[str, str] = {}
    for label, (name, _sim) in matched.items():
        effective_of[label] = canonical_of[name]

    # 规范标签 → 自动名字（先登记簇级匹配；重指派会补充新名字）
    mapping: dict[str, str] = {
        canonical: name for name, canonical in canonical_of.items()
    }
    label_of_name: dict[str, str] = dict(canonical_of)

    verify_by_index = dict(plan.verify_windows)

    # 第 3 步：逐段验证与重指派
    corrected: list[SpeakerSegment] = []
    for i, seg in enumerate(segments):
        eff = effective_of.get(seg.speaker, seg.speaker)
        window = verify_by_index.get(i)
        v = vectors_by_window.get(window) if window is not None else None
        if v is not None and (seg.end - seg.start) >= VERIFY_MIN_SECONDS:
            best = _best_match(list(v), name_centroids)
            if best is not None:
                cand_name, cand_sim = best
                if cand_sim >= VERIFY_THRESHOLD:
                    current_name = matched.get(seg.speaker, (None, 0.0))[0]
                    current_sim = (
                        cosine_similarity(v, name_centroids[current_name])
                        if current_name is not None
                        else 0.0
                    )
                    if cand_sim - current_sim > REASSIGN_MARGIN:
                        eff = label_of_name.setdefault(cand_name, cand_name)
                        mapping[eff] = cand_name
        corrected.append(SpeakerSegment(eff, seg.start, seg.end))

    return RecognitionResult(
        mapping=mapping,
        segments=tuple(corrected),
        library_revision=library_revision,
        used_library=True,
    )


def _best_match(
    vector: Sequence[float], name_centroids: dict[str, list[float]]
) -> tuple[str, float] | None:
    """与库中全部名字比较，返回相似度最高的 (名字, 相似度)；空库返回 None。"""
    best: tuple[str, float] | None = None
    for name, nc in name_centroids.items():
        sim = cosine_similarity(vector, nc)
        if best is None or sim > best[1]:
            best = (name, sim)
    return best
