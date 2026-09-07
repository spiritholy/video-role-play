"""认人/纠错纯函数：向量比较、抽声纹窗口规划、匹配/合并/重指派算法。

算法是 spec #9 的核心，作为纯函数直接单测（先例：samples.select_samples）。
只断言外部可观察行为（修正后时段、自动命名映射、阈值/margin 边界），
不触碰 ML 推理与文件系统。
"""

import pytest

from app.inference import SpeakerSegment
from app.recognition import (
    RecognitionPlan,
    apply_recognition,
    centroid,
    cosine_similarity,
    middle_window,
    plan_embeddings,
)


def seg(label, start, end):
    return SpeakerSegment(label, start, end)


# 独热基向量：第 i 维为 1，其余为 0，相似度断言平凡（同维=1，异维=0）
def oh(dim: int, size: int = 4) -> list[float]:
    v = [0.0] * size
    v[dim] = 1.0
    return v


def test_余弦相似度_中心化_窗口():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 2.0], [2.0, 4.0]) == pytest.approx(1.0)
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [0.0]) == 0.0, "长度不一致按 0"
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0, "全零向量按 0"

    assert centroid([[1.0, 3.0], [3.0, 1.0]]) == [2.0, 2.0]
    assert centroid([]) == []
    assert centroid([[1.0, 2.0]]) == [1.0, 2.0]

    assert middle_window(seg("S", 0.0, 10.0)) == (3.0, 7.0), "中间 4 秒"
    assert middle_window(seg("S", 1.0, 2.0)) == (1.0, 2.0), "不足 4 秒取整段"


def test_plan_embeddings_每簇最长时段与逐段验证窗口():
    turns = [
        seg("A", 0.0, 1.0),    # 1 秒：不进验证
        seg("A", 2.0, 9.0),    # 7 秒：簇 + 验证
        seg("A", 10.0, 15.0),  # 5 秒：簇 + 验证
    ]
    plan = plan_embeddings(turns)
    assert [label for label, _w in plan.cluster_windows] == ["A", "A", "A"], (
        "每簇取最长时段（3 个都取）"
    )
    cluster_windows = {w for _l, w in plan.cluster_windows}
    assert middle_window(turns[1]) in cluster_windows
    assert middle_window(turns[2]) in cluster_windows

    verify = dict(plan.verify_windows)
    assert set(verify) == {1, 2}, "≥2 秒的时段逐段验证；1 秒段跳过"
    assert verify[1] == middle_window(turns[1])
    assert verify[2] == middle_window(turns[2])


def test_plan_embeddings_全短段不验证_每簇超过5段只取最长5段():
    short = [seg("A", 0.0, 0.5), seg("B", 1.0, 1.5)]
    plan = plan_embeddings(short)
    assert plan.verify_windows == (), "全部短于 2 秒：不验证"
    assert {label for label, _w in plan.cluster_windows} == {"A", "B"}

    many = [seg("A", float(i), float(i) + 0.6) for i in range(7)]
    assert len([l for l, _w in plan_embeddings(many).cluster_windows if l == "A"]) == 5


def _apply(segments, name_vectors, vectors, revision=7):
    plan = plan_embeddings(segments)
    return apply_recognition(segments, plan, vectors, name_vectors, revision)


def test_空库恒等_不改时段且标记未用库():
    segments = [seg("A", 0.0, 3.0), seg("B", 4.0, 7.0)]
    result = _apply(segments, {}, {})
    assert result.mapping == {}
    assert result.segments == tuple(segments), "空库时修正后时段与原始一致"
    assert result.used_library is False
    assert result.library_revision == 7


def test_匹配命中_簇质心与库名字一致则自动命名():
    segments = [seg("A", 0.0, 4.0), seg("A", 5.0, 9.0)]
    vectors = {middle_window(s): oh(0) for s in segments}
    result = _apply(segments, {"Ross": [oh(0)]}, vectors)
    assert result.mapping == {"A": "Ross"}
    assert [s.speaker for s in result.segments] == ["A", "A"], "命中不改时段归属"


def test_阈值拒绝_相似度不足保持未知名():
    segments = [seg("A", 0.0, 4.0)]
    vectors = {middle_window(segments[0]): oh(1)}
    result = _apply(segments, {"Ross": [oh(0)]}, vectors)
    assert result.mapping == {}, "匹配不上应保持未知名"
    assert result.segments == tuple(segments)


def test_同名多簇合并到说话时长最长的规范标签():
    segments = [
        seg("SPEAKER_00", 0.0, 4.0),   # 4 秒
        seg("SPEAKER_03", 6.0, 16.0),  # 10 秒：更长，应为规范标签
    ]
    vectors = {middle_window(s): oh(0) for s in segments}
    result = _apply(segments, {"Ross": [oh(0)]}, vectors)
    assert result.mapping == {"SPEAKER_03": "Ross"}, "规范标签是说话时长最长的簇"
    assert [s.speaker for s in result.segments] == ["SPEAKER_03", "SPEAKER_03"], (
        "同人拆成的两个簇合并为一个人"
    )


def test_逐段重指派_未匹配簇中标错的段回到库中人物():
    # SPEAKER_00 是 Ross（簇级命中）；SPEAKER_01 的簇质心介于两簇之间
    # 匹配不上（相似度低于阈值），但其中一长段其实是 Ross → 重指派
    segments = [
        seg("SPEAKER_00", 0.0, 4.0),   # oh(0) → Ross
        seg("SPEAKER_01", 5.0, 9.0),   # 长段其实是 Ross
        seg("SPEAKER_01", 10.0, 14.0),  # 长段确实是别人（oh(2)）
    ]
    vectors = {
        middle_window(segments[0]): oh(0),
        middle_window(segments[1]): oh(0),
        middle_window(segments[2]): oh(2),
    }
    result = _apply(segments, {"Ross": [oh(0)]}, vectors)
    assert [s.speaker for s in result.segments] == [
        "SPEAKER_00", "SPEAKER_00", "SPEAKER_01",
    ], "标错的段重指派回 Ross 的规范标签，其余段不动"
    assert result.mapping == {"SPEAKER_00": "Ross"}


def test_重指派到本集无标签的名字_新建规范标签():
    # 两个簇都不匹配（质心被稀释到阈值以下）；其中一簇里的一长段是
    # 库中的 Ross → 本集没有 Ross 标签，新建规范标签（名字即标签）
    segments = [
        seg("SPEAKER_00", 0.0, 4.0),   # oh(0) → Ross
        seg("SPEAKER_00", 5.0, 9.0),   # oh(2) → 稀释质心
        seg("SPEAKER_01", 10.0, 14.0),  # oh(1) → 无关
    ]
    vectors = {
        middle_window(segments[0]): oh(0),
        middle_window(segments[1]): oh(2),
        middle_window(segments[2]): oh(1),
    }
    result = _apply(segments, {"Ross": [oh(0)]}, vectors)
    assert [s.speaker for s in result.segments] == [
        "Ross", "SPEAKER_00", "SPEAKER_01",
    ], "该名字本集无标签时新建规范标签（名字即标签）"
    assert result.mapping == {"Ross": "Ross"}


def test_逐段重指派_margin守卫_领先不足不重指派():
    # 簇质心更靠近 Ross（簇级命中 Ross）；其中一长段的声纹略偏向 Rachel，
    # 但领先 Ross 的幅度不足 margin → 不重指派，防抖动
    segments = [
        seg("SPEAKER_00", 0.0, 4.0),   # 纯 Ross
        seg("SPEAKER_00", 5.0, 9.0),   # 介于 Ross 与 Rachel 之间
    ]
    ross = [1.0, 0.0]
    rachel = [0.95, 0.05]
    vectors = {
        middle_window(segments[0]): ross,
        middle_window(segments[1]): [0.97, 0.03],
    }
    result = _apply(segments, {"Ross": [ross], "Rachel": [rachel]}, vectors)
    assert result.mapping == {"SPEAKER_00": "Ross"}, "簇级命中 Ross，不被小领先覆盖"
    assert [s.speaker for s in result.segments] == ["SPEAKER_00", "SPEAKER_00"], (
        "领先不足 margin 的候选不得重指派"
    )


def test_短段跳过逐段验证():
    # SPEAKER_00 的簇质心被稀释匹配不上；其中一段是 Ross 但只有 1.5 秒，
    # 短于验证下限 → 不逐段验证，保持标错的原状
    segments = [
        seg("SPEAKER_00", 0.0, 4.0),    # oh(1)：稀释质心
        seg("SPEAKER_00", 5.0, 6.5),    # 1.5 秒：其实是 Ross，但太短
    ]
    vectors = {
        middle_window(segments[0]): oh(1),
        middle_window(segments[1]): oh(0),
    }
    result = _apply(segments, {"Ross": [oh(0)]}, vectors)
    assert result.mapping == {}, "短段不验证：不得重指派"
    assert [s.speaker for s in result.segments] == ["SPEAKER_00", "SPEAKER_00"]


def test_有效库为空条目则恒等():
    segments = [seg("A", 0.0, 3.0)]
    result = _apply(segments, {"Ross": []}, {})
    assert result.mapping == {} and result.segments == tuple(segments)
