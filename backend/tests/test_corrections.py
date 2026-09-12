"""人工校正核心逻辑：生效时段优先级、改派与新建说话人（纯函数）。"""

import pytest

from app.corrections import (
    apply_reassignment,
    apply_segment_delete,
    apply_segment_split,
    diarization_fingerprint,
    effective_segments,
    next_speaker_label,
    snapshot_state,
    apply_voiceprint_merge,
    apply_voiceprint_reassign,
    apply_voiceprint_rename,
)

BASE = [
    {"speaker": "SPEAKER_00", "start": 0.2, "end": 0.9},
    {"speaker": "SPEAKER_01", "start": 0.95, "end": 1.25},
    {"speaker": "SPEAKER_00", "start": 1.3, "end": 1.8},
]


def test_生效时段以完整校正快照优先():
    snapshot = {
        "version": 1,
        "base_fingerprint": diarization_fingerprint(BASE),
        "segments": BASE[:1] + [{"speaker": "SPEAKER_01", "start": 1.3, "end": 1.8}],
    }
    recognition = {"segments": BASE[1:]}
    assert effective_segments(BASE, recognition, snapshot) == snapshot["segments"]
    assert effective_segments(BASE, recognition, None) == BASE[1:]
    assert effective_segments(BASE, recognition, {}) == BASE[1:]  # 无效快照回自动分析


def test_改派现有说话人保持原始区间并返回新快照():
    snapshot = apply_reassignment(BASE, None, 0, "SPEAKER_01")
    assert snapshot["segments"][0] == {"speaker": "SPEAKER_01", "start": 0.2, "end": 0.9}
    assert snapshot["segments"][1:] == BASE[1:]
    assert snapshot["revision"] == 1
    assert snapshot["base_fingerprint"] == diarization_fingerprint(BASE)
    # 纯函数不改输入
    assert BASE[0]["speaker"] == "SPEAKER_00"


def test_改派新建说话人顺延编号():
    label = next_speaker_label(BASE)
    assert label == "SPEAKER_02"
    snapshot = apply_reassignment(BASE, None, 2, "__new__")
    assert snapshot["segments"][2]["speaker"] == "SPEAKER_02"
    # 新编号从全部已知标签里顺延，不与历史新建标签冲突
    assert next_speaker_label(snapshot["segments"]) == "SPEAKER_03"


def test_重复改派覆盖且快照版本递增():
    first = apply_reassignment(BASE, None, 0, "SPEAKER_01")
    second = apply_reassignment(BASE, first, 0, "SPEAKER_00")
    assert second["revision"] == 2
    assert second["segments"] == BASE
    with pytest.raises(ValueError):
        apply_reassignment(BASE, second, 0, "SPEAKER_00")  # 目标与当前相同


def test_快照状态按基础分离指纹作废():
    fingerprint = diarization_fingerprint(BASE)
    snapshot = {
        "version": 1,
        "base_fingerprint": fingerprint,
        "segments": BASE,
        "revision": 3,
    }
    assert snapshot_state(BASE, snapshot) == "active"
    changed = BASE[:-1] + [{"speaker": "SPEAKER_00", "start": 1.3, "end": 1.9}]
    assert snapshot_state(changed, snapshot) == "invalidated"
    assert snapshot_state(BASE, {"segments": BASE}) == "invalidated"
    assert snapshot_state(BASE, None) == "none"


def test_重叠时段快照仍视为有效():
    """基础分离可能产生相邻/嵌套重叠（如 pyannote 输出），快照继承后不应
    因重叠整体作废：否则任何校正写入都会立刻被锁定。"""
    base = [
        {"speaker": "SPEAKER_00", "start": 0.2, "end": 0.9},
        {"speaker": "SPEAKER_01", "start": 0.95, "end": 1.25},
        {"speaker": "SPEAKER_01", "start": 2.0, "end": 4.0},
        {"speaker": "SPEAKER_00", "start": 2.5, "end": 2.8},
    ]
    snapshot = {
        "version": 1,
        "base_fingerprint": diarization_fingerprint(base),
        "segments": base,
    }
    assert snapshot_state(base, snapshot) == "active"
    assert effective_segments(base, None, snapshot) == snapshot["segments"]


def test_结构非法时段仍判作废():
    """放宽重叠后仍保留结构校验：类型/有限数值/start<end/非空说话人。"""
    for bad in (
        [{"speaker": "SPEAKER_00", "start": 0.9, "end": 0.2}],  # start >= end
        [{"speaker": "", "start": 0.2, "end": 0.9}],  # 空说话人
        [{"speaker": "SPEAKER_00", "start": float("nan"), "end": 0.9}],  # 非有限
        [{"speaker": 123, "start": 0.2, "end": 0.9}],  # 说话人类型非法
    ):
        snapshot = {
            "version": 1,
            "base_fingerprint": diarization_fingerprint(bad),
            "segments": bad,
        }
        assert snapshot_state(bad, snapshot) == "invalidated"


def test_删除会留下结构非法时段时拒绝生成快照():
    """删除护栏：不写一份写入即作废的快照（结构非法时抛 ValueError）。"""
    base = [
        {"speaker": "SPEAKER_00", "start": 0.2, "end": 0.9},
        {"speaker": "SPEAKER_01", "start": 1.0, "end": 0.5},
    ]
    with pytest.raises(ValueError, match="无法生成有效校正快照"):
        apply_segment_delete(base, None, 0, diarization_fingerprint(base))


def test_切分至少需要两个结果区间():
    with pytest.raises(ValueError, match="至少需要两个结果区间"):
        apply_segment_split(
            BASE, None, 0,
            [{"start": 0.2, "end": 0.9, "target": "SPEAKER_01"}],
        )


def test_切分每个结果区间无条件至少零点一秒():
    # 两区间中一段不足 0.1 秒：不再因"整段改说话人"豁免
    with pytest.raises(ValueError, match="每个结果区间至少 0.1 秒"):
        apply_segment_split(
            BASE, None, 0,
            [
                {"start": 0.2, "end": 0.25, "target": "SPEAKER_00"},
                {"start": 0.25, "end": 0.9, "target": "SPEAKER_01"},
            ],
        )


def test_声纹改派建议按时段并入并允许命名目标():
    snapshot = apply_voiceprint_reassign(
        BASE,
        None,
        {
            "type": "reassign",
            "from_label": "SPEAKER_00",
            "to_label": "Ross",
            "start": 0.2,
            "end": 0.9,
        },
    )
    assert snapshot["segments"][0] == {"speaker": "Ross", "start": 0.2, "end": 0.9}
    assert snapshot["segments"][1:] == BASE[1:]
    assert snapshot["revision"] == 1


def test_声纹合并建议把非规范簇并入规范标签():
    suggestion = {
        "type": "merge",
        "labels": ["SPEAKER_00", "SPEAKER_01"],
        "canonical": "SPEAKER_01",
    }
    snapshot = apply_voiceprint_merge(BASE, None, suggestion)
    assert all(s["speaker"] == "SPEAKER_01" for s in snapshot["segments"])
    assert [s["start"] for s in snapshot["segments"]] == [0.2, 0.95, 1.3]


def test_接受命名建议写入校正快照且不改时段():
    snapshot = apply_voiceprint_rename(BASE, None, "SPEAKER_00", "Ross")
    assert snapshot["segments"] == BASE
    assert snapshot["renames"] == {"SPEAKER_00": "Ross"}
    assert snapshot["revision"] == 1
    assert apply_voiceprint_rename(BASE, snapshot, "SPEAKER_01", "Rachel")[
        "renames"
    ] == {"SPEAKER_00": "Ross", "SPEAKER_01": "Rachel"}
