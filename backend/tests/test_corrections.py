"""人工校正核心逻辑：生效时段优先级、改派与新建说话人（纯函数）。"""

import pytest

from app.corrections import (
    apply_boundary_move,
    apply_reassignment,
    diarization_fingerprint,
    effective_segments,
    next_speaker_label,
    snapshot_state,
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


SHARED = [
    {"speaker": "SPEAKER_00", "start": 0.2, "end": 0.9},
    {"speaker": "SPEAKER_01", "start": 0.9, "end": 1.25},
    {"speaker": "SPEAKER_00", "start": 1.3, "end": 1.8},
]


def test_相邻段共享边界双向联动且无重叠空洞():
    snapshot = apply_boundary_move(SHARED, None, 0, "end", 1.05)
    assert snapshot["segments"][0]["end"] == 1.05
    assert snapshot["segments"][1]["start"] == 1.05
    assert snapshot["revision"] == 1

    snapshot = apply_boundary_move(SHARED, snapshot, 1, "start", 1.0)
    assert snapshot["segments"][0]["end"] == 1.0
    assert snapshot["segments"][1]["start"] == 1.0
    assert snapshot["revision"] == 2


def test_边界可自由延伸进空隙但撞邻居钳制():
    snapshot = apply_boundary_move(BASE, None, 0, "end", 0.93)
    assert snapshot["segments"][0]["end"] == 0.93
    assert snapshot["segments"][1]["start"] == 0.95

    snapshot = apply_boundary_move(BASE, None, 0, "end", 1.4)
    assert snapshot["segments"][0]["end"] == snapshot["segments"][1]["start"]
    assert snapshot["segments"][0]["end"] <= 0.95
    assert snapshot["segments"][0]["start"] < snapshot["segments"][0]["end"]

    snapshot = apply_boundary_move(BASE, None, 1, "start", 0.93)
    assert snapshot["segments"][1]["start"] == 0.93
    assert snapshot["segments"][0]["end"] == 0.9

    snapshot = apply_boundary_move(BASE, None, 1, "start", 0.4)
    assert snapshot["segments"][1]["start"] == 0.9
    assert snapshot["segments"][0]["end"] <= snapshot["segments"][1]["start"]


def test_边界修改拒绝不存在时段并保持互斥():
    with pytest.raises(ValueError):
        apply_boundary_move(BASE, None, 99, "end", 1.0)
    with pytest.raises(ValueError):
        apply_boundary_move(BASE, None, 0, "end", 0.2)  # 会使前段零时长

    snapshot = apply_boundary_move(BASE, None, 0, "end", 1.05)
    ordered = sorted(snapshot["segments"], key=lambda s: s["start"])
    assert all(a["end"] <= b["start"] for a, b in zip(ordered, ordered[1:]))
