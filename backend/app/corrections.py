"""人工校正：说话人时段的最终裁决层与完整生效时段解析。

原始说话人分离是底层事实；认人/纠错可产生修正时段；人工校正快照是最终
生效层。快照按完整时段列表落盘，并在基础说话人分离指纹变化时自动作废。
掩码合成、擦除校验与时间轴展示都应经本模块读取同一份生效时段。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

_CORRECTIONS = "corrections.json"
CORRECTIONS_VERSION = 1
_LABEL_NUMBER = re.compile(r"SPEAKER_(\d+)\Z")
# 时段最小宽度：用于声纹改派定位的容差与浮点比较，避免退化区间。
MIN_SEGMENT_SECONDS = 1e-6
# 切分结果区间最小宽度（秒）：避免产生难以操作的极碎时段。
MIN_SPLIT_SECONDS = 0.1
# 切分结果区间时间比较容差：拖动/换算产生的浮点误差在此范围内视为连续覆盖。
_SPLIT_TOLERANCE = 1e-6
# 新建说话人槽位的 API 协议标记；配合 slot 槽位标识把多个区间映射到同一标签。
_NEW_SLOT_TOKEN = "__new__"



def diarization_fingerprint(
    segments: list[dict],
    *,
    requested_num_speakers: int | None = None,
    num_speakers: int | None = None,
    model: str | None = None,
) -> str:
    """基础说话人分离指纹：校正快照只跟随这份分析作废。

    除时段本身外，请求人数、结果人数与模型也参与指纹：即使两次罐头/
    模型结果碰巧相同，重新分离仍是新的说话人分离，旧校正应作废。
    """
    payload = {
        "segments": [
            {
                "speaker": str(s.get("speaker", "")),
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
            }
            for s in segments or []
        ],
        "requested_num_speakers": requested_num_speakers,
        "num_speakers": num_speakers,
        "model": model,
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _valid_segments(value) -> list[dict] | None:
    """读取快照里的完整时段；结构/数值非法时按没有可生效快照处理。"""
    if not isinstance(value, list):
        return None
    result: list[dict] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        speaker = item.get("speaker")
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except (TypeError, ValueError):
            return None
        if not isinstance(speaker, str) or not speaker or not math.isfinite(start) or not math.isfinite(end) or start >= end:
            return None
        result.append({"speaker": speaker, "start": start, "end": end})
    # 基础说话人分离（如 pyannote）可能产生相邻/嵌套重叠时段，时间轴与
    # 擦除/声纹路径都按原样消费；快照继承后应立即可用，重叠不整体作废，
    # 这里只校验结构合法性（类型/有限数值/start<end/非空说话人）。
    return result


def read_corrections(ws: Path) -> dict | None:
    """读取人工校正快照；缺失或损坏返回 None（不阻断展示与擦除）。"""
    path = Path(ws) / _CORRECTIONS
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def snapshot_state(
    base_segments: list[dict], snapshot: dict | None, base_fingerprint: str | None = None
) -> str:
    """快照相对基础说话人分离的状态：none / active / invalidated。"""
    if not isinstance(snapshot, dict):
        return "none"
    fingerprint = base_fingerprint or diarization_fingerprint(base_segments)
    valid = _valid_segments(snapshot.get("segments"))
    return (
        "active"
        if snapshot.get("version") == CORRECTIONS_VERSION
        and snapshot.get("base_fingerprint") == fingerprint
        and valid is not None
        else "invalidated"
    )


def effective_segments(
    raw_segments: list[dict],
    recognition: dict | None = None,
    snapshot: dict | None = None,
    base_fingerprint: str | None = None,
) -> list[dict]:
    """统一生效时段解析：人工校正 > 原始说话人分离。"""
    if snapshot_state(raw_segments, snapshot, base_fingerprint) == "active":
        return _valid_segments(snapshot.get("segments")) or []
    segments = (
        recognition.get("segments")
        if isinstance(recognition, dict)
        else recognition
    )
    if isinstance(segments, list):
        return segments
    return raw_segments


def next_speaker_label(segments: list[dict], created_labels=()) -> str:
    """新建说话人的顺延编号标签；不占用自动分离已用/历史新建编号。"""
    known = {str(s.get("speaker", "")) for s in segments} | set(created_labels)
    numbers = [int(m.group(1)) for label in known if (m := _LABEL_NUMBER.fullmatch(label))]
    return f"SPEAKER_{max(numbers, default=-1) + 1:02d}"


def _effective_snapshot_segments(
    base_segments: list[dict],
    snapshot: dict | None,
    base_fingerprint: str | None,
) -> list[dict]:
    """编辑前使用的完整时段：有效校正快照优先，否则回到原始分离。"""
    if snapshot_state(base_segments, snapshot, base_fingerprint) != "active":
        snapshot = None
    return _valid_segments((snapshot or {}).get("segments")) or [
        {"speaker": str(s["speaker"]), "start": float(s["start"]), "end": float(s["end"])}
        for s in base_segments
    ]


def apply_reassignment(
    base_segments: list[dict],
    snapshot: dict | None,
    segment_index: int,
    target: str,
    base_fingerprint: str | None = None,
) -> dict:
    """把一个生效时段改派给现有/新建说话人，返回新的完整时段快照。

    segment_index 是生效时段在原始分离顺序中的索引；快照保持该顺序，因
    此改派可以跨任务稳定定位。函数是纯函数，调用方负责持久化。
    """
    current = _effective_snapshot_segments(base_segments, snapshot, base_fingerprint)
    if not isinstance(segment_index, int) or not 0 <= segment_index < len(current):
        raise ValueError("说话人时段不存在")
    created_labels = list((snapshot or {}).get("created_labels") or [])
    if target == "__new__":
        target = next_speaker_label(current, created_labels)
        created_labels.append(target)
    elif target not in {s["speaker"] for s in current}:
        raise ValueError("说话人不存在")
    if current[segment_index]["speaker"] == target:
        raise ValueError("该时段已属于目标说话人")
    segments = [dict(s) for s in current]
    segments[segment_index]["speaker"] = target
    revision = int((snapshot or {}).get("revision") or 0) + 1
    return {
        "version": CORRECTIONS_VERSION,
        "base_fingerprint": base_fingerprint or diarization_fingerprint(base_segments),
        "segments": segments,
        "created_labels": created_labels,
        "revision": revision,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _validate_split_results(results) -> list[dict]:
    """归一化并校验切分结果区间列表；返回按时间排序的规范字典列表。"""
    if not isinstance(results, list) or not results:
        raise ValueError("切分结果非法")
    normalized: list[dict] = []
    for item in results:
        if not isinstance(item, dict):
            raise ValueError("切分结果非法")
        target = item.get("target")
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except (TypeError, ValueError) as e:
            raise ValueError("结果区间时间非法") from e
        if not isinstance(target, str) or not target:
            raise ValueError("结果区间必须指定说话人")
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError("结果区间时间非法")
        normalized.append(
            {
                "start": start,
                "end": end,
                "target": target,
                "slot": str(item.get("slot") or ""),
            }
        )
    normalized.sort(key=lambda item: (item["start"], item["end"]))
    return normalized


def apply_segment_split(
    base_segments: list[dict],
    snapshot: dict | None,
    segment_index: int,
    results,
    base_fingerprint: str | None = None,
) -> dict:
    """按间隔线把一个生效时段拆成相邻结果区间，返回新的完整时段快照。

    结果区间必须至少两个、每段至少 0.1 秒、按时间连续且精确覆盖原时段、
    全部指定说话人、至少一个区间改变归属；整段改说话人只走改派端点
    （PUT /segments/{i}/speaker）。results 是「按时间排序的连续区间」列表，
    每项含 start/end/target：target 为现有说话人标签，或 "__new__"（此时
    必须带 slot 槽位标识，同一个 slot 映射为同一个顺延的 SPEAKER_XX 标签）。
    校验失败抛 ValueError；函数是纯函数，调用方负责把返回的快照持久化。
    """
    if snapshot_state(base_segments, snapshot, base_fingerprint) != "active":
        snapshot = None  # 无效快照不继承修订号或历史新建说话人
    current = _effective_snapshot_segments(base_segments, snapshot, base_fingerprint)
    if not isinstance(segment_index, int) or not 0 <= segment_index < len(current):
        raise ValueError("说话人时段不存在")
    original = current[segment_index]

    normalized = _validate_split_results(results)
    if len(normalized) < 2:
        raise ValueError("至少需要两个结果区间")

    tol = _SPLIT_TOLERANCE
    if abs(normalized[0]["start"] - original["start"]) > tol:
        raise ValueError("结果区间必须精确覆盖原时段")
    if abs(normalized[-1]["end"] - original["end"]) > tol:
        raise ValueError("结果区间必须精确覆盖原时段")
    for a, b in zip(normalized, normalized[1:]):
        if b["start"] < a["end"] - tol:
            raise ValueError("结果区间必须按时间连续且互不重叠")
        if b["start"] > a["end"] + tol:
            raise ValueError("结果区间必须按时间连续且精确覆盖原时段")
    for item in normalized:
        if item["end"] - item["start"] + tol < MIN_SPLIT_SECONDS:
            raise ValueError("每个结果区间至少 0.1 秒")

    known = {s["speaker"] for s in current}
    created_labels = list((snapshot or {}).get("created_labels") or [])
    slot_labels: dict[str, str] = {}
    final: list[dict] = []
    for item in normalized:
        target = item["target"]
        if target == _NEW_SLOT_TOKEN:
            slot = item.get("slot")
            if not slot:
                raise ValueError("新建说话人必须指定槽位")
            if slot not in slot_labels:
                slot_labels[slot] = next_speaker_label(current, created_labels)
                created_labels.append(slot_labels[slot])
            speaker = slot_labels[slot]
        else:
            if target not in known:
                raise ValueError(f"说话人不存在: {target}")
            speaker = target
        final.append(
            {"speaker": speaker, "start": item["start"], "end": item["end"]}
        )
    if all(seg["speaker"] == original["speaker"] for seg in final):
        raise ValueError("至少一个结果区间不是原说话人才允许保存")

    segments: list[dict] = []
    for i, seg in enumerate(current):
        if i == segment_index:
            segments.extend(final)
        else:
            segments.append(dict(seg))
    return {
        "version": CORRECTIONS_VERSION,
        "base_fingerprint": base_fingerprint or diarization_fingerprint(base_segments),
        "segments": segments,
        "created_labels": created_labels,
        "revision": int((snapshot or {}).get("revision") or 0) + 1,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def apply_segment_delete(
    base_segments: list[dict],
    snapshot: dict | None,
    segment_index: int,
    base_fingerprint: str | None = None,
) -> dict:
    """删除一个生效时段：该区间不再归属任何说话人，返回新的完整时段快照。

    segment_index 是生效时段在原始分离顺序中的索引；删除不改变其余时段的
    相对顺序，快照保持该顺序以便跨任务稳定定位。函数是纯函数，调用方负责
    持久化。
    """
    if snapshot_state(base_segments, snapshot, base_fingerprint) != "active":
        snapshot = None  # 无效快照不继承修订号或历史新建说话人
    current = _effective_snapshot_segments(base_segments, snapshot, base_fingerprint)
    if not isinstance(segment_index, int) or not 0 <= segment_index < len(current):
        raise ValueError("说话人时段不存在")
    segments = [dict(s) for i, s in enumerate(current) if i != segment_index]
    if _valid_segments(segments) is None:
        # 护栏：不写一份写入即作废的快照；调用方把 ValueError 映射为 4xx。
        raise ValueError("删除失败：无法生成有效校正快照")
    return {
        "version": CORRECTIONS_VERSION,
        "base_fingerprint": base_fingerprint or diarization_fingerprint(base_segments),
        "segments": segments,
        "created_labels": list((snapshot or {}).get("created_labels") or []),
        "revision": int((snapshot or {}).get("revision") or 0) + 1,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _suggestion_snapshot_payload(
    base_segments: list[dict],
    snapshot: dict | None,
    base_fingerprint: str | None,
    segments: list[dict],
    renames: dict[str, str],
) -> dict:
    """三个建议并入函数共享的完整校正快照结构。"""
    return {
        "version": CORRECTIONS_VERSION,
        "base_fingerprint": base_fingerprint or diarization_fingerprint(base_segments),
        "segments": segments,
        "created_labels": list((snapshot or {}).get("created_labels") or []),
        "renames": renames,
        "revision": int((snapshot or {}).get("revision") or 0) + 1,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def apply_voiceprint_rename(
    base_segments: list[dict],
    snapshot: dict | None,
    label: str,
    name: str,
    base_fingerprint: str | None = None,
) -> dict:
    """接受命名建议：把建议命名记入校正层，不改任何生效时段。"""
    if snapshot_state(base_segments, snapshot, base_fingerprint) != "active":
        snapshot = None
    current = _effective_snapshot_segments(base_segments, snapshot, base_fingerprint)
    clean = str(name or "").strip()
    if not clean:
        raise ValueError("说话人名字不能为空")
    if str(label) not in {s["speaker"] for s in current}:
        raise ValueError("说话人不存在")
    renames = dict((snapshot or {}).get("renames") or {})
    renames[str(label)] = clean
    return _suggestion_snapshot_payload(
        base_segments, snapshot, base_fingerprint, current, renames
    )


def apply_voiceprint_merge(
    base_segments: list[dict],
    snapshot: dict | None,
    suggestion: dict,
    base_fingerprint: str | None = None,
) -> dict:
    """接受合并建议：把同名非规范簇整体改派到规范标签。"""
    if snapshot_state(base_segments, snapshot, base_fingerprint) != "active":
        snapshot = None
    current = _effective_snapshot_segments(base_segments, snapshot, base_fingerprint)
    labels = suggestion.get("labels")
    canonical = suggestion.get("canonical")
    valid_labels = isinstance(labels, list) and len(labels) >= 2
    valid_canonical = isinstance(canonical, str) and bool(canonical)
    if not valid_labels or not valid_canonical:
        raise ValueError("合并建议非法")
    labels = [str(label) for label in labels]
    known = {s["speaker"] for s in current}
    if canonical not in known or any(label not in known for label in labels):
        raise ValueError("合并建议包含不存在的说话人")
    segments = [dict(s) for s in current]
    changed = False
    for segment in segments:
        if segment["speaker"] in labels and segment["speaker"] != canonical:
            segment["speaker"] = canonical
            changed = True
    if not changed:
        raise ValueError("该合并建议已生效")
    renames = dict((snapshot or {}).get("renames") or {})
    return _suggestion_snapshot_payload(
        base_segments, snapshot, base_fingerprint, segments, renames
    )


def apply_voiceprint_reassign(
    base_segments: list[dict],
    snapshot: dict | None,
    suggestion: dict,
    base_fingerprint: str | None = None,
) -> dict:
    """接受改派建议：按原始时段定位单段并改派到建议目标。"""
    if snapshot_state(base_segments, snapshot, base_fingerprint) != "active":
        snapshot = None
    current = _effective_snapshot_segments(base_segments, snapshot, base_fingerprint)
    from_label = suggestion.get("from_label")
    target = str(suggestion.get("to_label") or suggestion.get("name") or "").strip()
    if not isinstance(from_label, str) or not target:
        raise ValueError("改派建议非法")
    try:
        start = float(suggestion.get("start"))
        end = float(suggestion.get("end"))
    except (TypeError, ValueError) as e:
        raise ValueError("改派建议非法") from e
    matches = [
        i for i, s in enumerate(current)
        if s["speaker"] == from_label
        and math.isclose(s["start"], start, abs_tol=MIN_SEGMENT_SECONDS)
        and math.isclose(s["end"], end, abs_tol=MIN_SEGMENT_SECONDS)
    ]
    if not matches:
        raise ValueError("说话人时段不存在")
    if len(matches) > 1:
        raise ValueError("改派建议定位到多个时段")
    if current[matches[0]]["speaker"] == target:
        raise ValueError("该时段已属于目标说话人")
    segments = [dict(s) for s in current]
    segments[matches[0]]["speaker"] = target
    renames = dict((snapshot or {}).get("renames") or {})
    return _suggestion_snapshot_payload(
        base_segments, snapshot, base_fingerprint, segments, renames
    )


def write_corrections(ws: Path, snapshot: dict) -> None:
    """校正快照随工作区原子持久化，供同源新任务与擦除管线继承。"""
    path = Path(ws) / _CORRECTIONS
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def discard(ws: Path) -> None:
    """放弃全部校正：删除快照后回到自动分析结果。"""
    (Path(ws) / _CORRECTIONS).unlink(missing_ok=True)


def correction_revision(snapshot: dict | None) -> int:
    """校正层版本；掩码合成据此判断同目标但校正已变的过期混音。"""
    return int((snapshot or {}).get("revision") or 0)
