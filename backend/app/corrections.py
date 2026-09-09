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
    """统一生效时段解析：人工校正 > 认人/纠错 > 原始说话人分离。"""
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
    if snapshot_state(base_segments, snapshot, base_fingerprint) != "active":
        snapshot = None
    current = _valid_segments((snapshot or {}).get("segments")) or [
        {"speaker": str(s["speaker"]), "start": float(s["start"]), "end": float(s["end"])}
        for s in base_segments
    ]
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
