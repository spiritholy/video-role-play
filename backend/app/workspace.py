"""工作区：以源视频命名，落盘阶段产物与阶段清单、说话人命名元数据。"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

_MANIFEST = "manifest.json"
_SPEAKER_NAMES = "speaker_names.json"
_UNSAFE = re.compile(r"[^\w.\-]", re.UNICODE)


def sanitize_stem(stem: str) -> str:
    cleaned = _UNSAFE.sub("_", stem).strip("._-")
    return cleaned or "workspace"


def prepare(root: Path, video_path: str) -> Path:
    """创建（或复用）源视频的工作区目录，返回路径。"""
    src = Path(video_path)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    ws = root / sanitize_stem(src.stem)
    if ws.exists():
        existing = read_manifest(ws)
        if existing is not None and existing.get("source", {}).get("path") != str(src):
            ws = root / f"{ws.name}__{hashlib.sha1(str(src).encode()).hexdigest()[:8]}"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def read_manifest(ws: Path) -> dict | None:
    path = Path(ws) / _MANIFEST
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_manifest(ws: Path, manifest: dict) -> None:
    path = Path(ws) / _MANIFEST
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(tmp, path)


def read_speaker_names(ws: Path) -> dict[str, str]:
    """说话人命名元数据：{说话人编号标签: 名字}；损坏或缺失视为未命名。

    独立于阶段清单落盘：清单由管线线程在每个阶段边界整体重写，
    命名走 API 线程，分文件避免互相覆盖，命名因此天然随工作区跨任务存续。
    """
    path = Path(ws) / _SPEAKER_NAMES
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    names = data.get("names", {}) if isinstance(data, dict) else {}
    return {k: v for k, v in names.items() if isinstance(k, str) and isinstance(v, str)}


def write_speaker_names(ws: Path, names: dict[str, str]) -> None:
    path = Path(ws) / _SPEAKER_NAMES
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps({"version": 1, "names": names}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)
