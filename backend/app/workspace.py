"""工作区：以源视频命名，落盘阶段产物与阶段清单。"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

_MANIFEST = "manifest.json"
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
