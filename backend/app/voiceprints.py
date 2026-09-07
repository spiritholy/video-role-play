"""声纹库：全局单文件，存储于工作区根目录（跨视频累积，按名字组织）。

每次写入递增 revision 号：认人/纠错产物按 revision 失效，改库后重开工作区
只重跑认人/纠错与样本转写（不重跑昂贵的说话人分离与音源分离）。模型标识
与当前 embedder 不一致的条目视为不存在；v1 不做跨模型迁移，写入时整库
对齐当前模型。
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

from .models import now_iso

VOICEPRINTS_FILENAME = "voiceprints.json"
LIBRARY_VERSION = 1
# 每个名字最多保留的声纹向量数（spec：最多 6 个）
MAX_VECTORS_PER_NAME = 6


def empty_library() -> dict:
    """空库：revision 0、无模型、无条目。"""
    return {
        "version": LIBRARY_VERSION,
        "revision": 0,
        "model": "",
        "names": {},
    }


def read_library(root: Path) -> dict:
    """读声纹库；缺失/损坏按空库处理（revision 0），不抛错。"""
    path = Path(root) / VOICEPRINTS_FILENAME
    if not path.is_file():
        return empty_library()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return empty_library()
    if not isinstance(data, dict):
        return empty_library()
    names = data.get("names")
    if not isinstance(names, dict):
        names = {}
    return {
        "version": data.get("version", LIBRARY_VERSION),
        "revision": int(data.get("revision") or 0),
        "model": data.get("model") or "",
        "names": names,
    }


def write_library(root: Path, library: dict) -> dict:
    """原子写库并递增 revision；返回写入后的库。

    每次写入都使旧认人/纠错产物失效（revision 是关联失效依据）。
    """
    path = Path(root) / VOICEPRINTS_FILENAME
    Path(root).mkdir(parents=True, exist_ok=True)
    library = dict(library)
    library["revision"] = int(library.get("revision") or 0) + 1
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(library, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(tmp, path)
    return library


def effective_entries(library: dict, model: str) -> dict[str, dict]:
    """模型与当前 embedder 一致的库条目；不一致或模型为空视为不存在。

    条目结构：{"vectors": [[...], ...], "source_workspace": str,
    "updated_at": str}。向量缺失/损坏的条目同样视为不存在。
    """
    if library.get("model") != model:
        return {}
    names = library.get("names") or {}
    return {
        name: entry
        for name, entry in names.items()
        if isinstance(entry, dict) and isinstance(entry.get("vectors"), list)
    }


def add_vectors(
    root: Path,
    name: str,
    vectors: Sequence[Sequence[float]],
    model: str,
    source_workspace: str,
) -> dict:
    """向名字追加向量（每个名字最多 MAX_VECTORS_PER_NAME 个），写入并递增 revision。

    模型与库不一致时旧向量整库作废（v1 不做跨模型迁移）。返回写入后的库。
    """
    library = read_library(root)
    if library.get("model") and library.get("model") != model:
        library["names"] = {}
    library["model"] = model
    entry = library["names"].get(name)
    if not isinstance(entry, dict):
        entry = {
            "vectors": [],
            "source_workspace": source_workspace,
            "updated_at": now_iso(),
        }
    merged = [list(v) for v in entry.get("vectors") or []]
    merged += [list(v) for v in vectors if isinstance(v, (list, tuple))]
    entry["vectors"] = merged[-MAX_VECTORS_PER_NAME:]
    entry["source_workspace"] = source_workspace
    entry["updated_at"] = now_iso()
    library["names"][name] = entry
    return write_library(root, library)


def delete_name(root: Path, name: str) -> bool:
    """删除名字条目；不存在返回 False。删除也递增 revision（匹配结果会变）。"""
    library = read_library(root)
    names = library.get("names") or {}
    if name not in names:
        return False
    names.pop(name, None)
    write_library(root, library)
    return True


def list_names(root: Path) -> list[dict]:
    """库条目列表（含模型不匹配的旧条目，供用户查看/清理）。"""
    library = read_library(root)
    names = library.get("names") or {}
    items = [
        {
            "name": name,
            "vectors_count": len(entry.get("vectors") or []),
            "source_workspace": entry.get("source_workspace"),
            "updated_at": entry.get("updated_at"),
        }
        for name, entry in sorted(names.items())
        if isinstance(entry, dict)
    ]
    return items


def revision(root: Path) -> int:
    """当前库 revision；缺失视为 0。"""
    return int(read_library(root).get("revision") or 0)
