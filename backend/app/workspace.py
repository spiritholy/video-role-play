"""工作区：以源视频命名，落盘阶段产物与阶段清单、说话人命名元数据。

阶段清单是断点续跑与同源复用的依据：哪些阶段已完成、产物在哪、属于哪一份
分析（源视频路径与体积 + 音轨选择），全部以清单为准，任务内存状态不作数。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

_MANIFEST = "manifest.json"
_SPEAKER_NAMES = "speaker_names.json"
_UNSAFE = re.compile(r"[^\w.\-]", re.UNICODE)

MANIFEST_VERSION = 1

# 分析阶段（①–④）：产物昂贵且与目标说话人无关，可跨任务复用。
# 声纹校验是按需建议动作，不属于阶段链，也不会让分析产物失效。
ANALYSIS_STAGES = ("audio_extract", "separate", "diarize", "transcribe")


class WorkspaceNotFound(Exception):
    """工作区不存在，或名字越出了工作区根目录。"""


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


def ensure_stage_entries(manifest: dict | None, stage_defs) -> None:
    """确保清单包含全部阶段条目；旧清单缺失的按待运行物化。

    旧清单可能含已废弃的认人/纠错条目；这里保留历史条目不重写，新的阶段
    列表统一由任务重建。物化只补新条目不写盘，待任务运行时随清单落盘。
    """
    if not isinstance(manifest, dict):
        return
    present = {
        e.get("id")
        for e in manifest.get("stages") or []
        if isinstance(e, dict) and isinstance(e.get("id"), str)
    }
    for stage_id, name in stage_defs:
        if stage_id in present:
            continue
        manifest.setdefault("stages", []).append(
            {
                "id": stage_id,
                "name": name,
                "status": "pending",
                "artifacts": [],
                "outputs": {},
            }
        )


def write_manifest(ws: Path, manifest: dict) -> None:
    path = Path(ws) / _MANIFEST
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(tmp, path)


def source_path(manifest: dict | None) -> str | None:
    source = (manifest or {}).get("source")
    return source.get("path") if isinstance(source, dict) else None


def is_for_source(
    manifest: dict | None, video_path: str, audio_track: int | None
) -> bool:
    """清单是否属于同一份分析：同源视频、同音轨选择。

    源视频按路径 + 体积认定：同路径被换成另一个文件时，提取出的音频与
    分离、说话人分离结果全都不再对应，复用会把旧音频的擦除结果封装进新
    视频。不比 mtime——复制/还原同一个文件就改 mtime，白丢数小时的产物。
    换了音轨同样是另一份分析（提取出的音频不同）。
    """
    if not isinstance(manifest, dict) or manifest.get("version") != MANIFEST_VERSION:
        return False
    source = manifest.get("source")
    if not isinstance(source, dict) or source.get("path") != str(video_path):
        return False
    try:
        size = Path(video_path).stat().st_size
    except OSError:
        return False
    return (
        source.get("size_bytes") == size
        and manifest.get("requested_audio_track") == audio_track
    )


def done_stages(ws: Path, manifest: dict | None) -> set[str]:
    """清单中标记完成、且产物文件齐全的阶段 id。

    产物缺失（被清理、写入中断）按未完成处理：宁可重算，也不能拿着不
    存在的产物继续跑。
    """
    if not isinstance(manifest, dict):
        return set()
    ws = Path(ws)
    done: set[str] = set()
    for entry in manifest.get("stages") or []:
        if not isinstance(entry, dict) or entry.get("status") != "done":
            continue
        stage_id = entry.get("id")
        if not isinstance(stage_id, str):
            continue
        artifacts = list(entry.get("artifacts") or [])
        artifacts += list((entry.get("outputs") or {}).values())
        if all(isinstance(a, str) and a and (ws / a).is_file() for a in artifacts):
            done.add(stage_id)
    return done


def analysis_reusable(ws: Path, manifest: dict | None) -> bool:
    """全部分析阶段是否齐全（同源新任务可直接复用，几分钟出新成品）。

    旧认人/纠错产物只是历史证据，不要求其条目齐全；分析链以 ①–④ 为准。
    """
    return set(ANALYSIS_STAGES) <= done_stages(ws, manifest)


# 打开擦除选择所需的"核心分析"：昂贵阶段齐全即可。认人/纠错与样本转写
# 会在打开任务里快速补跑（无声纹库时认人是恒等），不阻塞老工作区进入
# 擦除选择（spec #9：升级前工作区重开不崩）。
CORE_STAGES = ("audio_extract", "separate", "diarize")


def openable(ws: Path, manifest: dict | None) -> bool:
    """核心分析是否齐全：可打开进入擦除选择（认人/转写由任务补跑）。"""
    return set(CORE_STAGES) <= done_stages(ws, manifest)


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


def resolve(root: Path, name: str) -> Path:
    """把工作区名字解析为根目录下的真实子目录。

    名字来自 HTTP 路径，越出根目录（`..`、绝对路径）一律按不存在处理：
    删除是不可逆操作，只允许作用于工作区根目录的直接子目录。
    """
    root = Path(root).resolve()
    ws = (root / name).resolve()
    if ws == root or ws.parent != root or not ws.is_dir():
        raise WorkspaceNotFound(name)
    return ws


def list_workspaces(root: Path) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)


def size_bytes(path: Path) -> int:
    """目录树占用的字节数（不跟随符号链接）。"""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                continue  # 正在被任务写入或清理的产物：跳过即可
    return total


def delete(ws: Path) -> None:
    shutil.rmtree(ws)


def format_bytes(n: float) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{n:.0f} B"


def resumable(manifest: dict | None) -> bool:
    """清单是否代表一轮未结束的任务（重启后可从工作区一键继续）。"""
    return isinstance(manifest, dict) and manifest.get("state") in (
        "running", "failed", "cancelled",
    )


def requested_audio_track(manifest: dict | None) -> int | None:
    """清单记录的音轨选择（None=默认第一条）。"""
    if not isinstance(manifest, dict):
        return None
    value = manifest.get("requested_audio_track")
    return value if isinstance(value, int) else None


def requested_num_speakers(manifest: dict | None) -> int | None:
    """从清单恢复用户指定的说话人人数；旧清单回退到说话人分离产物。

    None 表示让模型自动估计。新清单在创建时就把请求人数写在顶层，
    说话人分离还没跑到也能在重启后原样续跑。
    """
    if not isinstance(manifest, dict):
        return None
    value = manifest.get("requested_num_speakers")
    if isinstance(value, int):
        return value
    diarization = manifest.get("diarization")
    if isinstance(diarization, dict):
        value = diarization.get("requested_num_speakers")
        if isinstance(value, int):
            return value
    return None


def erase_request(manifest: dict | None) -> dict | None:
    """清单记录的最近一次擦除请求；没有则返回 None（续跑回到分析断点）。"""
    if not isinstance(manifest, dict):
        return None
    erase = manifest.get("erase")
    if not isinstance(erase, dict) or not isinstance(erase.get("speakers"), list):
        return None
    return {
        "speakers": sorted({str(s) for s in erase["speakers"]}),
        "output_dir": erase.get("output_dir"),
    }
