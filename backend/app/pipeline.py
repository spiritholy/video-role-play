"""处理任务管线：音轨提取/降混 → 人声/伴奏分离 → 编码封装基线。

分离经统一 ML 推理接口（inference.Inference）调用，生产为 Demucs、
测试注入 canned stems；工作区阶段清单为后续断点续跑与复用提供结构。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import threading

from . import workspace
from .inference import Inference, SeparationStems
from .media import TaskCancelled, TaskError, audio_tracks, ffprobe_json, run_ffmpeg
from .models import StageStatus, now_iso

# 官方支持 AAC 的容器；其余（如 WebM）落到 MKV
AAC_CONTAINERS = {".mp4", ".m4v", ".mov", ".mkv"}
_MUXER_BY_EXT = {".mp4": "mp4", ".m4v": "mp4", ".mov": "mov", ".mkv": "matroska"}
# 成品命名：临时后缀，命名规范在擦除闭环中定稿
OUTPUT_SUFFIX = "__baseline"

PCM_SAMPLE_RATE = 48000


@dataclass
class RunContext:
    workspace_root: Path
    cancel_event: threading.Event
    inference: Inference


def execute(task, ctx: RunContext) -> None:
    """运行任务的全部阶段；取消/失败时保留已完成产物并写入阶段清单。"""
    src = Path(task.video_path)
    ws = workspace.prepare(ctx.workspace_root, task.video_path)
    task.workspace_dir = str(ws)
    manifest = {
        "version": 1,
        "source": {"path": str(src), "container": src.suffix.lower()},
        "requested_audio_track": task.audio_track,
        "audio_track": None,
        "separation": None,
        "output": None,
        "state": "running",
        "error": None,
        "stages": [
            {
                "id": s.id,
                "name": s.name,
                "status": StageStatus.PENDING.value,
                "artifacts": [],
                "outputs": {},
            }
            for s in task.stages
        ],
    }
    _check_cancelled(ctx.cancel_event)
    try:
        wav = _stage_audio_extract(task, ctx, ws, manifest)
        _stage_separate(task, ctx, ws, manifest, wav)
        _stage_encode_mux(task, ctx, ws, manifest, src, wav)
        manifest["state"] = "completed"
    except TaskCancelled:
        _mark_stage_terminal(task, manifest, StageStatus.CANCELLED)
        manifest["state"] = "cancelled"
        raise
    except TaskError as e:
        _mark_stage_terminal(task, manifest, StageStatus.FAILED, str(e))
        manifest["state"] = "failed"
        manifest["error"] = str(e)
        raise
    except Exception as e:
        _mark_stage_terminal(task, manifest, StageStatus.FAILED, repr(e))
        manifest["state"] = "failed"
        manifest["error"] = repr(e)
        raise TaskError(f"内部错误: {e}") from e
    finally:
        workspace.write_manifest(ws, manifest)


def _stage_audio_extract(task, ctx: RunContext, ws: Path, manifest: dict) -> Path:
    stage = _begin_stage(task, manifest, "audio_extract")
    src = Path(task.video_path)
    if not src.is_file():
        raise TaskError(f"源视频路径不存在: {src}")
    info = ffprobe_json(src)
    audio = audio_tracks(info)
    if not audio:
        raise TaskError("源视频没有音轨")
    if task.audio_track is None:
        track_no = 0
    elif 0 <= task.audio_track < len(audio):
        track_no = task.audio_track
    else:
        raise TaskError(
            f"音轨序号 {task.audio_track} 超出范围（共 {len(audio)} 条音轨）"
        )
    stream = audio[track_no]

    out_dir = ws / "01-audio-extract"
    out_dir.mkdir(exist_ok=True)
    wav = out_dir / f"track{stream['index']}.wav"
    run_ffmpeg(
        [
            "-i", str(src),
            "-vn",
            "-map", f"0:{stream['index']}",
            "-ac", "2",
            "-ar", str(PCM_SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            str(wav),
        ],
        ctx.cancel_event,
    )
    artifact = str(wav.relative_to(ws))
    stage.artifacts.append(artifact)
    manifest["audio_track"] = {
        "index": track_no,
        "stream_index": stream["index"],
        "codec": stream.get("codec_name"),
        "channels": stream.get("channels"),
        "sample_rate": stream.get("sample_rate"),
    }
    _end_stage(task, manifest, stage, StageStatus.DONE)
    return ws / artifact


def _stage_separate(
    task, ctx: RunContext, ws: Path, manifest: dict, wav: Path
) -> SeparationStems:
    stage = _begin_stage(task, manifest, "separate")
    out_dir = ws / "02-separate"
    out_dir.mkdir(exist_ok=True)

    def on_progress(fraction: float) -> None:
        # 分离是 CPU 上最耗时的阶段之一：进度实时可见，取消经此注入
        stage.progress = max(0.0, min(1.0, float(fraction)))
        _check_cancelled(ctx.cancel_event)

    try:
        stems = ctx.inference.separator.separate(wav, out_dir, on_progress)
    except (TaskCancelled, TaskError):
        raise
    except Exception as e:
        raise TaskError(f"人声/伴奏分离失败: {e}") from e

    vocal = Path(stems.vocal)
    accomp = Path(stems.accompaniment)
    missing = [p for p in (vocal, accomp) if not p.is_file()]
    if missing:
        raise TaskError(f"分离结果缺少产物: {', '.join(str(p.name) for p in missing)}")
    stage.artifacts = [str(vocal.relative_to(ws)), str(accomp.relative_to(ws))]
    stage.outputs = {
        "vocal": stage.artifacts[0],
        "accompaniment": stage.artifacts[1],
    }
    stage.progress = 1.0
    manifest["separation"] = {
        "model": ctx.inference.separator.name,
        "vocal": stage.outputs["vocal"],
        "accompaniment": stage.outputs["accompaniment"],
    }
    _end_stage(task, manifest, stage, StageStatus.DONE)
    return stems


def _stage_encode_mux(
    task, ctx: RunContext, ws: Path, manifest: dict, src: Path, wav: Path
) -> None:
    stage = _begin_stage(task, manifest, "encode_mux")
    ext = src.suffix.lower()
    out_ext = ext if ext in AAC_CONTAINERS else ".mkv"
    final = src.with_name(f"{src.stem}{OUTPUT_SUFFIX}{out_ext}")
    tmp = src.with_name(f"{src.stem}{OUTPUT_SUFFIX}{out_ext}.part-{os.getpid()}")
    try:
        run_ffmpeg(
            [
                "-i", str(src),
                "-i", str(wav),
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "256k",
                "-ar", str(PCM_SAMPLE_RATE),
                "-ac", "2",
                "-f", _MUXER_BY_EXT[out_ext],
                str(tmp),
            ],
            ctx.cancel_event,
        )
        os.replace(tmp, final)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    manifest["output"] = {"path": str(final), "container": out_ext}
    task.output_path = str(final)
    _end_stage(task, manifest, stage, StageStatus.DONE)


def _begin_stage(task, manifest: dict, stage_id: str):
    _check_cancelled(task.cancel_event)
    stage = next(s for s in task.stages if s.id == stage_id)
    entry = next(m for m in manifest["stages"] if m["id"] == stage_id)
    stage.status = StageStatus.RUNNING
    stage.started_at = now_iso()
    entry["status"] = stage.status.value
    entry["started_at"] = stage.started_at
    workspace.write_manifest(Path(task.workspace_dir), manifest)
    return stage


def _end_stage(task, manifest: dict, stage, status: StageStatus) -> None:
    stage.status = status
    stage.finished_at = now_iso()
    _sync_stage_entry(task, manifest, stage)


def _sync_stage_entry(task, manifest: dict, stage) -> dict:
    entry = next(m for m in manifest["stages"] if m["id"] == stage.id)
    entry["status"] = stage.status.value
    entry["finished_at"] = stage.finished_at
    entry["error"] = stage.error
    entry["artifacts"] = list(stage.artifacts)
    entry["outputs"] = dict(stage.outputs)
    workspace.write_manifest(Path(task.workspace_dir), manifest)
    return entry


def _mark_stage_terminal(
    task, manifest: dict, status: StageStatus, error: str | None = None
) -> None:
    stage = next((s for s in task.stages if s.status == StageStatus.RUNNING), None)
    if stage is None:
        return
    stage.status = status
    stage.error = error
    stage.finished_at = now_iso()
    _sync_stage_entry(task, manifest, stage)


def _check_cancelled(cancel_event: threading.Event) -> None:
    if cancel_event.is_set():
        raise TaskCancelled()
