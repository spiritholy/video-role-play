"""FastAPI 应用：任务 API（提交/取消/续跑/擦除）+ 音轨探测 + 阶段产物与
说话人时段试听 + 说话人命名 + 工作区列表与手动删除 + 服务已构建前端。"""

from __future__ import annotations

import json
import mimetypes
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import workspace
from .inference import Inference
from .manager import (
    TaskManager,
    TaskNotCancellable,
    TaskNotErasable,
    TaskNotResumable,
)
from .media import TaskError, audio_tracks, ffprobe_json, run_ffmpeg_bytes
from .models import TaskState, snapshot

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE_ROOT = REPO_ROOT / "workspaces"
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"

# 时段试听切片上限（秒）：认人只需秒级片段；超长时段截断到上限而非拒绝
MAX_SEGMENT_SECONDS = 120.0


def _stage_of(task, stage_id: str):
    return next((s for s in task.stages if s.id == stage_id), None)


class TaskSubmitIn(BaseModel):
    video_path: str = Field(min_length=1)
    audio_track: int | None = None


class ProbeIn(BaseModel):
    video_path: str = Field(min_length=1)


class SpeakerNameIn(BaseModel):
    name: str = Field(max_length=100)


class EraseIn(BaseModel):
    speakers: list[str] = Field(min_length=1)
    output_dir: str | None = None


def _require_task(manager: TaskManager, task_id: str):
    task = manager.get(task_id)
    if task is None or task.workspace_dir is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task


def _load_diarization(task) -> dict:
    """读取说话人时段产物；阶段未完成或产物缺失时按 404 拒绝。"""
    stage = _stage_of(task, "diarize")
    artifact = stage.outputs.get("diarization") if stage else None
    if artifact is None:
        raise HTTPException(status_code=404, detail="说话人分离尚未完成")
    path = Path(task.workspace_dir) / artifact
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise HTTPException(status_code=404, detail="说话人分离产物缺失")


def create_app(
    workspace_root: Path | None = None,
    on_start: Callable[[], None] | None = None,
    inference: Inference | None = None,
) -> FastAPI:
    root = workspace_root or Path(
        os.environ.get("VRP_WORKSPACE_ROOT", DEFAULT_WORKSPACE_ROOT)
    )
    manager = TaskManager(root, inference=inference)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        manager.start()
        if on_start is not None:
            on_start()
        yield
        manager.stop()

    app = FastAPI(title="视频人声擦除", lifespan=lifespan)

    @app.post("/api/tasks")
    def create_task(body: TaskSubmitIn):
        task = manager.submit(body.video_path, body.audio_track)
        return snapshot(task)

    @app.get("/api/tasks")
    def list_tasks():
        return {"tasks": [snapshot(t) for t in manager.list()]}

    @app.get("/api/tasks/{task_id}")
    def get_task(task_id: str):
        task = manager.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        return snapshot(task)

    @app.post("/api/tasks/{task_id}/cancel")
    def cancel_task(task_id: str):
        try:
            task = manager.cancel(task_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="任务不存在")
        except TaskNotCancellable:
            raise HTTPException(status_code=409, detail="任务已结束，无法取消")
        return snapshot(task)

    @app.post("/api/tasks/{task_id}/resume")
    def resume_task(task_id: str):
        """续跑失败/已取消的任务：跳过工作区里已完成的阶段，从断点继续。

        已完成阶段的产物仍在工作区，数小时的分离与说话人分离不重算。
        """
        try:
            task = manager.resume(task_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="任务不存在")
        except TaskNotResumable:
            raise HTTPException(
                status_code=409, detail="任务当前状态不可续跑（只有失败或已取消的任务有断点）"
            )
        return snapshot(task)

    @app.post("/api/tasks/{task_id}/erase")
    def erase_task(task_id: str, body: EraseIn):
        """发起擦除：多选目标说话人，可选指定输出目录（默认源视频同目录）。

        任务重新排队运行掩码合成与编码封装；说话人标签须来自说话人分离
        结果，输出目录须可创建。
        """
        task = _require_task(manager, task_id)
        if task.state in (TaskState.QUEUED, TaskState.RUNNING):
            raise HTTPException(
                status_code=409, detail="任务正在排队或运行中，无法发起擦除"
            )
        data = _load_diarization(task)
        known = {s["speaker"] for s in data.get("segments", [])}
        unknown = sorted({s for s in body.speakers if s not in known})
        if unknown:
            raise HTTPException(
                status_code=404, detail=f"说话人不存在: {', '.join(unknown)}"
            )
        output_dir = body.output_dir.strip() if body.output_dir else None
        if output_dir:
            try:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise HTTPException(status_code=400, detail=f"输出目录不可用: {e}")
        try:
            task = manager.erase(task_id, body.speakers, output_dir)
        except KeyError:
            raise HTTPException(status_code=404, detail="任务不存在")
        except TaskNotErasable:
            raise HTTPException(status_code=409, detail="任务当前状态不可擦除")
        return snapshot(task)

    @app.get("/api/workspaces")
    def get_workspaces():
        """工作区列表：磁盘占用与可复用性，供用户决定删哪个释放空间。

        reusable 表示四个分析阶段产物齐全——同源新任务可直接复用，删掉
        就意味着下次要重跑数小时的分析。busy 表示正被排队/运行中任务使用。
        """
        items = []
        for ws in workspace.list_workspaces(root):
            manifest = workspace.read_manifest(ws)
            size = workspace.size_bytes(ws)
            items.append(
                {
                    "name": ws.name,
                    "path": str(ws),
                    "source_path": workspace.source_path(manifest),
                    "size_bytes": size,
                    "size_label": workspace.format_bytes(size),
                    "reusable": workspace.analysis_reusable(ws, manifest),
                    "busy": manager.workspace_busy(str(ws)),
                }
            )
        total = sum(w["size_bytes"] for w in items)
        return {
            "workspaces": items,
            "total_bytes": total,
            "total_label": workspace.format_bytes(total),
        }

    @app.delete("/api/workspaces/{name}")
    def delete_workspace(name: str):
        """手动删除工作区释放磁盘；只影响该源视频的工作区。"""
        try:
            ws = workspace.resolve(root, name)
        except workspace.WorkspaceNotFound:
            raise HTTPException(status_code=404, detail="工作区不存在")
        if manager.workspace_busy(str(ws)):
            raise HTTPException(
                status_code=409, detail="工作区正在被任务使用，无法删除"
            )
        try:
            workspace.delete(ws)
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"删除工作区失败: {e}")
        return {"deleted": ws.name, "path": str(ws)}

    @app.post("/api/probe")
    def probe_video(body: ProbeIn):
        src = Path(body.video_path)
        if not src.is_file():
            raise HTTPException(status_code=400, detail=f"路径不存在: {src}")
        try:
            info = ffprobe_json(src)
        except TaskError as e:
            raise HTTPException(status_code=400, detail=str(e))
        tracks = [
            {
                "index": i,
                "stream_index": s["index"],
                "codec": s.get("codec_name"),
                "channels": s.get("channels"),
                "channel_layout": s.get("channel_layout"),
                "language": (s.get("tags") or {}).get("language"),
            }
            for i, s in enumerate(audio_tracks(info))
        ]
        return {
            "container": src.suffix.lower(),
            "duration": float(info.get("format", {}).get("duration") or 0),
            "tracks": tracks,
        }

    @app.get("/api/tasks/{task_id}/diarization")
    def get_diarization(task_id: str):
        """说话人时间轴数据：按说话人分组的时段（纵览全片谁在何时说话）。

        每个说话人附带工作区元数据中的命名（未命名为 null，界面回退编号标签）。
        """
        task = _require_task(manager, task_id)
        data = _load_diarization(task)
        turns = [
            (s["speaker"], float(s["start"]), float(s["end"]))
            for s in data.get("segments", [])
        ]
        turns.sort(key=lambda t: t[1])
        speakers: dict[str, dict] = {}
        for label, start, end in turns:
            sp = speakers.setdefault(
                label, {"label": label, "speech_time": 0.0, "segments": []}
            )
            sp["speech_time"] += end - start
            sp["segments"].append({"start": start, "end": end})
        names = workspace.read_speaker_names(Path(task.workspace_dir))
        return {
            "model": data.get("model"),
            "num_speakers": data.get("num_speakers", len(speakers)),
            "duration": max((e for _, _, e in turns), default=0.0),
            "speakers": [
                {**speakers[k], "name": names.get(k)} for k in sorted(speakers)
            ],
        }

    @app.put("/api/tasks/{task_id}/speakers/{label}/name")
    def rename_speaker(task_id: str, label: str, body: SpeakerNameIn):
        """命名/改名/清除：写入工作区元数据，随工作区存续，不强制唯一。"""
        task = _require_task(manager, task_id)
        data = _load_diarization(task)
        known = {s["speaker"] for s in data.get("segments", [])}
        if label not in known:
            raise HTTPException(status_code=404, detail="说话人不存在")
        name = body.name.strip()
        ws = Path(task.workspace_dir)
        names = workspace.read_speaker_names(ws)
        if name:
            names[label] = name
        else:
            names.pop(label, None)  # 空白名字 = 清除命名，界面回退编号标签
        workspace.write_speaker_names(ws, names)
        return {"label": label, "name": name or None}

    @app.get("/api/tasks/{task_id}/transcripts")
    def get_transcripts(task_id: str):
        """样本转写数据：每个说话人的样本段区间与文字台词（"看台词认人"）。"""
        task = manager.get(task_id)
        if task is None or task.workspace_dir is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        stage = _stage_of(task, "transcribe")
        artifact = stage.outputs.get("transcripts") if stage else None
        if artifact is None:
            raise HTTPException(status_code=404, detail="样本转写尚未完成")
        path = Path(task.workspace_dir) / artifact
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise HTTPException(status_code=404, detail="样本转写产物缺失")
        samples = [
            {
                "speaker": s["speaker"],
                "start": float(s["start"]),
                "end": float(s["end"]),
                "text": s["text"],
            }
            for s in data.get("samples", [])
        ]
        samples.sort(key=lambda s: (s["speaker"], s["start"]))
        return {"model": data.get("model"), "samples": samples}

    @app.get("/api/tasks/{task_id}/audio-segment")
    def listen_segment(
        task_id: str,
        start: float = Query(),
        end: float = Query(),
    ):
        """试听说话人时段：从降混源音频实时切片返回 WAV；超长时段截断到上限。"""
        task = manager.get(task_id)
        if task is None or task.workspace_dir is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        if not 0 <= start < end:
            raise HTTPException(
                status_code=400, detail="时间区间非法：需满足 0 ≤ 开始 < 结束"
            )
        duration = min(end - start, MAX_SEGMENT_SECONDS)
        stage = _stage_of(task, "audio_extract")
        artifacts = stage.artifacts if stage else []
        if not artifacts:
            raise HTTPException(status_code=404, detail="源音频尚未提取，无法试听")
        wav = Path(task.workspace_dir) / artifacts[0]
        if not wav.is_file():
            raise HTTPException(status_code=404, detail="音频产物文件缺失")
        try:
            clip = run_ffmpeg_bytes(
                [
                    "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
                    "-i", str(wav),
                    "-c:a", "pcm_s16le", "-f", "wav", "-",
                ]
            )
        except TaskError as e:
            raise HTTPException(status_code=500, detail=str(e))
        return Response(content=clip, media_type="audio/wav")

    @app.get("/api/tasks/{task_id}/audio/{artifact_path:path}")
    def listen_artifact(task_id: str, artifact_path: str):
        """试听任务阶段产物（人声轨/伴奏轨）；仅允许阶段清单登记过的产物。"""
        task = manager.get(task_id)
        if task is None or task.workspace_dir is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        allowed = {a for s in task.stages for a in s.artifacts}
        if artifact_path not in allowed:
            raise HTTPException(status_code=404, detail="产物不存在或不可试听")
        path = Path(task.workspace_dir) / artifact_path
        if not path.is_file():
            raise HTTPException(status_code=404, detail="产物文件缺失")
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return FileResponse(path, media_type=media_type)

    if FRONTEND_DIST.is_dir():
        app.mount(
            "/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend"
        )

    return app
