"""FastAPI 应用：任务 API + 音轨探测 + 阶段产物/说话人时段试听 + 服务已构建前端。"""

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

from .inference import Inference
from .manager import TaskManager, TaskNotCancellable
from .media import TaskError, audio_tracks, ffprobe_json, run_ffmpeg_bytes
from .models import snapshot

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
        """说话人时间轴数据：按说话人分组的时段（纵览全片谁在何时说话）。"""
        task = manager.get(task_id)
        if task is None or task.workspace_dir is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        stage = _stage_of(task, "diarize")
        artifact = stage.outputs.get("diarization") if stage else None
        if artifact is None:
            raise HTTPException(status_code=404, detail="说话人分离尚未完成")
        path = Path(task.workspace_dir) / artifact
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise HTTPException(status_code=404, detail="说话人分离产物缺失")
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
        return {
            "model": data.get("model"),
            "num_speakers": data.get("num_speakers", len(speakers)),
            "duration": max((e for _, _, e in turns), default=0.0),
            "speakers": [speakers[k] for k in sorted(speakers)],
        }

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
