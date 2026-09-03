"""FastAPI 应用：任务 API + 音轨探测 + 服务已构建前端。"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .manager import TaskManager, TaskNotCancellable
from .media import TaskError, audio_tracks, ffprobe_json
from .models import snapshot

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE_ROOT = REPO_ROOT / "workspaces"
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"


class TaskSubmitIn(BaseModel):
    video_path: str = Field(min_length=1)
    audio_track: int | None = None


class ProbeIn(BaseModel):
    video_path: str = Field(min_length=1)


def create_app(
    workspace_root: Path | None = None,
    on_start: Callable[[], None] | None = None,
) -> FastAPI:
    root = workspace_root or Path(
        os.environ.get("VRP_WORKSPACE_ROOT", DEFAULT_WORKSPACE_ROOT)
    )
    manager = TaskManager(root)

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

    if FRONTEND_DIST.is_dir():
        app.mount(
            "/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend"
        )

    return app
