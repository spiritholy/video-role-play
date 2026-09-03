"""处理任务领域模型：状态、阶段与序列化。

词汇遵循 CONTEXT.md：处理任务、工作区、成品视频。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class TaskState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# 状态的中文显示名（前端直接可用）
STATE_LABELS = {
    TaskState.QUEUED: "排队中",
    TaskState.RUNNING: "运行中",
    TaskState.COMPLETED: "完成",
    TaskState.FAILED: "失败",
    TaskState.CANCELLED: "已取消",
}

# 处理任务的两个阶段（骨架闭环，暂不含擦除相关阶段）
STAGE_DEFS = [
    ("audio_extract", "音轨提取/降混"),
    ("encode_mux", "编码封装基线"),
]


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class StageState:
    id: str
    name: str
    status: StageStatus = StageStatus.PENDING
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    artifacts: list[str] = field(default_factory=list)


@dataclass
class Task:
    id: str
    video_path: str
    audio_track: int | None
    state: TaskState = TaskState.QUEUED
    error: str | None = None
    stages: list[StageState] = field(
        default_factory=lambda: [StageState(id=i, name=n) for i, n in STAGE_DEFS]
    )
    created_at: str = field(default_factory=now_iso)
    started_at: str | None = None
    finished_at: str | None = None
    output_path: str | None = None
    workspace_dir: str | None = None
    cancel_event: threading.Event = field(
        default_factory=threading.Event, repr=False, compare=False
    )


def snapshot(task: Task) -> dict:
    """任务的可序列化视图（API 响应体）。"""
    return {
        "id": task.id,
        "video_path": task.video_path,
        "audio_track": task.audio_track,
        "state": task.state.value,
        "state_label": STATE_LABELS[task.state],
        "error": task.error,
        "stages": [
            {
                "id": s.id,
                "name": s.name,
                "status": s.status.value,
                "started_at": s.started_at,
                "finished_at": s.finished_at,
                "error": s.error,
                "artifacts": list(s.artifacts),
            }
            for s in task.stages
        ],
        "created_at": task.created_at,
        "started_at": task.started_at,
        "finished_at": task.finished_at,
        "output_path": task.output_path,
        "workspace_dir": task.workspace_dir,
    }
