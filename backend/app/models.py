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

# 处理任务阶段（spec #1：①提取 → ②分离 → ③说话人分离 → ④样本转写
# → ⑤掩码合成 → ⑥编码封装；①–④ 为分析阶段，⑤–⑥ 由用户选定目标说话人
# 后经擦除动作触发）
STAGE_DEFS = [
    ("audio_extract", "音轨提取/降混"),
    ("separate", "人声/伴奏分离"),
    ("diarize", "说话人分离"),
    ("transcribe", "样本转写"),
    ("mask_erase", "掩码合成"),
    ("encode_mux", "编码封装"),
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
    # 阶段内进度 0..1（如分离阶段的段级进度）；无细粒度进度时为 None
    progress: float | None = None
    # 命名产物：角色 → 工作区相对路径（如 separate: vocal/accompaniment）
    outputs: dict[str, str] = field(default_factory=dict)


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
    # 最近一次擦除请求：{"speakers": [标签...], "output_dir": str | None}；
    # None 表示尚未发起擦除（分析完成后由用户经 API 发起）
    erase: dict | None = None
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
                "progress": s.progress,
                "outputs": dict(s.outputs),
            }
            for s in task.stages
        ],
        "created_at": task.created_at,
        "started_at": task.started_at,
        "finished_at": task.finished_at,
        "output_path": task.output_path,
        "workspace_dir": task.workspace_dir,
        "erase": dict(task.erase) if task.erase else None,
    }
