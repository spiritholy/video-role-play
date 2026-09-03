"""任务管理器：单实例运行 + FIFO 排队 + 运行中/排队中可取消。"""

from __future__ import annotations

import threading
import uuid
from collections import deque
from pathlib import Path

from . import pipeline
from .inference import DemucsSeparator, Inference, PyannoteDiarizer
from .models import Task, TaskState, now_iso
from .pipeline import RunContext


class TaskNotCancellable(Exception):
    """任务已结束，无法取消。"""


class TaskManager:
    def __init__(self, workspace_root: Path, inference: Inference | None = None):
        self._workspace_root = Path(workspace_root)
        self._inference = inference or Inference(
            separator=DemucsSeparator(), diarizer=PyannoteDiarizer()
        )
        self._tasks: dict[str, Task] = {}
        self._queue: deque[str] = deque()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="task-worker"
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join(timeout=10)

    def submit(self, video_path: str, audio_track: int | None = None) -> Task:
        task = Task(
            id=uuid.uuid4().hex[:12],
            video_path=str(video_path),
            audio_track=audio_track,
        )
        with self._lock:
            self._tasks[task.id] = task
            self._queue.append(task.id)
        self._wake.set()
        return task

    def get(self, task_id: str) -> Task | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list(self) -> list[Task]:
        with self._lock:
            tasks = list(self._tasks.values())
        tasks.sort(key=lambda t: (t.created_at, t.id), reverse=True)
        return tasks

    def cancel(self, task_id: str) -> Task:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(task_id)
            if task.state not in (TaskState.QUEUED, TaskState.RUNNING):
                raise TaskNotCancellable(task.state.value)
            task.cancel_event.set()
            if task.state == TaskState.QUEUED:
                task.state = TaskState.CANCELLED
                task.finished_at = now_iso()
            return task

    def _worker(self) -> None:
        while not self._stop.is_set():
            task = self._claim_next()
            if task is None:
                self._wake.wait(timeout=0.2)
                self._wake.clear()
                continue
            self._run(task)

    def _claim_next(self) -> Task | None:
        with self._lock:
            while self._queue:
                task_id = self._queue[0]
                task = self._tasks[task_id]
                if task.state == TaskState.QUEUED:
                    return task
                self._queue.popleft()  # 清理已取消/已结束的滞留条目
            return None

    def _run(self, task: Task) -> None:
        with self._lock:
            if task.state != TaskState.QUEUED:  # 排队期间被取消
                return
            task.state = TaskState.RUNNING
            task.started_at = now_iso()
        try:
            # 管线对取消事件做阶段边界与子进程内的检查：正常返回即完成
            # （即使取消请求恰好晚于最后一个阶段结束，成品有效，按完成处理），
            # 抛 TaskCancelled 才是取消。
            pipeline.execute(
                task,
                RunContext(
                    workspace_root=self._workspace_root,
                    cancel_event=task.cancel_event,
                    inference=self._inference,
                ),
            )
            self._finalize(task, TaskState.COMPLETED)
        except pipeline.TaskCancelled:
            self._finalize(task, TaskState.CANCELLED)
        except pipeline.TaskError as e:
            self._finalize(task, TaskState.FAILED, str(e))
        except Exception as e:  # noqa: BLE001 — 任何意外都让任务干净失败
            self._finalize(task, TaskState.FAILED, f"内部错误: {e!r}")

    def _finalize(self, task: Task, state: TaskState, error: str | None = None) -> None:
        with self._lock:
            task.state = state
            task.error = error
            task.finished_at = now_iso()
