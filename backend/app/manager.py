"""任务管理器：单实例运行 + FIFO 排队 + 运行中/排队中可取消 + 终态可续跑。"""

from __future__ import annotations

import threading
import uuid
from collections import deque
from pathlib import Path

from . import pipeline, workspace
from .inference import (
    DemucsSeparator,
    Inference,
    PyannoteDiarizer,
    PyannoteEmbedder,
    WhisperTranscriber,
)
from .models import Task, TaskState, now_iso
from .pipeline import RunContext


class TaskNotCancellable(Exception):
    """任务已结束，无法取消。"""


class TaskNotErasable(Exception):
    """任务当前状态不可发起擦除（需要已完成说话人分析）。"""


class TaskNotResumable(Exception):
    """任务当前状态没有断点可续（只有失败/已取消的任务可续跑）。"""


class TaskCheckActive(Exception):
    """已有声纹校验在运行（校验须显式取消后才能重启）。"""


class TaskManager:
    def __init__(self, workspace_root: Path, inference: Inference | None = None):
        self._workspace_root = Path(workspace_root)
        self.inference = inference or Inference(
            separator=DemucsSeparator(),
            diarizer=PyannoteDiarizer(),
            transcriber=WhisperTranscriber(),
            embedder=PyannoteEmbedder(),
        )
        self._tasks: dict[str, Task] = {}
        self._queue: deque[str] = deque()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="task-worker"
        )
        self._check_thread: threading.Thread | None = None
        self._check_event = threading.Event()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join(timeout=10)

    def submit(
        self,
        video_path: str,
        audio_track: int | None = None,
        num_speakers: int | None = None,
    ) -> Task:
        task = Task(
            id=uuid.uuid4().hex[:12],
            video_path=str(video_path),
            audio_track=audio_track,
            num_speakers=num_speakers,
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

    def erase(self, task_id: str, speakers: list[str], output_dir: str | None) -> Task:
        """对任务发起擦除：重新排队运行掩码合成与编码封装。

        放行已完成分析终态的任务（完成/失败/已取消）：昂贵的分析产物仍
        在工作区，擦除失败后腾出磁盘重试、或换目标再擦都不该重跑分析；
        擦除的可行性由说话人分离产物校验兜底（分析没跑到说话人分离的
        任务没有该产物）。
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(task_id)
            if task.state in (TaskState.QUEUED, TaskState.RUNNING):
                raise TaskNotErasable(task.state.value)
            task.erase = {
                "speakers": sorted(set(speakers)),  # 目标是说话人集合
                "output_dir": output_dir,
            }
            self._requeue(task)
        self._wake.set()
        return task

    def resume(self, task_id: str) -> Task:
        """续跑失败/已取消的任务：重新排队，管线按工作区清单跳过已完成阶段。

        已发起过的擦除请求原样保留（目标说话人与输出目录不变），因此擦除
        中途失败/取消的任务续跑的是擦除，而不是回到分析。
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(task_id)
            if task.state not in (TaskState.FAILED, TaskState.CANCELLED):
                raise TaskNotResumable(task.state.value)
            self._requeue(task)
        self._wake.set()
        return task

    def resume_workspace(self, ws: Path, manifest: dict) -> Task:
        """按工作区清单重建断点任务并入队（重启后从工作区一键续跑）。

        重启会清空任务内存，清单才是断点依据：恢复源视频、音轨、说话人
        人数与擦除请求，管线按清单跳过已完成阶段直接续跑。入队前就登记
        工作区占用，避免同一工作区被重复排队。
        """
        return self._workspace_task(ws, manifest, workspace.erase_request(manifest))

    def open_workspace(self, ws: Path, manifest: dict) -> Task:
        """把分析产物齐全的工作区物化为可操作任务并入队（进入擦除选择）。

        与续跑的区别：不带历史擦除请求——这是新的擦除目标选择，不是继续
        上一次擦除；分析阶段按清单全部跳过、任务几乎瞬间完成，卡片上就是
        说话人时间轴与擦除目标多选表单。入队前同样登记工作区占用。
        """
        return self._workspace_task(ws, manifest, None)

    def _workspace_task(self, ws: Path, manifest: dict, erase: dict | None) -> Task:
        """按工作区清单重建任务并入队（调用方不持锁）。

        重启会清空任务内存，清单才是断点依据：恢复源视频、音轨与说话人
        人数，管线按清单跳过已完成阶段直接续跑。入队前就登记工作区占用，
        避免同一工作区被重复排队。erase 由调用方决定：续跑带历史擦除请求，
        打开擦除选择不带。
        """
        source = workspace.source_path(manifest)
        if not source:
            raise TaskNotResumable("工作区清单缺少源视频路径")
        task = Task(
            id=uuid.uuid4().hex[:12],
            video_path=source,
            audio_track=workspace.requested_audio_track(manifest),
            num_speakers=workspace.requested_num_speakers(manifest),
        )
        task.workspace_dir = str(ws)
        task.erase = erase
        with self._lock:
            if self.workspace_busy(str(ws)):
                raise TaskNotResumable("工作区正在被任务使用")
            self._tasks[task.id] = task
            self._queue.append(task.id)
        self._wake.set()
        return task

    def _requeue(self, task: Task) -> None:
        """终态任务重新排队（调用方持锁）：不带上一轮的失败/取消痕迹。"""
        task.error = None
        task.finished_at = None
        task.cancel_event.clear()
        task.state = TaskState.QUEUED
        self._queue.append(task.id)

    def workspace_busy(self, workspace_dir: str) -> bool:
        """该工作区是否正被排队中/运行中的任务使用。

        删除工作区前的占用检查：删掉正在写的工作区会把任务掏空。
        """
        with self._lock:
            return any(
                task.workspace_dir == workspace_dir
                and task.state in (TaskState.QUEUED, TaskState.RUNNING)
                for task in self._tasks.values()
            )

    def start_voiceprint_check(self, task_id: str) -> Task:
        """显式启动按需声纹校验；同一时刻全局只允许一个校验线程。"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.workspace_dir is None:
                raise KeyError(task_id)
            if task.state in (TaskState.QUEUED, TaskState.RUNNING):
                raise TaskNotErasable(task.state.value)
            if self._check_thread is not None and self._check_thread.is_alive():
                raise TaskCheckActive(task_id)
            cancel_event = threading.Event()
            task.voiceprint_check = {
                "state": "queued",
                "progress": 0.0,
                "error": None,
                "cancel_requested": False,
                "cancel_event": cancel_event,
            }
            self._check_event = cancel_event
            self._check_thread = threading.Thread(
                target=self._run_voiceprint_check,
                args=(task, cancel_event),
                daemon=True,
                name=f"voiceprint-check-{task_id}",
            )
        self._check_thread.start()
        return task

    def cancel_voiceprint_check(self, task_id: str) -> Task:
        """请求取消声纹校验；校验线程在下一个取消点停止。"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.voiceprint_check is None:
                raise KeyError(task_id)
            state = task.voiceprint_check.get("state")
            if state not in ("queued", "running"):
                raise TaskNotCancellable(state)
            task.voiceprint_check["cancel_requested"] = True
            event = task.voiceprint_check.get("cancel_event")
            if event is not None:
                event.set()
            return task

    def _run_voiceprint_check(
        self, task: Task, cancel_event: threading.Event
    ) -> None:
        def set_state(**changes) -> None:
            with self._lock:
                if task.voiceprint_check is None:
                    return
                task.voiceprint_check.update(changes)

        set_state(state="running", started_at=now_iso())
        try:
            data = pipeline.run_voiceprint_check(
                task,
                self.inference,
                cancel_event,
                on_progress=lambda fraction: set_state(
                    progress=max(0.0, min(1.0, float(fraction)))
                ),
            )
            set_state(
                state="completed",
                progress=1.0,
                result=data,
                finished_at=now_iso(),
            )
        except Exception as e:
            cancelled = cancel_event.is_set()
            set_state(
                state="cancelled" if cancelled else "failed",
                error=None if cancelled else (str(e) or repr(e)),
                finished_at=now_iso(),
            )

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
            ctx = RunContext(
                workspace_root=self._workspace_root,
                cancel_event=task.cancel_event,
                inference=self.inference,
            )
            if task.erase:
                # 管线对取消事件做阶段边界与子进程内的检查：正常返回即完成
                # （即使取消请求恰好晚于最后一个阶段结束，产物有效，按完成
                # 处理），抛 TaskCancelled 才是取消。
                pipeline.run_erase(task, ctx)
            else:
                pipeline.run_analysis(task, ctx)
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
