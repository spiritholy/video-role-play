"""FastAPI 应用：任务 API（提交/取消/续跑/擦除）+ 音轨探测 + 阶段产物与
说话人时段试听 + 说话人命名与显式声纹入库 + 工作区列表与手动删除 + 服务已构建前端。"""

from __future__ import annotations

import json
import mimetypes
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import corrections, voiceprints, workspace
from .inference import Inference, SpeakerSegment
from .manager import (
    TaskManager,
    TaskNotCancellable,
    TaskNotErasable,
    TaskNotResumable,
)
from .media import TaskError, audio_tracks, ffprobe_json, run_ffmpeg_bytes
from .models import TaskState, snapshot
from .recognition import middle_window
from .samples import select_samples

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
    # 说话人数量：None=模型自动估计；1..20=强制分离出指定人数。
    # 自动估计在短片段上容易偏少（如 4 人只出 2 人），知道原片人数时
    # 指定后能强制物化出目标人数。
    num_speakers: int | None = Field(default=None, ge=1, le=20)


class ProbeIn(BaseModel):
    video_path: str = Field(min_length=1)


class SpeakerNameIn(BaseModel):
    name: str = Field(max_length=100)


class EraseIn(BaseModel):
    speakers: list[str] = Field(min_length=1)
    output_dir: str | None = None


class CorrectionIn(BaseModel):
    # "__new__" 是 API 协议里的新建说话人；后端会顺延生成 SPEAKER_XX。
    target: str = Field(min_length=1)


class BoundaryIn(BaseModel):
    # start/end 表示拖拽时段的左/右边缘；new_time 是请求的新的交接点。
    side: Literal["start", "end"]
    new_time: float = Field(ge=0)


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


def _read_recognition(task) -> dict | None:
    """认人/纠错产物；阶段未完成或产物缺失返回 None（回退原始结果）。"""
    stage = _stage_of(task, "recognize")
    artifact = stage.outputs.get("recognition") if stage else None
    if artifact is None:
        return None
    path = Path(task.workspace_dir) / artifact
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _diarization_fingerprint(raw: dict) -> str:
    """当前基础说话人分离的稳定指纹（含人数/模型元数据）。"""
    return corrections.diarization_fingerprint(
        raw.get("segments", []),
        requested_num_speakers=raw.get("requested_num_speakers"),
        num_speakers=raw.get("num_speakers"),
        model=raw.get("model"),
    )


def _correction_snapshot(task, raw: dict) -> tuple[dict | None, str]:
    """读取人工校正层并计算它相对当前说话人分离的状态。"""
    snapshot = corrections.read_corrections(Path(task.workspace_dir))
    base = raw.get("segments", [])
    fingerprint = _diarization_fingerprint(raw)
    return snapshot, corrections.snapshot_state(base, snapshot, fingerprint)


def _effective_segments(task, raw: dict, snapshot: dict | None) -> list[dict]:
    """统一生效时段：人工校正 > 认人/纠错 > 原始说话人分离。

    时间轴、命名与擦除校验统一走这里，保证界面看到的人物与擦除语义一致。
    """
    recognition = _read_recognition(task)
    return corrections.effective_segments(
        raw.get("segments", []),
        recognition,
        snapshot,
        _diarization_fingerprint(raw),
    )


def _correction_status(
    task, snapshot: dict | None, state: str, workspace_busy: bool
) -> dict:
    """时间轴使用的校正层元数据：可编辑状态、作废提示与工作区锁定。"""
    if workspace_busy:
        return {
            "state": state,
            "revision": int((snapshot or {}).get("revision") or 0),
            "message": None,
            "editable": False,
            "locked_reason": "工作区有排队中或运行中的处理任务，编辑已锁定",
        }
    if state == "invalidated":
        return {
            "state": state,
            "revision": int((snapshot or {}).get("revision") or 0),
            "message": "人工校正已作废：说话人分离已重新进行，请放弃作废校正后重新检查",
            "editable": False,
            "locked_reason": "人工校正快照已作废，请先放弃作废校正",
        }
    return {
        "state": state,
        "revision": int((snapshot or {}).get("revision") or 0),
        "message": None,
        "editable": True,
        "locked_reason": None,
    }


def _ingest_voiceprint(
    root: Path, manager, task, label: str, name: str, segments: list[dict]
) -> str | None:
    """显式入库：从该说话人的当前生效时段抽声纹写入全局声纹库。

    返回 None 表示入库成功；返回字符串是用户可读的失败原因——调用方
    保持其他数据不变，并把原因展示给用户。
    """
    try:
        ws = Path(task.workspace_dir)
        manifest = workspace.read_manifest(ws) or {}
        vocal_rel = (manifest.get("separation") or {}).get("vocal")
        if not vocal_rel:
            return "工作区缺少人声轨产物，无法抽取声纹入库"
        vocal_path = ws / vocal_rel
        if not vocal_path.is_file():
            return "人声轨产物文件缺失，无法抽取声纹入库"
        turns = [
            SpeakerSegment(s["speaker"], float(s["start"]), float(s["end"]))
            for s in segments
            if s["speaker"] == label
        ]
        if not turns:
            return
        samples = select_samples(turns, max((t.end for t in turns), default=0.0))
        windows = [middle_window(s) for s in samples]
        vectors = manager.inference.embedder.embed(vocal_path, windows, lambda _f: None)
        voiceprints.add_vectors(
            root,
            name,
            vectors,
            model=manager.inference.embedder.name,
            source_workspace=ws.name,
        )
        return None
    except Exception as e:
        # 声纹模型不可用/抽声纹失败：不影响命名，库保持原样；
        # 失败原因带回命名响应（界面展示），并打到后端控制台便于排查。
        message = str(e) or repr(e)
        print(f"[声纹入库] 失败：{message}", file=sys.stderr, flush=True)
        return message


def _warmup_embedder(manager: TaskManager) -> None:
    """启动时后台预热声纹模型：门控权重 pip 依赖装不了，只能联网下载。

    提前触发下载，首次显式入库不再卡在权重下载上；失败不阻塞启动，
    入库时会带着同样的原因再次尝试（并打印）。
    测试注入的 canned 替身没有 load_model，直接跳过。
    """
    embedder = manager.inference.embedder
    if not hasattr(embedder, "load_model"):
        return
    name = getattr(embedder, "name", type(embedder).__name__)

    def run():
        try:
            embedder.load_model()
            print(f"[声纹模型] {name} 已就绪（权重已缓存）", flush=True)
        except Exception as e:
            message = str(e) or repr(e)
            print(
                f"[声纹模型] 预热失败：{message}\n"
                "[声纹模型] 不影响使用；首次显式入库时会再次尝试下载。",
                flush=True,
            )

    threading.Thread(target=run, name="voiceprint-warmup", daemon=True).start()


def create_app(
    workspace_root: Path | None = None,
    on_start: Callable[[], None] | None = None,
    inference: Inference | None = None,
) -> FastAPI:
    root = Path(
        workspace_root or os.environ.get("VRP_WORKSPACE_ROOT", DEFAULT_WORKSPACE_ROOT)
    ).resolve()
    manager = TaskManager(root, inference=inference)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        manager.start()
        _warmup_embedder(manager)
        if on_start is not None:
            on_start()
        yield
        manager.stop()

    app = FastAPI(title="视频人声擦除", lifespan=lifespan)

    @app.post("/api/tasks")
    def create_task(body: TaskSubmitIn):
        task = manager.submit(
            body.video_path, body.audio_track, num_speakers=body.num_speakers
        )
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
        raw = _load_diarization(task)
        correction_snapshot, _state = _correction_snapshot(task, raw)
        segments = _effective_segments(task, raw, correction_snapshot)
        known = {s["speaker"] for s in segments}
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
                    "openable": workspace.openable(ws, manifest),
                    "resumable": workspace.resumable(manifest),
                    "busy": manager.workspace_busy(str(ws)),
                }
            )
        total = sum(w["size_bytes"] for w in items)
        return {
            "workspaces": items,
            "total_bytes": total,
            "total_label": workspace.format_bytes(total),
        }

    @app.post("/api/workspaces/{name}/resume")
    def resume_workspace(name: str):
        """重启后从工作区一键续跑：按清单重建任务并入队。

        重启会清空任务内存，工作区清单才是断点依据。源视频缺失时不建任务
        直接报错，避免排队半天才发现白跑；工作区被占用时拒绝（重复点击、
        正在运行的旧任务都算占用）。
        """
        try:
            ws = workspace.resolve(root, name)
        except workspace.WorkspaceNotFound:
            raise HTTPException(status_code=404, detail="工作区不存在")
        manifest = workspace.read_manifest(ws)
        if manifest is None:
            raise HTTPException(status_code=404, detail="工作区没有阶段清单，无法继续")
        if not workspace.resumable(manifest):
            raise HTTPException(
                status_code=409, detail="当前工作区没有可继续的断点"
            )
        source = workspace.source_path(manifest)
        if not source or not Path(source).is_file():
            raise HTTPException(status_code=400, detail=f"源视频不存在: {source}")
        if manager.workspace_busy(str(ws)):
            raise HTTPException(status_code=409, detail="工作区正在被任务使用，无法继续")
        try:
            task = manager.resume_workspace(ws, manifest)
        except TaskNotResumable:
            raise HTTPException(
                status_code=409, detail="工作区正在被任务使用，无法继续"
            )
        return snapshot(task)

    @app.post("/api/workspaces/{name}/open")
    def open_workspace(name: str):
        """打开「分析可复用」的工作区进入擦除选择：物化复用分析产物的任务。

        已完成的任务没有断点，下一步是发起擦除而非续跑（CONTEXT.md）：该
        接口为分析产物齐全的工作区重建一张任务卡片，分析阶段按清单全部跳过、
        秒级完成，卡片上就是说话人时间轴与「擦除目标说话人」选择表单。不带
        历史擦除请求——这是新的擦除目标选择。源视频缺失或已变化时不建任务
        直接报错，避免复用产物与当前文件对不上。
        """
        try:
            ws = workspace.resolve(root, name)
        except workspace.WorkspaceNotFound:
            raise HTTPException(status_code=404, detail="工作区不存在")
        manifest = workspace.read_manifest(ws)
        if manifest is None:
            raise HTTPException(status_code=404, detail="工作区没有阶段清单，无法打开")
        if not workspace.openable(ws, manifest):
            raise HTTPException(
                status_code=409, detail="分析产物不完整，无法进入擦除选择，请先完成分析"
            )
        source = workspace.source_path(manifest)
        if not source or not Path(source).is_file():
            raise HTTPException(status_code=400, detail=f"源视频不存在: {source}")
        track = workspace.requested_audio_track(manifest)
        if not workspace.is_for_source(manifest, source, track):
            raise HTTPException(
                status_code=400,
                detail="源视频已变化，工作区分析产物不再对应，请重新提交处理任务",
            )
        if manager.workspace_busy(str(ws)):
            raise HTTPException(status_code=409, detail="工作区正在被任务使用，无法打开")
        try:
            task = manager.open_workspace(ws, manifest)
        except TaskNotResumable:
            raise HTTPException(
                status_code=409, detail="工作区正在被任务使用，无法打开"
            )
        return snapshot(task)

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

    def _timeline_payload(task) -> dict:
        raw = _load_diarization(task)
        snapshot, state = _correction_snapshot(task, raw)
        segments = _effective_segments(task, raw, snapshot)
        turns = [
            {"speaker": s["speaker"], "start": float(s["start"]), "end": float(s["end"]), "index": i}
            for i, s in enumerate(segments)
        ]
        turns.sort(key=lambda t: t["start"])
        speakers: dict[str, dict] = {}
        for turn in turns:
            sp = speakers.setdefault(
                turn["speaker"],
                {"label": turn["speaker"], "speech_time": 0.0, "segments": []},
            )
            sp["speech_time"] += turn["end"] - turn["start"]
            sp["segments"].append(
                {"start": turn["start"], "end": turn["end"], "index": turn["index"]}
            )
        names = workspace.read_speaker_names(Path(task.workspace_dir))
        recognition = _read_recognition(task)
        auto_names = {
            str(k): str(v)
            for k, v in (recognition.get("mapping") or {}).items()
        } if recognition else {}
        rows = []
        for k in sorted(speakers):
            manual = names.get(k)
            auto = auto_names.get(k)
            rows.append(
                {
                    **speakers[k],
                    "name": manual if manual is not None else auto,
                    "auto": manual is None and auto is not None,
                }
            )
        return {
            "model": raw.get("model"),
            "num_speakers": len(rows),
            "duration": max((t["end"] for t in turns), default=0.0),
            "speakers": rows,
            "correction": _correction_status(
                task, snapshot, state, manager.workspace_busy(str(task.workspace_dir))
            ),
        }

    @app.get("/api/tasks/{task_id}/diarization")
    def get_diarization(task_id: str):
        """说话人时间轴数据：按生效时段分组的说话人（人工校正优先）。"""
        task = _require_task(manager, task_id)
        return _timeline_payload(task)

    @app.put("/api/tasks/{task_id}/segments/{segment_index}/speaker")
    def reassign_segment(task_id: str, segment_index: int, body: CorrectionIn):
        """人工改派一个时段：立即保存完整生效时段快照到工作区。"""
        task = _require_task(manager, task_id)
        raw = _load_diarization(task)
        snapshot, state = _correction_snapshot(task, raw)
        status = _correction_status(
            task, snapshot, state, manager.workspace_busy(str(task.workspace_dir))
        )
        if not status["editable"]:
            raise HTTPException(status_code=409, detail=status["locked_reason"])
        try:
            updated = corrections.apply_reassignment(
                raw.get("segments", []),
                snapshot,
                segment_index,
                body.target,
                _diarization_fingerprint(raw),
            )
            corrections.write_corrections(Path(task.workspace_dir), updated)
        except ValueError as e:
            message = str(e)
            status_code = 400 if message == "该时段已属于目标说话人" else 404
            raise HTTPException(status_code=status_code, detail=message)
        return _timeline_payload(task)

    @app.put("/api/tasks/{task_id}/segments/{segment_index}/boundary")
    def move_segment_boundary(task_id: str, segment_index: int, body: BoundaryIn):
        """人工移动一个时段边界：相邻共享边界联动并立即保存生效快照。"""
        task = _require_task(manager, task_id)
        raw = _load_diarization(task)
        snapshot, state = _correction_snapshot(task, raw)
        status = _correction_status(
            task, snapshot, state, manager.workspace_busy(str(task.workspace_dir))
        )
        if not status["editable"]:
            raise HTTPException(status_code=409, detail=status["locked_reason"])
        try:
            updated = corrections.apply_boundary_move(
                raw.get("segments", []),
                snapshot,
                segment_index,
                body.side,
                body.new_time,
                _diarization_fingerprint(raw),
            )
            corrections.write_corrections(Path(task.workspace_dir), updated)
        except ValueError as e:
            message = str(e)
            status_code = 404 if message == "说话人时段不存在" else 400
            raise HTTPException(status_code=status_code, detail=message)
        return _timeline_payload(task)

    @app.delete("/api/tasks/{task_id}/corrections")
    def discard_corrections(task_id: str):
        """放弃全部人工校正：删除快照，回到自动分析结果。"""
        task = _require_task(manager, task_id)
        raw = _load_diarization(task)
        snapshot, state = _correction_snapshot(task, raw)
        status = _correction_status(
            task, snapshot, state, manager.workspace_busy(str(task.workspace_dir))
        )
        if manager.workspace_busy(str(task.workspace_dir)):
            raise HTTPException(status_code=409, detail=status["locked_reason"])
        corrections.discard(Path(task.workspace_dir))
        return _timeline_payload(task)

    @app.put("/api/tasks/{task_id}/speakers/{label}/name")
    def rename_speaker(task_id: str, label: str, body: SpeakerNameIn):
        """命名/改名/清除：纯本地写入工作区元数据，不触碰声纹库。

        名字随工作区存续且不强制唯一；声纹库只经显式入库端点变化。
        """
        task = _require_task(manager, task_id)
        raw = _load_diarization(task)
        snapshot, state = _correction_snapshot(task, raw)
        status = _correction_status(
            task, snapshot, state, manager.workspace_busy(str(task.workspace_dir))
        )
        if not status["editable"]:
            raise HTTPException(status_code=409, detail=status["locked_reason"])
        segments = _effective_segments(task, raw, snapshot)
        known = {s["speaker"] for s in segments}
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

    @app.post("/api/tasks/{task_id}/speakers/{label}/voiceprint")
    def store_speaker_voiceprint(task_id: str, label: str):
        """显式存入声纹库：按当前生效时段抽样，同名人物追加向量。

        入库失败只影响本次跨视频复用；改名等本地操作与工作区数据保持
        不变，失败原因经 voiceprint_error 带回界面展示。
        """
        task = _require_task(manager, task_id)
        raw = _load_diarization(task)
        correction_snapshot, _state = _correction_snapshot(task, raw)
        segments = _effective_segments(task, raw, correction_snapshot)
        known = {s["speaker"] for s in segments}
        if label not in known:
            raise HTTPException(status_code=404, detail="说话人不存在")
        ws = Path(task.workspace_dir)
        recognition = _read_recognition(task)
        auto_names = (
            {str(k): str(v) for k, v in (recognition.get("mapping") or {}).items()}
            if recognition
            else {}
        )
        name = workspace.read_speaker_names(ws).get(label) or auto_names.get(label)
        if not name:
            raise HTTPException(status_code=400, detail="请先命名说话人，再存入声纹库")
        error = _ingest_voiceprint(root, manager, task, label, name, segments)
        return {
            "label": label,
            "name": name,
            "voiceprint_error": error,
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

    @app.get("/api/voiceprints")
    def get_voiceprints():
        """声纹库列表：全局累积的人物声纹，供查看与清理误入库条目。"""
        return {
            "revision": voiceprints.revision(root),
            "model": voiceprints.read_library(root).get("model"),
            "names": voiceprints.list_names(root),
        }

    @app.delete("/api/voiceprints/{name}")
    def delete_voiceprint(name: str):
        """删除声纹库条目：错误命名不再污染后续剧集的认人/纠错。"""
        if not name:
            raise HTTPException(status_code=400, detail="名字不能为空")
        if not voiceprints.delete_name(root, name):
            raise HTTPException(status_code=404, detail="声纹库中没有该名字")
        return {"deleted": name}

    if FRONTEND_DIST.is_dir():
        app.mount(
            "/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend"
        )

    return app
