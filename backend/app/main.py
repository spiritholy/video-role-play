"""FastAPI 应用：任务 API（提交/取消/续跑/擦除）+ 音轨探测 + 阶段产物与
说话人时段试听 + 说话人命名 + 工作区列表与手动删除 + 服务已构建前端。"""

from __future__ import annotations

import json
import mimetypes
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import voiceprints, workspace
from .inference import Inference, SpeakerSegment
from .manager import (
    TaskManager,
    TaskCheckActive,
    TaskNotCancellable,
    TaskNotErasable,
    TaskNotResumable,
)
from .media import TaskError, audio_tracks, ffprobe_json, run_ffmpeg_bytes
from .models import TaskState, snapshot
from .pipeline import VOICEPRINT_CHECK_DIR, VOICEPRINT_CHECK_FILENAME
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


def _read_voiceprint_check(ws: Path) -> dict | None:
    """按需声纹校验建议；缺失/损坏返回 None（不当作已校验）。"""
    path = ws / VOICEPRINT_CHECK_DIR / VOICEPRINT_CHECK_FILENAME
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _effective_segments(task, raw: dict) -> list[dict]:
    """当前生效时段：原始说话人分离结果。

    旧自动认人产物保留在磁盘，但不再作为生效层读取。人工校正层就绪后，
    这里改为优先读取其完整时段快照。
    """
    return raw.get("segments", [])


def _ingest_voiceprint(
    root: Path, manager, task, label: str, name: str, segments: list[dict]
) -> str | None:
    """命名即入库：从该说话人的样本段（取自人声轨）抽声纹写入全局声纹库。

    返回 None 表示入库成功（或本工作区没有可入库的样本段）；返回字符串
    是用户可读的失败原因——命名照常生效，只是不参与跨视频认人（手工
    命名永远优先于自动命名），失败原因交由命名响应带回界面展示。
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

    提前触发下载，首次命名入库不再卡在权重下载上；失败不阻塞启动，
    命名照常生效，入库时会带着同样的原因再次尝试（并打印）。
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
                "[声纹模型] 不影响使用；首次命名入库时会再次尝试下载。",
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
        segments = _effective_segments(task, raw)
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

    @app.get("/api/tasks/{task_id}/diarization")
    def get_diarization(task_id: str):
        """说话人时间轴数据：按当前生效时段分组（旧认人结果不再是生效层）。

        人工命名继续优先显示；自动命名映射只存在于旧产物中，不再注入。
        """
        task = _require_task(manager, task_id)
        raw = _load_diarization(task)
        segments = _effective_segments(task, raw)
        turns = [
            (s["speaker"], float(s["start"]), float(s["end"]))
            for s in segments
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
        rows = []
        for k in sorted(speakers):
            manual = names.get(k)
            rows.append(
                {
                    **speakers[k],
                    "name": manual,
                }
            )
        return {
            "model": raw.get("model"),
            "num_speakers": len(rows),
            "duration": max((e for _, _, e in turns), default=0.0),
            "speakers": rows,
        }

    @app.post("/api/tasks/{task_id}/voiceprint-check")
    def start_voiceprint_check(task_id: str):
        """显式触发声纹库校验：按需运行，只产出待确认建议。"""
        task = _require_task(manager, task_id)
        if task.state in (TaskState.QUEUED, TaskState.RUNNING):
            raise HTTPException(status_code=409, detail="任务正在排队或运行中，无法校验")
        if manager.workspace_busy(str(task.workspace_dir)):
            raise HTTPException(status_code=409, detail="工作区正在被任务使用，无法校验")
        library = voiceprints.read_library(root)
        if not voiceprints.effective_entries(
            library, manager.inference.embedder.name
        ):
            raise HTTPException(
                status_code=409,
                detail="声纹库为空：请先给说话人命名并存入声纹，再执行校验",
            )
        try:
            task = manager.start_voiceprint_check(task_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="任务不存在")
        except TaskCheckActive:
            raise HTTPException(status_code=409, detail="声纹校验正在运行，请先取消")
        except TaskNotErasable:
            raise HTTPException(status_code=409, detail="任务正在排队或运行中，无法校验")
        return snapshot(task)

    @app.get("/api/tasks/{task_id}/voiceprint-check")
    def get_voiceprint_check(task_id: str):
        """读取校验进度/取消状态；完成后返回工作区持久化的建议列表。"""
        task = manager.get(task_id)
        if task is not None and task.voiceprint_check is not None:
            state = dict(task.voiceprint_check)
            return {
                "state": state.get("state"),
                "progress": state.get("progress", 0.0),
                "error": state.get("error"),
                "result": state.get("result"),
            }
        if task is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        persisted = _read_voiceprint_check(Path(task.workspace_dir))
        return {
            "state": "completed" if persisted else "not_run",
            "progress": 1.0 if persisted else 0.0,
            "error": None,
            "result": persisted,
        }

    @app.delete("/api/tasks/{task_id}/voiceprint-check")
    def cancel_voiceprint_check(task_id: str):
        """取消正在运行的声纹校验（与阶段链任务取消相互独立）。"""
        try:
            task = manager.cancel_voiceprint_check(task_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="任务不存在")
        except TaskNotCancellable:
            raise HTTPException(status_code=409, detail="声纹校验已结束，无法取消")
        return snapshot(task)

    @app.put("/api/tasks/{task_id}/speakers/{label}/name")
    def rename_speaker(task_id: str, label: str, body: SpeakerNameIn):
        """命名/改名/清除：写入工作区元数据，随工作区存续，不强制唯一。

        非空命名写名后同步对该名字入库（从该说话人的样本段抽声纹写入全局
        声纹库，跨视频累积）；空名只清手工命名、不动库（spec #9）。
        入库失败不影响命名本身，失败原因经 voiceprint_error 带回界面。
        """
        task = _require_task(manager, task_id)
        raw = _load_diarization(task)
        segments = _effective_segments(task, raw)
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
        voiceprint_error = None
        if name:
            voiceprint_error = _ingest_voiceprint(
                root, manager, task, label, name, segments
            )
        return {
            "label": label,
            "name": name or None,
            "voiceprint_error": voiceprint_error,
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
