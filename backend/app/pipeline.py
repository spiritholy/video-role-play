"""处理任务管线：分析阶段（音轨提取/降混 → 人声/伴奏分离 → 说话人分离
→ 样本转写）与擦除阶段（掩码合成 → 编码封装）。

ML 阶段经统一 ML 推理接口（inference.Inference）调用，生产为 Demucs /
pyannote / Whisper、测试注入 canned 输出；工作区阶段清单为断点续跑与
复用提供结构。擦除语义按 ADR-0001：掩码 = 目标说话人时段之并集，只作用
于人声轨，重叠时段整段擦除（目标优先），伴奏轨原样保留。

阶段跳过一律以工作区清单为准（workspace.done_stages：清单标记完成且产物
文件齐全）：任务中断/取消/失败后重跑从断点阶段继续，同一源视频的新任务
直接复用四个分析阶段的产物。任务内存里的阶段状态不作数——工作区被删过
之后它是过期的。阶段产物是链式的：某阶段一旦要重算，其后阶段的产物一并
作废重做（_adopt_chain），不留旧分析的描述被静默复用。
"""

from __future__ import annotations

import json
import os
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Iterable

from . import workspace
from .inference import DiarizationResult, Inference, SeparationStems, SpeakerSegment
from .media import TaskCancelled, TaskError, audio_tracks, ffprobe_json, run_ffmpeg
from .models import StageStatus, now_iso
from .samples import select_samples

# 官方支持 AAC 的容器；其余（如 WebM）落到 MKV
AAC_CONTAINERS = {".mp4", ".m4v", ".mov", ".mkv"}
_MUXER_BY_EXT = {".mp4": "mp4", ".m4v": "mp4", ".mov": "mov", ".mkv": "matroska"}

PCM_SAMPLE_RATE = 48000

# 磁盘预检：按源视频体积的倍数估算工作区需求（spec：约 3–5 倍，取上限保守）
WORKSPACE_FACTOR = 5

DIARIZATION_DIR = "03-diarize"
DIARIZATION_FILENAME = "diarization.json"
TRANSCRIBE_DIR = "04-transcribe"
TRANSCRIPTION_FILENAME = "transcripts.json"
MASK_ERASE_DIR = "05-mask-erase"
ERASED_MIX_FILENAME = "erased_mix.wav"


@dataclass
class RunContext:
    workspace_root: Path
    cancel_event: threading.Event
    inference: Inference


def _disk_free(path: Path) -> int:
    """工作区所在磁盘的可用字节数（测试经 monkeypatch 注入固定余量）。"""
    return shutil.disk_usage(path).free


def ensure_disk_space(src: Path, workspace_root: Path) -> None:
    """任务启动前磁盘预检：按源视频体积估算工作区需求，不足立即失败。

    对任务的全部阶段生效（分析与擦除启动时各执行一次）。
    """
    if not src.is_file():
        raise TaskError(f"源视频路径不存在: {src}")
    size = src.stat().st_size
    need = size * WORKSPACE_FACTOR
    Path(workspace_root).mkdir(parents=True, exist_ok=True)  # 预检需真实路径
    free = _disk_free(workspace_root)
    if free < need:
        raise TaskError(
            f"磁盘空间不足：任务需要约 {workspace.format_bytes(need)}"
            f"（源视频 {workspace.format_bytes(size)} × {WORKSPACE_FACTOR}），"
            f"工作区所在磁盘仅剩 {workspace.format_bytes(free)}"
        )


def run_analysis(task, ctx: RunContext) -> None:
    """运行分析阶段（①–④）；已完成阶段按工作区清单跳过。

    取消/失败时保留已完成产物并写入阶段清单，供续跑或同源新任务复用。
    """
    src = Path(task.video_path)
    ensure_disk_space(src, ctx.workspace_root)
    ws = workspace.prepare(ctx.workspace_root, task.video_path)
    task.workspace_dir = str(ws)
    manifest = _resume_manifest(task, ws, src)
    _check_cancelled(ctx.cancel_event)

    def stages():
        wav = _stage_audio_extract(task, ctx, ws, manifest)
        _stage_separate(task, ctx, ws, manifest, wav)
        diarization = _stage_diarize(task, ctx, ws, manifest, wav)
        _stage_transcribe(task, ctx, ws, manifest, wav, diarization)

    _run_stages(task, manifest, stages)


def run_erase(task, ctx: RunContext) -> None:
    """运行擦除阶段（⑤–⑥）：读取工作区分析产物，掩码合成后编码封装。

    目标说话人与上次合成一致且擦除混音仍在工作区时跳过掩码合成（续跑
    失败/取消的擦除、同目标重复擦除）；编码封装总是重跑，成品总是新的。
    """
    src = Path(task.video_path)
    ensure_disk_space(src, ctx.workspace_root)
    ws = Path(task.workspace_dir or "")
    manifest = workspace.read_manifest(ws) if str(ws) else None
    if manifest is None or not manifest.get("diarization"):
        raise TaskError("工作区缺少说话人分离产物，无法擦除")
    erase_req = task.erase or {}
    targets = [str(s) for s in erase_req.get("speakers", [])]
    _prepare_erase_stages(task, ws, manifest, targets)
    manifest["state"] = "running"
    manifest["error"] = None
    _check_cancelled(ctx.cancel_event)

    def stages():
        mix = _stage_mask_erase(task, ctx, ws, manifest, targets)
        _stage_encode_mux(task, ctx, ws, manifest, src, mix, targets)
        manifest["erase"] = {
            "speakers": targets,
            "output_dir": erase_req.get("output_dir"),
        }

    _run_stages(task, manifest, stages)


def _run_stages(task, manifest: dict, stages) -> None:
    """运行阶段序列；取消/失败时保留已完成产物并写入阶段清单。"""
    try:
        stages()
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
        workspace.write_manifest(Path(task.workspace_dir), manifest)


def _new_manifest(task, src: Path) -> dict:
    return {
        "version": workspace.MANIFEST_VERSION,
        "source": {
            "path": str(src),
            "container": src.suffix.lower(),
            "size_bytes": src.stat().st_size,
        },
        "requested_audio_track": task.audio_track,
        "audio_track": None,
        "separation": None,
        "diarization": None,
        "transcription": None,
        "mask_erase": None,
        "erase": None,
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


def _resume_manifest(task, ws: Path, src: Path) -> dict:
    """接续工作区清单：同一份分析则采纳已完成阶段，否则新起一份。

    清单不可用（首次处理、换了音轨、源视频变了、工作区被删）时把任务的
    全部阶段复位：任务内存里可能还记着上一轮的"已完成"，而产物已经不在了。
    """
    manifest = workspace.read_manifest(ws)
    if workspace.is_for_source(manifest, task.video_path, task.audio_track):
        _adopt_chain(
            task, manifest, workspace.ANALYSIS_STAGES,
            workspace.done_stages(ws, manifest),
        )
        if not workspace.analysis_reusable(ws, manifest):
            # 分析链断过：旧分析合成的擦除混音不再对应，一并作废
            _reset_stages(task, manifest, ("mask_erase", "encode_mux"))
        manifest["state"] = "running"
        manifest["error"] = None
        return manifest
    manifest = _new_manifest(task, src)
    _reset_stages(task, manifest, [s.id for s in task.stages])
    return manifest


def _prepare_erase_stages(task, ws: Path, manifest: dict, targets: list[str]) -> None:
    """擦除前的阶段采纳：目标未变的掩码合成可续用，成品总是重新封装。"""
    done = workspace.done_stages(ws, manifest)
    if (manifest.get("mask_erase") or {}).get("targets") != targets:
        done.discard("mask_erase")  # 换了目标，已合成的擦除混音不再对应
    done.discard("encode_mux")
    _adopt_chain(task, manifest, ("mask_erase", "encode_mux"), done)


def _adopt_chain(task, manifest: dict, stage_ids: Iterable[str], done: set[str]) -> None:
    """按顺序采纳已完成阶段；某阶段一旦要重算，其后阶段全部作废重做。

    阶段产物是链式的：说话人时段重算后，旧的样本台词与擦除混音不再描述
    同一份分析，留着只会被静默复用出错——宁可重算。
    """
    adopt: list[str] = []
    stale: list[str] = []
    for stage_id in stage_ids:
        if stale or stage_id not in done:
            stale.append(stage_id)  # 断链：本阶段及其后一律重做
        else:
            adopt.append(stage_id)
    _adopt_stages(task, manifest, adopt)
    _reset_stages(task, manifest, stale)


def _adopt_stages(task, manifest: dict, stage_ids: Iterable[str]) -> None:
    """把工作区清单里的阶段产物采纳到任务上，跳过重算。"""
    for stage_id in stage_ids:
        entry = _entry(manifest, stage_id)
        stage = _stage(task, stage_id)
        stage.status = StageStatus.DONE
        stage.artifacts = list(entry.get("artifacts") or [])
        stage.outputs = dict(entry.get("outputs") or {})
        stage.started_at = entry.get("started_at")
        stage.finished_at = entry.get("finished_at")
        stage.error = None
        stage.progress = 1.0


def _reset_stages(task, manifest: dict, stage_ids: Iterable[str]) -> None:
    """把阶段复位为待运行（上次的产物将被重做）。"""
    for stage_id in stage_ids:
        stage = _stage(task, stage_id)
        stage.status = StageStatus.PENDING
        stage.started_at = None
        stage.finished_at = None
        stage.error = None
        stage.progress = None
        stage.artifacts = []
        stage.outputs = {}
        _entry(manifest, stage_id).update(
            status=StageStatus.PENDING.value,
            started_at=None,
            finished_at=None,
            error=None,
            artifacts=[],
            outputs={},
        )


def _stage_audio_extract(task, ctx: RunContext, ws: Path, manifest: dict) -> Path:
    stage = _stage(task, "audio_extract")
    if stage.status == StageStatus.DONE:
        return ws / stage.artifacts[0]  # 复用工作区里的降混 wav
    stage = _begin_stage(task, manifest, "audio_extract")
    src = Path(task.video_path)
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
    stage = _stage(task, "separate")
    if stage.status == StageStatus.DONE:
        # 复用工作区里的人声轨/伴奏轨：分离是最昂贵的 ML 阶段
        return SeparationStems(
            vocal=ws / stage.outputs["vocal"],
            accompaniment=ws / stage.outputs["accompaniment"],
        )
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


def _stage_diarize(
    task, ctx: RunContext, ws: Path, manifest: dict, wav: Path
) -> DiarizationResult:
    stage = _stage(task, "diarize")
    if stage.status == StageStatus.DONE:
        return _load_diarization(ws)  # 复用工作区里的说话人时段产物
    stage = _begin_stage(task, manifest, "diarize")
    out_dir = ws / DIARIZATION_DIR
    out_dir.mkdir(exist_ok=True)

    def on_progress(fraction: float) -> None:
        stage.progress = max(0.0, min(1.0, float(fraction)))
        _check_cancelled(ctx.cancel_event)

    try:
        diarization = ctx.inference.diarizer.diarize(wav, out_dir, on_progress)
    except (TaskCancelled, TaskError):
        raise
    except Exception as e:
        raise TaskError(f"说话人分离失败: {e}") from e

    if not diarization.segments:
        raise TaskError(
            "未检测到任何说话人：所选音轨可能没有人声，请确认选择了正确的音轨"
        )

    speakers = sorted({s.speaker for s in diarization.segments})
    data = {
        "version": 1,
        "model": ctx.inference.diarizer.name,
        "num_speakers": len(speakers),
        "segments": [
            {"speaker": s.speaker, "start": s.start, "end": s.end}
            for s in diarization.segments
        ],
    }
    artifact = out_dir / DIARIZATION_FILENAME
    artifact.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rel = str(artifact.relative_to(ws))
    stage.artifacts.append(rel)
    stage.outputs = {"diarization": rel}
    stage.progress = 1.0
    manifest["diarization"] = {
        "model": ctx.inference.diarizer.name,
        "num_speakers": len(speakers),
        "segments": rel,
    }
    _end_stage(task, manifest, stage, StageStatus.DONE)
    return diarization


def _stage_transcribe(
    task,
    ctx: RunContext,
    ws: Path,
    manifest: dict,
    wav: Path,
    diarization: DiarizationResult,
) -> None:
    if _stage(task, "transcribe").status == StageStatus.DONE:
        return  # 复用工作区里的样本台词产物
    stage = _begin_stage(task, manifest, "transcribe")
    out_dir = ws / TRANSCRIBE_DIR
    out_dir.mkdir(exist_ok=True)

    samples = select_samples(list(diarization.segments), _wav_duration(wav))

    def on_progress(fraction: float) -> None:
        stage.progress = max(0.0, min(1.0, float(fraction)))
        _check_cancelled(ctx.cancel_event)

    try:
        result = ctx.inference.transcriber.transcribe(wav, samples, on_progress)
    except (TaskCancelled, TaskError):
        raise
    except Exception as e:
        raise TaskError(f"样本转写失败: {e}") from e

    data = {
        "version": 1,
        "model": ctx.inference.transcriber.name,
        "samples": [
            {
                "speaker": t.sample.speaker,
                "start": t.sample.start,
                "end": t.sample.end,
                "text": t.text,
            }
            for t in result.transcripts
        ],
    }
    artifact = out_dir / TRANSCRIPTION_FILENAME
    artifact.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rel = str(artifact.relative_to(ws))
    stage.artifacts.append(rel)
    stage.outputs = {"transcripts": rel}
    stage.progress = 1.0
    manifest["transcription"] = {
        "model": ctx.inference.transcriber.name,
        "samples": rel,
    }
    _end_stage(task, manifest, stage, StageStatus.DONE)


def _wav_duration(wav: Path) -> float:
    """降混 PCM wav 的时长（秒）；读取失败按 0 处理（仅影响分散约束）。"""
    try:
        with wave.open(str(wav), "rb") as w:
            return w.getnframes() / w.getframerate()
    except (wave.Error, OSError):
        return 0.0


def _read_diarization_segments(ws: Path) -> list[dict]:
    path = ws / DIARIZATION_DIR / DIARIZATION_FILENAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise TaskError("说话人分离产物缺失或损坏，无法擦除") from e
    return list(data.get("segments", []))


def _load_diarization(ws: Path) -> DiarizationResult:
    """把工作区里的说话人时段产物读回结构化结果（复用/续跑用）。"""
    return DiarizationResult(
        segments=tuple(
            SpeakerSegment(str(s["speaker"]), float(s["start"]), float(s["end"]))
            for s in _read_diarization_segments(ws)
        )
    )


def union_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """时间区间并集（纯函数）：排序后合并重叠与相接的区间。"""
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def build_mix_filter(intervals: list[tuple[float, float]]) -> str:
    """掩码混音的 filter_complex 脚本：目标时段在人声轨整段擦除，再与伴奏求和。

    amix normalize=0 保持可加信号的原始电平关系（人声+伴奏即源混音）。
    走 filter_complex_script 文件而非命令行参数：长片说话人时段可达数千段，
    不受单参数长度限制。
    """
    if intervals:
        cond = "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in intervals)
        head = f"[0:a]volume=volume=0:enable='{cond}'[masked];"
        inputs = "[masked][1:a]"
    else:
        head = ""
        inputs = "[0:a][1:a]"
    return f"{head}{inputs}amix=inputs=2:duration=longest:normalize=0[mix]"


def erased_output_path(
    src: Path, targets: list[str], names: dict[str, str], output_dir: str | None
) -> Path:
    """成品路径：`<源片名>__erased-<名字或编号标签以+连接>.<原扩展名>`。

    容器跟随源（源容器官方不支持 AAC 时落 MKV）；名字优先于编号标签，
    按目标标签排序保证同请求幂等；标签段经文件名净化，名字里的 + 一并
    转写，保证文件名中的 + 都是目标分隔符。
    """
    ext = src.suffix.lower()
    out_ext = ext if ext in AAC_CONTAINERS else ".mkv"
    labels = "+".join(
        workspace.sanitize_stem(names.get(t) or t) for t in sorted(targets)
    )
    parent = Path(output_dir) if output_dir else src.parent
    return parent / f"{src.stem}__erased-{labels}{out_ext}"


def _stage_mask_erase(
    task, ctx: RunContext, ws: Path, manifest: dict, targets: list[str]
) -> Path:
    stage = _stage(task, "mask_erase")
    if stage.status == StageStatus.DONE:
        return ws / stage.outputs["erased_mix"]  # 目标未变，复用已合成的擦除混音
    stage = _begin_stage(task, manifest, "mask_erase")
    separation = manifest.get("separation") or {}
    if not separation.get("vocal") or not separation.get("accompaniment"):
        raise TaskError("工作区缺少人声/伴奏分离产物，无法擦除")
    vocal = ws / separation["vocal"]
    accomp = ws / separation["accompaniment"]
    if not vocal.is_file() or not accomp.is_file():
        raise TaskError("人声/伴奏分离产物文件缺失，无法擦除")
    target_set = set(targets)
    intervals = union_intervals(
        (float(s["start"]), float(s["end"]))
        for s in _read_diarization_segments(ws)
        if s["speaker"] in target_set
    )

    out_dir = ws / MASK_ERASE_DIR
    out_dir.mkdir(exist_ok=True)
    mix = out_dir / ERASED_MIX_FILENAME
    script = out_dir / "mask_filter.txt"
    script.write_text(build_mix_filter(intervals), encoding="utf-8")
    run_ffmpeg(
        [
            "-i", str(vocal),
            "-i", str(accomp),
            "-filter_complex_script", str(script),
            "-map", "[mix]",
            "-ar", str(PCM_SAMPLE_RATE),
            "-ac", "2",
            "-c:a", "pcm_s16le",
            str(mix),
        ],
        ctx.cancel_event,
    )
    rel = str(mix.relative_to(ws))
    stage.artifacts.append(rel)
    stage.outputs = {"erased_mix": rel}
    stage.progress = 1.0
    manifest["mask_erase"] = {"targets": targets, "intervals": intervals}
    _end_stage(task, manifest, stage, StageStatus.DONE)
    return mix


def _mux_args(src: Path, mix: Path, tmp: Path, ext: str, full_streams: bool) -> list[str]:
    args = ["-i", str(src), "-i", str(mix)]
    if full_streams:
        # 字幕/章节/附件等在容器允许时一并复制；源音频替换为擦除混音
        args += ["-map", "0", "-map", "-0:a", "-map", "1:a:0"]
    else:
        args += ["-map", "0:v:0", "-map", "1:a:0"]
    args += [
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "256k",
        "-ar", str(PCM_SAMPLE_RATE),
        "-ac", "2",
        "-f", _MUXER_BY_EXT[ext],
        str(tmp),
    ]
    return args


def _stage_encode_mux(
    task, ctx: RunContext, ws: Path, manifest: dict, src: Path, mix: Path,
    targets: list[str],
) -> None:
    stage = _begin_stage(task, manifest, "encode_mux")
    names = workspace.read_speaker_names(ws)
    out = erased_output_path(src, targets, names, (task.erase or {}).get("output_dir"))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".part-{os.getpid()}")
    full_streams = True
    try:
        try:
            run_ffmpeg(_mux_args(src, mix, tmp, out.suffix, True), ctx.cancel_event)
        except TaskError:
            # 容器装不下全部流（如 MP4 装 PGS 字幕）：回退到仅视频+音频复制
            full_streams = False
            tmp.unlink(missing_ok=True)
            run_ffmpeg(_mux_args(src, mix, tmp, out.suffix, False), ctx.cancel_event)
        os.replace(tmp, out)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    manifest["output"] = {
        "path": str(out),
        "container": out.suffix,
        "full_streams": full_streams,
    }
    task.output_path = str(out)
    _end_stage(task, manifest, stage, StageStatus.DONE)


def _stage(task, stage_id: str):
    return next(s for s in task.stages if s.id == stage_id)


def _entry(manifest: dict, stage_id: str) -> dict:
    return next(m for m in manifest["stages"] if m["id"] == stage_id)


def _begin_stage(task, manifest: dict, stage_id: str):
    _check_cancelled(task.cancel_event)
    stage = _stage(task, stage_id)
    stage.status = StageStatus.RUNNING
    stage.started_at = now_iso()
    stage.finished_at = None
    stage.error = None  # 重跑的阶段不背着上次失败/取消的原因
    stage.progress = None
    stage.artifacts = []  # 产物重新登记，续跑不留重复条目
    stage.outputs = {}
    entry = _entry(manifest, stage_id)
    entry["status"] = stage.status.value
    entry["started_at"] = stage.started_at
    workspace.write_manifest(Path(task.workspace_dir), manifest)
    return stage


def _end_stage(task, manifest: dict, stage, status: StageStatus) -> None:
    stage.status = status
    stage.finished_at = now_iso()
    _sync_stage_entry(task, manifest, stage)


def _sync_stage_entry(task, manifest: dict, stage) -> dict:
    entry = _entry(manifest, stage.id)
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
