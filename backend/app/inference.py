"""ML 推理注入边界：全项目唯一的测试注入点。

生产环境经此调用真实模型（Demucs 是人声/伴奏分离的接入者，pyannote 是
说话人分离的接入者，Whisper 由后续 issue 接入）；行为测试注入 canned 输出，
真实模型与权重下载不进测试回路。
接口约定：separate() 在独立目录产出 stems 文件，经 on_progress(0..1) 汇报
进度；diarize() 返回结构化说话人时段（由管线序列化落盘）。取消通过
on_progress 抛 TaskCancelled 实现（管线负责注入）。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from huggingface_hub.errors import (
    GatedRepoError,
    RepositoryNotFoundError,
)

from .media import TaskCancelled, TaskError

ProgressCallback = Callable[[float], None]

# 分离产物的文件名契约：管线按返回路径登记，名称仅由 separator 决定
VOCALS_FILENAME = "vocals.wav"
ACCOMPANIMENT_FILENAME = "accompaniment.wav"

# HF token 的候选环境变量（huggingface 官方惯例，任一命中即可）
HF_TOKEN_VARS = ("HF_TOKEN", "HUGGINGFACE_TOKEN")


@dataclass(frozen=True)
class SeparationStems:
    vocal: Path
    accompaniment: Path


@dataclass(frozen=True)
class SpeakerSegment:
    """说话人时段（CONTEXT.md：某个说话人正在说话的时间区间）。"""

    speaker: str
    start: float
    end: float


@dataclass(frozen=True)
class DiarizationResult:
    segments: tuple[SpeakerSegment, ...]


class VocalSeparator(Protocol):
    name: str

    def separate(
        self, wav: Path, out_dir: Path, on_progress: ProgressCallback
    ) -> SeparationStems: ...


class SpeakerDiarizer(Protocol):
    name: str

    def diarize(
        self, wav: Path, out_dir: Path, on_progress: ProgressCallback
    ) -> DiarizationResult: ...


@dataclass
class Inference:
    """统一注入容器：pyannote 已接入，Whisper 由后续 issue 接口挂在这里。"""

    separator: VocalSeparator
    diarizer: SpeakerDiarizer


def read_hf_token() -> str | None:
    for var in HF_TOKEN_VARS:
        token = os.environ.get(var)
        if token:
            return token
    return None


class DemucsSeparator:
    """htdemucs（CPU）人声/伴奏分离；权重缺失时自动从公开地址下载。

    基于 demucs.api.Separator（4.1.0 官方高层 API），参数与 CLI 默认一致
    （shifts=1、overlap=0.25、split、rescale 防削波、16bit wav、伴奏=其余
    stem 求和）。进度用官方 callback 的分段 end 事件折算。
    """

    name = "htdemucs"

    def __init__(self, model_name: str = "htdemucs", separator_factory=None):
        self._model_name = model_name
        # separator_factory 仅用于测试注入加载失败；生产走 demucs.api.Separator
        # （构造时经 torch.hub 自动下载权重到本地缓存，无需账号）
        self._separator_factory = separator_factory
        self._separator = None
        self._chunks_done = 0
        self._stride = 0

    def load_model(self):
        """懒加载并缓存分离器；首次调用触发权重下载（无需账号）。"""
        if self._separator is not None:
            return self._separator
        if self._separator_factory is None:
            try:
                from demucs.api import Separator
            except ImportError as e:
                raise TaskError(
                    "缺少依赖 demucs/torch：请先安装 backend/requirements.txt"
                ) from e
            self._separator_factory = (
                lambda: Separator(model=self._model_name, device="cpu")
            )
        try:
            self._separator = self._separator_factory()
        except Exception as e:
            raise TaskError(
                f"Demucs 模型权重获取失败（{self._model_name}）：{e}。"
                "首次使用需联网自动下载权重（公开地址，无需账号），"
                "请检查网络连接后重试。"
            ) from e
        return self._separator

    def separate(
        self, wav: Path, out_dir: Path, on_progress: ProgressCallback
    ) -> SeparationStems:
        from demucs.api import save_audio

        sep = self.load_model()
        out_dir.mkdir(parents=True, exist_ok=True)
        vocal_path = out_dir / VOCALS_FILENAME
        accomp_path = out_dir / ACCOMPANIMENT_FILENAME

        def on_chunk(info: dict) -> None:
            if info.get("state") != "end":
                return
            self._chunks_done += 1
            audio_length = info.get("audio_length") or 0
            if audio_length > 0 and self._stride > 0:
                total = max(1, math.ceil(audio_length / self._stride))
                on_progress(min(1.0, self._chunks_done / total))

        self._chunks_done = 0
        self._stride = self._compute_stride(sep)
        sep.update_parameter(callback=on_chunk)
        try:
            _, stems = sep.separate_audio_file(Path(wav))
        finally:
            sep.update_parameter(callback=None)

        vocals = stems["vocals"]
        others = [stems[s] for s in sep.model.sources if s != "vocals"]
        accompaniment = others[0]
        for extra in others[1:]:
            accompaniment = accompaniment + extra
        for path, tensor in ((vocal_path, vocals), (accomp_path, accompaniment)):
            save_audio(
                tensor, str(path), samplerate=sep.samplerate,
                clip="rescale", bits_per_sample=16,
            )
        return SeparationStems(vocal=vocal_path, accompaniment=accomp_path)

    def _compute_stride(self, sep) -> int:
        """与 demucs 分段逻辑一致的重叠步长，用于把段数折算成进度。"""
        model = sep.model
        segment = getattr(model, "segment", None)
        if segment is None:
            segment = model.max_allowed_segment
        segment_length = int(sep.samplerate * float(segment))
        return int((1 - 0.25) * segment_length)


# HF token / 协议失败的用户指引（界面上直接展示）
_TOKEN_MISSING_MESSAGE = (
    "HF token 未配置：pyannote 为门控模型，需一次性配置后完全离线运行。"
    "① 注册 huggingface.com；② 在模型页 pyannote/speaker-diarization-3.1 与"
    " pyannote/segmentation-3.0 接受使用协议；③ 设置环境变量 HF_TOKEN=<你的token>"
    " 后重启应用再试。"
)
_TOKEN_REJECTED_MESSAGE = (
    "pyannote 模型下载被拒绝：HF token 无效或模型协议未接受。"
    "请到 huggingface.com 模型页 pyannote/speaker-diarization-3.1 与"
    " pyannote/segmentation-3.0 接受使用协议，确认 HF_TOKEN 有效后重试。"
)
# pyannote 两个耗时步骤的进度权重；其余步骤瞬时，不参与折算
_PYANNOTE_STEP_WEIGHTS = {"segmentation": 0.8, "embeddings": 0.2}


class _PyannoteProgress:
    """把 pyannote 的 step 级 hook 调用折算为整体 0..1 进度。

    pyannote hook 协议（speaker-diarization 3.1）：hook(step_name,
    step_artifact, completed=i, total=n, file=file)；不带 completed/total
    的调用表示该步骤整体完成。折算保持单调：segmentation 段 0→0.8，
    embeddings 段 0.8→1.0。取消经管线注入的 on_progress 抛 TaskCancelled。
    """

    def __init__(self, on_progress: ProgressCallback):
        self._on_progress = on_progress
        self._fractions: dict[str, float] = {}

    def __call__(self, step_name, step_artifact, completed=None, total=None, file=None):
        if step_name not in _PYANNOTE_STEP_WEIGHTS:
            return
        if completed is None or not total:
            self._fractions[step_name] = 1.0
        else:
            current = min(1.0, completed / total)
            self._fractions[step_name] = max(
                self._fractions.get(step_name, 0.0), current
            )
        overall = sum(
            self._fractions.get(step, 0.0) * weight
            for step, weight in _PYANNOTE_STEP_WEIGHTS.items()
        )
        self._on_progress(overall)


class PyannoteDiarizer:
    """pyannote.audio 3.1 speaker-diarization（CPU），第二个接入推理接口的模型。

    门控模型：首次使用需 HF token（HF_TOKEN / HUGGINGFACE_TOKEN）并在线接受
    模型协议，权重下载后完全离线。说话人数量全自动估计——刻意不传任何
    人数提示参数（不确定人数时乱填反而干扰估计，spec #1 用户决策）。
    """

    name = "pyannote/speaker-diarization-3.1"

    def __init__(self, pipeline_factory=None):
        # pipeline_factory 仅用于测试注入；生产走 Pipeline.from_pretrained
        self._pipeline_factory = pipeline_factory
        self._pipeline = None

    def load_model(self):
        """懒加载并缓存 pipeline；首次调用触发权重下载（需 token 与协议）。"""
        if self._pipeline is not None:
            return self._pipeline
        token = read_hf_token()
        if token is None:
            raise TaskError(_TOKEN_MISSING_MESSAGE)
        factory = self._pipeline_factory or (
            lambda t: self._create_pipeline(self.name, t)
        )
        try:
            self._pipeline = factory(token)
        except TaskError:
            raise
        except Exception as e:
            raise TaskError(
                f"pyannote 模型加载失败：{e}。首次使用需联网下载模型"
                "（需 HF token），请检查网络连接后重试。"
            ) from e
        return self._pipeline

    def diarize(
        self, wav: Path, out_dir: Path, on_progress: ProgressCallback
    ) -> DiarizationResult:
        pipeline = self.load_model()
        try:
            annotation = pipeline(str(wav), hook=_PyannoteProgress(on_progress))
        except TaskCancelled:
            raise
        except TaskError:
            raise
        except Exception as e:
            raise TaskError(f"pyannote 说话人分离推理失败: {e}") from e
        segments = [
            SpeakerSegment(speaker=label, start=seg.start, end=seg.end)
            for seg, _, label in annotation.itertracks(yield_label=True)
        ]
        segments.sort(key=lambda s: s.start)
        return DiarizationResult(segments=tuple(segments))

    @staticmethod
    def _create_pipeline(model_id: str, token: str):
        try:
            from pyannote.audio import Pipeline
        except ImportError as e:
            raise TaskError(
                "缺少依赖 pyannote.audio：请先安装 backend/requirements.txt"
            ) from e
        try:
            return Pipeline.from_pretrained(model_id, use_auth_token=token)
        except (GatedRepoError, RepositoryNotFoundError) as e:
            # 403 = 协议未接受；401/404（hub 将无效 token 折叠成"仓库不存在"）
            raise TaskError(_TOKEN_REJECTED_MESSAGE) from e
