"""ML 推理注入边界：全项目唯一的测试注入点。

生产环境经此调用真实模型（Demucs 是第一个接入者，pyannote、Whisper 由
后续 issue 接入）；行为测试注入 canned 输出，真实模型与权重下载不进测试回路。
接口约定：separate() 在独立目录产出 stems 文件，经 on_progress(0..1) 汇报
进度；取消通过 on_progress 抛 TaskCancelled 实现（管线负责注入）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from .media import TaskError

ProgressCallback = Callable[[float], None]

# 分离产物的文件名契约：管线按返回路径登记，名称仅由 separator 决定
VOCALS_FILENAME = "vocals.wav"
ACCOMPANIMENT_FILENAME = "accompaniment.wav"


@dataclass(frozen=True)
class SeparationStems:
    vocal: Path
    accompaniment: Path


class VocalSeparator(Protocol):
    name: str

    def separate(
        self, wav: Path, out_dir: Path, on_progress: ProgressCallback
    ) -> SeparationStems: ...


@dataclass
class Inference:
    """统一注入容器：后续 pyannote / Whisper 接口挂在这里。"""

    separator: VocalSeparator


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
