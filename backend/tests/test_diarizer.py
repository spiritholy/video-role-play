"""PyannoteDiarizer：token 缺失/协议未接受/token 无效/网络失败映射为明确指引。

单元接缝仅注入 token 读取与 pipeline 构造这两点（进度折算 _PyannoteProgress
为纯函数直接单测）；diarize 行为本身由 API 级测试经统一注入边界（canned
说话人时段）覆盖，真实模型不进测试回路。
"""

from dataclasses import dataclass
import os
import sys
import types
from pathlib import Path

import pytest

from huggingface_hub.constants import HF_HUB_CACHE

from app.inference import PyannoteDiarizer, SpeakerSegment, _PyannoteProgress
from app.media import TaskError


def clear_hf_token(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)


def test_生产加载指向HF缓存而不是pyannote默认缓存(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_ok")
    calls = []

    def fake_from_pretrained(model_id, use_auth_token=None, cache_dir=None):
        calls.append((model_id, use_auth_token, cache_dir))
        return object()

    fake_module = types.ModuleType("pyannote.audio")
    fake_module.Pipeline = type(
        "FakePipeline", (), {"from_pretrained": staticmethod(fake_from_pretrained)}
    )
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_module)
    PyannoteDiarizer._create_pipeline("pyannote/speaker-diarization-3.1", "hf_ok")
    assert len(calls) == 1
    model_id, token, cache_dir = calls[0]
    assert model_id == "pyannote/speaker-diarization-3.1"
    assert token == "hf_ok"
    assert cache_dir == HF_HUB_CACHE, "缓存目录必须与 huggingface_hub 默认缓存一致"
    assert os.environ.get("PYANNOTE_CACHE") == HF_HUB_CACHE


def test_token未配置时给出配置指引(monkeypatch):
    clear_hf_token(monkeypatch)

    def factory(token):
        raise AssertionError("token 未配置时不应尝试加载模型")

    with pytest.raises(TaskError) as ei:
        PyannoteDiarizer(pipeline_factory=factory).load_model()
    message = str(ei.value)
    assert "HF_TOKEN" in message
    assert "协议" in message
    assert "speaker-diarization" in message


def test_token可从HUGGINGFACE_TOKEN读取(monkeypatch):
    clear_hf_token(monkeypatch)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_from_huggingface_token")
    seen = []

    def factory(token):
        seen.append(token)
        return object()

    PyannoteDiarizer(pipeline_factory=factory).load_model()
    assert seen == ["hf_from_huggingface_token"]


def test_协议未接受时映射为接受协议指引(monkeypatch):
    from huggingface_hub.errors import GatedRepoError

    monkeypatch.setenv("HF_TOKEN", "hf_invalid")

    def factory(token):
        raise GatedRepoError(
            "Access to model pyannote/speaker-diarization-3.1 is restricted"
        )

    with pytest.raises(TaskError) as ei:
        PyannoteDiarizer(pipeline_factory=factory).load_model()
    message = str(ei.value)
    assert "协议" in message
    assert "HF_TOKEN" in message


def test_token无效时映射为接受协议指引(monkeypatch):
    # hub 将无效 token（401）折叠成"仓库不存在"（RepositoryNotFoundError），
    # 与 403 协议未接受同样归入"接受协议 + 确认 token"指引
    from huggingface_hub.errors import RepositoryNotFoundError

    monkeypatch.setenv("HF_TOKEN", "hf_invalid")

    def factory(token):
        raise RepositoryNotFoundError(
            "401 User does not have permission to access this resource"
        )

    with pytest.raises(TaskError) as ei:
        PyannoteDiarizer(pipeline_factory=factory).load_model()
    message = str(ei.value)
    assert "协议" in message
    assert "HF_TOKEN" in message


def test_网络失败时映射为检查网络指引(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_ok")

    def factory(token):
        raise ConnectionError("网络不可达")

    with pytest.raises(TaskError) as ei:
        PyannoteDiarizer(pipeline_factory=factory).load_model()
    message = str(ei.value)
    assert "网络" in message
    assert "网络不可达" in message


def test_加载成功则缓存管线只构造一次(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_ok")
    sentinel = object()
    calls = []

    def factory(token):
        calls.append(token)
        return sentinel

    d = PyannoteDiarizer(pipeline_factory=factory)
    assert d.load_model() is sentinel
    assert d.load_model() is sentinel
    assert calls == ["hf_ok"], "管线应只构造一次（模型只下载一次）"


@dataclass
class _FakeSeg:
    start: float
    end: float


class _FakeAnnotation:
    def itertracks(self, yield_label):
        assert yield_label is True
        return iter(
            [
                (_FakeSeg(1.3, 1.8), "a", "SPEAKER_00"),
                (_FakeSeg(0.2, 0.9), "b", "SPEAKER_01"),
            ]
        )


def test_说话人时段映射为结构化数据并按时间排序(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_ok")

    def fake_pipeline(file, hook=None):
        return _FakeAnnotation()

    d = PyannoteDiarizer(pipeline_factory=lambda token: fake_pipeline)
    result = d.diarize(Path("任意.wav"), Path("."), lambda f: None)
    assert result.segments == (
        SpeakerSegment("SPEAKER_01", 0.2, 0.9),
        SpeakerSegment("SPEAKER_00", 1.3, 1.8),
    )


class _FakeParamPipeline:
    """带参数的假 pyannote 管线：记录 instantiate/调用参数。"""

    def __init__(self):
        self.instantiated: list[dict] = []
        self.calls: list[dict] = []

    def parameters(self, instantiated=False):
        assert instantiated is True
        return {"clustering": {"min_cluster_size": 12}}

    def instantiate(self, params):
        self.instantiated.append(params)

    def __call__(self, file, hook=None, **kwargs):
        self.calls.append(kwargs)
        return _FakeAnnotation()


def test_指定说话人数量透传管线并还原聚类参数(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_ok")
    pipe = _FakeParamPipeline()
    d = PyannoteDiarizer(pipeline_factory=lambda token: pipe)
    d.diarize(Path("任意.wav"), Path("."), lambda f: None, num_speakers=4)
    assert pipe.calls == [{"num_speakers": 4}], "指定的说话人数量应透传给管线"
    assert pipe.instantiated == [
        {"clustering": {"min_cluster_size": 3}},  # 强制人数时调低 min_cluster_size
        {"clustering": {"min_cluster_size": 12}},  # 用完还原，不污染跨任务共享管线
    ]


def test_pyannote进度折算单调且到达1():
    reported = []
    p = _PyannoteProgress(reported.append)

    p("segmentation", None, completed=0, total=10)
    assert reported[-1] == pytest.approx(0.0)
    p("segmentation", None, completed=5, total=10)
    assert reported[-1] == pytest.approx(0.4), "segmentation 占 0.8 权重，一半即 0.4"
    p("segmentation", None, completed=20, total=10)
    assert reported[-1] == pytest.approx(0.8), "completed 超过 total 应封顶"
    p("speaker_counting", None)  # 瞬时步骤不影响整体折算
    assert reported[-1] == pytest.approx(0.8)
    p("embeddings", None, completed=1, total=4)
    assert reported[-1] == pytest.approx(0.85)
    p("embeddings", None, completed=4, total=4)
    assert reported[-1] == pytest.approx(1.0)
    assert reported == sorted(reported), "整体进度应保持单调不回跳"
