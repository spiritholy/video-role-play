"""DemucsSeparator：权重下载失败映射为界面可读的明确错误。

单元接缝仅注入"加载失败"这一点；分离行为本身由 API 级测试经
统一注入边界（canned stems）覆盖，真实模型不进测试回路。
"""

import pytest

from app.inference import DemucsSeparator
from app.media import TaskError


def test_权重下载失败映射为明确错误():
    def broken_factory():
        raise RuntimeError("URLError: <urlopen error 网络不可达>")

    sep = DemucsSeparator(separator_factory=broken_factory)
    with pytest.raises(TaskError) as ei:
        sep.load_model()
    message = str(ei.value)
    assert "权重获取失败" in message
    assert "无需账号" in message
    assert "网络不可达" in message


def test_加载成功则缓存分离器():
    sentinel = object()
    calls = []

    def factory():
        calls.append(1)
        return sentinel

    sep = DemucsSeparator(separator_factory=factory)
    assert sep.load_model() is sentinel
    assert sep.load_model() is sentinel
    assert len(calls) == 1, "分离器应只构造一次（权重只下载一次）"
