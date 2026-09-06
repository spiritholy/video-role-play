"""启动端口选择：两侧占用都避让、问不了宿主不瞎换、显式指定时不偷偷改口。

宿主监听表靠 Windows 自带的 NETSTAT.EXE，测试里用假脚本顶替（喂一份真实机器
上抄来的输出，含 CRLF、GBK 乱码表头、IPv6 行与非监听行），不真的跑 interop；
本侧判定用真 socket 绑一个端口验证。
"""

import socket
from pathlib import Path

import pytest

from app import ports
from app.ports import (
    PortUnavailable,
    choose_port,
    local_port_free,
    windows_port_free,
)

# 中文 Windows 上 netstat -ano 的输出形态（表头 GBK，数据行 ASCII）
NETSTAT_SAMPLE = (
    "\r\n"
    "\x00\xef活动连接\r\n"
    "\r\n"
    "  Э  صַ          ⲿַ        ״̬           PID\r\n"
    "  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       1776\r\n"
    "  TCP    127.0.0.1:8765         0.0.0.0:0              LISTENING       25016\r\n"
    "  TCP    [::]:135               [::]:0                 LISTENING       1776\r\n"
    "  TCP    127.0.0.1:8765         127.0.0.1:52299        TIME_WAIT       0\r\n"
    "  TCP    127.0.0.1:52299        127.0.0.1:8765         ESTABLISHED     4321\r\n"
    "  UDP    0.0.0.0:5353           *:*                                    296\r\n"
)


class CallCounter:
    """记录被问过的端口，用于断言"不该白问一次"。"""

    def __init__(self, verdict):
        self.verdict = verdict
        self.ports: list[int] = []

    def __call__(self, port: int):
        self.ports.append(port)
        return self.verdict(port) if callable(self.verdict) else self.verdict


def free_local(host: str, port: int) -> bool:
    return True


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    ports.windows_listening_ports.cache_clear()
    yield
    ports.windows_listening_ports.cache_clear()


@pytest.fixture
def fake_netstat(tmp_path: Path, monkeypatch):
    """把宿主的 netstat 换成假脚本；返回记录调用次数的日志路径的工厂。"""

    def install(output: str = NETSTAT_SAMPLE, returncode: int = 0) -> Path:
        sample = tmp_path / "netstat.txt"
        sample.write_bytes(output.encode("utf-8", errors="replace"))
        calls = tmp_path / "calls.log"
        script = tmp_path / "NETSTAT.EXE"
        script.write_text(
            "#!/bin/sh\n"
            f'echo called >> "{calls}"\n'
            f'cat "{sample}"\n'
            f"exit {returncode}\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        monkeypatch.setattr(ports, "WINDOWS_NETSTAT", script)
        monkeypatch.setattr(ports, "is_wsl", lambda: True)
        return calls

    return install


def test_默认端口避开AnkiConnect常驻的8765():
    assert ports.DEFAULT_PORT != 8765


def test_首选端口空闲时原样选中():
    choice = choose_port(8770, local_free=free_local, windows_free=CallCounter(True))
    assert choice == ports.PortChoice(port=8770, avoided=())


def test_本侧被占用则避让到下一个端口():
    local = lambda host, port: port != 8770  # noqa: E731
    choice = choose_port(8770, local_free=local, windows_free=CallCounter(True))
    assert choice.port == 8771
    assert choice.avoided == (8770,)


def test_宿主被占用也避让_AnkiConnect场景():
    # 本侧绑得上、宿主上却已被 AnkiConnect 占住：正是浏览器打开成 AnkiConnect
    # 应答的那种冲突，必须换端口
    windows = CallCounter(lambda port: port != 8765)
    choice = choose_port(8765, local_free=free_local, windows_free=windows)
    assert choice.port == 8766
    assert choice.avoided == (8765,)
    assert windows.ports == [8765, 8766]


def test_本侧已占用时不再去问宿主():
    windows = CallCounter(True)
    local = lambda host, port: port != 8770  # noqa: E731
    choose_port(8770, local_free=local, windows_free=windows)
    assert windows.ports == [8771]  # 8770 本侧就否掉了


def test_问不了宿主时只信本侧且不反复重试():
    windows = CallCounter(None)  # 不在 WSL / interop 被禁 / netstat 找不到
    local = lambda host, port: port != 8770  # noqa: E731
    choice = choose_port(8770, local_free=local, windows_free=windows)
    assert choice.port == 8771
    assert windows.ports == [8771]  # 只白问一次，后续候选不再问


def test_连续占满尝试次数后报错并给出可操作提示():
    local = lambda host, port: False  # noqa: E731
    with pytest.raises(PortUnavailable) as excinfo:
        choose_port(8770, local_free=local, windows_free=CallCounter(True), attempts=3)
    message = str(excinfo.value)
    assert "8770, 8771, 8772" in message
    assert "--port" in message


def test_显式指定端口时不避让直接报错():
    local = lambda host, port: False  # noqa: E731
    with pytest.raises(PortUnavailable):
        choose_port(9999, local_free=local, windows_free=CallCounter(True), attempts=1)


def test_本侧判定认得真实socket占用():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        taken = busy.getsockname()[1]
        assert local_port_free("127.0.0.1", taken) is False

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        freed = probe.getsockname()[1]
    assert local_port_free("127.0.0.1", freed) is True


def test_监听表解析只认LISTENING行并兼容IPv6():
    assert ports._listening_ports(NETSTAT_SAMPLE) == frozenset({135, 8765})


def test_宿主监听表里有该端口判为占用(fake_netstat):
    fake_netstat()
    assert windows_port_free(8765) is False  # AnkiConnect 在宿主上监听
    assert windows_port_free(8770) is True


def test_宿主监听表一次启动只问一次(fake_netstat):
    calls = fake_netstat()
    assert windows_port_free(8770) is True
    assert windows_port_free(8771) is True
    assert len(calls.read_text(encoding="utf-8").splitlines()) == 1


def test_netstat跑不起来时无从判断(fake_netstat):
    fake_netstat(output="", returncode=1)
    assert windows_port_free(8770) is None


def test_不在WSL下不去问宿主(fake_netstat, monkeypatch):
    calls = fake_netstat()
    monkeypatch.setattr(ports, "is_wsl", lambda: False)
    assert windows_port_free(8770) is None
    assert not calls.exists()  # 假脚本压根没被调用


def test_宿主netstat缺失时无从判断(monkeypatch):
    monkeypatch.setattr(ports, "is_wsl", lambda: True)
    monkeypatch.setattr(ports, "WINDOWS_NETSTAT", Path("/nonexistent/NETSTAT.EXE"))
    assert windows_port_free(8770) is None


def test_选择端口时默认探测走宿主监听表(monkeypatch):
    monkeypatch.setattr(ports, "windows_listening_ports", lambda: frozenset({8765}))
    choice = choose_port(8765, local_free=free_local)
    assert choice.port == 8766
    assert choice.avoided == (8765,)
