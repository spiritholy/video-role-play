"""启动端口选择：默认端口避开 AnkiConnect，被占用时自动往后避让。

为什么不用 8765：那是 Anki 插件 AnkiConnect 的固定端口，装了 Anki 的机器上它
常驻监听。这个冲突在 WSL 下还看不出端倪——WSL 与 Windows 是两个网络命名空间，
uvicorn 绑 127.0.0.1:8765 照样成功，run.py 照样打印「已启动」；但打开浏览器走
的是 wslview，落地的是 Windows 浏览器，它访问的 localhost:8765 属于 Windows
那一侧，而 WSL 的 localhost 转发因为该端口已被 AnkiConnect 占住接不上，页面就
成了 {"apiVersion": "AnkiConnect v.6"}。

所以「端口空闲」两侧都要判：
- WSL 内：socket 试绑（与 uvicorn 一样开 SO_REUSEADDR，TIME_WAIT 不算占用）；
- Windows 侧：读宿主的 netstat 监听表。只能借 Windows 自己的进程问——从 WSL
  里连不到 Windows 的 loopback。用 netstat 一次性取全表而不是逐端口探：一发
  约 60ms，而 loopback 上逐个 curl 探在这类机器上要等两秒才拿到「连接被拒」。
  问不了（不在 WSL、interop 被禁、找不到 NETSTAT.EXE）就退化为只信 WSL 内那
  一侧：探测能力缺失不该反过来逼着换端口。
"""

from __future__ import annotations

import re
import socket
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_PORT = 8770
# 未显式指定 --port 时向后避让的尝试次数（含首选端口本身）
ATTEMPTS = 20
# 宿主 netstat 的等待上限（秒）：正常几十毫秒，给足冷启动余量
NETSTAT_TIMEOUT = 5.0

# Windows 自带的 netstat（interop 可达）；不在默认挂载点就当作问不了
WINDOWS_NETSTAT = Path("/mnt/c/Windows/System32/NETSTAT.EXE")

# 监听行：协议列 + 本地地址（0.0.0.0:135 / [::]:135）+ 状态列。表头是本地化
# 的（中文 Windows 输出 GBK，解码后是乱码），但状态词恒为英文，故只认它。
_LISTENING_LINE = re.compile(r"LISTENING", re.IGNORECASE)


class PortUnavailable(RuntimeError):
    """首选端口起连续若干个都被占用，没有可启动的端口。"""


@dataclass(frozen=True)
class PortChoice:
    """选中的端口，以及为躲开占用而跳过的端口（供启动日志解释为何改口）。"""

    port: int
    avoided: tuple[int, ...] = ()


@lru_cache(maxsize=1)
def is_wsl() -> bool:
    try:
        version = Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False
    return "microsoft" in version


def local_port_free(host: str, port: int) -> bool:
    """本侧（WSL 内）端口是否可绑。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _listening_ports(netstat_output: str) -> frozenset[int]:
    """从 netstat -ano 输出里取所有 LISTENING 的本地端口（IPv4 与 IPv6 一并）。"""
    ports: set[int] = set()
    for line in netstat_output.splitlines():
        if not _LISTENING_LINE.search(line):
            continue
        fields = line.split()
        if len(fields) < 4:
            continue
        try:
            ports.add(int(fields[1].rsplit(":", 1)[1]))
        except (IndexError, ValueError):
            continue  # 认不出的行跳过：探测不该因为一行格式变化就整体失效
    return frozenset(ports)


@lru_cache(maxsize=1)
def windows_listening_ports(timeout: float = NETSTAT_TIMEOUT) -> frozenset[int] | None:
    """Windows 宿主上正在 LISTENING 的 TCP 端口；None 表示无从判断。

    结果按进程缓存：选端口时逐个候选都要问一次，而宿主监听表在一次启动里
    不会变。必须在 uvicorn 起来之前问——起来之后 WSL 的 localhost 转发会让
    宿主同一端口也显示为监听，那时探到的「占用」是自己。
    """
    if not is_wsl() or not WINDOWS_NETSTAT.is_file():
        return None
    try:
        proc = subprocess.run(
            [str(WINDOWS_NETSTAT), "-ano"],
            capture_output=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None  # interop 被禁、二进制跑不起来：判断不了
    if proc.returncode != 0:
        return None
    # 表头随 Windows 显示语言变化（中文机器上是 GBK），数据行是 ASCII
    return _listening_ports(proc.stdout.decode("utf-8", errors="replace"))


def windows_port_free(port: int) -> bool | None:
    """宿主上该端口是否空闲；None 表示无从判断。"""
    listening = windows_listening_ports()
    if listening is None:
        return None
    return port not in listening


def choose_port(
    preferred: int = DEFAULT_PORT,
    *,
    host: str = "127.0.0.1",
    attempts: int = ATTEMPTS,
    local_free=local_port_free,
    windows_free=windows_port_free,
) -> PortChoice:
    """从 preferred 起找第一个两侧都空闲的端口；找不到就 PortUnavailable。

    attempts=1 即「不许避让」——用户显式 --port 时该报错而不是偷偷换端口。
    """
    avoided: list[int] = []
    probe_windows = True
    for offset in range(max(1, attempts)):
        port = preferred + offset
        if port > 65535:
            break
        if not local_free(host, port):
            avoided.append(port)
            continue  # 本侧就占了，不必再问宿主
        if probe_windows:
            verdict = windows_free(port)
            if verdict is None:
                probe_windows = False  # 问不了宿主，后续候选也不再白问
            elif not verdict:
                avoided.append(port)
                continue
        return PortChoice(port=port, avoided=tuple(avoided))
    raise PortUnavailable(
        f"端口被占用，无可用端口：{', '.join(str(p) for p in avoided)}。"
        "用 --port 指定一个空闲端口（WSL 下 Windows 宿主的占用也算，"
        "例如 AnkiConnect 常驻 8765）"
    )
