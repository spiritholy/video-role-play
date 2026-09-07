"""一条命令启动：FastAPI 后端 + 已构建前端 + 自动打开浏览器。

用法：python3 run.py [--port 8770] [--no-browser] [--env-file .env]

配置来源：进程环境变量优先，缺失的键由仓库根 .env 补（模板 .env.example，
读取规则见 backend/app/envfile.py）。模型是懒加载的，所以缺 HF token 不拦停
启动，只在「说话人分离」阶段失败——这里提前提示一次。

端口：默认 8770（VRP_PORT 可改），刻意避开 AnkiConnect 常驻的 8765；未显式
指定 --port 时遇到占用会自动往后找空闲端口，显式指定则直接报错。WSL 下
「占用」连 Windows 宿主那一侧一起判，否则浏览器打开的是宿主上别人的服务
（判定与理由见 backend/app/ports.py）。
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

import uvicorn  # noqa: E402

from app.envfile import load_env_file  # noqa: E402
from app.main import create_app  # noqa: E402
from app.ports import ATTEMPTS, DEFAULT_PORT, PortUnavailable, choose_port  # noqa: E402

# 说话人分离的 HF token 候选变量（与 inference.HF_TOKEN_VARS 同一组约定）
HF_TOKEN_VARS = ("HF_TOKEN", "HUGGINGFACE_TOKEN")


def preferred_port() -> int:
    """未显式 --port 时的首选端口：VRP_PORT（进程环境或 .env）优先。"""
    raw = os.environ.get("VRP_PORT", "").strip()
    if not raw:
        return DEFAULT_PORT
    try:
        return int(raw)
    except ValueError:
        print(f"[warn] VRP_PORT={raw!r} 不是整数，忽略；用默认端口 {DEFAULT_PORT}")
        return DEFAULT_PORT


def main() -> None:
    parser = argparse.ArgumentParser(description="启动视频人声擦除工具")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"监听端口，默认 {DEFAULT_PORT}（VRP_PORT 可改）；不指定时遇占用自动往后避让",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument(
        "--env-file",
        default=str(REPO_ROOT / ".env"),
        help="环境变量文件，默认仓库根 .env（不存在则跳过）",
    )
    args = parser.parse_args()

    # 必须先于 create_app 与端口选择：工作区根目录、VRP_PORT 都在环境变量里
    loaded = load_env_file(Path(args.env_file))
    if loaded:
        print(f"[env] 已从 {args.env_file} 注入：{', '.join(loaded)}")
    if not any(os.environ.get(var) for var in HF_TOKEN_VARS):
        print(
            "[warn] 未配置 HF token：任务会在「说话人分离」阶段失败；"
            "认人/纠错与命名入库静默跳过（命名照常生效）。"
            "cp .env.example .env 并填入 token（详见 .env.example 注释）"
        )

    explicit_port = args.port is not None
    try:
        choice = choose_port(
            args.port if explicit_port else preferred_port(),
            host=args.host,
            attempts=1 if explicit_port else ATTEMPTS,
        )
    except PortUnavailable as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
    if choice.avoided:
        skipped = ", ".join(str(p) for p in choice.avoided)
        print(f"[port] {skipped} 已被占用，改用 {choice.port}（WSL 下 Windows 宿主的占用也算）")

    url = f"http://{args.host}:{choice.port}"
    opened = threading.Event()

    def on_start() -> None:
        if not args.no_browser and not opened.is_set():
            opened.set()
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    print(f"视频人声擦除已启动：{url}（Ctrl+C 退出）")
    uvicorn.run(
        create_app(on_start=on_start),
        host=args.host,
        port=choice.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
