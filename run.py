"""一条命令启动：FastAPI 后端 + 已构建前端 + 自动打开浏览器。

用法：python3 run.py [--port 8765] [--no-browser]
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

import uvicorn  # noqa: E402

from app.main import create_app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="启动视频人声擦除工具")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    opened = threading.Event()

    def on_start() -> None:
        if not args.no_browser and not opened.is_set():
            opened.set()
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    print(f"视频人声擦除已启动：{url}（Ctrl+C 退出）")
    uvicorn.run(
        create_app(on_start=on_start),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
