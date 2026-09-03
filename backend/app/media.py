"""ffmpeg / ffprobe 进程封装：支持取消、干净错误。"""

from __future__ import annotations

import json
import subprocess
import threading

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


class TaskError(Exception):
    """任务干净失败的带原因错误（界面上直接展示）。"""


class TaskCancelled(Exception):
    """任务被用户取消。"""


def ffprobe_json(path) -> dict:
    proc = subprocess.run(
        [FFPROBE, "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise TaskError(f"无法读取视频文件（ffprobe 失败）: {_tail(proc.stderr)}")
    return json.loads(proc.stdout)


def audio_tracks(info: dict) -> list[dict]:
    """从 ffprobe 结果中取音轨列表（按流序号排序）。"""
    return sorted(
        (s for s in info.get("streams", []) if s.get("codec_type") == "audio"),
        key=lambda s: s["index"],
    )


def run_ffmpeg(args: list[str], cancel_event: threading.Event) -> None:
    """运行 ffmpeg；取消事件触发时杀死进程并抛 TaskCancelled。"""
    proc = subprocess.Popen(
        [FFMPEG, "-v", "error", "-y", *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    stderr_lines: list[str] = []

    def _drain() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_lines.append(line)

    reader = threading.Thread(target=_drain, daemon=True)
    reader.start()
    try:
        while True:
            code = proc.poll()
            if code is not None:
                break
            if cancel_event.is_set():
                proc.kill()
                raise TaskCancelled()
            reader.join(timeout=0.1)
    finally:
        proc.wait()
        reader.join(timeout=2)
    if proc.returncode != 0:
        raise TaskError(f"ffmpeg 执行失败: {_tail(''.join(stderr_lines))}")


def _tail(text: str, limit: int = 400) -> str:
    text = (text or "").strip()
    return text[-limit:]
