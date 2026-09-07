"""ffmpeg / ffprobe 进程封装：支持取消、干净错误。"""

from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Callable

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


def run_ffmpeg(
    args: list[str],
    cancel_event: threading.Event,
    on_progress: Callable[[float], None] | None = None,
    total_seconds: float = 0.0,
) -> None:
    """运行 ffmpeg；取消事件触发时杀死进程并抛 TaskCancelled。

    传入 on_progress 时，用 ffmpeg 的 -progress 输出按总时长折算 0..1；
    缺少总时长时不折算，界面可继续显示不确定进度。
    """
    proc = subprocess.Popen(
        [FFMPEG, *(["-progress", "pipe:1", "-nostats"] if on_progress else []),
         "-v", "error", "-y", *args],
        stdout=subprocess.PIPE if on_progress else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    stderr_lines: list[str] = []
    last_fraction = 0.0

    def _drain() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_lines.append(line)

    def _read_progress() -> None:
        assert proc.stdout is not None
        assert on_progress is not None
        nonlocal last_fraction
        for line in proc.stdout:
            key, _, value = line.strip().partition("=")
            if key not in ("out_time_us", "out_time_ms"):
                continue
            try:
                # ffmpeg 的 out_time_ms 实际也是微秒。
                seconds = int(value) / 1_000_000
            except ValueError:
                continue
            if total_seconds <= 0 or seconds < 0:
                continue
            last_fraction = max(last_fraction, min(1.0, seconds / total_seconds))
            on_progress(last_fraction)

    reader = threading.Thread(target=_drain, daemon=True)
    reader.start()
    progress_reader = (
        threading.Thread(target=_read_progress, daemon=True)
        if on_progress else None
    )
    if progress_reader is not None:
        progress_reader.start()
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
        if progress_reader is not None:
            progress_reader.join(timeout=2)
    if proc.returncode == 0:
        if on_progress is not None and total_seconds > 0:
            on_progress(1.0)
    else:
        raise TaskError(f"ffmpeg 执行失败: {_tail(''.join(stderr_lines))}")


def run_ffmpeg_bytes(args: list[str]) -> bytes:
    """运行 ffmpeg 并返回 stdout 字节流（切片等内存输出）；失败抛 TaskError。"""
    proc = subprocess.run(
        [FFMPEG, "-v", "error", *args],
        capture_output=True,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace")
        raise TaskError(f"ffmpeg 执行失败: {_tail(stderr)}")
    return proc.stdout


def _tail(text: str, limit: int = 400) -> str:
    text = (text or "").strip()
    return text[-limit:]
