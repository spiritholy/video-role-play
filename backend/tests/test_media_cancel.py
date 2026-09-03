"""run_ffmpeg 的取消机制：取消事件触发时真正杀死 ffmpeg 子进程。"""

import threading

import pytest

from app.media import TaskCancelled, run_ffmpeg


def test_取消事件杀死正在运行的ffmpeg(tmp_path):
    out = tmp_path / "never_finished.mp4"
    cancel_event = threading.Event()
    threading.Timer(0.5, cancel_event.set).start()

    with pytest.raises(TaskCancelled):
        run_ffmpeg(
            [
                "-f", "lavfi", "-i",
                "testsrc2=size=640x360:rate=15:duration=600",
                "-c:v", "libx264", "-preset", "ultrafast", str(out),
            ],
            cancel_event,
        )
    # TaskCancelled 抛出即代表子进程被杀死（run_ffmpeg 内部 proc.kill + wait）；
    # 半成品清理由管线的 .part 临时文件机制负责。
