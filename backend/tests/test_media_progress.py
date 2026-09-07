"""run_ffmpeg 的进度机制：进度回调按媒体时长折算到 0..1。"""

import threading

from app.media import run_ffmpeg


def test_run_ffmpeg按时长汇报进度(tmp_path):
    out = tmp_path / "progress.wav"
    fractions: list[float] = []
    run_ffmpeg(
        [
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-c:a", "pcm_s16le", str(out),
        ],
        threading.Event(),
        on_progress=fractions.append,
        total_seconds=1.0,
    )
    assert out.is_file()
    assert fractions
    assert fractions[-1] == 1.0
    assert all(0 <= f <= 1 for f in fractions)
