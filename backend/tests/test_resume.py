"""断点续跑与取消语义：已完成阶段不重算，从断点阶段继续。

"已完成阶段零调用"以 ML 推理注入边界的调用计数断言（canned 替身记录每次
调用）；非 ML 阶段（音轨提取、掩码合成）是否重算以工作区产物文件的 mtime
佐证——两者都是外部可观察行为，状态断言全部走 HTTP API。
"""

import json
import os
import shutil
import threading
from pathlib import Path

import pytest

from app.media import TaskError
from conftest import (
    CannedDiarizer,
    CannedSeparator,
    CannedTranscriber,
    call_counts,
    erase,
    get_task,
    make_client,
    make_video,
    run_to_analysis,
    stage_of,
    statuses,
    submit_task,
    wait_for,
)

ERASED_MIX = Path("05-mask-erase") / "erased_mix.wav"


def resume(client, task_id: str):
    return client.post(f"/api/tasks/{task_id}/resume")


class FailOnceSeparator(CannedSeparator):
    """首次调用失败、之后正常：模拟可重试的人声/伴奏分离失败。"""

    def __init__(self, error: Exception):
        super().__init__()
        self._error = error

    def separate(self, wav: Path, out_dir: Path, on_progress):
        self.fail = None if self.calls else self._error
        return super().separate(wav, out_dir, on_progress)


class FailOnceTranscriber(CannedTranscriber):
    """首次调用失败、之后正常：模拟可重试的样本转写失败。"""

    def __init__(self, error: Exception):
        super().__init__()
        self._error = error

    def transcribe(self, wav: Path, samples, on_progress):
        self.fail = None if self.calls else self._error
        return super().transcribe(wav, samples, on_progress)


def test_转写失败后续跑_分离与说话人分离零调用(tmp_path):
    separator, diarizer = CannedSeparator(), CannedDiarizer()
    transcriber = FailOnceTranscriber(TaskError("转写失败（模拟）"))
    video = make_video(tmp_path, "断点.mp4")

    with make_client(
        tmp_path, separator=separator, diarizer=diarizer, transcriber=transcriber
    ) as client:
        task = submit_task(client, video)
        wait_for(lambda: get_task(client, task["id"])["state"] == "failed")
        failed = get_task(client, task["id"])
        assert "转写失败" in failed["error"]
        assert statuses(failed)["diarize"] == "done"
        assert statuses(failed)["transcribe"] == "failed"

        resp = resume(client, task["id"])
        assert resp.status_code == 200, resp.text
        assert resp.json()["state"] in ("queued", "running")
        wait_for(lambda: get_task(client, task["id"])["state"] == "completed")

        done = get_task(client, task["id"])
        assert done["error"] is None
        assert call_counts(separator, diarizer, transcriber) == (1, 1, 2), (
            "已完成阶段的 ML 推理不得重算，只有断点阶段（样本转写）重跑"
        )
        assert statuses(done) == {
            "audio_extract": "done",
            "separate": "done",
            "diarize": "done",
            "transcribe": "done",
            "mask_erase": "pending",
            "encode_mux": "pending",
        }
        assert client.get(f"/api/tasks/{task['id']}/transcripts").status_code == 200


def test_分离失败后续跑_已完成的音轨提取不重算(tmp_path):
    separator = FailOnceSeparator(TaskError("分离失败（模拟）"))
    video = make_video(tmp_path, "提取.mp4")
    ws = tmp_path / "workspaces" / "提取"

    with make_client(tmp_path, separator=separator) as client:
        task = submit_task(client, video)
        wait_for(lambda: get_task(client, task["id"])["state"] == "failed")
        assert statuses(get_task(client, task["id"]))["audio_extract"] == "done"

        wavs = list((ws / "01-audio-extract").glob("*.wav"))
        assert len(wavs) == 1
        before = wavs[0].stat().st_mtime_ns

        assert resume(client, task["id"]).status_code == 200
        wait_for(lambda: get_task(client, task["id"])["state"] == "completed")

        assert wavs[0].stat().st_mtime_ns == before, "已完成的音轨提取不应重算"
        assert len(separator.calls) == 2, "断点阶段（人声/伴奏分离）应重跑"
        assert list((ws / "01-audio-extract").glob("*.wav")) == wavs, (
            "续跑不应产出第二份音轨提取产物"
        )


def test_取消运行中任务_已完成产物保留且可续跑(tmp_path):
    gate = threading.Event()
    separator, transcriber = CannedSeparator(), CannedTranscriber()
    diarizer = CannedDiarizer(gate=gate)
    video = make_video(tmp_path, "取消.mp4")
    ws = tmp_path / "workspaces" / "取消"

    with make_client(
        tmp_path, separator=separator, diarizer=diarizer, transcriber=transcriber
    ) as client:
        task = submit_task(client, video)
        wait_for(lambda: stage_of(get_task(client, task["id"]), "diarize")["status"] == "running")

        assert client.post(f"/api/tasks/{task['id']}/cancel").status_code == 200
        gate.set()  # diarizer 在进度回调处观察到取消并中止
        wait_for(lambda: get_task(client, task["id"])["state"] == "cancelled")

        cancelled = get_task(client, task["id"])
        assert cancelled["state_label"] == "已取消"
        assert statuses(cancelled) == {
            "audio_extract": "done",
            "separate": "done",
            "diarize": "cancelled",
            "transcribe": "pending",
            "mask_erase": "pending",
            "encode_mux": "pending",
        }, "取消应停止当前及后续阶段"
        assert cancelled["output_path"] is None

        vocal = ws / "02-separate" / "vocals.wav"
        assert vocal.is_file(), "已完成阶段的产物保留在工作区"
        manifest = json.loads((ws / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["state"] == "cancelled"
        assert [s["status"] for s in manifest["stages"]][:2] == ["done", "done"]
        before = vocal.stat().st_mtime_ns

        assert resume(client, task["id"]).status_code == 200
        wait_for(lambda: get_task(client, task["id"])["state"] == "completed")

        done = get_task(client, task["id"])
        assert call_counts(separator, diarizer, transcriber) == (1, 2, 1), (
            "续跑从断点阶段（说话人分离）继续，之前的分离不重算"
        )
        assert vocal.stat().st_mtime_ns == before
        assert statuses(done)["transcribe"] == "done"
        assert client.get(f"/api/tasks/{task['id']}/diarization").status_code == 200


def test_工作区被清空后续跑从头重算(tmp_path):
    """清单不可用（工作区被删）时不得凭任务内存状态跳过阶段。"""
    separator = FailOnceSeparator(TaskError("分离失败（模拟）"))
    video = make_video(tmp_path, "清空.mp4")
    ws = tmp_path / "workspaces" / "清空"

    with make_client(tmp_path, separator=separator) as client:
        task = submit_task(client, video)
        wait_for(lambda: get_task(client, task["id"])["state"] == "failed")
        assert statuses(get_task(client, task["id"]))["audio_extract"] == "done"

        shutil.rmtree(ws)
        assert resume(client, task["id"]).status_code == 200
        wait_for(lambda: get_task(client, task["id"])["state"] == "completed")

        done = get_task(client, task["id"])
        assert statuses(done)["audio_extract"] == "done"
        assert list((ws / "01-audio-extract").glob("*.wav")), (
            "工作区被清空后音轨提取必须重算，产物重新落盘"
        )
        assert len(separator.calls) == 2
        assert json.loads((ws / "manifest.json").read_text(encoding="utf-8"))[
            "state"
        ] == "completed"


@pytest.mark.skipif(os.geteuid() == 0, reason="root 不受目录写权限约束")
def test_编码封装失败后续跑_掩码合成不重算(tmp_path):
    video = make_video(tmp_path, "封装.mp4")
    out_dir = tmp_path / "成品"
    out_dir.mkdir()
    ws = tmp_path / "workspaces" / "封装"

    with make_client(tmp_path) as client:
        task = run_to_analysis(client, video)
        try:
            out_dir.chmod(0o500)  # 只读目录：成品写不进去，编码封装干净失败
            assert erase(client, task["id"], ["SPEAKER_00"], str(out_dir)).status_code == 200
            wait_for(lambda: get_task(client, task["id"])["state"] == "failed")
        finally:
            out_dir.chmod(0o700)

        failed = get_task(client, task["id"])
        assert statuses(failed)["mask_erase"] == "done"
        assert statuses(failed)["encode_mux"] == "failed"
        assert failed["output_path"] is None
        before = (ws / ERASED_MIX).stat().st_mtime_ns

        assert resume(client, task["id"]).status_code == 200
        wait_for(lambda: get_task(client, task["id"])["state"] == "completed")

        done = get_task(client, task["id"])
        assert (ws / ERASED_MIX).stat().st_mtime_ns == before, "已完成的掩码合成不应重算"
        assert done["output_path"] == str(out_dir / "封装__erased-SPEAKER_00.mp4")
        assert Path(done["output_path"]).is_file()


def test_排队与运行中的任务不可续跑(tmp_path):
    gate = threading.Event()
    with make_client(tmp_path, diarizer=CannedDiarizer(gate=gate)) as client:
        running = submit_task(client, make_video(tmp_path, "运行中.mp4"))
        wait_for(lambda: get_task(client, running["id"])["state"] == "running")
        queued = submit_task(client, make_video(tmp_path, "排队中.mp4"))
        assert get_task(client, queued["id"])["state"] == "queued"

        assert resume(client, running["id"]).status_code == 409
        assert resume(client, queued["id"]).status_code == 409
        gate.set()
        wait_for(lambda: get_task(client, queued["id"])["state"] == "completed")


def test_完成任务不可续跑(client, tmp_path):
    task = run_to_analysis(client, make_video(tmp_path, "已完成.mp4"))
    resp = resume(client, task["id"])
    assert resp.status_code == 409, "分析已完成，下一步是发起擦除而非续跑"


def test_任务不存在时续跑被拒绝(client):
    assert resume(client, "不存在").status_code == 404
