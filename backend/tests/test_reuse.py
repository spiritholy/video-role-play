"""同源复用：同一源视频的新任务直接复用工作区四个分析阶段的产物。

复用以 ML 推理注入边界的调用计数断言（前四阶段零调用）；命名继承经时间轴
端点与成品文件名断言。复用条件不成立（换音轨、源视频变了、产物缺失）时
按未完成处理；某阶段重算则其后阶段一并作废，不复用旧分析的产物。
"""

import json
from pathlib import Path

from conftest import (
    CannedDiarizer,
    CannedSeparator,
    CannedTranscriber,
    call_counts,
    erase_to_completion,
    make_client,
    make_video,
    run_to_analysis,
    stage_of,
    statuses,
)

ANALYSIS_STATUSES = {
    "audio_extract": "done",
    "separate": "done",
    "diarize": "done",
    "recognize": "done",
    "transcribe": "done",
    "mask_erase": "pending",
    "encode_mux": "pending",
}

ERASED_MIX = Path("05-mask-erase") / "erased_mix.wav"


def rename(client, task_id: str, label: str, name: str):
    resp = client.put(
        f"/api/tasks/{task_id}/speakers/{label}/name", json={"name": name}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def name_of(client, task_id: str, label: str) -> str | None:
    resp = client.get(f"/api/tasks/{task_id}/diarization")
    assert resp.status_code == 200, resp.text
    speaker = next(s for s in resp.json()["speakers"] if s["label"] == label)
    return speaker["name"]


def manifest_of(task: dict) -> dict:
    return json.loads(
        (Path(task["workspace_dir"]) / "manifest.json").read_text(encoding="utf-8")
    )


def test_同源第二个任务复用分析产物_前四阶段零调用(tmp_path):
    separator, diarizer, transcriber = CannedSeparator(), CannedDiarizer(), CannedTranscriber()
    video = make_video(tmp_path, "同源.mp4")

    with make_client(
        tmp_path, separator=separator, diarizer=diarizer, transcriber=transcriber
    ) as client:
        first = run_to_analysis(client, video)
        assert call_counts(separator, diarizer, transcriber) == (1, 1, 1)

        second = run_to_analysis(client, video)
        assert second["id"] != first["id"]
        assert second["workspace_dir"] == first["workspace_dir"], "同源任务共用一个工作区"
        assert call_counts(separator, diarizer, transcriber) == (1, 1, 1), (
            "复用工作区产物时，音轨提取/分离/说话人分离/样本转写零推理调用"
        )
        assert statuses(second) == ANALYSIS_STATUSES

        # 复用后新任务的时间轴、样本台词与试听端点立即可用
        assert client.get(f"/api/tasks/{second['id']}/diarization").status_code == 200
        assert client.get(f"/api/tasks/{second['id']}/transcripts").status_code == 200
        assert client.get(
            f"/api/tasks/{second['id']}/audio-segment?start=0.2&end=0.9"
        ).status_code == 200
        assert client.get(
            f"/api/tasks/{second['id']}/audio/02-separate/vocals.wav"
        ).status_code == 200


def test_同源第二个任务继承命名并用于成品命名(tmp_path):
    separator, diarizer, transcriber = CannedSeparator(), CannedDiarizer(), CannedTranscriber()
    video = make_video(tmp_path, "继承.mp4")

    with make_client(
        tmp_path, separator=separator, diarizer=diarizer, transcriber=transcriber
    ) as client:
        first = run_to_analysis(client, video)
        rename(client, first["id"], "SPEAKER_00", "男主")

        second = run_to_analysis(client, video)
        assert name_of(client, second["id"], "SPEAKER_00") == "男主", (
            "命名随工作区存续，同源新任务自动继承"
        )

        done = erase_to_completion(client, second["id"], ["SPEAKER_00"])
        # 纯命名不改声纹库、认人产物不失效：同源新任务复用全部分析产物
        assert call_counts(separator, diarizer, transcriber) == (1, 1, 1)
        assert done["output_path"] == str(video.with_name("继承__erased-男主.mp4")), (
            "继承的命名应可用于成品命名"
        )
        assert Path(done["output_path"]).is_file()


def test_换音轨的同源任务不复用分析产物(tmp_path):
    separator = CannedSeparator()
    video = make_video(tmp_path, "多音轨.mp4", extra_tracks=1)

    with make_client(tmp_path, separator=separator) as client:
        first = run_to_analysis(client, video, audio_track=0)
        assert stage_of(first, "audio_extract")["artifacts"] == ["01-audio-extract/track1.wav"]

        second = run_to_analysis(client, video, audio_track=1)
        assert len(separator.calls) == 2, "换音轨是另一份分析，不得复用"
        assert stage_of(second, "audio_extract")["artifacts"] == ["01-audio-extract/track2.wav"]
        manifest = manifest_of(second)
        assert manifest["requested_audio_track"] == 1
        assert manifest["audio_track"]["index"] == 1


def test_源视频被换成另一个文件时不复用分析产物(tmp_path):
    """同路径不同体积＝另一个源视频：复用会把旧音频的擦除结果封装进新视频。"""
    separator = CannedSeparator()
    video = make_video(tmp_path, "换片.mp4", duration=2)

    with make_client(tmp_path, separator=separator) as client:
        run_to_analysis(client, video)
        old_size = video.stat().st_size
        make_video(tmp_path, "换片.mp4", duration=4, audio_freq=880)  # 同路径换成另一个文件
        assert video.stat().st_size != old_size

        second = run_to_analysis(client, video)
        assert len(separator.calls) == 2, "源视频变了，分析产物不得复用"
        assert statuses(second) == ANALYSIS_STATUSES
        assert manifest_of(second)["source"]["size_bytes"] == video.stat().st_size


def test_清单标记完成但产物文件缺失时该阶段及其后阶段重算(tmp_path):
    separator, diarizer, transcriber = CannedSeparator(), CannedDiarizer(), CannedTranscriber()
    video = make_video(tmp_path, "缺产物.mp4")

    with make_client(
        tmp_path, separator=separator, diarizer=diarizer, transcriber=transcriber
    ) as client:
        first = run_to_analysis(client, video)
        (Path(first["workspace_dir"]) / "03-diarize" / "diarization.json").unlink()

        second = run_to_analysis(client, video)
        assert call_counts(separator, diarizer, transcriber) == (1, 2, 2), (
            "产物齐全的阶段仍复用；说话人分离重算后，依赖它的样本转写一并重做"
        )
        assert statuses(second) == ANALYSIS_STATUSES
        assert client.get(f"/api/tasks/{second['id']}/transcripts").status_code == 200


def test_说话人分离重算后旧擦除混音作废(tmp_path):
    """分析链断过：目标未变也不得复用基于旧说话人时段合成的擦除混音。"""
    separator, diarizer, transcriber = CannedSeparator(), CannedDiarizer(), CannedTranscriber()
    video = make_video(tmp_path, "作废.mp4")

    with make_client(
        tmp_path, separator=separator, diarizer=diarizer, transcriber=transcriber
    ) as client:
        first = run_to_analysis(client, video)
        erase_to_completion(client, first["id"], ["SPEAKER_00"])
        mix = Path(first["workspace_dir"]) / ERASED_MIX
        before = mix.stat().st_mtime_ns
        (Path(first["workspace_dir"]) / "03-diarize" / "diarization.json").unlink()

        second = run_to_analysis(client, video)
        assert call_counts(separator, diarizer, transcriber) == (1, 2, 2)
        assert statuses(second)["mask_erase"] == "pending", "旧擦除混音应随分析作废"

        done = erase_to_completion(client, second["id"], ["SPEAKER_00"])
        assert mix.stat().st_mtime_ns != before, "掩码合成必须按新的说话人时段重做"
        assert Path(done["output_path"]).is_file()


def test_同目标再次擦除复用已合成的擦除混音(client, tmp_path):
    video = make_video(tmp_path, "复用掩码.mp4")
    task = run_to_analysis(client, video)
    first = erase_to_completion(client, task["id"], ["SPEAKER_00"])

    mix = tmp_path / "workspaces" / "复用掩码" / ERASED_MIX
    before = mix.stat().st_mtime_ns
    second = erase_to_completion(client, task["id"], ["SPEAKER_00"])

    assert mix.stat().st_mtime_ns == before, "同一组目标说话人的擦除混音应复用"
    assert second["output_path"] == first["output_path"]
    assert Path(second["output_path"]).is_file(), "复用混音仍要产出新成品"


def test_换目标再次擦除重做掩码合成(client, tmp_path):
    video = make_video(tmp_path, "换目标.mp4")
    task = run_to_analysis(client, video)
    erase_to_completion(client, task["id"], ["SPEAKER_00"])

    mix = tmp_path / "workspaces" / "换目标" / ERASED_MIX
    before = mix.stat().st_mtime_ns
    second = erase_to_completion(client, task["id"], ["SPEAKER_01"])

    assert mix.stat().st_mtime_ns != before, "目标变了必须重做掩码合成"
    assert second["output_path"] == str(video.with_name("换目标__erased-SPEAKER_01.mp4"))


def test_改变说话人数量只重算说话人分离与样本转写(tmp_path):
    separator, diarizer, transcriber = CannedSeparator(), CannedDiarizer(), CannedTranscriber()
    video = make_video(tmp_path, "改人数.mp4")

    with make_client(
        tmp_path, separator=separator, diarizer=diarizer, transcriber=transcriber
    ) as client:
        first = run_to_analysis(client, video)  # 自动估计
        assert call_counts(separator, diarizer, transcriber) == (1, 1, 1)
        assert diarizer.num_speakers == [None]

        second = run_to_analysis(client, video, num_speakers=4)
        assert call_counts(separator, diarizer, transcriber) == (1, 2, 2), (
            "说话人数量变了只重算说话人分离与样本转写，昂贵的分离与音轨提取不复算"
        )
        assert diarizer.num_speakers == [None, 4]
        assert statuses(second) == ANALYSIS_STATUSES
        assert manifest_of(second)["diarization"]["requested_num_speakers"] == 4


def test_相同说话人数量再次提交复用分析产物(tmp_path):
    separator, diarizer, transcriber = CannedSeparator(), CannedDiarizer(), CannedTranscriber()
    video = make_video(tmp_path, "同人数.mp4")

    with make_client(
        tmp_path, separator=separator, diarizer=diarizer, transcriber=transcriber
    ) as client:
        first = run_to_analysis(client, video, num_speakers=4)
        assert call_counts(separator, diarizer, transcriber) == (1, 1, 1)

        second = run_to_analysis(client, video, num_speakers=4)
        assert call_counts(separator, diarizer, transcriber) == (1, 1, 1), (
            "相同说话人数量不是新的分析，应直接复用工作区产物"
        )
        assert statuses(second) == ANALYSIS_STATUSES
