"""擦除闭环：掩码合成语义、成品命名与输出目录、磁盘预检、API 错误路径。

掩码语义是本工单重心：注入内容可区分的 canned 人声轨（3000Hz）/伴奏轨
（200Hz）与含重叠时段的 canned 说话人时段，用 ffmpeg 音频检测断言成品——
目标说话人活跃时段人声内容被擦除（含与保留说话人的重叠段）、保留说话人
非重叠时段人声内容完整、伴奏内容全程保留。
"""

import pytest

from conftest import (
    CannedDiarizer,
    ToneStemsSeparator,
    band_mean_volume,
    dominant_freq,
    erase,
    erase_to_completion,
    get_task,
    make_client,
    make_video,
    run_to_analysis,
    submit_task,
    wait_for,
)
from app import pipeline

VOCAL_HZ = ToneStemsSeparator.VOCAL_HZ
ACCOMP_HZ = ToneStemsSeparator.ACCOMP_HZ

# 说话人时段（含重叠）：SPEAKER_00 是目标，SPEAKER_01 是保留说话人
TURNS = (
    ("SPEAKER_00", 0.2, 1.0),
    ("SPEAKER_01", 0.6, 1.4),  # 与目标重叠 [0.6, 1.0]
    ("SPEAKER_00", 2.0, 2.8),
    ("SPEAKER_01", 3.0, 3.8),
)

MEASURE_RATE = 24000  # 零交叉法在接近奈奎斯特频率时失真，需远高于 3000Hz


def assert_vocal_erased(path, start, end):
    freq = dominant_freq(path, start, end, rate=MEASURE_RATE)
    assert abs(freq - ACCOMP_HZ) < 80, (
        f"[{start}, {end}] 目标活跃时段人声应被擦除（只剩 {ACCOMP_HZ}Hz 伴奏），实测 {freq:.0f}Hz"
    )
    # 人声频段能量应落到底噪水平（实测 AAC 成品擦除段约 -72dB，保留段约 -33dB）
    high = band_mean_volume(path, start, end, prefilter="highpass=f=2000")
    assert high < -50, (
        f"[{start}, {end}] 目标活跃时段人声频段能量应被擦除，实测 {high:.1f}dB"
    )


def assert_vocal_kept(path, start, end):
    freq = dominant_freq(path, start, end, rate=MEASURE_RATE)
    assert abs(freq - VOCAL_HZ) < 700, (
        f"[{start}, {end}] 保留说话人非重叠时段人声应完整（{VOCAL_HZ}Hz），实测 {freq:.0f}Hz"
    )
    high = band_mean_volume(path, start, end, prefilter="highpass=f=2000")
    assert high > -40, (
        f"[{start}, {end}] 保留说话人人声频段能量应完整，实测 {high:.1f}dB"
    )


def assert_accomp_present(path, start, end):
    # canned 伴奏为 0.4 × 默认振幅（1/8）的正弦，RMS 约 -29dB；经 AAC 后
    # 约 -32dB，远高于无人声段底噪（<-70dB）
    low = band_mean_volume(path, start, end, prefilter="lowpass=f=500")
    assert low > -40, f"[{start}, {end}] 伴奏应全程保留，低频响度实测 {low:.1f}dB"


@pytest.fixture()
def tone_workspace(tmp_path):
    """注入定调 stems + 重叠说话人时段的客户端上下文。"""
    separator = ToneStemsSeparator(duration=4.0)
    from app.inference import SpeakerSegment

    diarizer = CannedDiarizer(
        turns=tuple(SpeakerSegment(s, a, b) for s, a, b in TURNS)
    )
    video = make_video(tmp_path, "样片.mp4", duration=4)
    with make_client(tmp_path, separator=separator, diarizer=diarizer) as client:
        task = run_to_analysis(client, video)
        yield client, task, video


def test_目标说话人时段人声被擦除_重叠段整段擦除_伴奏全程保留(tone_workspace):
    client, task, video = tone_workspace
    done = erase_to_completion(client, task["id"], ["SPEAKER_00"])
    out = done["output_path"]
    assert out is not None

    # 目标说话人的两个时段（含与保留说话人的重叠段 [0.6, 1.0]）都被擦除
    assert_vocal_erased(out, 0.25, 0.55)  # SPEAKER_00 非重叠活跃
    assert_vocal_erased(out, 0.65, 0.95)  # 重叠段：目标优先，整段擦除
    assert_vocal_erased(out, 2.1, 2.7)  # SPEAKER_00 第二段
    # 保留说话人的非重叠时段人声完整
    assert_vocal_kept(out, 1.05, 1.35)  # SPEAKER_01 活跃，不与目标重叠
    assert_vocal_kept(out, 3.1, 3.7)
    # 伴奏内容全程保留（含被擦除时段）
    for start, end in ((0.25, 0.55), (0.65, 0.95), (1.05, 1.35), (3.1, 3.7)):
        assert_accomp_present(out, start, end)


def test_成品命名_未命名用编号标签_命名后用名字(tone_workspace, tmp_path):
    client, task, video = tone_workspace
    done = erase_to_completion(client, task["id"], ["SPEAKER_00"])
    expected = video.with_name("样片__erased-SPEAKER_00.mp4")
    assert done["output_path"] == str(expected)
    assert expected.is_file()

    resp = client.put(
        f"/api/tasks/{task['id']}/speakers/SPEAKER_00/name", json={"name": "男主"}
    )
    assert resp.status_code == 200
    done = erase_to_completion(client, task["id"], ["SPEAKER_00"])
    assert done["output_path"] == str(video.with_name("样片__erased-男主.mp4"))
    assert video.with_name("样片__erased-男主.mp4").is_file()


def test_多目标以加号连接_按编号标签排序(tone_workspace, tmp_path):
    client, task, video = tone_workspace
    client.put(
        f"/api/tasks/{task['id']}/speakers/SPEAKER_01/name", json={"name": "旁白"}
    )
    done = erase_to_completion(client, task["id"], ["SPEAKER_01", "SPEAKER_00"])
    assert done["output_path"] == str(video.with_name("样片__erased-SPEAKER_00+旁白.mp4"))
    # 两个目标都被擦除：整段只剩伴奏
    out = done["output_path"]
    assert_vocal_erased(out, 0.25, 0.55)
    assert_vocal_erased(out, 1.05, 1.35)
    assert_vocal_erased(out, 3.1, 3.7)


def test_重复目标被去重(tone_workspace, tmp_path):
    client, task, video = tone_workspace
    done = erase_to_completion(client, task["id"], ["SPEAKER_00", "SPEAKER_00"])
    assert done["erase"]["speakers"] == ["SPEAKER_00"], "重复目标应去重"
    assert done["output_path"] == str(video.with_name("样片__erased-SPEAKER_00.mp4"))


def test_输出目录默认源视频同目录_可修改(tone_workspace, tmp_path):
    client, task, video = tone_workspace
    target_dir = tmp_path / "成品"
    done = erase_to_completion(
        client, task["id"], ["SPEAKER_00"], output_dir=str(target_dir)
    )
    expected = target_dir / "样片__erased-SPEAKER_00.mp4"
    assert done["output_path"] == str(expected), "成品应写入指定输出目录"
    assert expected.is_file()


def test_换目标再次擦除_两个成品共存且掩码各自正确(tone_workspace, tmp_path):
    client, task, video = tone_workspace
    first = erase_to_completion(client, task["id"], ["SPEAKER_00"])
    second = erase_to_completion(client, task["id"], ["SPEAKER_01"])

    assert first["output_path"] == str(video.with_name("样片__erased-SPEAKER_00.mp4"))
    assert second["output_path"] == str(video.with_name("样片__erased-SPEAKER_01.mp4"))
    assert video.with_name("样片__erased-SPEAKER_00.mp4").is_file()

    out = second["output_path"]
    assert_vocal_erased(out, 0.65, 0.95)  # SPEAKER_01 时段（含重叠段）被擦
    assert_vocal_erased(out, 1.05, 1.35)
    assert_vocal_erased(out, 3.1, 3.7)
    assert_vocal_kept(out, 0.25, 0.55)  # SPEAKER_00 非重叠时段保留


def test_分析未完成时发起擦除被拒绝(tmp_path):
    import threading

    gate = threading.Event()
    video = make_video(tmp_path, "运行中.mp4")
    with make_client(tmp_path, diarizer=CannedDiarizer(gate=gate)) as client:
        task = submit_task(client, video)
        wait_for(lambda: get_task(client, task["id"])["state"] == "running")
        resp = erase(client, task["id"], ["SPEAKER_00"])
        assert resp.status_code == 409, resp.text
        gate.set()


def test_擦除未知说话人被拒绝_任务保持可擦除(tone_workspace):
    client, task, video = tone_workspace
    resp = erase(client, task["id"], ["SPEAKER_99"])
    assert resp.status_code == 404, resp.text

    again = erase(client, task["id"], ["SPEAKER_00"])
    assert again.status_code == 200, "拒绝后任务状态不应被破坏"


def test_擦除空目标列表被拒绝(tone_workspace):
    client, task, video = tone_workspace
    resp = erase(client, task["id"], [])
    assert resp.status_code == 422, resp.text


def test_输出目录无法创建时被拒绝(tone_workspace, tmp_path):
    client, task, video = tone_workspace
    blocker = tmp_path / "挡路"
    blocker.write_text("不是目录")
    resp = erase(client, task["id"], ["SPEAKER_00"], output_dir=str(blocker))
    assert resp.status_code == 400, resp.text


def test_任务不存在时擦除被拒绝(tmp_path):
    with make_client(tmp_path) as client:
        resp = erase(client, "不存在", ["SPEAKER_00"])
        assert resp.status_code == 404


def test_失败任务无说话人产物时擦除被拒绝(tmp_path):
    """失败任务可重擦，但损坏视频没跑到说话人分离，按 404 拒绝。"""
    from conftest import make_corrupt_video

    video = make_corrupt_video(tmp_path, "损坏.mp4")
    with make_client(tmp_path) as client:
        task = submit_task(client, video)
        wait_for(lambda: get_task(client, task["id"])["state"] == "failed")
        resp = erase(client, task["id"], ["SPEAKER_00"])
        assert resp.status_code == 404, resp.text
        assert "说话人分离" in resp.json()["detail"]


def test_磁盘空间不足_分析任务立即失败(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "_disk_free", lambda path: 0)
    video = make_video(tmp_path, "大.mp4")
    with make_client(tmp_path) as client:
        task = submit_task(client, video)
        wait_for(lambda: get_task(client, task["id"])["state"] == "failed")
        done = get_task(client, task["id"])
        assert "磁盘空间不足" in done["error"]
        assert done["output_path"] is None
        assert not (tmp_path / "workspaces" / "大").exists(), (
            "预检失败不应创建工作区"
        )


def test_磁盘空间不足_擦除任务立即失败(tmp_path, monkeypatch):
    video = make_video(tmp_path, "预检.mp4")
    with make_client(tmp_path) as client:
        task = run_to_analysis(client, video)
        monkeypatch.setattr(pipeline, "_disk_free", lambda path: 0)
        resp = erase(client, task["id"], ["SPEAKER_00"])
        assert resp.status_code == 200
        wait_for(lambda: get_task(client, task["id"])["state"] == "failed")
        done = get_task(client, task["id"])
        assert "磁盘空间不足" in done["error"]
        assert done["output_path"] is None
        assert not video.with_name("预检__erased-SPEAKER_00.mp4").exists()


def test_磁盘预检按源视频体积五倍估算边界判定(tmp_path, monkeypatch):
    """余量低于 源体积×5 立即失败；略高于则通过。"""
    video = make_video(tmp_path, "估算.mp4")
    size = video.stat().st_size

    with make_client(tmp_path) as client:
        task = run_to_analysis(client, video)
        wait_for(lambda: get_task(client, task["id"])["state"] == "completed")

        monkeypatch.setattr(pipeline, "_disk_free", lambda path: int(size * 4.9))
        resp = erase(client, task["id"], ["SPEAKER_00"])
        assert resp.status_code == 200
        wait_for(lambda: get_task(client, task["id"])["state"] == "failed")
        done = get_task(client, task["id"])
        assert "磁盘空间不足" in done["error"]
        assert "5" in done["error"], "错误应说明估算倍数"

        # 擦除失败的任务可重试：恢复余量到 5.1 倍再擦，成功
        monkeypatch.setattr(pipeline, "_disk_free", lambda path: int(size * 5.1))
        resp = erase(client, task["id"], ["SPEAKER_00"])
        assert resp.status_code == 200, resp.text
        wait_for(lambda: get_task(client, task["id"])["state"] == "completed")
        assert video.with_name("估算__erased-SPEAKER_00.mp4").exists()
