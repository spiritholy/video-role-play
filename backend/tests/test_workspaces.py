"""工作区列表与手动删除：磁盘占用可见、删除只影响目标工作区。

工作区是几 GB 到十几 GB 的磁盘占用单位，用户手动删除释放空间；正在被
排队/运行中任务使用的工作区不可删（删了会把跑到一半的任务掏空）。
"""

import json
import threading
from pathlib import Path

from app.media import TaskError

from conftest import (
    CannedDiarizer,
    CannedSeparator,
    FailOnceSeparator,
    erase,
    erase_to_completion,
    get_task,
    run_ffmpeg,
    make_client,
    make_video,
    run_to_analysis,
    submit_task,
    wait_for,
)


def list_workspaces(client) -> dict:
    resp = client.get("/api/workspaces")
    assert resp.status_code == 200, resp.text
    return resp.json()


def workspace_items(client) -> dict[str, dict]:
    return {w["name"]: w for w in list_workspaces(client)["workspaces"]}


def actual_size(ws: Path) -> int:
    return sum(f.stat().st_size for f in ws.rglob("*") if f.is_file())


def test_列表显示各工作区的源视频与磁盘占用(client, tmp_path):
    v1 = make_video(tmp_path, "甲.mp4")
    v2 = make_video(tmp_path, "乙.mp4")
    run_to_analysis(client, v1)
    run_to_analysis(client, v2)

    data = list_workspaces(client)
    items = {w["name"]: w for w in data["workspaces"]}
    assert set(items) == {"甲", "乙"}

    w = items["甲"]
    assert w["source_path"] == str(v1)
    assert w["path"] == str(tmp_path / "workspaces" / "甲")
    assert w["size_bytes"] == actual_size(tmp_path / "workspaces" / "甲") > 0
    assert "MB" in w["size_label"] or "KB" in w["size_label"]
    assert w["reusable"] is True, "分析产物齐全的工作区可直接复用"
    assert w["busy"] is False
    assert data["total_bytes"] == items["甲"]["size_bytes"] + items["乙"]["size_bytes"]


def test_没有工作区时返回空列表(client):
    data = list_workspaces(client)
    assert data["workspaces"] == []
    assert data["total_bytes"] == 0


def test_运行中任务的工作区标记使用中且删除被拒绝(tmp_path):
    gate = threading.Event()
    with make_client(tmp_path, diarizer=CannedDiarizer(gate=gate)) as client:
        task = submit_task(client, make_video(tmp_path, "运行中.mp4"))
        wait_for(lambda: get_task(client, task["id"])["state"] == "running")

        w = workspace_items(client)["运行中"]
        assert w["busy"] is True, "正在被任务使用的工作区应标记使用中"
        assert w["reusable"] is False

        resp = client.delete("/api/workspaces/运行中")
        assert resp.status_code == 409, resp.text
        assert (tmp_path / "workspaces" / "运行中").is_dir(), "拒绝的删除不应动目录"

        gate.set()
        wait_for(lambda: get_task(client, task["id"])["state"] == "completed")
        assert workspace_items(client)["运行中"]["busy"] is False


def test_删除工作区移除目录且不影响其他工作区(client, tmp_path):
    v1 = make_video(tmp_path, "删我.mp4")
    v2 = make_video(tmp_path, "留我.mp4")
    t1 = run_to_analysis(client, v1)
    run_to_analysis(client, v2)

    resp = client.delete("/api/workspaces/删我")
    assert resp.status_code == 200, resp.text

    assert not (tmp_path / "workspaces" / "删我").exists(), "删除后目录移除"
    assert (tmp_path / "workspaces" / "留我" / "manifest.json").is_file(), (
        "删除不影响其他源视频的工作区"
    )
    assert set(workspace_items(client)) == {"留我"}

    # 被删工作区的任务仍在列表里，只是产物不再可读
    assert get_task(client, t1["id"])["state"] == "completed"
    assert client.get(f"/api/tasks/{t1['id']}/diarization").status_code == 404


def test_删除后同源任务重新分析(tmp_path):
    separator = CannedSeparator()
    video = make_video(tmp_path, "重来.mp4")
    with make_client(tmp_path, separator=separator) as client:
        run_to_analysis(client, video)
        assert len(separator.calls) == 1

        assert client.delete("/api/workspaces/重来").status_code == 200
        run_to_analysis(client, video)
        assert len(separator.calls) == 2, "工作区删了就没有可复用的产物"


def test_删除不存在的工作区被拒绝(client, tmp_path):
    run_to_analysis(client, make_video(tmp_path, "甲.mp4"))
    assert client.delete("/api/workspaces/不存在").status_code == 404
    assert (tmp_path / "workspaces" / "甲").is_dir()


def test_删除不接受越出工作区根目录的名字(client, tmp_path):
    run_to_analysis(client, make_video(tmp_path, "甲.mp4"))
    outside = tmp_path / "外面"
    outside.mkdir()

    resp = client.delete("/api/workspaces/%2E%2E")
    assert resp.status_code == 404, resp.text
    assert outside.is_dir()
    assert tmp_path.is_dir()
    assert (tmp_path / "workspaces" / "甲").is_dir()

def test_重启后从工作区继续_已完成阶段不重算且保留人数(tmp_path):
    separator = FailOnceSeparator(TaskError("分离失败（模拟）"))
    diarizer = CannedDiarizer()
    video = make_video(tmp_path, "重启.mp4")

    with make_client(tmp_path, separator=separator) as client:
        task = submit_task(client, video, num_speakers=4)
        wait_for(lambda: get_task(client, task["id"])["state"] == "failed")

    with make_client(tmp_path, separator=separator, diarizer=diarizer) as client:
        w = workspace_items(client)["重启"]
        assert w["resumable"] is True, "未完成的工作区应可继续"

        resp = client.post("/api/workspaces/重启/resume")
        assert resp.status_code == 200, resp.text
        new_task = resp.json()
        assert new_task["num_speakers"] == 4, "清单应保留用户指定的说话人人数"
        assert new_task["state"] in ("queued", "running")

        wait_for(lambda: get_task(client, new_task["id"])["state"] == "completed")
        assert len(separator.calls) == 2, "续跑只重跑断点阶段，不重算已完成阶段"
        assert diarizer.num_speakers == [4], "续跑应继续沿用指定人数"


def test_分析已完成的工作区不可继续(client, tmp_path):
    run_to_analysis(client, make_video(tmp_path, "已完成.mp4"))

    w = workspace_items(client)["已完成"]
    assert w["resumable"] is False, "分析已完成没有断点可继续"
    resp = client.post("/api/workspaces/已完成/resume")
    assert resp.status_code == 409, resp.text


def test_无清单工作区不可继续(client, tmp_path):
    (tmp_path / "workspaces" / "空").mkdir(parents=True)

    w = workspace_items(client)["空"]
    assert w["resumable"] is False
    resp = client.post("/api/workspaces/空/resume")
    assert resp.status_code == 404, resp.text


def test_源视频缺失时继续被拒绝且不创建任务(tmp_path):
    separator = FailOnceSeparator(TaskError("分离失败（模拟）"))
    video = make_video(tmp_path, "消失.mp4")

    with make_client(tmp_path, separator=separator) as client:
        task = submit_task(client, video)
        wait_for(lambda: get_task(client, task["id"])["state"] == "failed")
    video.unlink()

    with make_client(tmp_path) as client:
        resp = client.post("/api/workspaces/消失/resume")
        assert resp.status_code == 400, resp.text
        assert "源视频不存在" in resp.json()["detail"]
        assert client.get("/api/tasks").json()["tasks"] == [], (
            "源视频缺失不应创建排队任务"
        )


def test_任务使用中的工作区不可继续(tmp_path):
    gate = threading.Event()
    video = make_video(tmp_path, "占用中.mp4")
    with make_client(tmp_path, diarizer=CannedDiarizer(gate=gate)) as client:
        task = submit_task(client, video)
        wait_for(lambda: get_task(client, task["id"])["state"] == "running")

        w = workspace_items(client)["占用中"]
        assert w["busy"] is True
        resp = client.post("/api/workspaces/占用中/resume")
        assert resp.status_code == 409, resp.text

        gate.set()
        wait_for(lambda: get_task(client, task["id"])["state"] == "completed")


def test_擦除请求提前落盘且可从工作区续跑擦除(tmp_path):
    video = make_video(tmp_path, "擦除失败.mp4")
    with make_client(tmp_path) as client:
        task = run_to_analysis(client, video)
        ws = tmp_path / "workspaces" / "擦除失败"
        manifest = json.loads((ws / "manifest.json").read_text(encoding="utf-8"))
        (ws / manifest["separation"]["vocal"]).unlink()  # 逼掩码合成失败

        resp = erase(client, task["id"], ["SPEAKER_00"])
        assert resp.status_code == 200, resp.text
        wait_for(lambda: get_task(client, task["id"])["state"] == "failed")

        manifest = json.loads((ws / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["erase"] == {
            "speakers": ["SPEAKER_00"],
            "output_dir": None,
        }, "擦除请求应在阶段运行前就落盘，失败后仍可恢复"

        assert workspace_items(client)["擦除失败"]["resumable"] is True
        r = client.post("/api/workspaces/擦除失败/resume")
        assert r.status_code == 200, r.text
        new_task = r.json()
        assert new_task["erase"]["speakers"] == ["SPEAKER_00"], (
            "工作区续跑应重建擦除任务而非退回分析"
        )


def test_中断擦除可从工作区续跑且不重算掩码(tmp_path):
    video = make_video(tmp_path, "中断擦除.mp4")
    ws = tmp_path / "workspaces" / "中断擦除"
    mix_dir = ws / "05-mask-erase"
    mix_dir.mkdir(parents=True)
    mix = mix_dir / "mix.wav"
    run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=2", str(mix)])

    manifest = {
        "version": 1,
        "source": {
            "path": str(video),
            "container": ".mp4",
            "size_bytes": video.stat().st_size,
        },
        "requested_audio_track": None,
        "requested_num_speakers": None,
        "audio_track": None,
        "separation": None,
        "diarization": {"segments": []},
        "transcription": None,
        "mask_erase": {"targets": ["SPEAKER_00"], "intervals": []},
        "erase": {"speakers": ["SPEAKER_00"], "output_dir": None},
        "output": None,
        "state": "running",
        "error": None,
        "stages": [
            {"id": sid, "status": "done", "artifacts": [], "outputs": {}}
            for sid in ("audio_extract", "separate", "diarize", "transcribe")
        ]
        + [
            {
                "id": "mask_erase",
                "status": "done",
                "artifacts": ["05-mask-erase/mix.wav"],
                "outputs": {"erased_mix": "05-mask-erase/mix.wav"},
            },
            {"id": "encode_mux", "status": "pending", "artifacts": [], "outputs": {}},
        ],
    }
    (ws / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )

    with make_client(tmp_path) as client:
        w = workspace_items(client)["中断擦除"]
        assert w["resumable"] is True
        before = mix.stat().st_mtime_ns

        r = client.post("/api/workspaces/中断擦除/resume")
        assert r.status_code == 200, r.text
        new_task = r.json()
        assert new_task["erase"]["speakers"] == ["SPEAKER_00"], (
            "中断擦除续跑应重建擦除任务而非退回分析"
        )
        wait_for(lambda: get_task(client, new_task["id"])["state"] == "completed")

        assert mix.stat().st_mtime_ns == before, "掩码合成产物应复用，不重算"
        assert list(video.parent.glob("中断擦除__*")), "成品视频应生成"


def test_从可复用工作区打开进入擦除选择_不重算分析且不带历史擦除(tmp_path):
    separator = CannedSeparator()
    video = make_video(tmp_path, "可复用.mp4")
    with make_client(tmp_path, separator=separator) as client:
        run_to_analysis(client, video)
        assert len(separator.calls) == 1

        w = workspace_items(client)["可复用"]
        assert w["reusable"] is True
        resp = client.post("/api/workspaces/可复用/open")
        assert resp.status_code == 200, resp.text
        task = resp.json()
        assert task["erase"] is None, "打开擦除选择不应携带历史擦除请求"

        wait_for(lambda: get_task(client, task["id"])["state"] == "completed")
        assert len(separator.calls) == 1, "分析产物应整体复用，不重算分离"
        assert get_task(client, task["id"])["output_path"] is None
        assert client.get(f"/api/tasks/{task['id']}/diarization").status_code == 200, (
            "打开后应能读取说话人时间轴，供擦除表单渲染"
        )


def test_打开已完成擦除的工作区不携带旧擦除请求(tmp_path):
    video = make_video(tmp_path, "已擦除.mp4")
    with make_client(tmp_path) as client:
        task = run_to_analysis(client, video)
        erase_to_completion(client, task["id"], ["SPEAKER_00"])

        w = workspace_items(client)["已擦除"]
        assert w["reusable"] is True
        resp = client.post("/api/workspaces/已擦除/open")
        assert resp.status_code == 200, resp.text
        new_task = resp.json()
        assert new_task["erase"] is None, "打开擦除选择不应继承上次擦除目标"

        wait_for(lambda: get_task(client, new_task["id"])["state"] == "completed")
        done = get_task(client, new_task["id"])
        assert done["output_path"] is None, "打开擦除选择只物化分析，不重复封装成品"
        assert client.get(f"/api/tasks/{new_task['id']}/diarization").status_code == 200


def test_分析未完成或无清单的工作区不可打开(tmp_path):
    separator = FailOnceSeparator(TaskError("分离失败（模拟）"))
    video = make_video(tmp_path, "分析失败.mp4")
    with make_client(tmp_path, separator=separator) as client:
        task = submit_task(client, video)
        wait_for(lambda: get_task(client, task["id"])["state"] == "failed")

        w = workspace_items(client)["分析失败"]
        assert w["reusable"] is False
        resp = client.post("/api/workspaces/分析失败/open")
        assert resp.status_code == 409, resp.text
        assert "分析产物不完整" in resp.json()["detail"]
        tasks = client.get("/api/tasks").json()["tasks"]
        assert [t["id"] for t in tasks] == [task["id"]], (
            "拒绝打开不应额外创建任务"
        )

    (tmp_path / "workspaces" / "空").mkdir(parents=True)
    with make_client(tmp_path) as client:
        assert client.post("/api/workspaces/空/open").status_code == 404
        assert client.post("/api/workspaces/不存在/open").status_code == 404


def test_源视频缺失或变化时打开被拒绝且不创建任务(tmp_path):
    video = make_video(tmp_path, "消失.mp4")
    with make_client(tmp_path) as client:
        run_to_analysis(client, video)
    video.unlink()

    with make_client(tmp_path) as client:
        resp = client.post("/api/workspaces/消失/open")
        assert resp.status_code == 400, resp.text
        assert "源视频不存在" in resp.json()["detail"]
        assert client.get("/api/tasks").json()["tasks"] == [], (
            "源视频缺失不应创建排队任务"
        )

    # 同路径换成另一个文件（体积不同）：复用依据失效，拒绝而非静默从头重算
    make_video(tmp_path, "消失.mp4", duration=3)
    with make_client(tmp_path) as client:
        resp = client.post("/api/workspaces/消失/open")
        assert resp.status_code == 400, resp.text
        assert "源视频已变化" in resp.json()["detail"]
        assert client.get("/api/tasks").json()["tasks"] == [], (
            "源视频变化不应创建排队任务"
        )


def test_运行中的工作区不可打开(tmp_path):
    gate = threading.Event()
    video = make_video(tmp_path, "占用中.mp4")
    with make_client(tmp_path, diarizer=CannedDiarizer(gate=gate)) as client:
        task = submit_task(client, video)
        wait_for(lambda: get_task(client, task["id"])["state"] == "running")

        w = workspace_items(client)["占用中"]
        assert w["busy"] is True
        resp = client.post("/api/workspaces/占用中/open")
        assert resp.status_code == 409, resp.text
        tasks = client.get("/api/tasks").json()["tasks"]
        assert [t["id"] for t in tasks] == [task["id"]], (
            "占用中的工作区不应额外创建任务"
        )

        gate.set()
        wait_for(lambda: get_task(client, task["id"])["state"] == "completed")
