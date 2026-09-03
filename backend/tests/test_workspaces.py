"""工作区列表与手动删除：磁盘占用可见、删除只影响目标工作区。

工作区是几 GB 到十几 GB 的磁盘占用单位，用户手动删除释放空间；正在被
排队/运行中任务使用的工作区不可删（删了会把跑到一半的任务掏空）。
"""

import threading
from pathlib import Path

from conftest import (
    CannedDiarizer,
    CannedSeparator,
    get_task,
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
