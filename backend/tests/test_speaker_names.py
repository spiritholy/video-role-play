"""说话人命名：持久化于工作区元数据、显示回退数据、不强制唯一。

命名是工作区级数据（随工作区存续、跨任务继承），不是任务级状态；
接口经 HTTP API 接缝验证，ML 推理照例注入 canned 输出。
"""

import json
import threading

from conftest import (
    CannedDiarizer,
    get_task,
    make_client,
    make_video,
    submit_task,
    wait_for,
)

DIARIZATION = "03-diarize/diarization.json"


def speakers_of(client, task_id: str) -> list[dict]:
    resp = client.get(f"/api/tasks/{task_id}/diarization")
    assert resp.status_code == 200, resp.text
    return resp.json()["speakers"]


def rename(client, task_id: str, label: str, name: str):
    return client.put(
        f"/api/tasks/{task_id}/speakers/{label}/name", json={"name": name}
    )


def test_命名写入工作区元数据且时间轴即时可见(client, tmp_path):
    video = make_video(tmp_path, "命名.mp4")
    task = submit_task(client, video)
    wait_for(lambda: get_task(client, task["id"])["state"] == "completed")

    resp = rename(client, task["id"], "SPEAKER_00", "男主")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"label": "SPEAKER_00", "name": "男主"}

    ws = tmp_path / "workspaces" / "命名"
    stored = json.loads((ws / "speaker_names.json").read_text(encoding="utf-8"))
    assert stored["names"]["SPEAKER_00"] == "男主"

    rows = {s["label"]: s for s in speakers_of(client, task["id"])}
    assert rows["SPEAKER_00"]["name"] == "男主"
    assert rows["SPEAKER_01"]["name"] is None, "未命名说话人应返回空名字供回退显示"


def test_未命名说话人返回空名字供回退显示(client, tmp_path):
    video = make_video(tmp_path, "回退.mp4")
    task = submit_task(client, video)
    wait_for(lambda: get_task(client, task["id"])["state"] == "completed")

    rows = speakers_of(client, task["id"])
    assert [s["name"] for s in rows] == [None, None], "未命名时应返回空名字，由界面回退编号标签"


def test_改名覆盖旧名字并去除首尾空白(client, tmp_path):
    video = make_video(tmp_path, "改名.mp4")
    task = submit_task(client, video)
    wait_for(lambda: get_task(client, task["id"])["state"] == "completed")

    assert rename(client, task["id"], "SPEAKER_00", "男主").status_code == 200
    resp = rename(client, task["id"], "SPEAKER_00", " 旁白 ")
    assert resp.status_code == 200
    assert resp.json()["name"] == "旁白", "改名应覆盖旧名字并去除首尾空白"

    rows = {s["label"]: s for s in speakers_of(client, task["id"])}
    assert rows["SPEAKER_00"]["name"] == "旁白"


def test_清除命名回退编号标签(client, tmp_path):
    video = make_video(tmp_path, "清除.mp4")
    task = submit_task(client, video)
    wait_for(lambda: get_task(client, task["id"])["state"] == "completed")

    assert rename(client, task["id"], "SPEAKER_00", "男主").status_code == 200
    resp = rename(client, task["id"], "SPEAKER_00", "   ")
    assert resp.status_code == 200
    assert resp.json()["name"] is None, "空白名字应视为清除命名"

    rows = {s["label"]: s for s in speakers_of(client, task["id"])}
    assert rows["SPEAKER_00"]["name"] is None


def test_不强制唯一_同一人拆出的两个标签可同名(client, tmp_path):
    video = make_video(tmp_path, "同名.mp4")
    task = submit_task(client, video)
    wait_for(lambda: get_task(client, task["id"])["state"] == "completed")

    assert rename(client, task["id"], "SPEAKER_00", "男主").status_code == 200
    assert rename(client, task["id"], "SPEAKER_01", "男主").status_code == 200

    rows = {s["label"]: s for s in speakers_of(client, task["id"])}
    assert rows["SPEAKER_00"]["name"] == "男主"
    assert rows["SPEAKER_01"]["name"] == "男主"


def test_应用重启后读取同一工作区名字仍在(tmp_path):
    video = make_video(tmp_path, "重启.mp4")
    with make_client(tmp_path) as client:
        task = submit_task(client, video)
        wait_for(lambda: get_task(client, task["id"])["state"] == "completed")
        assert rename(client, task["id"], "SPEAKER_00", "男主").status_code == 200

    with make_client(tmp_path) as client:
        task = submit_task(client, video)
        wait_for(lambda: get_task(client, task["id"])["state"] == "completed")
        rows = {s["label"]: s for s in speakers_of(client, task["id"])}
        assert rows["SPEAKER_00"]["name"] == "男主", "重启后重跑同一工作区，命名应继承"
        assert rows["SPEAKER_01"]["name"] is None


def test_命名未知说话人被拒绝(client, tmp_path):
    video = make_video(tmp_path, "未知说话人.mp4")
    task = submit_task(client, video)
    wait_for(lambda: get_task(client, task["id"])["state"] == "completed")

    resp = rename(client, task["id"], "SPEAKER_99", "幽灵")
    assert resp.status_code == 404
    ws = tmp_path / "workspaces" / "未知说话人"
    assert not (ws / "speaker_names.json").exists(), "拒绝的命名不应落盘"


def test_任务不存在时命名被拒绝(tmp_path):
    with make_client(tmp_path) as client:
        resp = rename(client, "不存在", "SPEAKER_00", "男主")
        assert resp.status_code == 404


def test_说话人分离阶段未完成时命名被拒绝(tmp_path):
    gate = threading.Event()
    with make_client(tmp_path, diarizer=CannedDiarizer(gate=gate)) as client:
        video = make_video(tmp_path, "未完成命名.mp4")
        task = submit_task(client, video)
        wait_for(
            lambda: get_task(client, task["id"])["state"] == "running"
            and next(
                s for s in get_task(client, task["id"])["stages"] if s["id"] == "diarize"
            )["status"]
            == "running"
        )

        resp = rename(client, task["id"], "SPEAKER_00", "男主")
        assert resp.status_code == 404, "说话人分离未完成前没有可命名的说话人"
        gate.set()
