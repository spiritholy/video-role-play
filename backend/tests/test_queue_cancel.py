"""单实例运行 + FIFO 排队 + 取消（运行中与排队中）。

用节流替身控制管线停驻以便确定性观察中间状态；状态断言全部走 HTTP API，
真实 ffmpeg 行为由其他测试覆盖。
"""

import threading
import time

import pytest

import app.pipeline as pipeline_module
from app.pipeline import TaskCancelled
from conftest import get_task, make_video, submit_task, wait_for


@pytest.fixture()
def paced_pipeline(monkeypatch):
    """让每个任务的执行停在门禁处，直到 release 或取消。

    语义与真实管线一致：取消事件置位后抛 TaskCancelled（即取消在阶段
    边界被观察到）。仅用于驻停以便确定性观察中间状态；真实 ffmpeg 行为
    由其他测试走完整管线覆盖。
    """
    started = threading.Event()
    release = threading.Event()
    order: list[str] = []

    def fake_execute(task, ctx):
        order.append(task.id)
        started.set()
        while not (release.is_set() or ctx.cancel_event.is_set()):
            time.sleep(0.01)
        if ctx.cancel_event.is_set():
            raise TaskCancelled()

    monkeypatch.setattr(pipeline_module, "run_analysis", fake_execute)
    return started, release, order


def test_运行期间提交的新任务排队等待(client, tmp_path, paced_pipeline):
    started, release, order = paced_pipeline
    v1 = make_video(tmp_path, "一.mp4")
    v2 = make_video(tmp_path, "二.mp4")

    t1 = submit_task(client, v1)
    wait_for(lambda: get_task(client, t1["id"])["state"] == "running")

    t2 = submit_task(client, v2)
    assert get_task(client, t2["id"])["state"] == "queued"

    release.set()
    wait_for(lambda: get_task(client, t1["id"])["state"] == "completed")
    wait_for(lambda: get_task(client, t2["id"])["state"] == "completed")
    assert order == [t1["id"], t2["id"]], "必须按提交顺序 FIFO 执行"


def test_取消排队中任务(client, tmp_path, paced_pipeline):
    started, release, order = paced_pipeline
    v1 = make_video(tmp_path, "一.mp4")
    v2 = make_video(tmp_path, "二.mp4")

    t1 = submit_task(client, v1)
    wait_for(lambda: get_task(client, t1["id"])["state"] == "running")
    t2 = submit_task(client, v2)
    assert get_task(client, t2["id"])["state"] == "queued"

    resp = client.post(f"/api/tasks/{t2['id']}/cancel")
    assert resp.status_code == 200
    assert resp.json()["state"] == "cancelled"

    release.set()
    wait_for(lambda: get_task(client, t1["id"])["state"] == "completed")
    assert get_task(client, t2["id"])["state"] == "cancelled"
    assert order == [t1["id"]], "已取消的排队任务不得执行"
    for s in get_task(client, t2["id"])["stages"]:
        assert s["status"] == "pending"


def test_取消运行中任务(client, tmp_path, paced_pipeline):
    started, release, _ = paced_pipeline
    v1 = make_video(tmp_path, "一.mp4")

    t1 = submit_task(client, v1)
    wait_for(lambda: get_task(client, t1["id"])["state"] == "running")

    resp = client.post(f"/api/tasks/{t1['id']}/cancel")
    assert resp.status_code == 200
    wait_for(lambda: get_task(client, t1["id"])["state"] == "cancelled")
    done = get_task(client, t1["id"])
    assert done["output_path"] is None


def test_已结束的任务不可再取消(client, tmp_path, paced_pipeline):
    started, release, _ = paced_pipeline
    v1 = make_video(tmp_path, "一.mp4")
    t1 = submit_task(client, v1)
    wait_for(lambda: get_task(client, t1["id"])["state"] == "running")
    client.post(f"/api/tasks/{t1['id']}/cancel")
    wait_for(lambda: get_task(client, t1["id"])["state"] == "cancelled")

    resp = client.post(f"/api/tasks/{t1['id']}/cancel")
    assert resp.status_code == 409
