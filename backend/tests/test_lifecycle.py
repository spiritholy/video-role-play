"""任务生命周期：提交 → 分析运行 → 分析完成（等待擦除动作）。"""

from pathlib import Path


def test_提交源视频创建任务并运行到分析完成(client, tmp_path):
    from conftest import get_task, make_video, submit_task, wait_for

    video = make_video(tmp_path, "样本.mp4")

    task = submit_task(client, video)
    assert task["state"] in ("queued", "running")
    assert task["video_path"] == str(video)
    assert task["error"] is None

    wait_for(lambda: get_task(client, task["id"])["state"] == "completed")
    done = get_task(client, task["id"])
    assert done["error"] is None
    assert done["erase"] is None, "尚未发起擦除"
    assert done["output_path"] is None, "分析完成不产成品，成品由擦除动作生成"

    # 状态与阶段轨迹：①–④ 分析完成，⑤–⑥ 待用户多选目标说话人后触发
    stages = {s["id"]: s["status"] for s in done["stages"]}
    assert stages == {
        "audio_extract": "done",
        "separate": "done",
        "diarize": "done",
        "transcribe": "done",
        "mask_erase": "pending",
        "encode_mux": "pending",
    }

    # 发起擦除后任务重跑 ⑤–⑥ 并产出成品
    resp = client.post(
        f"/api/tasks/{done['id']}/erase", json={"speakers": ["SPEAKER_00"]}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["state"] in ("queued", "running")
    wait_for(lambda: get_task(client, done["id"])["state"] == "completed")
    final = get_task(client, done["id"])
    assert final["output_path"] is not None
    assert Path(final["output_path"]).exists()
    assert final["erase"] == {
        "speakers": ["SPEAKER_00"],
        "output_dir": None,
    }
    stages = {s["id"]: s["status"] for s in final["stages"]}
    assert stages["mask_erase"] == "done"
    assert stages["encode_mux"] == "done"
