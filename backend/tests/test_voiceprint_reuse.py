"""跨视频说话人声纹复用（issue #9）：命名即入库 → 02 自动认人/合并/纠错。

装配缝隙走 HTTP TestClient：命名端点、说话人时间轴端点、声纹库端点、擦除
与阶段清单状态都是外部可观察行为；ML 推理注入 canned 替身。CannedEmbedder
按窗口中心命中已登记时间区间返回独热向量，相似度断言平凡（同向量=1.0、
异向量=0.0），阈值命中/拒绝由纯函数测试覆盖，这里只验证装配与用户可见结果。
"""

import json
import shutil
from pathlib import Path

import pytest
from app import voiceprints
from app.inference import SpeakerSegment

from conftest import (
    CannedDiarizer,
    CannedEmbedder,
    CannedSeparator,
    CannedTranscriber,
    call_counts,
    erase_to_completion,
    get_task,
    make_client,
    make_video,
    run_to_analysis,
    wait_for,
)

V0 = [1.0, 0.0, 0.0, 0.0]
V1 = [0.0, 1.0, 0.0, 0.0]

# 01/02 共用：同一时间区间 → 同一向量（模拟"同一个人声音稳定"）
TWO_SPEAKER_REGIONS = [
    (0.2, 0.9, V0),      # SPEAKER_00
    (0.95, 1.25, V1),    # SPEAKER_01
    (1.3, 1.8, V0),      # SPEAKER_00 第二段
]

DEFAULT_TURNS = tuple(
    SpeakerSegment(s, a, b) for s, a, b in (
        ("SPEAKER_00", 0.2, 0.9),
        ("SPEAKER_01", 0.95, 1.25),
        ("SPEAKER_00", 1.3, 1.8),
    )
)


def speakers_of(client, task_id: str) -> list[dict]:
    resp = client.get(f"/api/tasks/{task_id}/diarization")
    assert resp.status_code == 200, resp.text
    return resp.json()["speakers"]


def by_label(speakers: list[dict]) -> dict[str, dict]:
    return {s["label"]: s for s in speakers}


def rename(client, task_id: str, label: str, name: str):
    resp = client.put(
        f"/api/tasks/{task_id}/speakers/{label}/name", json={"name": name}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def recognition_of(ws: Path) -> dict:
    return json.loads((ws / "03b-recognize" / "recognition.json").read_text(
        encoding="utf-8"
    ))


def name_of(client, task_id: str, label: str) -> str | None:
    return by_label(speakers_of(client, task_id))[label]["name"]


def test_命名即入库_声纹库列表与删除端点(client, tmp_path):
    video = make_video(tmp_path, "第一集.mp4")
    task = run_to_analysis(client, video)

    # 无声纹库：库为空、阶段不产声纹，与现状一致
    data = client.get("/api/voiceprints").json()
    assert data["revision"] == 0
    assert data["names"] == []

    rename(client, task["id"], "SPEAKER_00", "男主")

    data = client.get("/api/voiceprints").json()
    assert data["revision"] == 1
    assert data["model"] == "canned-embedder"
    assert [(n["name"], n["vectors_count"]) for n in data["names"]] == [("男主", 2)]

    lib = voiceprints.read_library(tmp_path / "workspaces")
    assert len(lib["names"]["男主"]["vectors"]) == 2  # 纯函数读回一致

    # 删除误入库条目
    resp = client.delete("/api/voiceprints/男主")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"deleted": "男主"}
    data = client.get("/api/voiceprints").json()
    assert data["revision"] == 2
    assert data["names"] == []

    resp = client.delete("/api/voiceprints/男主")
    assert resp.status_code == 404, resp.text


def test_02自动认人_时间轴auto标记_手工名覆盖自动名(tmp_path):
    embedder = CannedEmbedder(regions=TWO_SPEAKER_REGIONS)
    with make_client(tmp_path, embedder=embedder) as client:
        v1 = make_video(tmp_path, "第一集.mp4")
        v2 = make_video(tmp_path, "第二集.mp4")

        first = run_to_analysis(client, v1)
        rename(client, first["id"], "SPEAKER_00", "男主")

        second = run_to_analysis(client, v2)
        rows = by_label(speakers_of(client, second["id"]))
        assert rows["SPEAKER_00"]["name"] == "男主", "02 应沿用 01 的手工命名"
        assert rows["SPEAKER_00"]["auto"] is True, "自动认出来的名字应带 auto 标记"
        assert rows["SPEAKER_01"]["name"] is None, "匹配不上的说话人保持未知名"
        assert rows["SPEAKER_01"]["auto"] is False

        # 01 是手工命名：不标 auto
        first_rows = by_label(speakers_of(client, first["id"]))
        assert first_rows["SPEAKER_00"]["name"] == "男主"
        assert first_rows["SPEAKER_00"]["auto"] is False

        # 手工命名永远覆盖自动名
        rename(client, second["id"], "SPEAKER_00", "旁白")
        rows = by_label(speakers_of(client, second["id"]))
        assert rows["SPEAKER_00"]["name"] == "旁白"
        assert rows["SPEAKER_00"]["auto"] is False

        # 清除手工名后回退到自动名
        rename(client, second["id"], "SPEAKER_00", "   ")
        rows = by_label(speakers_of(client, second["id"]))
        assert rows["SPEAKER_00"]["name"] == "男主"
        assert rows["SPEAKER_00"]["auto"] is True


def test_同名多簇合并_时间轴一个人_转写与擦除覆盖全部时段(tmp_path):
    v1 = make_video(tmp_path, "第一集.mp4")
    v2 = make_video(tmp_path, "第二集.mp4")

    with make_client(
        tmp_path, embedder=CannedEmbedder(regions=[(0.2, 0.9, V0), (1.3, 1.8, V0)])
    ) as client:
        first = run_to_analysis(client, v1)
        rename(client, first["id"], "SPEAKER_00", "男主")

    # 02 里同一个人被拆成 SPEAKER_00 与 SPEAKER_02 两个簇
    merge_turns = tuple(
        SpeakerSegment(s, a, b) for s, a, b in (
            ("SPEAKER_00", 0.2, 0.9),
            ("SPEAKER_02", 1.0, 1.5),
        )
    )
    with make_client(
        tmp_path,
        diarizer=CannedDiarizer(turns=merge_turns),
        embedder=CannedEmbedder(regions=[(0.2, 0.9, V0), (1.0, 1.5, V0)]),
    ) as client:
        second = run_to_analysis(client, v2)
        rows = by_label(speakers_of(client, second["id"]))
        assert set(rows) == {"SPEAKER_00"}, "同名多簇应合并为一个人，不再显示两个"
        assert rows["SPEAKER_00"]["name"] == "男主"
        assert rows["SPEAKER_00"]["auto"] is True
        assert rows["SPEAKER_00"]["speech_time"] == pytest.approx(1.2)
        assert [s["start"] for s in rows["SPEAKER_00"]["segments"]] == [0.2, 1.0]

        # 转写样本取自修正后时段：只有合并后的规范说话人
        trans = client.get(f"/api/tasks/{second['id']}/transcripts").json()
        assert {s["speaker"] for s in trans["samples"]} == {"SPEAKER_00"}
        assert {s["start"] for s in trans["samples"]} == {0.2, 1.0}

        # 已并入规范标签的旧标签不再可擦除
        resp = client.post(
            f"/api/tasks/{second['id']}/erase", json={"speakers": ["SPEAKER_02"]}
        )
        assert resp.status_code == 404, resp.text

        # 擦除规范标签：掩码覆盖合并后的全部时段
        erase_to_completion(client, second["id"], ["SPEAKER_00"])
        ws = Path(second["workspace_dir"])
        manifest = json.loads((ws / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["mask_erase"]["intervals"] == [[0.2, 0.9], [1.0, 1.5]]  # JSON 落盘后为列表


def test_逐段验证重指派_标错段回到正确人物(tmp_path):
    v1 = make_video(tmp_path, "第一集.mp4")
    v2 = make_video(tmp_path, "第二集.mp4", duration=6)

    with make_client(
        tmp_path, embedder=CannedEmbedder(regions=[(0.2, 0.9, V0), (1.3, 1.8, V0)])
    ) as client:
        first = run_to_analysis(client, v1)
        rename(client, first["id"], "SPEAKER_00", "男主")

    # 02：一段 2 秒的声音被错标成 SPEAKER_01，但其声纹其实是 男主
    reassign_turns = tuple(
        SpeakerSegment(s, a, b) for s, a, b in (
            ("SPEAKER_00", 0.2, 1.0),
            ("SPEAKER_01", 2.0, 4.0),   # 2 秒，达到逐段验证门槛
            ("SPEAKER_01", 4.5, 4.8),   # 短段，跳过验证（可能是别人）
        )
    )
    with make_client(
        tmp_path,
        diarizer=CannedDiarizer(turns=reassign_turns),
        embedder=CannedEmbedder(regions=[
            (0.2, 1.0, V0),
            (2.0, 4.0, V0),
            (4.5, 4.8, V1),
        ]),
    ) as client:
        second = run_to_analysis(client, v2)
        rows = by_label(speakers_of(client, second["id"]))
        assert rows["SPEAKER_00"]["name"] == "男主"
        assert rows["SPEAKER_00"]["auto"] is True
        assert rows["SPEAKER_00"]["speech_time"] == pytest.approx(2.8), (
            "标错的 2 秒段应重指派回 男主，并入其总时长"
        )
        assert [s["start"] for s in rows["SPEAKER_00"]["segments"]] == [0.2, 2.0]
        assert rows["SPEAKER_01"]["speech_time"] == pytest.approx(0.3)
        assert rows["SPEAKER_01"]["name"] is None


def test_改库后重开_只重跑认人与转写_昂贵阶段零二次调用(tmp_path):
    separator, diarizer, transcriber = CannedSeparator(), CannedDiarizer(), CannedTranscriber()
    embedder = CannedEmbedder(regions=TWO_SPEAKER_REGIONS)
    v1 = make_video(tmp_path, "第一集.mp4")
    v2 = make_video(tmp_path, "第二集.mp4")
    with make_client(
        tmp_path, separator=separator, diarizer=diarizer,
        transcriber=transcriber, embedder=embedder,
    ) as client:
        first = run_to_analysis(client, v1)
        rename(client, first["id"], "SPEAKER_00", "男主")

        second = run_to_analysis(client, v2)
        assert name_of(client, second["id"], "SPEAKER_00") == "男主"
        ws = Path(second["workspace_dir"])
        assert recognition_of(ws)["revision"] == 1

        counts_before = call_counts(separator, diarizer, transcriber)

        # 改库：01 里再命名 SPEAKER_01 → 反派，库 revision 递增
        rename(client, first["id"], "SPEAKER_01", "反派")
        assert voiceprints.revision(tmp_path / "workspaces") == 2

        # 重开 02：只重跑认人/纠错与样本转写
        resp = client.post("/api/workspaces/第二集/open")
        assert resp.status_code == 200, resp.text
        opened = resp.json()
        wait_for(lambda: get_task(client, opened["id"])["state"] == "completed")

        after = call_counts(separator, diarizer, transcriber)
        assert (after[0], after[1]) == (counts_before[0], counts_before[1]), (
            "改库后重开不应重算音轨提取/分离/说话人分离"
        )
        assert after[2] == counts_before[2] + 1, "样本转写应重跑（基于修正后时段）"

        rows = by_label(speakers_of(client, opened["id"]))
        assert rows["SPEAKER_00"]["name"] == "男主"
        assert rows["SPEAKER_00"]["auto"] is True
        assert rows["SPEAKER_01"]["name"] == "反派", "新入库的名字应在重开后被认出来"
        assert rows["SPEAKER_01"]["auto"] is True
        assert recognition_of(ws)["revision"] == 2


def test_旧6阶段清单重开不崩_补跑认人与转写(tmp_path):
    with make_client(tmp_path) as client:
        video = make_video(tmp_path, "老片.mp4")
        task = run_to_analysis(client, video)
        ws = Path(task["workspace_dir"])

        # 模拟升级前的工作区：清单没有认人阶段条目，也没有认人产物
        manifest = json.loads((ws / "manifest.json").read_text(encoding="utf-8"))
        manifest["stages"] = [
            s for s in manifest["stages"] if s["id"] != "recognize"
        ]
        (ws / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        shutil.rmtree(ws / "03b-recognize")

        w = next(
            w for w in client.get("/api/workspaces").json()["workspaces"]
            if w["name"] == "老片"
        )
        assert w["openable"] is True, "旧工作区核心分析齐全，应可打开进入擦除选择"
        assert w["reusable"] is False, "缺认人阶段，不算全量可复用"

        resp = client.post("/api/workspaces/老片/open")
        assert resp.status_code == 200, resp.text
        opened = resp.json()
        wait_for(lambda: get_task(client, opened["id"])["state"] == "completed")

        done = get_task(client, opened["id"])
        statuses = {s["id"]: s["status"] for s in done["stages"]}
        assert statuses["recognize"] == "done"
        assert statuses["transcribe"] == "done"
        assert client.get(f"/api/tasks/{opened['id']}/diarization").status_code == 200

        # 认人/纠错已按空库恒等补跑并落盘
        rec = recognition_of(ws)
        assert rec["used_library"] is False
        assert rec["revision"] == 0
        assert [s["speaker"] for s in rec["segments"]] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]

        w = next(
            w for w in client.get("/api/workspaces").json()["workspaces"]
            if w["name"] == "老片"
        )
        assert w["reusable"] is True, "补跑认人/纠错后，工作区恢复全量可复用"
