"""跨视频说话人声纹复用：显式入库为按需校验提供精选声纹来源。

装配缝隙走 HTTP TestClient：命名端点、说话人时间轴端点、声纹库端点、擦除
与阶段清单状态都是外部可观察行为；ML 推理注入 canned 替身。CannedEmbedder
按窗口中心命中已登记时间区间返回独热向量，相似度断言平凡（同向量=1.0、
异向量=0.0），阈值命中/拒绝由纯函数测试覆盖，这里只验证装配与用户可见结果。
"""

from pathlib import Path

from app import voiceprints

from conftest import (
    CannedEmbedder,
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

class FailingEmbedder(CannedEmbedder):
    """embed 恒抛错：模拟声纹模型不可用/下载失败的显式入库失败。"""

    def __init__(self, error: Exception):
        super().__init__()
        self._error = error

    def embed(self, wav, windows, on_progress):
        raise self._error


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


def store(client, task_id: str, label: str):
    resp = client.post(f"/api/tasks/{task_id}/speakers/{label}/voiceprint")
    assert resp.status_code == 200, resp.text
    return resp.json()


def name_of(client, task_id: str, label: str) -> str | None:
    return by_label(speakers_of(client, task_id))[label]["name"]


def test_显式入库_声纹库列表与删除端点(client, tmp_path):
    video = make_video(tmp_path, "第一集.mp4")
    task = run_to_analysis(client, video)

    # 无声纹库：库为空、阶段不产声纹，与现状一致
    data = client.get("/api/voiceprints").json()
    assert data["revision"] == 0
    assert data["names"] == []

    rename(client, task["id"], "SPEAKER_00", "男主")
    store(client, task["id"], "SPEAKER_00")

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


def test_入库失败_命名不受影响_响应带出原因(tmp_path):
    embedder = FailingEmbedder(RuntimeError("声纹模型下载失败"))
    with make_client(tmp_path, embedder=embedder) as client:
        video = make_video(tmp_path, "第一集.mp4")
        task = run_to_analysis(client, video)

        data = rename(client, task["id"], "SPEAKER_00", "男主")
        data = store(client, task["id"], "SPEAKER_00")

        assert data["voiceprint_error"] == "声纹模型下载失败"
        assert data["name"] == "男主"
        assert name_of(client, task["id"], "SPEAKER_00") == "男主"

        data = client.get("/api/voiceprints").json()
        assert data["revision"] == 0
        assert data["names"] == []


def test_入库成功_响应不带错误(tmp_path):
    embedder = CannedEmbedder(regions=TWO_SPEAKER_REGIONS)
    with make_client(tmp_path, embedder=embedder) as client:
        video = make_video(tmp_path, "第一集.mp4")
        task = run_to_analysis(client, video)

        data = rename(client, task["id"], "SPEAKER_00", "男主")
        data = store(client, task["id"], "SPEAKER_00")

        assert data["voiceprint_error"] is None
        assert data["name"] == "男主"
        lib = client.get("/api/voiceprints").json()
        assert lib["revision"] == 1
        assert [(n["name"], n["vectors_count"]) for n in lib["names"]] == [("男主", 2)]


def test_启动预热_声纹模型后台加载(tmp_path):
    class TrackingEmbedder(CannedEmbedder):
        def __init__(self):
            super().__init__()
            self.loaded = False

        def load_model(self):
            self.loaded = True

    embedder = TrackingEmbedder()
    with make_client(tmp_path, embedder=embedder) as client:
        wait_for(lambda: embedder.loaded)
        assert client.get("/api/voiceprints").status_code == 200


def test_预热失败_不阻塞启动(tmp_path):
    class ExplodingEmbedder(CannedEmbedder):
        def load_model(self):
            raise RuntimeError("声纹模型下载失败")

    with make_client(tmp_path, embedder=ExplodingEmbedder()) as client:
        resp = client.get("/api/voiceprints")
        assert resp.status_code == 200
        assert resp.json()["revision"] == 0
