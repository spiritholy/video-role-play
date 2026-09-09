"""显式声纹入库（issue #14）：改名是纯本地操作，库只经按钮变化。

装配缝隙走 HTTP TestClient：改名端点、逐说话人入库端点与声纹库端点是外部
可观察行为；声纹抽取注入 CannedEmbedder，真实模型与权重下载不进入测试。
"""

import json
from pathlib import Path

from app import voiceprints

from conftest import (
    CannedEmbedder,
    make_client,
    make_video,
    run_to_analysis,
)


def rename(client, task_id: str, label: str, name: str):
    resp = client.put(
        f"/api/tasks/{task_id}/speakers/{label}/name", json={"name": name}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def store(client, task_id: str, label: str):
    return client.post(
        f"/api/tasks/{task_id}/speakers/{label}/voiceprint"
    )


def test_改名不入库_按钮显式入库(tmp_path):
    embedder = CannedEmbedder()
    with make_client(tmp_path, embedder=embedder) as client:
        video = make_video(tmp_path, "第一集.mp4")
        task = run_to_analysis(client, video)
        calls_after_analysis = len(embedder.calls)

        data = rename(client, task["id"], "SPEAKER_00", "男主")

        assert data == {"label": "SPEAKER_00", "name": "男主"}
        assert len(embedder.calls) == calls_after_analysis, "改名不应抽取声纹"
        assert client.get("/api/voiceprints").json()["revision"] == 0

        resp = store(client, task["id"], "SPEAKER_00")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "label": "SPEAKER_00",
            "name": "男主",
            "voiceprint_error": None,
        }
        assert len(embedder.calls) == calls_after_analysis + 1
        library = client.get("/api/voiceprints").json()
        assert library["revision"] == 1
        assert [(n["name"], n["vectors_count"]) for n in library["names"]] == [
            ("男主", 2)
        ]


def test_入库失败_不影响命名_原因可见(tmp_path):
    class FailingEmbedder(CannedEmbedder):
        def embed(self, wav, windows, on_progress):
            raise RuntimeError("声纹模型下载失败")

    with make_client(tmp_path, embedder=FailingEmbedder()) as client:
        video = make_video(tmp_path, "第一集.mp4")
        task = run_to_analysis(client, video)

        data = rename(client, task["id"], "SPEAKER_00", "男主")
        assert data["name"] == "男主"
        assert client.get("/api/voiceprints").json()["revision"] == 0

        resp = store(client, task["id"], "SPEAKER_00")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["name"] == "男主"
        assert data["voiceprint_error"] == "声纹模型下载失败"
        assert client.get("/api/voiceprints").json()["revision"] == 0


def test_同名人物再次入库_追加向量且不超过6个(tmp_path):
    with make_client(tmp_path) as client:
        first = run_to_analysis(client, make_video(tmp_path, "第一集.mp4"))
        rename(client, first["id"], "SPEAKER_00", "男主")
        store(client, first["id"], "SPEAKER_00")

        second = run_to_analysis(client, make_video(tmp_path, "第二集.mp4"))
        rename(client, second["id"], "SPEAKER_00", "男主")
        store(client, second["id"], "SPEAKER_00")

        library = client.get("/api/voiceprints").json()
        assert library["revision"] == 2
        assert library["names"][0]["vectors_count"] == 4
        assert voiceprints.read_library(tmp_path / "workspaces")["names"]["男主"][
            "vectors"
        ] == ([[0.0] * 4] * 4)


def test_入库使用当前生效时段(tmp_path):
    embedder = CannedEmbedder()
    with make_client(tmp_path, embedder=embedder) as client:
        task = run_to_analysis(client, make_video(tmp_path, "校正.mp4"))
        ws = Path(task["workspace_dir"])
        recognition_path = ws / "03b-recognize" / "recognition.json"
        recognition = json.loads(recognition_path.read_text(encoding="utf-8"))
        recognition["segments"] = [
            {"speaker": "SPEAKER_00", "start": 0.2, "end": 1.8}
        ]
        recognition_path.write_text(
            json.dumps(recognition, ensure_ascii=False), encoding="utf-8"
        )

        rename(client, task["id"], "SPEAKER_00", "男主")
        resp = store(client, task["id"], "SPEAKER_00")
        assert resp.status_code == 200, resp.text
        assert embedder.calls[-1][1] == ((0.2, 1.8),), "应抽取合并后的生效时段"


def test_未命名说话人拒绝入库(tmp_path):
    with make_client(tmp_path) as client:
        task = run_to_analysis(client, make_video(tmp_path, "未命名.mp4"))
        resp = store(client, task["id"], "SPEAKER_00")
        assert resp.status_code == 400, resp.text
        assert resp.json()["detail"] == "请先命名说话人，再存入声纹库"
