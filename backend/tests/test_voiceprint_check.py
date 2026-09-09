"""按需声纹校验：不进分析阶段链，只产出并持久化待确认建议。"""

import json
from pathlib import Path

import conftest
from conftest import (
    CannedDiarizer,
    CannedEmbedder,
    CannedSeparator,
    make_client,
    make_video,
    run_to_analysis,
    statuses,
    wait_for,
    get_task,
)
from app.inference import SpeakerSegment
from app.voiceprints import add_vectors


def oh(index: int, size: int = 4) -> list[float]:
    return [1.0 if i == index else 0.0 for i in range(size)]


def add_library(root: Path, name: str, vector: list[float]):
    return add_vectors(
        root,
        name,
        [vector],
        model="canned-embedder",
        source_workspace="test",
    )


def check_path(task: dict) -> Path:
    return Path(task["workspace_dir"]) / "03b-voiceprint-check" / "voiceprint_check.json"


def test_新分析阶段链不含认人_库变动不触发重算(tmp_path):
    separator = CannedSeparator()
    diarizer = CannedDiarizer()
    transcriber = conftest.CannedTranscriber()
    root = tmp_path / "workspaces"
    video = make_video(tmp_path, "按需.mp4", duration=16)

    with make_client(
        tmp_path, separator=separator, diarizer=diarizer, transcriber=transcriber
    ) as client:
        first = run_to_analysis(client, video)
        assert list(statuses(first)) == [
            "audio_extract", "separate", "diarize", "transcribe",
            "mask_erase", "encode_mux",
        ]
        assert "recognize" not in statuses(first)
        counts = (len(separator.calls), len(diarizer.calls), len(transcriber.calls))

        add_library(root, "Ross", oh(0))
        second = run_to_analysis(client, video)
        assert (len(separator.calls), len(diarizer.calls), len(transcriber.calls)) == counts, (
            "声纹库变动不得触发任何自动重算"
        )
        assert second["workspace_dir"] == first["workspace_dir"]


def test_按需校验产出建议且不改生效时段(tmp_path):
    embedder = CannedEmbedder(
        regions=[(0.0, 4.0, oh(1)), (5.0, 9.0, oh(1)), (10.0, 14.0, oh(0)), (15.0, 19.0, oh(2))],
        default=[0.0, 0.0, 0.0, 0.0],
    )
    turns = (
        SpeakerSegment("SPEAKER_00", 0.0, 4.0),
        SpeakerSegment("SPEAKER_00", 5.0, 9.0),
        SpeakerSegment("SPEAKER_00", 10.0, 14.0),
        SpeakerSegment("SPEAKER_01", 15.0, 19.0),
    )
    diarizer = CannedDiarizer(turns=turns)
    video = make_video(tmp_path, "建议.mp4", duration=20)
    root = tmp_path / "workspaces"

    with make_client(
        tmp_path, diarizer=diarizer, embedder=embedder
    ) as client:
        task = run_to_analysis(client, video)
        add_library(root, "Ross", oh(0))
        add_library(root, "Rachel", oh(1))
        timeline_before = client.get(f"/api/tasks/{task['id']}/diarization").json()

        resp = client.post(f"/api/tasks/{task['id']}/voiceprint-check")
        assert resp.status_code == 200, resp.text
        wait_for(lambda: get_task(client, task["id"])["voiceprint_check"]["state"] == "completed")

        result = client.get(f"/api/tasks/{task['id']}/voiceprint-check").json()
        assert result["state"] == "completed"
        suggestions = result["result"]["suggestions"]
        types = {s["type"] for s in suggestions}
        assert "reassign" in types
        reassign = next(s for s in suggestions if s["type"] == "reassign")
        assert reassign["from_label"] == "SPEAKER_00"
        assert reassign["to_label"] == "Ross"
        assert reassign["start"] == 10.0 and reassign["end"] == 14.0
        assert check_path(task).is_file()

        timeline_after = client.get(f"/api/tasks/{task['id']}/diarization").json()
        assert timeline_after == timeline_before, "校验建议不得直接改变生效时段"


def test_按需校验建议命名与合并(tmp_path):
    turns = (
        SpeakerSegment("SPEAKER_00", 0.0, 4.0),
        SpeakerSegment("SPEAKER_03", 5.0, 15.0),
    )
    diarizer = CannedDiarizer(turns=turns)
    video = make_video(tmp_path, "合并.mp4", duration=16)
    root = tmp_path / "workspaces"

    with make_client(
        tmp_path,
        diarizer=diarizer,
        embedder=CannedEmbedder(regions=[(0.0, 15.0, oh(0))]),
    ) as client:
        task = run_to_analysis(client, video)
        add_library(root, "Ross", oh(0))
        client.post(f"/api/tasks/{task['id']}/voiceprint-check")
        wait_for(lambda: get_task(client, task["id"])["voiceprint_check"]["state"] == "completed")
        suggestions = client.get(f"/api/tasks/{task['id']}/voiceprint-check").json()["result"]["suggestions"]
        assert [s["type"] for s in suggestions].count("rename") == 2
        merge = next(s for s in suggestions if s["type"] == "merge")
        assert merge["name"] == "Ross"
        assert merge["labels"] == ["SPEAKER_00", "SPEAKER_03"]
        assert merge["canonical"] == "SPEAKER_03"


def test_声纹库为空得到明确提示(tmp_path):
    video = make_video(tmp_path, "空库.mp4", duration=8)
    with make_client(tmp_path) as client:
        task = run_to_analysis(client, video)
        resp = client.post(f"/api/tasks/{task['id']}/voiceprint-check")
        assert resp.status_code == 409
        assert "声纹库为空" in resp.json()["detail"]


def test_校验有进度且可取消(tmp_path):
    gate = __import__("threading").Event()
    embedder = CannedEmbedder(gate=gate)
    video = make_video(tmp_path, "取消.mp4", duration=12)
    root = tmp_path / "workspaces"

    with make_client(tmp_path, embedder=embedder) as client:
        task = run_to_analysis(client, video)
        add_library(root, "Ross", oh(0))
        assert client.post(f"/api/tasks/{task['id']}/voiceprint-check").status_code == 200
        wait_for(lambda: get_task(client, task["id"])["voiceprint_check"]["state"] == "running")
        assert client.delete(f"/api/tasks/{task['id']}/voiceprint-check").status_code == 200
        gate.set()
        wait_for(lambda: get_task(client, task["id"])["voiceprint_check"]["state"] == "cancelled")
        assert not check_path(task).exists()


def test_旧认人产物不作为生效层_旧清单打开不崩(tmp_path):
    separator = CannedSeparator()
    diarizer = CannedDiarizer()
    transcriber = conftest.CannedTranscriber()
    video = make_video(tmp_path, "旧产物.mp4", duration=8)

    with make_client(
        tmp_path, separator=separator, diarizer=diarizer, transcriber=transcriber
    ) as client:
        first = run_to_analysis(client, video)
        ws = Path(first["workspace_dir"])
        manifest_path = ws / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        old = {
            "id": "recognize",
            "name": "认人/纠错",
            "status": "done",
            "started_at": "2025-01-01T00:00:00+00:00",
            "finished_at": "2025-01-01T00:00:00+00:00",
            "artifacts": ["03b-recognize/recognition.json"],
            "outputs": {"recognition": "03b-recognize/recognition.json"},
            "progress": 1.0,
        }
        manifest["stages"].append(old)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        rec_dir = ws / "03b-recognize"
        rec_dir.mkdir()
        (rec_dir / "recognition.json").write_text(json.dumps({
            "version": 1, "revision": 9, "model": "canned-embedder",
            "used_library": True,
            "mapping": {"SPEAKER_00": "旧自动名", "SPEAKER_01": "旧自动名"},
            "segments": [{"speaker": "SPEAKER_99", "start": 0.0, "end": 999.0}],
        }, ensure_ascii=False), encoding="utf-8")

        counts = (len(separator.calls), len(diarizer.calls), len(transcriber.calls))
        resp = client.post("/api/workspaces/旧产物/open")
        assert resp.status_code == 200, resp.text
        opened_id = resp.json()["id"]
        wait_for(lambda: get_task(client, opened_id)["state"] == "completed")
        assert (len(separator.calls), len(diarizer.calls), len(transcriber.calls)) == counts
        timeline = client.get(f"/api/tasks/{opened_id}/diarization").json()
        labels = {s["label"] for s in timeline["speakers"]}
        assert labels == {"SPEAKER_00", "SPEAKER_01"}, "旧认人产物不得改派时间轴"
        assert all(s["name"] is None for s in timeline["speakers"]), "旧自动命名不再生效"
