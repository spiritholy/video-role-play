"""声纹校验建议裁决：接受并入人工校正层，忽略只标记不删除。"""

from pathlib import Path

from conftest import (
    CannedDiarizer,
    CannedEmbedder,
    make_client,
    make_video,
    run_to_analysis,
    wait_for,
    get_task,
)
from app.inference import SpeakerSegment
from app.voiceprints import add_vectors


def add_library(root: Path, name: str, vector: list[float]):
    return add_vectors(root, name, [vector], model="canned-embedder", source_workspace="test")


def oh(index: int, size: int = 4) -> list[float]:
    return [1.0 if i == index else 0.0 for i in range(size)]


def decision(client, task_id: str, index: int, value: str):
    return client.post(
        f"/api/tasks/{task_id}/voiceprint-check/suggestions/{index}/decision",
        json={"decision": value},
    )


def wait_for_check(client, task_id: str) -> list[dict]:
    resp = client.post(f"/api/tasks/{task_id}/voiceprint-check")
    assert resp.status_code == 200, resp.text
    wait_for(
        lambda: get_task(client, task_id)["voiceprint_check"]["state"] == "completed"
    )
    return client.get(f"/api/tasks/{task_id}/voiceprint-check").json()["result"][
        "suggestions"
    ]


def test_接受改派建议进校正快照_忽略建议只标记(tmp_path):
    turns = (
        SpeakerSegment("SPEAKER_00", 0.0, 4.0),
        SpeakerSegment("SPEAKER_00", 5.0, 9.0),
        SpeakerSegment("SPEAKER_00", 10.0, 14.0),
        SpeakerSegment("SPEAKER_01", 15.0, 19.0),
    )
    diarizer = CannedDiarizer(turns=turns)
    embedder = CannedEmbedder(
        regions=[
            (0.0, 4.0, oh(1)),
            (5.0, 9.0, oh(1)),
            (10.0, 14.0, oh(0)),
            (15.0, 19.0, oh(2)),
        ]
    )
    video = make_video(tmp_path, "裁决.mp4", duration=20)
    root = tmp_path / "workspaces"

    with make_client(tmp_path, diarizer=diarizer, embedder=embedder) as client:
        task = run_to_analysis(client, video)
        add_library(root, "Ross", oh(0))
        add_library(root, "Rachel", oh(1))
        suggestions = wait_for_check(client, task["id"])
        reassign_index = next(
            i for i, s in enumerate(suggestions) if s["type"] == "reassign"
        )
        rename_index = next(
            i for i, s in enumerate(suggestions) if s["type"] == "rename"
        )

        ignored = decision(client, task["id"], rename_index, "ignore")
        assert ignored.status_code == 200, ignored.text
        data = ignored.json()["result"]["suggestions"][rename_index]
        assert data["decision"] == "ignored"

        accepted = decision(client, task["id"], reassign_index, "accept")
        assert accepted.status_code == 200, accepted.text
        timeline = accepted.json()["timeline"]
        assert timeline["correction"]["state"] == "active"
        segments = {s["index"]: s for row in timeline["speakers"] for s in row["segments"]}
        assert segments[2]["start"] == 10.0 and segments[2]["end"] == 14.0
        assert next(r for r in timeline["speakers"] if r["label"] == "Ross")[
            "segments"
        ] == [{"start": 10.0, "end": 14.0, "index": 2}]
        suggestions = accepted.json()["result"]["suggestions"]
        assert suggestions[reassign_index]["decision"] == "accepted"
        assert suggestions[rename_index]["decision"] == "ignored"

        reopened = run_to_analysis(client, video)
        transcripts = client.get(f"/api/tasks/{reopened['id']}/transcripts").json()
        assert "Ross" not in {s["speaker"] for s in transcripts["samples"]}
        assert "SPEAKER_00" in {s["speaker"] for s in transcripts["samples"]}


def test_接受命名与合并建议_状态随工作区持久化(tmp_path):
    turns = (
        SpeakerSegment("SPEAKER_00", 0.0, 4.0),
        SpeakerSegment("SPEAKER_03", 5.0, 15.0),
    )
    diarizer = CannedDiarizer(turns=turns)
    video = make_video(tmp_path, "持久.mp4", duration=16)
    root = tmp_path / "workspaces"

    with make_client(
        tmp_path, diarizer=diarizer, embedder=CannedEmbedder(regions=[(0.0, 15.0, oh(0))])
    ) as client:
        task = run_to_analysis(client, video)
        add_library(root, "Ross", oh(0))
        suggestions = wait_for_check(client, task["id"])
        rename_index = next(
            i for i, s in enumerate(suggestions)
            if s["type"] == "rename" and s["label"] == "SPEAKER_00"
        )
        merge_index = next(i for i, s in enumerate(suggestions) if s["type"] == "merge")

        resp = decision(client, task["id"], rename_index, "accept")
        assert resp.status_code == 200, resp.text
        assert resp.json()["timeline"]["correction"]["state"] == "active"

        resp = decision(client, task["id"], merge_index, "accept")
        assert resp.status_code == 200, resp.text
        timeline = resp.json()["timeline"]
        assert [s["label"] for s in timeline["speakers"]] == ["SPEAKER_03"]
        check = client.get(f"/api/tasks/{task['id']}/voiceprint-check").json()
        decisions = [s["decision"] for s in check["result"]["suggestions"]]
        assert decisions.count("accepted") == 2

    with make_client(
        tmp_path, diarizer=diarizer, embedder=CannedEmbedder(regions=[(0.0, 15.0, oh(0))])
    ) as client:
        second = run_to_analysis(client, video)
        check = client.get(f"/api/tasks/{second['id']}/voiceprint-check").json()
        assert check["state"] == "completed"
        decisions = [s["decision"] for s in check["result"]["suggestions"]]
        assert decisions.count("accepted") == 2
        timeline = client.get(f"/api/tasks/{second['id']}/diarization").json()
        assert [s["label"] for s in timeline["speakers"]] == ["SPEAKER_03"]


def test_重新说话人分离后建议状态保留但作废(tmp_path):
    turns = (
        SpeakerSegment("SPEAKER_00", 0.0, 4.0),
        SpeakerSegment("SPEAKER_03", 5.0, 15.0),
    )
    diarizer = CannedDiarizer(turns=turns)
    video = make_video(tmp_path, "作废.mp4", duration=16)
    root = tmp_path / "workspaces"

    with make_client(
        tmp_path, diarizer=diarizer, embedder=CannedEmbedder(regions=[(0.0, 15.0, oh(0))])
    ) as client:
        task = run_to_analysis(client, video)
        add_library(root, "Ross", oh(0))
        suggestions = wait_for_check(client, task["id"])
        merge_index = next(i for i, s in enumerate(suggestions) if s["type"] == "merge")
        assert decision(client, task["id"], merge_index, "ignore").status_code == 200

    with make_client(
        tmp_path, diarizer=diarizer, embedder=CannedEmbedder(regions=[(0.0, 15.0, oh(0))])
    ) as client:
        second = run_to_analysis(client, video, num_speakers=4)
        check = client.get(f"/api/tasks/{second['id']}/voiceprint-check").json()
        assert check["state"] == "completed"
        assert check["result"]["status"] == "invalidated"
        assert check["result"]["suggestions"][merge_index]["decision"] == "ignored"
        assert all("decision" in s for s in check["result"]["suggestions"])
        locked = decision(client, second["id"], merge_index, "ignore")
        assert locked.status_code == 409, locked.text
        assert "已作废" in locked.json()["detail"]
