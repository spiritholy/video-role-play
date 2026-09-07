"""声纹库读写：全局单文件、revision 递增、模型过滤、条目删除与上限。

库文件是认人/纠错阶段的关联失效依据（revision）与数据源（名字向量），
作为纯函数直接单测（先例：samples.select_samples）。
"""

import json
from pathlib import Path

import pytest

from app import voiceprints


def test_缺失库按空库处理_revision为0():
    lib = voiceprints.read_library(Path("/nonexistent"))
    assert lib["revision"] == 0
    assert lib["model"] == ""
    assert lib["names"] == {}
    assert voiceprints.revision(Path("/nonexistent")) == 0


def test_损坏库按空库处理(tmp_path):
    (tmp_path / voiceprints.VOICEPRINTS_FILENAME).write_text("{bad json", encoding="utf-8")
    assert voiceprints.read_library(tmp_path)["revision"] == 0


def test_写入递增revision并原子落盘(tmp_path):
    lib = voiceprints.add_vectors(tmp_path, "Ross", [[1.0, 0.0]], model="m1", source_workspace="ws1")
    assert lib["revision"] == 1
    assert lib["model"] == "m1"
    data = json.loads((tmp_path / voiceprints.VOICEPRINTS_FILENAME).read_text(encoding="utf-8"))
    assert data["revision"] == 1
    assert data["names"]["Ross"]["vectors"] == [[1.0, 0.0]]
    assert data["names"]["Ross"]["source_workspace"] == "ws1"
    assert data["names"]["Ross"]["updated_at"]

    lib = voiceprints.add_vectors(tmp_path, "Ross", [[2.0, 0.0]], model="m1", source_workspace="ws2")
    assert lib["revision"] == 2, "每次写入递增 revision"
    assert lib["names"]["Ross"]["vectors"] == [[1.0, 0.0], [2.0, 0.0]]


def test_每个名字最多保留6个向量(tmp_path):
    for i in range(8):
        voiceprints.add_vectors(tmp_path, "Ross", [[float(i)]], model="m1", source_workspace="ws")
    lib = voiceprints.read_library(tmp_path)
    assert len(lib["names"]["Ross"]["vectors"]) == 6, "最多保留 6 个向量"
    assert lib["names"]["Ross"]["vectors"][-1] == [7.0], "保留最新向量"


def test_模型不一致条目视为不存在():
    lib = {"revision": 3, "model": "m1", "names": {"Ross": {"vectors": [[1.0]]}}}
    assert voiceprints.effective_entries(lib, "m1") == {"Ross": lib["names"]["Ross"]}
    assert voiceprints.effective_entries(lib, "m2") == {}, "模型不一致视为不存在"
    assert voiceprints.effective_entries({"revision": 0, "model": "", "names": {}}, "m1") == {}


def test_模型不一致时写入整库对齐当前模型(tmp_path):
    voiceprints.add_vectors(tmp_path, "Ross", [[1.0]], model="m1", source_workspace="ws")
    lib = voiceprints.add_vectors(tmp_path, "Rachel", [[2.0]], model="m2", source_workspace="ws")
    assert lib["model"] == "m2"
    assert set(lib["names"]) == {"Rachel"}, "跨模型写入：旧向量整库作废"


def test_删除条目并递增revision(tmp_path):
    lib = voiceprints.add_vectors(tmp_path, "Ross", [[1.0]], model="m1", source_workspace="ws")
    rev = lib["revision"]
    assert voiceprints.delete_name(tmp_path, "Ross") is True
    assert voiceprints.delete_name(tmp_path, "Ross") is False, "已删除条目再删返回 False"
    assert voiceprints.revision(tmp_path) == rev + 1, "删除也递增 revision（匹配结果会变）"


def test_列表包含来源与向量数(tmp_path):
    voiceprints.add_vectors(tmp_path, "Ross", [[1.0], [2.0]], model="m1", source_workspace="e01")
    voiceprints.add_vectors(tmp_path, "Rachel", [[3.0]], model="m1", source_workspace="e02")
    items = voiceprints.list_names(tmp_path)
    assert [i["name"] for i in items] == ["Rachel", "Ross"]
    assert items[1]["vectors_count"] == 2
    assert items[1]["source_workspace"] == "e01"
    assert items[1]["updated_at"]
