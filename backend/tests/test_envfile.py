""".env 读取：非空环境变量优先、解析规则、缺文件静默。

纯进程环境操作，不触碰模型与网络；每条断言都清理相关变量，避免测试间串味。
"""

import os
from pathlib import Path

import pytest

from app.envfile import load_env_file

VARS = (
    "HF_TOKEN", "HF_ENDPOINT", "HUGGINGFACE_TOKEN",
    "VRP_WORKSPACE_ROOT", "SOME_KEY", "COMMENTED_KEY",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in VARS:
        monkeypatch.delenv(var, raising=False)


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


def test_文件不存在时静默跳过(tmp_path):
    assert load_env_file(tmp_path / "nope.env") == []
    assert "HF_TOKEN" not in os.environ


def test_目录当文件传时静默跳过(tmp_path):
    assert load_env_file(tmp_path) == []


def test_写入缺失的键并返回键名(tmp_path):
    env = write(tmp_path, "HF_TOKEN=hf_abc\nVRP_WORKSPACE_ROOT=/data/ws\n")
    assert load_env_file(env) == ["HF_TOKEN", "VRP_WORKSPACE_ROOT"]
    assert os.environ["HF_TOKEN"] == "hf_abc"
    assert os.environ["VRP_WORKSPACE_ROOT"] == "/data/ws"


def test_已有的非空环境变量优先不被覆盖(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "from-shell")
    env = write(tmp_path, "HF_TOKEN=from-file\nSOME_KEY=1\n")
    assert load_env_file(env) == ["SOME_KEY"]  # 只补没设过的
    assert os.environ["HF_TOKEN"] == "from-shell"


def test_空字符串变量视为缺口由env补齐(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "")
    env = write(tmp_path, "HF_TOKEN=hf_from_file\n")
    assert load_env_file(env) == ["HF_TOKEN"]
    assert os.environ["HF_TOKEN"] == "hf_from_file"


def test_override为真时覆盖(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "from-shell")
    env = write(tmp_path, "HF_TOKEN=from-file\n")
    assert load_env_file(env, override=True) == ["HF_TOKEN"]
    assert os.environ["HF_TOKEN"] == "from-file"


def test_解析export引号与行尾注释(tmp_path):
    env = write(
        tmp_path,
        "\n".join(
            [
                "export HF_TOKEN=hf_export",
                'SOME_KEY="带 空格 # 号"',
                'HF_ENDPOINT="https://hf-mirror.com" # 带注释的引号值',
                "export VRP_WORKSPACE_ROOT=/path/with 空格",
                "#COMMENTED_KEY=/commented/out",
                "HUGGINGFACE_TOKEN=hf_alias   # 行尾注释",
            ]
        ),
    )
    assert load_env_file(env) == [
        "HF_TOKEN",
        "SOME_KEY",
        "HF_ENDPOINT",
        "VRP_WORKSPACE_ROOT",
        "HUGGINGFACE_TOKEN",
    ]
    assert os.environ["SOME_KEY"] == "带 空格 # 号"  # 引号内不剥 #
    assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"  # 闭引号后的注释要剥
    assert os.environ["VRP_WORKSPACE_ROOT"] == "/path/with 空格"  # 未加引号保留空格
    assert os.environ["HUGGINGFACE_TOKEN"] == "hf_alias"
    assert "COMMENTED_KEY" not in os.environ  # 注释行不算配置


def test_非法行与空值不写入(tmp_path):
    env = write(tmp_path, "not a kv line\n=missing_name\nSOME_KEY=\n")
    assert load_env_file(env) == []
    assert "SOME_KEY" not in os.environ  # 空值补了也是缺口，不如留空让提示说话


def test_同名键后写的胜出且键列表保序去重(tmp_path):
    env = write(tmp_path, "SOME_KEY=first\nHF_TOKEN=t\nSOME_KEY=second\n")
    assert load_env_file(env) == ["SOME_KEY", "HF_TOKEN"]
    assert os.environ["SOME_KEY"] == "second"
