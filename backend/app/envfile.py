""".env 读取：零依赖、非覆盖式地把本地配置注入进程环境。

为什么自己写而不引 python-dotenv：requirements.txt 锁的是一整组互相牵制的
CPU 推理依赖版本，为一个二十行的键值读取扩大依赖面不划算。

语义（与 dotenv 惯例一致）：
- 只补缺口：进程里已有的**非空**同名变量优先，.env 不覆盖——临时
  `HF_TOKEN=xxx python run.py` 照常生效；
- 文件内部同名键重复时后者胜出（像 source 一遍），但注入判定一律以**调用前**
  的进程环境为准，避免同一份配置里前后两行互相打架；
- 支持 `export KEY=VALUE`、`#` 整行注释、未加引号值后的 ` #` 行尾注释、
  单/双引号包裹的值（引号内的 # 与空格保留）；
- 文件里值为空的键视为「没配」，不写入——留空的 HF_TOKEN 不该骗过预检；
- 文件不存在时静默返回——token 只放在 shell 环境里也是合法配置。
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# KEY=VALUE / export KEY=VALUE；键必须是合法 shell 标识符
_LINE = re.compile(r"\A(?:export[ \t]+)?([A-Za-z_][A-Za-z0-9_]*)[ \t]*=(.*)\Z")
# 未加引号的值：空格 + # 之后视为行尾注释
_TRAILING_COMMENT = re.compile(r"[ \t]+#.*\Z")


def _clean_value(raw: str) -> str:
    """剥引号与行尾注释：引号内原样保留（含 # 与空格），不展开转义。"""
    value = raw.strip()
    quote = value[:1]
    if quote in ("\"", "'"):
        end = value.find(quote, 1)
        tail = value[end + 1 :].strip() if end != -1 else None
        # 闭引号后只能跟空白或 # 注释，否则按未加引号的值处理
        if end != -1 and (not tail or tail.startswith("#")):
            return value[1:end]
    return _TRAILING_COMMENT.sub("", value).strip()


def load_env_file(path: Path, *, override: bool = False) -> list[str]:
    """把 .env 的键值写进进程环境，返回实际写入的键（按首次出现保序去重）。"""
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, IsADirectoryError):
        return []
    before = dict(os.environ)  # 判定基准固定为调用前的环境
    pending: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE.match(line)
        if match is None:
            continue  # 非法行跳过：配置不该因为一行手误就起不来
        key, value = match.group(1), _clean_value(match.group(2))
        if not value:
            continue  # 留空 = 未配置，不该骗过启动预检
        if not override and before.get(key):
            continue
        pending[key] = value
    for key, value in pending.items():
        os.environ[key] = value
    return list(pending)
