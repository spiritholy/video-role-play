#!/usr/bin/env bash
# 一条命令启动整个应用：准备 venv 与依赖、构建前端（如缺失）、启动后端并打开浏览器。
set -euo pipefail
cd "$(dirname "$0")"

VENV_PY="backend/.venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
  echo "[setup] 创建 Python 虚拟环境…"
  python3 -m venv backend/.venv
fi

if ! "$VENV_PY" -c "import fastapi, uvicorn, httpx" 2>/dev/null; then
  echo "[setup] 安装后端依赖…"
  "$VENV_PY/bin/pip" install -r backend/requirements.txt \
    || "$VENV_PY/bin/pip" install -r backend/requirements.txt \
      -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

if [ ! -f frontend/dist/index.html ]; then
  echo "[setup] 构建前端…"
  if [ ! -d frontend/node_modules ]; then
    (cd frontend && npm install)
  fi
  (cd frontend && npm run build)
fi

exec "$VENV_PY" run.py "$@"
