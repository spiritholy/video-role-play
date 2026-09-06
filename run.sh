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

# HF token 预检：pyannote 是门控模型，缺 token 会白跑几小时才撞上失败。
# 只提醒不拦停（run.py 会重复一次更权威的判断，--env-file 自定义路径时以它为准）。
has_env_token() { [ -n "${HF_TOKEN:-}" ] || [ -n "${HUGGINGFACE_TOKEN:-}" ]; }
has_file_token() {
  [ -f .env ] && grep -Eq \
    '^[[:space:]]*(export[[:space:]]+)?(HF_TOKEN|HUGGINGFACE_TOKEN)[[:space:]]*=[[:space:]]*[^[:space:]]' .env
}
if ! has_env_token && ! has_file_token; then
  echo "[warn] 未找到 HF token（进程环境与环境变量文件都没有）"
  echo "[warn] 建议：cp .env.example .env && chmod 600 .env，再把 token 填进 .env"
  echo "[warn] 现在仍能启动并使用到「人声/伴奏分离」，之后会在「说话人分离」失败。"
fi

exec "$VENV_PY" run.py "$@"
