#!/usr/bin/env bash
# 一条命令启动整个应用：准备 venv 与依赖、构建前端（如缺失）、启动后端并打开浏览器。
set -euo pipefail
cd "$(dirname "$0")"

VENV_PY="backend/.venv/bin/python"

echo "[setup] 同步后端依赖（uv sync）…"
if ! uv sync --project backend; then
  echo "[setup] 直连 PyPI 失败，改用清华镜像重试…"
  UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv sync --project backend
fi

if [ ! -f frontend/dist/index.html ]; then
  echo "[setup] 构建前端…"
  if [ ! -d frontend/node_modules ]; then
    (cd frontend && pnpm install)
  fi
  (cd frontend && pnpm run build)
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
  echo "[warn] 现在仍能启动并使用到「人声/伴奏分离」，之后会在「说话人分离」失败；"
  echo "[warn] 认人/纠错与命名入库会静默跳过（命名照常生效，只是不参与跨视频自动认人）。"
fi

exec "$VENV_PY" run.py "$@"
