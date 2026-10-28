# video-role-play

本地 Web 工具：从源视频音频中识别说话人，擦除你选定说话人的声音，生成成品视频写入目标文件夹。成品拿去做角色配音——配音发生在工具之外，不属于本项目范围。

全程在本机运行，不上传视频。

## 前置条件

| 依赖 | 版本 | 说明 |
|---|---|---|
| Python | 3.10 | `requires-python` 锁定 `>=3.10,<3.11` |
| [uv](https://docs.astral.sh/uv/) | — | 后端依赖管理 |
| Node.js + [pnpm](https://pnpm.io/) | — | 前端构建 |
| **ffmpeg / ffprobe** | — | **必须在 PATH 中**，用于抽音轨与合成成品视频 |
| Hugging Face token | — | 说话人分离与声纹功能需要，见下节 |

首次运行会自动下载约 320 MB 模型权重（demucs 81 MB、pyannote 98 MB、Whisper base.en 141 MB），下载到本地缓存后即可完全离线使用。

## 配置 HF token（说话人分离必需）

`pyannote/speaker-diarization-3.1`、`pyannote/segmentation-3.0`、`pyannote/embedding` 是 Hugging Face 上的**门控模型**：权重是 MIT 许可，但下载前必须由你本人在模型页接受使用条款。本项目不分发这些权重。

1. 注册并登录 [huggingface.co](https://huggingface.co)
2. 在下列三个模型页分别点 **Agree to the user license**：
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/embedding
3. https://huggingface.co/settings/tokens → Create new token → 类型选 **Read**
4. 填入 `.env`（见下节）

接受条款的账号必须与 token 属于同一账号。未配置 token 时应用仍能启动并跑到「人声/伴奏分离」，之后会在「说话人分离」阶段失败；认人/纠错与声纹入库会静默跳过。

## 快速开始

```bash
cp .env.example .env
chmod 600 .env          # 只让自己可读
$EDITOR .env            # 把 HF_TOKEN= 填上
./run.sh                # 装依赖 → 构建前端 → 启动 → 打开浏览器
```

也可以改用首次配置向导，它会逐步检查前置工具、引导接受模型许可、写入 token 并说明首次启动：

```bash
./scripts/setup-wizard.sh
```

启动后访问 http://127.0.0.1:8770（`Ctrl+C` 退出）。

`run.sh` 等价于 `uv sync` + `pnpm build` + `backend/.venv/bin/python run.py`，中国大陆网络下 PyPI 直连失败会自动回退到清华镜像。

常用选项：

```bash
./run.sh --port 8888      # 显式指定端口（被占用则直接报错）
./run.sh --no-browser     # 不自动打开浏览器
```

不指定 `--port` 时，端口被占用会自动往后避让（WSL 下连 Windows 宿主一侧的占用一起判）。也可以用 `.env` 里的 `VRP_PORT` 设置首选端口。

## 开发模式

前后端分开跑，前端热更新：

```bash
# 终端 1：后端（必须用 venv 解释器；裸 python 找不到 uvicorn）
cd backend && uv sync && cd .. && backend/.venv/bin/python run.py --no-browser

# 终端 2：前端 dev server（/api 代理到后端）
cd frontend && pnpm install && pnpm dev
```

dev server 默认代理到 `8770`。若后端避让到了别的端口，用 `VRP_PORT=<端口> pnpm dev`。

## 测试

```bash
cd backend && uv run pytest      # 后端行为测试
cd frontend && pnpm test         # vitest
```

行为测试对模型推理注入 canned 输出，不需要下载权重、不需要 HF token。

## 项目文档

- [`CONTEXT.md`](CONTEXT.md) — 领域术语表（说话人时段、切分草稿、人工校正、声纹库等的确切含义）
- [`docs/adr/`](docs/adr/) — 架构决策记录
- [`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md) — 第三方依赖与模型权重的许可归属

## 许可

[MIT](LICENSE) © 2026 lixuexuan

第三方依赖与模型权重的许可详情见 [`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md)。
