# Third-Party Notices

本项目 `video-role-play` 以 **MIT License** 发布（见根目录 `LICENSE`）。

本文件收录项目在**运行时**与**构建期**使用的第三方组件及其许可，仅作归属声明之用，
不修改、不限制任何第三方组件自身的许可条款。各组件的完整许可以其自身随附的
LICENSE 文件为准。

版本号取自审计快照：后端 `backend/uv.lock`（Python 3.10，120 个已安装包，含传递依赖），
前端 `frontend/pnpm-lock.yaml`（106 个包）。依赖升级后本文件需同步更新，重新生成方式见文末。

---

## 1. 模型权重

本项目**不随仓库分发任何模型权重**。所有权重在用户首次运行时按需从公开地址下载，
下载到用户本地缓存后即离线可用。

| 模型 | 用途 | 许可 | 门控仓库 |
|---|---|---|---|
| `adefossez/HTDemucs` | 人声 / 伴奏分离 | MIT | 否 |
| `pyannote/speaker-diarization-3.1` | 说话人分离流水线 | MIT | **是** |
| `pyannote/segmentation-3.0` | 语音活动检测与说话人分段 | MIT | **是** |
| `pyannote/embedding`<br>（`pyannote/wespeaker-voxceleb-resnet34-LM`）| 说话人声纹抽取 | MIT | **是** |
| `Systran/faster-whisper-base.en`<br>（OpenAI Whisper `base.en`）| 样本段台词转写 | MIT | 否 |

### pyannote 门控模型的使用条件

`pyannote/speaker-diarization-3.1`、`pyannote/segmentation-3.0`、`pyannote/embedding`
在 Hugging Face 上是**门控（gated）仓库**：权重本身以 MIT 发布，但下载前必须由使用者
本人在模型页接受用户条款，并使用自己的 Access Token。

因此：

- 本项目**不能、也不会**代为分发或镜像这些权重；
- 使用者需自行完成注册、接受条款、配置 `HF_TOKEN`——步骤见 `.env.example`；
- 门控机制由 pyannote 维护者用于了解用户构成以支撑开源维护，请如实填写其表单。

---

## 2. Python 依赖（backend）

### MIT（56）

`alembic==1.19.2`, `annotated-doc==0.0.5`, `annotated-types==0.8.0`, `anyio==4.15.1`,
`asteroid-filterbanks==0.4.0`, `attrs==26.1.0`, `cffi==2.1.1`, `charset-normalizer==2.1.1`,
`coloredlogs==15.0.1`, `colorlog==6.12.0`, `ctranslate2==4.8.2`, `demucs==4.1.0`,
`docopt==0.6.2`, `einops==0.8.2`, `exceptiongroup==1.3.1`, `fastapi==0.115.6`,
`faster-whisper==1.1.0`, `filelock==3.32.3`, `fonttools==4.64.0`, `h11==0.16.0`,
`humanfriendly==10.0`, `iniconfig==2.3.0`, `julius==0.2.8`, `Mako==1.4.1`,
`markdown-it-py==4.2.0`, `mdurl==0.1.2`, `onnxruntime==1.23.2`, `optuna==5.0.0`,
`pillow==12.3.0`, `pluggy==1.6.0`, `primePy==1.3` \*, `pyannote.audio==3.1.0`,
`pyannote.core==5.0.0`, `pyannote.database==5.1.3`, `pyannote.metrics==3.2.1` \*,
`pyannote.pipeline==3.0.1`, `pydantic==2.10.5`, `pydantic_core==2.27.2`,
`pyparsing==3.3.2`, `pytest==8.3.4`, `pytorch-metric-learning==2.9.0` \*,
`pytz==2026.3.post1`, `PyYAML==6.0.3`, `rich==15.0.0`, `ruamel.yaml==0.18.17`,
`ruamel.yaml.clib==0.2.15`, `setuptools==78.1.0`, `six==1.17.0`, `SQLAlchemy==2.0.52`,
`tabulate==0.10.0`, `tensorboardX==2.6.5`, `tomli==2.4.1`,
`torch-audiomentations==0.12.0`, `torch_pitch_shift==1.2.5`, `typer==0.27.2`,
`urllib3==1.26.13`

### BSD-3-Clause（32）

`antlr4-python3-runtime==4.9.3`, `av==17.1.0`, `click==8.5.0`, `cloudpickle==3.1.2`,
`contourpy==1.3.2`, `cycler==0.12.1`, `fsspec==2024.12.0`, `httpcore==1.0.9`,
`httpx==0.28.1`, `idna==3.4`, `Jinja2==3.1.6`, `joblib==1.6.0`, `kiwisolver==1.5.1`,
`MarkupSafe==3.0.3`, `mpmath==1.3.0`, `networkx==3.4.2`, `numpy==1.26.4`,
`omegaconf==2.3.1`, `pandas==2.3.3`, `protobuf==7.36.1`, `pycparser==3.0`,
`Pygments==2.21.0`, `scikit-learn==1.5.2`, `scipy==1.13.1`, `semver==3.0.4`,
`soundfile==0.14.0`, `starlette==0.41.3`, `sympy==1.14.0`, `threadpoolctl==3.6.0`,
`torch==2.4.1+cpu`, `torchaudio==2.4.1+cpu`, `uvicorn==0.34.0`

### Apache-2.0（23）

`aiohttp==3.14.3`, `aiosignal==1.4.0`, `async-timeout==5.0.1`, `flatbuffers==25.12.19`,
`frozenlist==1.8.0`, `huggingface-hub==0.25.2`, `HyperPyYAML==1.2.3`,
`lightning==2.1.4`, `lightning-utilities==0.11.8`, `multidict==6.7.1`,
`packaging==24.1`, `propcache==0.5.2`, `pytorch-lightning==2.6.5`, `requests==2.28.1`,
`safetensors==0.8.0`, `sentencepiece==0.2.2`, `sortedcontainers==2.4.0`,
`sphn==0.2.1` \*, `speechbrain==0.5.16`, `tokenizers==0.23.2`, `torchmetrics==1.0.3`,
`tzdata==2026.3`, `yarl==1.24.5`

### PSF-2.0（4）

`aiohappyeyeballs==2.7.1`, `greenlet==3.5.5`, `matplotlib==3.10.9`,
`typing_extensions==4.16.0`

（PSF License Agreement 与 Python 自身许可一致，为 BSD 类宽松许可。
`matplotlib` 使用的是 "License agreement for matplotlib versions 1.3.0 and later"，
同属 BSD 类宽松许可。）

### ISC（1）

`shellingham==1.5.4`

### MPL-2.0（2）— 弱著佐权

`certifi==2022.12.7`, `tqdm==4.66.5`（后者声明为 `MPL-2.0 AND MIT`）

Mozilla Public License 2.0 的著佐权范围为**文件级**：仅要求对被修改的 MPL 文件本身
保持开源。本项目未修改这些文件，因此不产生额外义务。

### LGPL-3.0-or-later（1）— 弱著佐权

`lameenc==1.8.4`（`demucs` 的传递依赖，用于 MP3 编码）

GNU LGPL 的著佐权范围为**库级**：以独立库形式动态链接调用不构成对本项目代码的传染。
本项目仅通过 `demucs` 间接调用，未修改其源码、未静态链接。若将来把依赖打包进
可分发产物（wheel / 容器镜像），需保留使用者替换该库并重新链接的能力。

### 双许可（1）

`python-dateutil==2.9.0.post0` — `BSD-3-Clause OR Apache-2.0`，使用者可任选其一。

### \* 关于 4 个 PyPI 元数据缺失许可字段的包

以下 4 个包在 PyPI 元数据中未声明 `License`，上表中的许可是读取其**已安装 wheel 内
随附的 LICENSE 文件**后确认的：

| 包 | 确认结果 | 依据 |
|---|---|---|
| `sphn==0.2.1` | Apache-2.0 | `sphn-0.2.1.dist-info/licenses/LICENSE` |
| `pyannote.metrics==3.2.1` | MIT | `pyannote.metrics-3.2.1.dist-info/LICENSE`<br>Copyright (c) 2014 CNRS (Hervé BREDIN) |
| `pytorch-metric-learning==2.9.0` | MIT | `pytorch_metric_learning-2.9.0.dist-info/LICENSE`<br>Copyright (c) 2019 Kevin Musgrave |
| `primePy==1.3` | MIT | wheel 内**无** LICENSE 文件；上游仓库<br>`github.com/janaindrajit/primePy` 为 MIT，按上游许可记录 |

---

## 3. 前端依赖（frontend）

全部为构建期 / 开发期依赖，**运行时只有 `react` 与 `react-dom`（均 MIT）会被打包进产物**。

### MIT（95）

`@babel/code-frame@7.29.7`, `@babel/compat-data@7.29.7`, `@babel/core@7.29.7`,
`@babel/generator@7.29.8`, `@babel/helper-compilation-targets@7.29.7`,
`@babel/helper-globals@7.29.7`, `@babel/helper-module-imports@7.29.7`,
`@babel/helper-module-transforms@7.29.7`, `@babel/helper-plugin-utils@7.29.7`,
`@babel/helper-string-parser@7.29.7`, `@babel/helper-validator-identifier@7.29.7`,
`@babel/helper-validator-option@7.29.7`, `@babel/helpers@7.29.7`,
`@babel/parser@7.29.8`, `@babel/plugin-transform-react-jsx-self@7.29.7`,
`@babel/plugin-transform-react-jsx-source@7.29.7`, `@babel/template@7.29.7`,
`@babel/traverse@7.29.8`, `@babel/types@7.29.8`, `@esbuild/linux-x64@0.21.5`,
`@jridgewell/gen-mapping@0.3.13`, `@jridgewell/remapping@2.3.5`,
`@jridgewell/resolve-uri@3.1.2`, `@jridgewell/sourcemap-codec@1.6.0`,
`@jridgewell/trace-mapping@0.3.31`, `@napi-rs/lzma-linux-x64-gnu@1.5.1`,
`@rolldown/pluginutils@1.0.0-beta.27`, `@rollup/rollup-linux-x64-gnu@4.63.1`,
`@types/babel__core@7.20.5`, `@types/babel__generator@7.27.0`,
`@types/babel__template@7.4.4`, `@types/babel__traverse@7.28.0`, `@types/chai@5.2.3`,
`@types/deep-eql@4.0.2`, `@types/estree@1.0.9`, `@types/node@26.5.0`,
`@types/prop-types@15.7.15`, `@types/react@18.3.31`, `@types/react-dom@18.3.7`,
`@vitejs/plugin-react@4.7.0`, `@vitest/expect@3.2.7`, `@vitest/mocker@3.2.7`,
`@vitest/pretty-format@3.2.7`, `@vitest/runner@3.2.7`, `@vitest/snapshot@3.2.7`,
`@vitest/spy@3.2.7`, `@vitest/utils@3.2.7`, `assertion-error@2.0.1`,
`browserslist@4.28.9`, `cac@6.7.14`, `chai@5.3.3`, `check-error@2.1.3`,
`convert-source-map@2.0.0`, `csstype@3.2.3`, `debug@4.4.3`, `deep-eql@5.0.2`,
`es-module-lexer@1.7.0`, `esbuild@0.21.5`, `escalade@3.2.0`, `estree-walker@3.0.3`,
`fdir@6.5.0`, `gensync@1.0.0-beta.2`, `js-tokens@4.0.0` / `9.0.1`, `jsesc@3.1.0`,
`json5@2.2.3`, `loose-envify@1.4.0`, `loupe@3.2.1`, `magic-string@0.30.21`,
`ms@2.1.3`, `nanoid@3.3.18`, `node-releases@2.0.54`, `pathe@2.0.3`,
`pathval@2.0.1`, `picomatch@4.0.7`, `postcss@8.5.28`, `react@18.3.1`,
`react-dom@18.3.1`, `react-refresh@0.17.0`, `rollup@4.63.1`, `scheduler@0.23.2`,
`stackback@0.0.2`, `std-env@3.10.0`, `strip-literal@3.1.0`, `tinybench@2.9.0`,
`tinyexec@0.3.2`, `tinyglobby@0.2.17`, `tinypool@1.1.1`, `tinyrainbow@2.0.0`,
`tinyspy@4.0.6`, `undici-types@8.9.0`, `update-browserslist-db@1.3.2`,
`vite@5.4.21`, `vite-node@3.2.4`, `vitest@3.2.7`, `why-is-node-running@2.3.0`

### ISC（6）

`electron-to-chromium@1.5.425`, `lru-cache@5.1.1`, `picocolors@1.1.1`,
`semver@6.3.1`, `siginfo@2.0.0`, `yallist@3.1.1`

### Apache-2.0（3）

`baseline-browser-mapping@2.11.21`, `expect-type@1.4.0`, `typescript@5.6.3`

### BSD-3-Clause（1）

`source-map-js@1.2.1`

### CC-BY-4.0（1）— 需署名

`caniuse-lite@1.0.30001810`

Creative Commons Attribution 4.0 International 要求**署名**。该包是 `browserslist`
使用的浏览器兼容性数据集，仅在构建期被读取、不以原样再分发，其数据经 Babel /
Autoprefixer 转换后体现在构建产物的目标浏览器范围中。

署名如下：

> caniuse-lite — browser compatibility data
> Copyright (c) Alexis Deveria and the caniuse.com community
> Licensed under CC-BY-4.0. https://github.com/browserslist/caniuse-lite
> Data sourced from https://caniuse.com/

---

## 4. 二进制分发注意事项

以下条目仅在把依赖打包成可分发产物（Python wheel、容器镜像、Electron 等）时才相关。
本仓库以**纯源码**形式发布，使用者自行执行 `uv sync` / `pnpm install` 获取依赖，
相应的许可义务落在分发方一侧。

- **`av`（PyAV）**：Python 绑定为 BSD-3-Clause，但官方 wheel **捆绑 FFmpeg 库**，
  FFmpeg 为 LGPL-2.1+；若 FFmpeg 以 `--enable-gpl` 构建则为 GPL-2+。
- **`torch` / `torchaudio`**：核心为 BSD-3-Clause，但 wheel 内含 MKL、OpenMP、
  sleef、fp16 等第三方组件，各有独立许可（详见其 `LICENSE` 与 `NOTICE`）。
- **`onnxruntime`**：MIT，但捆绑的 provider 可能引入额外第三方组件。
- **`certifi`**：MPL-2.0，其根证书集合另受各 CA 的条款约束。

---

## 5. 许可证正文

短文本许可证全文内联于下；长文本许可证给出官方地址，使用者安装包时已随附副本。

### MIT License

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### BSD 3-Clause License

```
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### ISC License

```
Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
```

### 其余许可证

| 许可证 | 官方地址 |
|---|---|
| Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0 |
| Mozilla Public License 2.0 | https://www.mozilla.org/MPL/2.0/ |
| GNU Lesser GPL v3.0 or later | https://www.gnu.org/licenses/lgpl-3.0.txt |
| Creative Commons Attribution 4.0 | https://creativecommons.org/licenses/by/4.0/legalcode |
| PSF License Agreement | https://docs.python.org/3/license.html |
| matplotlib License | https://matplotlib.org/stable/users/project/license.html |

---

## 重新生成本清单

后端（列出全部已安装包及其许可声明）：

```bash
cd backend && uv sync && .venv/bin/python -c "
import importlib.metadata as md
for d in sorted(md.distributions(), key=lambda d: (d.metadata['Name'] or '').lower()):
    cls = [c.split('::')[-1].strip() for c in d.metadata.get_all('Classifier') or [] if c.startswith('License')]
    print(d.metadata['Name'], d.version, d.metadata.get('License') or '; '.join(cls) or '(UNSPECIFIED)')
"
```

标记为 `(UNSPECIFIED)` 的包需手工读取其 `*.dist-info/LICENSE` 确认，见上文第 2 节脚注。

前端：

```bash
cd frontend && pnpm install && pnpm licenses list
```
