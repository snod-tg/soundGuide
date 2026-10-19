# SoundGuide

基于官方手册的耳机问答应用，支持型号隔离的混合检索和带来源引用的回答。使用 Python、FastAPI、LangChain 与 PostgreSQL/pgvector，提供无需额外构建的网页界面。

## 功能

- **官方资料处理**：采集 Sony PDF 与网页手册，校验 SHA256，按页或章节切分，保留型号、版本、原始页码与来源链接。
- **混合检索**：向量检索结合中文 BM25，以 RRF 融合排序；支持纯向量策略。检索前按型号、语言和文档版本过滤。
- **证据问答**：通过独立模型 API 生成回答，以 JSON Schema 约束输出。后端校验引用归属，并从原切片填入引文、页码和链接。
- **澄清与拒答**：意图不清时询问设备或故障现象，证据不足时明确说明；外部服务故障与资料不足分别处理。
- **事务入库**：每份手册的切片和向量在同一事务中替换；记录 Embedding 与处理流水线指纹，拒绝混用不兼容索引。
- **轻量网页**：选择型号、输入问题、查看引用原文与官方链接；每次问题独立处理，不保存对话历史。

## 支持型号

| 类型 | 型号 |
| --- | --- |
| 头戴式 | MDR-1000X、WH-1000XM2、WH-1000XM3、WH-1000XM4、WH-1000XM5、WH-1000XM6、WH-1000XM4C、WH-1000XX |
| 真无线 | WF-1000X、WF-1000XM3、WF-1000XM4、WF-1000XM5、WF-1000XM6 |
| 颈挂式 | WI-1000X、WI-1000XM2 |

来源目录见 [`soundguide/sources.json`](soundguide/sources.json)。部分官方手册为机器翻译，页面与引用会标注。下载文件和索引不随仓库分发，需要在部署环境获取并入库。

## 架构

```mermaid
flowchart LR
    M[官方 PDF / 网页手册] --> P[校验、解析与切分]
    P --> E[独立 Embedding API]
    E --> D[(PostgreSQL / pgvector)]
    W[网页 / API 客户端] --> F[FastAPI]
    F --> R[型号过滤、向量与 BM25、RRF]
    D --> R
    R --> G[独立生成模型 API]
    G --> V[结构和引用校验]
    V --> W
```

应用仅通过 HTTP 使用模型，不负责下载、启动或重启推理服务。

## 快速开始

### 1. 安装

需要 Python 3.11+、[uv](https://docs.astral.sh/uv/)，以及已准备好的 PostgreSQL/pgvector、Embedding 和生成模型服务。

```bash
git clone https://github.com/snod-tg/soundGuide.git
cd soundGuide
uv sync --frozen
cp .env.example .env
```

编辑 `.env`，填入自己的数据库地址、模型接口与密钥。`.env` 被 Git 忽略，请勿提交真实凭据。

### 2. 配置独立服务

| 配置 | 要求 |
| --- | --- |
| PostgreSQL | 提前创建数据库、应用账号，并由管理员在该库启用 `CREATE EXTENSION IF NOT EXISTS vector;`；应用在入库时创建自己的表 |
| Embedding | 默认 `BAAI/bge-small-zh-v1.5`，输出 **512 维**；支持 `POST /v1/embeddings` 的批量字符串输入与 `encoding_format=float` |
| Embedding 健康检查 | `GET /health` 返回 `model`、`revision`、`dimensions`；必须与 `.env` 配置一致 |
| 生成模型 | OpenAI 兼容 `POST /v1/chat/completions`，支持 `response_format=json_schema` 与 `chat_template_kwargs.enable_thinking=false`；已按 Qwen3.5 / vLLM 接口接入 |

Embedding 健康检查格式示例：

```json
{"model":"BAAI/bge-small-zh-v1.5","revision":"your-weight-sha256","dimensions":512}
```

主要环境变量：

```dotenv
SG_DATABASE_URL=postgresql://soundguide:change-me@127.0.0.1:55433/soundguide
SG_EMBEDDING_BASE_URL=http://127.0.0.1:8091/v1
SG_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
SG_EMBEDDING_API_KEY=your-embedding-key
SG_EMBEDDING_REVISION=your-weight-sha256
SG_EMBEDDING_DIMENSIONS=512
SG_GENERATION_BASE_URL=http://127.0.0.1:8080/v1
SG_GENERATION_MODEL=your-served-model-id
SG_GENERATION_API_KEY=your-model-key
SG_GENERATION_MAX_TOKENS=1536
```

地址与模型 ID 按自己的服务填写；无需认证的本地生成服务仍需设置一个非空占位 API Key。当前数据库结构固定为512维，切换其他维度需要同步修改结构并重新入库。Embedding 模型、权重或处理流水线改变后也需要重新入库。

### 3. 下载资料并建立索引

```bash
uv run python -m soundguide.download
uv run python -m soundguide.ingest
```

下载器只接受允许的官方 HTTPS 来源，并校验目录中固定的文件指纹。官网资料发生变化时会拒绝静默替换，应先审核新文档并更新来源目录。原文件、下载记录和预处理切片保存在被忽略的 `.data/` 中。

### 4. 启动网页与 API

```bash
uv run uvicorn soundguide.server:APP --host 127.0.0.1 --port 8001
```

- 网页：<http://127.0.0.1:8001/>
- API 文档：<http://127.0.0.1:8001/docs>
- 存活检查：`GET /health`，只表示应用进程正常
- 来源与实际索引状态：`GET /api/sources`

服务启动后可运行 `uv run python scripts/doctor.py` 检查依赖连接与索引状态；该脚本只检查，不启动外部服务。

## API 示例

问答：

```bash
curl --fail http://127.0.0.1:8001/api/answer \
  -H 'Content-Type: application/json' \
  -d '{"model":"WH-1000XM5","question":"自动NC优化需要按按钮吗？","strategy":"hybrid"}'
```

返回的 `status` 为 `answered`、`insufficient_evidence` 或 `clarification_needed`。有答案时包含 `answer` 与 `citations`，引用携带原文、型号、页码或章节、版本及官方链接。非法或截断模型输出返回502，外部服务不可用返回503。

仅检索：

```bash
curl --fail http://127.0.0.1:8001/api/search \
  -H 'Content-Type: application/json' \
  -d '{"model":"WH-1000XM5","query":"检查电池剩余电量","strategy":"hybrid","top_k":5}'
```

`strategy` 支持 `hybrid` 和 `vector`，型号为必填项。当前资料语言为 `zh-CN`，文档版本为 `v1`；`v1` 是指南版本，不代表耳机固件版本。

## 项目结构

```text
soundguide/       应用、资料处理、存储、检索与生成
  web/            网页静态资源
  sources.json    官方来源目录
langconnect/      保留的上游模块；使用其文档处理组件
scripts/doctor.py 外部依赖诊断
scripts/verify_series.py 资料索引一致性检查
tests/            应用单元测试与独立数据库集成测试
.env.example      配置模板
pyproject.toml    依赖与开发工具配置
uv.lock           固定依赖版本
```

## 开发检查

```bash
# 离线测试，不调用真实模型或数据库
uv run pytest tests/soundguide --disable-socket --allow-unix-socket

# 检查应用和公开脚本
uv run ruff check soundguide scripts/doctor.py scripts/verify_series.py tests/soundguide tests/integration
```

可选数据库集成测试需单独创建本机 `soundguide_test` 数据库并启用 pgvector，在 `.env` 配置 `SG_TEST_DATABASE_URL` 后运行：

```bash
SG_RUN_POSTGRES_TESTS=1 uv run pytest tests/integration
```

## 使用边界

这是单轮资料问答应用。当前不包含 OCR、图片或复杂表格理解、神经重排、多轮会话、用户认证与公网部署配置。小规模语料的 BM25 在内存计算，向量检索使用精确余弦排序。

真实引用与结构合法不保证回答语义正确。模型仍可能遗漏条件、误解操作或产生无依据细节，使用前应核对引用原文。项目与 Sony 无隶属关系，官方手册版权归其权利人所有。

## 来源与许可证

项目起始于 [LangConnect](https://github.com/langchain-ai/langconnect)，基于提交 `653db4db537e38a94afa0b03906091c2c655abb4`，保留上游模块及 MIT 许可证。SoundGuide 新增领域资料目录、来源校验、按页/章节处理、独立存储、HTTP Embedding、混合检索、证据问答和网页；未将上游集合 API 挂载到当前服务，运行 SoundGuide 无需 Supabase 服务。

代码使用 [MIT License](LICENSE)。
