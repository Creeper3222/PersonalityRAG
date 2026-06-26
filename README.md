# PersonalityRAG v0.1.0

PersonalityRAG 是一个脱离具体 Bot 框架独立运行的人格记忆 RAG 服务。首版以 AstrBot LivingMemory v2.3.5 为兼容基线，保留其主记忆、图谱、记忆原子、会话历史、BM25/FAISS 双路检索和 RRF 融合行为。

项目采用 **AGPL-3.0-only**。来源与版权信息见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，完整许可证见 [LICENSE](LICENSE)。

## 当前能力

- 独立 Windows Shell 版本，双击 `launcher.bat` 启动。
- FastAPI + Uvicorn 后端和本地 WebUI。
- 多记忆库管理，每个记忆库独立保存 LivingMemory 兼容数据库、会话库、FTS、FAISS 索引、备份和导入归档。
- 全局 Embedding Provider 池，支持 OpenAI Embedding、Ollama Embedding、vLLM Embedding。
- Provider revision 与 FAISS generation 绑定；切换模型需要重建索引，验证成功后原子切换。
- 记忆管理、知识图谱、召回测试、系统概览、日志与任务列表、基础设置。
- `livingmemory.db` 导入、全量索引重建、记忆库复制、核心数据库备份和删除回收。
- Bearer API Key 与 WebUI 登录密码。未设置登录密码时，WebUI 使用 API Key 登录；设置密码后，WebUI 使用密码登录，API Key 仍可用于脚本访问。

首版暂不包含 LLM 自动总结、Agent 工具、AstrBot/MaiBot/EchoBot 接入插件，也不持续采集聊天记录。

## 启动

双击：

```text
launcher.bat
```

启动器会自动完成：

1. 创建 `.venv`。
2. 安装或补装 `requirements.txt` 中缺失的依赖。
3. 生成 `config/config.json`、随机 API Key 和会话密钥。
4. 启动 WebUI，默认地址为 `http://127.0.0.1:8765/`。

如果配置端口被占用，服务会向后寻找可用端口，并在终端和日志中提示实际访问地址。

## 模型提供商

WebUI 的“模型提供商”页面可以新增、编辑、复制、测试、启用和删除 Embedding Provider。

| 类型 | 默认接口 | 说明 |
|---|---|---|
| OpenAI Embedding | `/v1/models`、`/v1/embeddings` | 支持 OpenAI 官方和兼容接口；维度大于 0 时发送 `dimensions`。 |
| Ollama Embedding | `/api/tags`、`/api/embed` | 支持单条和批量嵌入。 |
| vLLM Embedding | `/v1/models`、`/v1/embeddings` | 自动匹配 served-model-name；不会向 vLLM 发送 `dimensions`。 |

默认示例 Provider 为：

```json
{
  "id": "vllm_embedding",
  "display_name": "本机 bge-m3",
  "api_base": "http://127.0.0.1:8001/v1",
  "model": "BAAI/bge-m3",
  "dimensions": 1024
}
```

Provider 的 API Key 不会在查询接口中回传明文。编辑时留空表示保持原值，也可以显式清除。

## 记忆库

WebUI 的“记忆库”页面用于创建、进入、编辑、复制、备份、删除记忆库，并选择当前操作库。

每个记忆库目录结构类似：

```text
data/libraries/<library_id>/
├─ livingmemory.db
├─ conversations.db
├─ indexes/
├─ backups/
├─ imports/
├─ reports/
├─ stopwords/
└─ decay_state.json
```

`livingmemory.db` 保存核心记忆、图谱和原子；`conversations.db` 保存会话和消息历史。FAISS 索引是派生数据，可以随时通过重建恢复。

“立即备份”和“删除记忆库”的回收目录都会保留：

- `livingmemory.db`
- `conversations.db`

记忆库复制是全量复制，会复制数据库、索引、导入归档和报告等完整目录。

## 导入 LivingMemory

空记忆库可以通过“导入记忆”导入 LivingMemory 的 `livingmemory.db`。导入会先校验 SQLite integrity 和核心表结构，校验通过后替换目标空库的核心数据库，并自动重建 FTS 和 FAISS 索引。

当前单文件导入只接收 `livingmemory.db`。如果需要完整保留会话统计，需要同时迁移或备份源目录里的 `conversations.db`。

## 召回行为

PersonalityRAG 复刻 LivingMemory v2.3.5 的核心召回链路：

- 文档路：jieba 分词、SQLite FTS5 BM25、FAISS 向量检索。
- 图谱路：图关键词、邻居扩展、图向量检索。
- RRF 融合：默认 `rrf_k=60`。
- 文档评分：相关性 `0.5`、重要性 `0.25`、新鲜度 `0.25`。
- MMR 去重：默认 `lambda=0.7`。
- 双路融合：文档 `0.65`、图谱 `0.35`，双路命中加 `0.08`。
- 支持 persona/session 过滤，默认启用 persona 过滤、关闭 session 过滤。

## REST API

除健康检查、登录状态和静态资源外，接口默认需要：

```http
Authorization: Bearer <API_KEY>
```

常用接口：

```text
GET/POST   /api/v1/libraries
GET/PATCH  /api/v1/libraries/{library_id}
DELETE     /api/v1/libraries/{library_id}
POST       /api/v1/libraries/{library_id}/backup
POST       /api/v1/libraries/{library_id}/copy
POST       /api/v1/libraries/{library_id}/imports/livingmemory-db

GET/POST   /api/v1/providers
PATCH      /api/v1/providers/{provider_id}
DELETE     /api/v1/providers/{provider_id}
POST       /api/v1/providers/{provider_id}/copy
POST       /api/v1/providers/{provider_id}/test

GET/POST   /api/v1/libraries/{library_id}/memories
POST       /api/v1/libraries/{library_id}/recall
GET        /api/v1/libraries/{library_id}/graph/overview
POST       /api/v1/libraries/{library_id}/graph/query
POST       /api/v1/libraries/{library_id}/indexes/rebuild
GET        /api/v1/jobs
GET        /api/v1/jobs/{job_id}
```

旧的 `/api/v1/memories`、`/api/v1/recall`、`/api/v1/graph/...` 会代理到默认记忆库，便于早期脚本兼容。

## 发布版内容

公开仓库只包含源码、静态资源、测试、工具脚本、配置示例和文档。不会包含：

- `.venv/`
- `data/`
- `config/config.json`
- 日志、缓存、浏览器测试 profile
- 真实记忆库、FAISS 索引、导入上传文件、备份和 trash

运行后产生的数据都保存在本地 `data/` 和 `config/config.json` 中，请自行备份。

## 开发检查

```powershell
.\.venv\Scripts\python.exe -m compileall personalityrag
node --check static/app.js
.\.venv\Scripts\python.exe -m pytest -q
```

WebUI 验收脚本位于 `tools/acceptance_webui.py`，开发依赖见 `requirements-dev.txt`。
