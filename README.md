# PersonalityRAG v0.1.2

PersonalityRAG 是一个脱离具体 Bot 框架独立运行的类型化 RAG 数据库服务。数据库分为“记忆库”和“知识库”两个大类；当前提供兼容 AstrBot LivingMemory 数据结构的 `LivingMemory v8` 记忆库，以及以文本检索为核心、可关联图片附件的 `text_media_v1` 知识库。

项目采用 **AGPL-3.0-only**。PersonalityRAG 的存储、召回、图记忆、迁移和管理界面设计受到 [lxfight-s-Astrbot-Plugins/astrbot_plugin_livingmemory](https://github.com/lxfight-s-Astrbot-Plugins/astrbot_plugin_livingmemory) 启发，并以其 LivingMemory 数据库结构作为兼容目标。来源与版权信息见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，完整许可证见 [LICENSE](LICENSE)。

AstrBot 接入请按数据库类别使用配套插件：[记忆库适配器](https://github.com/Creeper3222/astrbot_plugin_personality_rag_adapter) 或 [知识库适配器](https://github.com/Creeper3222/astrbot_plugin_personality_knowledgebase_adapter)。

## 当前能力

- 独立 Windows Shell 版本，双击 `launcher.bat` 启动。
- FastAPI + Uvicorn 后端和本地 WebUI。
- 类型化数据库管理，资源身份为 `(database_type, id)`；同类型 ID 唯一，不同类型可使用相同 ID。
- 记忆库与知识库分类入口；当前提供 `livingmemory_v8` 记忆库驱动和 `text_media_v1` 知识库驱动。
- 全局 Embedding Provider 池，支持 OpenAI-compatible、Gemini、NVIDIA NIM、Ollama、vLLM。
- 全局 Rerank Provider 池，支持 vLLM/OpenAI-compatible Rerank、Xinference Rerank、阿里云百炼 qwen3-rerank/DashScope 分支、NVIDIA NIM Rerank。
- Provider revision 与 FAISS generation 绑定；切换 embedding 模型会提示并重建索引，重建验证成功后再原子切换。
- 最大上下文长度自动检测和手动设置；切换到更低上下文模型时会提示潜在截断风险，但不会修改或截断源文本。
- 记忆管理、知识图谱、召回测试、系统概览、文件管理、日志与任务列表、基础设置。
- WebUI 与适配器接入面使用显式隔离监听器；支持独立 HTTPS 公网适配器 URL
  展示，适合单节点云服务器经反向代理接入。
- `livingmemory.db` 单文件导入、全量索引重建、记忆库复制、核心数据库备份和删除回收。
- Bearer API Key、WebUI 登录密码和库级 PSK 接入密钥。

本体不内置 Bot 接入插件，也不会主动持续采集聊天记录；AstrBot 对话捕获、召回注入、LLM 总结和 Agent 工具由配套适配器插件完成。

## 标识符命名层次

运行时、API 响应、前端状态和新日志统一使用三层标识符：

- 跨类型通用字段：`database_id`、`database_type`。
- 记忆库专属字段：`memory_store_id`、`memory_store_type`。
- 知识库专属字段：`knowledge_base_id`、`knowledge_base_type`。

类型专属响应同时提供专属字段和通用字段。`library_id`、`library_type`、`memory_library_type`、`knowledge_library_type` 仅作为 v0.1.1 旧客户端的 deprecated 兼容输入或输出，不得用于新配置、业务状态或日志。读取兼容数据时按“专属字段 → 通用字段 → 旧兼容字段”回退。

旧 API 别名至少保留至整个 v0.1.x 系列结束；只有明确的下一代协议升级并提供迁移说明后才允许移除。冻结 SQL 列和已发布备份清单不受该期限影响，仍按格式兼容要求长期保留。

HTTP namespace 保持 `/memory-libraries/...` 与 `/knowledge-libraries/...` 不变；旧 SQL 列、任务兼容字段及 `.tmkb/.tmkbs/.prag` 已发布清单字段也保持原样。兼容双写集中在边界层，不能把旧名称重新扩散到活跃业务代码。busy 响应继续保留兼容错误码 `library_busy`，并提供规范条件字段 `condition: "database_busy"`。

日志只输出与目标类型一致的标识符，且不得记录 PSK/PKB、Authorization、查询正文、命中文本、完整签名 URL、Base64 或媒体内容。

Adapter 运行请求采用“共享元数据、类型级回显”结构：通用层只生成请求 ID并提取脱敏后的 Adapter ID、实例 ID与类型；每个数据库类型在自己的实现目录内记录其公开请求参数和结果摘要。`livingmemory_v8` 记录记忆召回数、Rerank 输出数、基线与过滤开关，`text_media_v1` 记录检索模式、媒体响应模式、Top-K、媒体阈值、Rerank 及文本/媒体结果数量。后续新增数据库类型必须实现自己的请求日志，不得把类型参数堆回通用 HTTP 中间件。

## 启动

运行环境支持 **CPython 3.10+ x64**。CPython 3.12 是项目开发与主要回归测试版本，不是启动硬门槛。启动器会校验已有 `.venv`；如果发现低于 3.10、32 位或非 CPython 环境，会明确拒绝启动，且不会删除或覆盖现有环境。

双击：

```text
launcher.bat
```

启动器会自动完成：

1. 在不存在 `.venv` 时优先使用 Python 3.12 创建虚拟环境，并在不可用时回退到其它兼容的 CPython 3.10+ x64。
2. 按 `requirements-runtime.lock` 安装与当前 Python 版本匹配的运行依赖。
3. 根据 Python 版本、`requirements.txt` 和锁文件生成依赖指纹；三者未变化时后续启动会跳过 `pip install`。
4. 生成 `config/config.json`、随机 API Key、会话密钥和库级 PSK 派生密钥。
5. 启动 WebUI，默认地址为 `http://127.0.0.1:8765/`。

如果配置端口被占用，服务会向后寻找可用端口，并在终端和日志中提示实际访问地址。基础设置页可以调整 WebUI 端口、接入端口和 URL 基址。

需要让远端适配器通过 HTTPS 使用记忆库或知识库时，请按
[Windows 远端适配器接入指南](docs/REMOTE_ADAPTER_DEPLOYMENT.md) 部署。公网
只开放反向代理的 `443`，核心继续绑定回环地址；WebUI 不应公开代理。

## 访问面与适配器地址

PersonalityRAG 使用两个显式监听面，不能互换：

- `http://127.0.0.1:8765`：本机 WebUI、管理 API 和静态资源，只用于管理员操作。
- `http://127.0.0.1:8766`：本机适配器接入地址，只开放探活和已登记的记忆库/知识库运行接口。
- `https://rag.example.com:443`：可选的公网适配器地址，由 Caddy/Nginx 等反向代理到本机 8766；公网不应直接开放 8765 或 8766。

适配器地址必须填写完整根级 origin，即“协议://主机或 IP:端口”。同机 Windows 使用 `http://127.0.0.1:8766`，同机 Docker 使用 `http://host.docker.internal:8766`，远端使用带端口的 HTTPS 地址。不要填写 WebUI 地址、业务路径、凭据、查询参数或片段。

接入监听器返回 `X-PersonalityRAG-Surface: adapter-access`；WebUI 监听器返回 `webui`。公网适配器会严格校验该标识，避免反向代理端口或 Host 头造成权限边界误判。

## 模型提供商

WebUI 的“模型提供商”页面可以新增、编辑、复制、测试、启用和删除 Provider，并在 Embedding 与 Rerank 子页间切换。

| 类型 | 默认接口 | 说明 |
|---|---|---|
| OpenAI Embedding | `/v1/models`, `/v1/embeddings` | 支持 OpenAI 官方和兼容接口；维度大于 0 时发送 `dimensions`。 |
| Gemini Embedding | `/v1beta/models`, `batchEmbedContents` | 对齐 Gemini Embedding 批量接口；维度大于 0 时发送 `outputDimensionality`。 |
| NVIDIA Embedding | `/v1/models`, `/v1/embeddings` | 对齐 NVIDIA NIM 接口；发送 `input_type` 和 float 编码。 |
| Ollama Embedding | `/api/tags`, `/api/embed` | 支持单条和批量嵌入。 |
| vLLM Embedding | `/v1/models`, `/v1/embeddings` | 自动匹配 `served-model-name`；不会向 vLLM 发送 `dimensions`。 |
| vLLM Rerank | 自定义 `api_suffix`，默认 `/v1/rerank` | 发送 `query`、`documents`、`model` 和 `top_n`。 |
| Xinference Rerank | 默认 `/v1/rerank` | 使用 Xinference Rerank REST 接口。 |
| 阿里云百炼 Rerank | DashScope/百炼 rerank endpoint | 支持 `qwen3-rerank` 与旧 DashScope payload 分支。 |
| NVIDIA Rerank | NIM reranking endpoint | 使用 `query/passages/rankings` 格式。 |

默认示例 Provider：

```json
{
  "id": "vllm_embedding",
  "display_name": "本机 bge-m3",
  "api_base": "http://127.0.0.1:8001/v1",
  "model": "BAAI/bge-m3",
  "dimensions": 1024
}
```

Provider 的 API Key 不会在查询接口中回传明文。编辑时留空表示保持原值，也可以显式清除。Rerank Provider 不参与索引构建，只在召回测试或 API 召回时对候选结果重排。

## 数据库类型

WebUI 的“数据库”页面包含平级的“记忆库”和“知识库”分类。记忆库由 bot 根据消息记录总结和写入，通常不由用户直接维护；知识库由用户上传文档或数据，经分片和向量化后用于常规 RAG，通常不由 bot 写入，也不依赖消息记录。

新增数据库时先选择类型。当前可选择记忆库分类下的 `LivingMemory v8`，或知识库分类下的“文本媒体知识库 v1”；类型注册表负责声明图标、能力、数据目录、运行时和密钥派生，未注册类型不会回退到其它类型链路。

## LivingMemory v8 记忆库

`LivingMemory v8` 沿用 v0.1.0 的目录结构，保证更新到 v0.1.1 后仍可回退读取现有数据：

每个记忆库目录结构类似：

```text
data/databases/memory_stores/livingmemory_v8/<memory_store_id>/
├── livingmemory.db
├── conversations.db
├── indexes/
├── backups/
├── imports/
├── reports/
├── stopwords/
└── decay_state.json
```

`livingmemory.db` 保存核心记忆、图谱和原子；`conversations.db` 保存会话和消息历史。FAISS 索引和 FTS 表是派生数据，可以随时通过重建恢复。

通用控制库同时记录复合身份 `(livingmemory_v8, <memory_store_id>)`。运行时物理目录统一位于 `data/databases/memory_stores/livingmemory_v8/<memory_store_id>/`；旧 `data/libraries/<library_id>/` 仅作为迁移输入，启动后会迁入新布局，稳态不再保留旧路径。

数据库格式兼容边界由 `database_type` 表达，不再为每个库维护独立的数据库版本字段。后续 LivingMemory 数据结构升级会注册为新的记忆库类型，并通过类型驱动实现迁移与兼容逻辑。

`livingmemory_v8` 不持久化用户可选的记忆类型字段，WebUI 与 API 将记录统一视为通用记忆。启动时会从历史 `documents.metadata` 中移除 PersonalityRAG 曾写入的 `memory_type`，确保 `livingmemory.db` 可直接回用于 LivingMemory v8；更丰富的分类字段由未来明确拥有该结构的新记忆库类型实现。

索引状态分为三类：

- 绿色“健康”：已有索引，且当前 embedding provider/revision/model 与索引 manifest 一致。
- 橙色“待构建”：还没有可用索引 generation。
- 红色“索引冲突”：已有索引，但 provider 绑定或索引计数已不匹配，需要重建。

“立即备份”和“删除记忆库”的回收目录都会保留：

- `livingmemory.db`
- `conversations.db`

记忆库复制是全量复制，会复制数据库、索引、导入归档和报告等完整目录。

## 导入 LivingMemory

空记忆库可以通过“导入记忆”导入 LivingMemory 的 `livingmemory.db`，并可同时选择配套的 `conversations.db`。导入会校验两个 SQLite 文件的 integrity、核心表结构、会话/消息计数、待总结范围和规范化哈希；通过后原子替换目标空库的数据，并自动重建 FTS、文档向量和图向量索引。

只提供 `livingmemory.db` 也可以完整派生长期记忆索引；`conversations.db` 不参与长期记忆向量计算，只负责保留短期会话、消息顺序、总结进度和待总结消息。需要让记忆适配器从原进度继续总结时，应成对导入两个文件。

## 文本媒体知识库 v1

`text_media_v1` 用于由用户维护的文本和图片资料。首版支持 UTF-8/UTF-8-SIG TXT、Markdown、静态 PNG/JPEG/WebP；不包含 OCR、图片向量或 PDF/DOCX。知识库 Adapter 只允许用库级 `pkb-` 密钥检索及读取命中媒体，不能修改内容、设置、Provider 绑定或索引。

运行目录固定为：

```text
data/databases/knowledge_bases/text_media_v1/<knowledge_base_id>/
├── textmediaknowledge.db
├── visual_intent_policy.csv              # 可选；缺失时继承类型默认词表
├── assets/documents/sha256/<prefix>/<hash>.<ext>
├── assets/images/sha256/<prefix>/<hash>.webp
├── derived/indexes/<generation>/chunks.faiss
└── derived/previews/<hash>.webp
```

`textmediaknowledge.db` 保存文档、知识条目、确定性分块、little-endian float32 向量、图片资源元数据，以及 document/entry/chunk 三层多对多附件关系。原始文档、规范图片和 SQLite 中的向量是可重建事实；FAISS、FTS5 与预览图都是派生缓存。

文本按段落优先进行确定性分块，默认目标 1200 字符、重叠 150 字符。检索使用归一化向量 FAISS IP 与 FTS5 BM25，再以 `k=60` 的 RRF 融合并按知识条目稳定聚合。命中文本后按 `chunk > entry > document` 合并图片关系，去重后返回鉴权媒体 URL，不在 JSON 中传输 Base64。

`POST .../{knowledge_base_id}/search` 默认使用兼容的 `media_response_mode=full`。知识库 Adapter 的智能媒体工具可改用内部协议值 `descriptions_only`：文本结果保持不变，媒体只返回已通过当前阈值并进入输出上限的 `media_candidates` 简述、分数、原因和可获取标记，同时将 `media_outputs/media_decisions` 置空且不生成内容或缩略图 URL。随后只有 LLM 在当前轮次明确选择的资源才会通过既有单资源读取或签名接口获取；该模式不增加任何写权限。

图片在浏览器端先预压缩，服务端仍会用 Pillow 重新解码、校验、修正 EXIF 方向、转 sRGB、清除元数据并输出静态 WebP。服务端不放大图片，最长边为 1536，规范图片硬上限为 1.5 MiB；原图不会保存进知识库。

知识库 Adapter 可在一次有效连接租约内为命中图片申请 300 秒短时签名 URL。签名不可篡改地绑定知识库类型、库 ID、资源 ID、图片变体、Adapter ID、实例 ID 与当前连接代次；每次匿名取图都会重新确认租约和连接代次。管理员强制下线后 URL 立即失效，手动重连产生新代次后旧 URL 也不会重新有效。连接级图片响应使用 `Cache-Control: private, no-store`，URL 可在有效期内重复读取，以兼容 AstrBot 视觉输入与主动生图工具；管理员签发的原有通用 URL 保持兼容。

PersonalityRAG 中的规范图片是唯一持久副本。本体不会为签名 URL 复制或生成临时图片文件，URL 到期仅撤销访问能力，不对应磁盘清理任务；Adapter 应在内存中完成 Data URI 或聊天回显处理，不建设媒体缓存目录。

视觉意图策略的四类可编辑词表使用 UTF-8 BOM CSV。类型级只读默认值位于 `text_media_v1` 自身资源 `visual_intent_policy.default.csv`；单库只有在自定义时才生成 `visual_intent_policy.csv`，恢复类型默认值会删除该覆盖文件。CSV 固定使用 `visual_object_terms`、`lookup_action_terms`、`generation_action_terms`、`reference_connector_terms` 四个英文列名；系统保护阻断词仍是类型级只读代码资源，不参与词表导入导出。

单个知识库使用明文 ZIP64 `.tmkb` 导入导出，包内包含 manifest、`textmediaknowledge.db`、原始文档、规范图片，以及存在时的 `settings/visual_intent_policy.csv`；所有成员都进入 SHA-256 清单。批量 `.tmkbs` 只包含 manifest 和多个可独立导入的 `libraries/<database_id>.tmkb`，已经压缩的子包以 `ZIP_STORED` 写入，避免重复压缩。

全环境 `.prag v2` 不会先把文本媒体库合成 `.tmkbs`，而是直接保存多个 `databases/text_media_v1/<id>.tmkb`。这些嵌套 `.tmkb` 同样使用 `ZIP_STORED`，JSON 等普通外层成员继续压缩；外层 `.prag` 统一负责 AES 加密。独立 `.tmkb` 与 `.tmkbs` **未加密**，格式版本均保持不变，`.prag v2` 仍可读取 v1 包。

## AstrBot 记忆库适配器用法

1. 在 WebUI 创建并测试 Embedding Provider；需要重排时再创建 Rerank Provider。
2. 新建 `LivingMemory v8` 记忆库并绑定 Provider。迁移现有 LivingMemory 时，在空库中导入 `livingmemory.db`，需要续接待总结消息时同时导入 `conversations.db`。
3. 等待索引任务完成；旧图索引或 Provider 发生变化时显式执行“重建索引”和“图谱重建”。
4. 在库卡片钥匙入口生成该库的 `psk-` 密钥，不要把 WebUI 全局 API Key 配给适配器。
5. 在 AstrBot 安装 [记忆库适配器](https://github.com/Creeper3222/astrbot_plugin_personality_rag_adapter)，填写 8766/HTTPS 接入 origin、`livingmemory_v8`、记忆库 ID 和匹配的 PSK。
6. 停用原 LivingMemory 插件以避免重复召回、重复捕获和重复写入；再用适配器 Dashboard 执行连接与召回测试。

适配器负责 AstrBot 事件身份、消息捕获、提示词、LLM 总结和上下文注入；PersonalityRAG 负责 `livingmemory.db`、`conversations.db`、索引、图谱、召回和维护任务。常用管理命令为 `/prag status`、`/prag search`、`/prag summarize`、`/prag webui`；可选 LLM 工具包括记忆召回、主动写入和主动总结。

## AstrBot 知识库适配器用法

1. 新建 `text_media_v1` 知识库，绑定 Embedding Provider，并按需绑定 Rerank Provider。
2. 在知识库管理页上传 TXT/Markdown 文档、知识条目和 PNG/JPEG/WebP 图片，建立图片与文档、条目或分块的显式关系。
3. 等待入库/索引任务完成，在本体检索测试页确认纯文本、图文和媒体置信度结果。
4. 在库卡片钥匙入口生成该库的 `pkb-` 密钥；每个知识库连接使用自己匹配的 ID 和 PKB。
5. 在 AstrBot 安装 [知识库适配器](https://github.com/Creeper3222/astrbot_plugin_personality_knowledgebase_adapter)，可配置多个独立连接，并在 Dashboard 分库测试。

默认自动检索只使用用户源消息。`Agentic 知识库检索` 允许 LLM 自主调用 `personalityrag_knowledge_retrieval`；`personalityrag_media_retrieval` 用于媒体候选复检。两者独立开关，同时启用时 Agentic 工具是父工具，媒体工具只能继续复检其候选，不能反向触发知识检索。知识适配器始终只读，不能修改文档、关系、Provider 或索引。

## 召回行为

PersonalityRAG 的 `livingmemory_v8` 类型复刻 LivingMemory v8 对应的核心召回链路：

- 文档路：jieba 分词、SQLite FTS5 BM25、FAISS 向量检索。
- 图谱路：图关键词、邻居扩展、图向量检索。
- RRF 融合：默认 `rrf_k=60`。
- 文档评分：相关性 `0.5`、重要性 `0.25`、新鲜度 `0.25`。
- MMR 去重：默认 `lambda=0.7`。
- 双路融合：文档 `0.65`、图谱 `0.35`，双路命中加 `0.08`。
- 支持 persona/session 过滤，默认启用 persona 过滤、关闭 session 过滤。

如果当前记忆库绑定了 Rerank Provider，召回 API 和 WebUI 召回测试页可以返回重排结果；Rerank 失败时会回退到未重排的 embedding 召回结果。

## 文件管理

WebUI 的“文件管理”页面复用 RocketCatShell 已验证的文件管理交互，用于为后续 Linux / Docker 部署提供不依赖宿主机桌面的文件浏览入口。页面支持目录浏览、UTF-8 文本查看和编辑、图片预览、新建、上传、重命名、移动、删除以及单项或批量下载。

- 文件管理只接受 PersonalityRAG 程序目录或运行数据目录内的相对路径；绝对路径、盘符、`..`、UNC 路径和符号链接越界目标都会被拒绝。
- 当程序目录和运行数据目录分离时，页面提供两个根入口；路径相同或互相包含时自动合并，不能借另一入口绕过保护。
- `.git`、虚拟环境、Python/测试缓存和 `node_modules` 不会显示，直接请求同样被拒绝。
- 本体源码、WebUI 静态资源、启动与依赖文件，以及 `config/`、`data/` 中的配置、数据库、索引和日志只允许浏览与下载，不能通过文件管理修改、移动或删除。
- 读取或下载运行数据需要再次输入 WebUI 登录/文件管理验证密码；密码只在请求体中传递，不会写入 URL 或日志。
- 文本预览和在线编辑上限为 `1 MiB`；单次最多上传 20 个文件，单文件上限为 `100 MiB`。大目录和批量下载使用临时 ZIP 流式返回，不会整体载入进程内存。
- 文件列表为目录、图片、TXT、JSON/Python/Markdown、PDF、Word 和其它文件显示不同图标；全部配色跟随 PersonalityRAG 明暗主题。

## REST API

除健康检查、登录状态和静态资源外，接口默认需要：

```http
Authorization: Bearer <API_KEY>
```

常用接口：

```text
GET        /api/v1/database-types?category=memory|knowledge
GET/POST   /api/v1/databases
POST       /api/v1/memory-libraries/livingmemory_v8
GET/PATCH  /api/v1/memory-libraries/livingmemory_v8/{memory_store_id}
DELETE     /api/v1/memory-libraries/livingmemory_v8/{memory_store_id}
POST       /api/v1/memory-libraries/livingmemory_v8/{memory_store_id}/access-key
POST       /api/v1/memory-libraries/livingmemory_v8/{memory_store_id}/backup
POST       /api/v1/memory-libraries/livingmemory_v8/{memory_store_id}/copy
POST       /api/v1/memory-libraries/livingmemory_v8/{memory_store_id}/imports/livingmemory-db

POST       /api/v1/knowledge-libraries/text_media_v1
GET/PATCH  /api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}
GET/POST   /api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/documents
GET/POST   /api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/entries
GET        /api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/assets
POST       /api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/assets/images
POST       /api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/assets/{asset_id}/signed-url
PUT        /api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/relations/{scope}/{target_id}/assets/{asset_id}
DELETE     /api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/relations/{scope}/{target_id}/assets/{asset_id}
POST       /api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/search
POST       /api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/adapters/heartbeat
POST       /api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/adapters/{adapter_id}/disconnect
POST       /api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/exports
POST       /api/v1/knowledge-libraries/text_media_v1/imports/inspect

GET/POST   /api/v1/providers
PATCH      /api/v1/providers/{provider_id}
DELETE     /api/v1/providers/{provider_id}
POST       /api/v1/providers/{provider_id}/copy
POST       /api/v1/providers/{provider_id}/test
POST       /api/v1/providers/detect-dimension
POST       /api/v1/providers/detect-context-length

GET/POST   /api/v1/memory-libraries/livingmemory_v8/{memory_store_id}/memories
POST       /api/v1/memory-libraries/livingmemory_v8/{memory_store_id}/recall
GET        /api/v1/memory-libraries/livingmemory_v8/{memory_store_id}/graph/overview
POST       /api/v1/memory-libraries/livingmemory_v8/{memory_store_id}/graph/query
POST       /api/v1/memory-libraries/livingmemory_v8/{memory_store_id}/indexes/rebuild
GET        /api/v1/jobs
GET        /api/v1/jobs/{job_id}
POST       /api/v1/jobs/{job_id}/pause
POST       /api/v1/jobs/{job_id}/resume
GET        /api/v1/updates/status
GET        /api/v1/updates/releases
POST       /api/v1/updates/switch
GET        /api/v1/updates/transactions/{transaction_id}
POST       /api/v1/jobs/{job_id}/stop
POST       /api/v1/jobs/{job_id}/cancel

GET        /api/v1/files
POST       /api/v1/files/read
POST       /api/v1/files/write
POST       /api/v1/files/create
POST       /api/v1/files/upload
POST       /api/v1/files/rename
POST       /api/v1/files/move
POST       /api/v1/files/delete
GET/POST   /api/v1/files/download
```

索引重建、图重建和 LivingMemory 导入/迁移任务支持安全断点暂停与继续。运行中的任务可停止并回滚到执行前状态，尚未开始的排队任务可取消；暂停或可恢复中断的任务会阻塞后续队列，直到继续、完成或停止。

v0.1.1 contract 已移除 `/api/v1/libraries/...` 和 `/api/v1/databases/{database_type}/{database_id}/...` 通用业务路径。记忆库与知识库 Adapter 都必须选择已实现的类型和独立 namespace；LivingMemory v8 允许召回、对话与总结记忆写入，`text_media_v1` Adapter 只允许搜索、读取命中媒体和连接控制。

## 数据库接入密钥

每个数据库类型拥有独立密钥派生实现。记忆库密钥统一使用 `psk-` 前缀，知识库密钥统一使用 `pkb-` 前缀；除 LivingMemory v8 的兼容算法外，新类型必须把 `database_type` 纳入独立派生域，避免不同类型的相同 ID 产生相同密钥。

LivingMemory v8 永久保留 v0.1.0 算法：以 `library_psk_secret` 为 HMAC-SHA256 密钥、以库 ID 为消息，URL-safe Base64 去除尾部 `=` 后添加 `psk-`。因此现有库升级前后的密钥逐字节一致。

Bot 适配器推荐使用与数据库类型匹配的库级密钥：

```http
Authorization: Bearer psk-...
Authorization: Bearer pkb-...
```

一个记忆库可连接多个适配器，但一个记忆适配器实例只绑定一个记忆库；知识库连接为多对多关系，一个知识库适配器实例可以分别连接并独立检索多个知识库。两类 Adapter 都通过各自类型路径下的长轮询 heartbeat 接收强制下线通知；被强制下线后，只有显式 `manual_reconnect` 心跳才能恢复该数据库连接，其他连接不受影响。

数据库密钥只在匹配类型和 ID 的路径上生效，例如 `/api/v1/memory-libraries/livingmemory_v8/Default/recall`。密钥不能重放到同 ID 的其它类型，前缀与分类不匹配时返回 `401 Unauthorized`。全局 API Key 仍保留给本机管理和迁移操作。

## 版本管理

Windows WebUI 的“基础设置”页提供版本管理，可读取官方 [PersonalityRAG Releases](https://github.com/Creeper3222/PersonalityRAG/releases) 中带有精确 Windows 资产的版本，并执行更新、兼容旧版本回退或当前版本重新安装。

- 版本切换只替换 Release Manifest 声明的运行代码；`.git`、`.venv`、`config/`、`data/`、数据库、索引、密钥、备份和用户文件始终受保护。
- 切换前会校验 GitHub 资产摘要、单一 ZIP 根目录、每个文件的 SHA-256、Python 兼容范围和持久化结构兼容范围。
- 有运行中、排队、暂停或异常中断的任务时不会开始版本切换。
- 新版本无法启动或健康版本不匹配时，独立更新助手会自动恢复原代码并重新启动原版本。
- GitHub 暂时不可用不会影响登录和记忆库功能；界面保留最近一次成功的版本列表。
- manifest v1 的受管目录和根级运行文件是 v0.1.x 的冻结更新契约。未来运行时代码必须放入既有受管目录；发布构建器会拒绝未分类的 Git 跟踪文件，避免新功能在升级 ZIP 中被静默遗漏。

## 发布版内容

公开仓库只包含源码、静态资源、测试、工具脚本、配置示例和文档。不会包含：

- `.venv/`
- `data/`
- `config/config.json`
- 日志、缓存、浏览器测试 profile
- 真实记忆库、SQLite 数据库、FAISS 索引、导入上传文件、备份、trash 和报告产物

运行后产生的数据都保存在本地 `data/` 和 `config/config.json` 中，请自行备份。

## 开发检查

```powershell
.\.venv\Scripts\python.exe -m compileall personalityrag
node --check static/app.js
.\.venv\Scripts\python.exe -m pytest -q
```

通用 WebUI 验收脚本位于 `tools/acceptance_webui.py`；文本媒体知识库的桌面/移动 Chromium 验收脚本位于 `tools/acceptance_text_media_webui.py`。开发依赖见 `requirements-dev.txt`。

## 相关仓库链接

- [PersonalityRAG Windows 版主仓库](https://github.com/Creeper3222/PersonalityRAG)
- [PersonalityRAG Linux/Docker 版分支](https://github.com/Creeper3222/PersonalityRAG/tree/Linux-Docker)
- [AstrBot 记忆库适配器仓库](https://github.com/Creeper3222/astrbot_plugin_personality_rag_adapter)
- [AstrBot 知识库适配器仓库](https://github.com/Creeper3222/astrbot_plugin_personality_knowledgebase_adapter)

## 鸣谢

PersonalityRAG 的兼容目标、核心记忆结构和召回思路来自 LivingMemory / `astrbot_plugin_livingmemory`：

https://github.com/lxfight-s-Astrbot-Plugins/astrbot_plugin_livingmemory

感谢 lxfight 及该项目贡献者提供的灵感与基础工作。
