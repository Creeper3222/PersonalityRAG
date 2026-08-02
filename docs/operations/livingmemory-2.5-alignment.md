# LivingMemory 2.3.6 → 2.5.0 对齐与归属矩阵

更新时间：2026-07-31

## 结论

- 上游 2.3.6、2.4.0、2.4.1、2.5.0 的 `DATABASE_VERSION` 均为 `8`，`db_migration.py` 没有数据库版本跃迁。
- 2.5.0 新增的 `memory_sources` 是 v8 运行时扩展表，不构成 v9。因此 PersonalityRAG 继续使用 `livingmemory_v8`，不新建库类型。
- PersonalityRAG 拥有数据库、索引、检索、归档、迁移任务和管理 API；AstrBot 记忆适配器拥有事件身份、白名单、作用域、对话 LLM、提示词和上下文注入。
- 上游算法与数据语义需要对齐；页面、任务系统、鉴权和 API 仍使用 PersonalityRAG 自身架构。
- 本文同时记录代码覆盖和实库验收状态；只有下方严格 A/B/A、数据完整性和正式插件更新全部通过后，才认定完成 2.5.0 对齐。

## 版本差异摘要

| 版本 | 上游主要变化 | PersonalityRAG 归属 | 适配器归属 |
|---|---|---|---|
| 2.4.0 | 全量图谱、社区布局、大图性能、最低重要性、扩展上下文年龄、批量删除、双通道摘要、六类提示词 | 全量图数据接口、可扩展图渲染、最低重要性、批量任务 | 上下文年龄、canonical/persona 双摘要、提示词管理 |
| 2.4.1 | memory 级聚合图向量、批量图重建、超大图分级渲染、旧双通道注入去重 | 聚合图向量、manifest、旧 entry 索引兼容、图重建 | 旧摘要兼容与注入正文选择 |
| 2.5.0 | 相似度阈值、近期槽位、event-only、来源时间、归档恢复、衰减保护、白名单、作用域、别名、原文、指定 N 条总结、JSON/CSV 迁移 | 检索策略、归档、衰减、原文表、迁移任务与接口 | 白名单、作用域、别名、来源时间、LLM 总结、`/prag summarize N`、按需原文 |

## 能力与实现矩阵

| 能力 | 数据/算法契约 | PersonalityRAG 实现 | 适配器实现 | 主要验证 |
|---|---|---|---|---|
| v8 原文表 | `memory_sources(memory_id, source_json, created_at, updated_at)`；读旧库不因读取原文建表 | `library_types/livingmemory_v8/source.py`、`storage.py` | `main.py` 按阈值提交结构化原文 | `test_livingmemory_250_alignment.py` |
| 原文生命周期 | 删除、替换、复制、备份、迁移同步处理；不进入 FTS、向量、图、原子 | `storage.py`、`service.py`、迁移任务 | 仅对命中且明确请求的 ID 读取原文 | `test_storage_and_indexes.py`、`test_livingmemory_250_alignment.py` |
| 双通道摘要 | `canonical_summary` 是检索正文；`persona_summary` 只用于人格化展示和注入 | 写入接口保存两类摘要，索引只读取 canonical 正文 | 自动总结、Agent 写入和迁移总结使用 2.5.0 回退次序 | `test_summary_alignment.py`、`test_summary_task_flow.py` |
| 最低重要性 | 统一出口过滤低重要性 active 记忆 | `retrieval.py`、`config.py`、库设置页 | 透传单库设置 | `test_livingmemory_250_alignment.py` |
| 最低相似度 | 取文档向量和图向量信号最大值；纯关键词命中不因缺向量分数被过滤 | `retrieval.py` | 无 | `test_livingmemory_250_alignment.py` |
| 近期记忆槽位 | 同 session/persona、active、时间窗口；去重后按上游顺序补位并截断 | `retrieval.py`、`storage.py` | 传入解析后的作用域 | `test_livingmemory_250_alignment.py` |
| event-only | 保留无类型旧记录，只排除可确定的纯偏好/纯关系 | `retrieval.py` | 无 | `test_livingmemory_250_alignment.py` |
| active-only | 文档、BM25、图关键词、图向量和近期路径均排除非 active | `storage.py`、`retrieval.py` | 无 | 检索及归档测试 |
| memory 级图向量 | 同一源记忆的去重图条目按换行拼接并截断 4000 字符，只生成一个图向量 | `indexes.py`、`storage.py` | 无 | `test_storage_and_indexes.py` |
| 旧图索引兼容 | 无 granularity 字段的历史 manifest 按 entry 读取；新库和显式重建按 memory | `indexes.py` | 无 | 索引单元测试及后续实库重建 |
| 图 manifest | 记录粒度、源记忆数、图条目数、向量数和内容哈希 | `indexes.py` | 无 | `test_storage_and_indexes.py` |
| 全量图与大图渲染 | 社区概览、连接束、内部骨架、选中精确邻接、空间命中索引、稳定后停止动画 | `routes/recall_graph.py`、`service.py`、`static/modules/graph.js` | 无 | API 测试；隔离 WebUI 视觉验收待执行 |
| 可恢复归档 | 保留 document 和 source；移除 FTS、向量、图和原子；恢复时重建派生数据 | `storage.py`、`service.py`、记忆详情 UI | 无 | `test_storage_and_indexes.py` |
| 清理策略 | `auto_archived_enabled=false` 保持删除；开启后归档 | `service.py`、库设置页 | 配置同步 | `test_tasks_import_and_backup.py` |
| 重要性保护 | 达到 `protected_importance_threshold` 的记忆不参与每日衰减 | `service.py`、`config.py` | 配置同步 | `test_tasks_import_and_backup.py` |
| 便携迁移 | 原生 JSON、CSV、常见外部字段；50 MiB/10,000 条；预检；作用域去重；不复用外部索引 | `transfer.py`、`routes/tasks_migration.py`、`resumable_tasks.py`、WebUI | 对仅原文且至少两条消息的项目调用 AstrBot LLM 后提交 | 核心迁移测试、适配器总结流程测试 |
| CSV 安全 | 导出防公式注入，导入可安全往返 | `transfer.py` | 无 | `test_livingmemory_250_alignment.py` |
| 来源时间 | 只写结构化来源起止时间/日期，不污染 canonical 正文 | 接收并保存 metadata | `main.py` 从原始消息时间确定性生成 | `test_summary_task_flow.py` |
| 白名单 | 启用且为空时拒绝全部记忆入口 | 无 AstrBot 身份判断 | `core/memory_scope.py`、`main.py` 覆盖捕获、总结、召回、写入和 Agent 工具 | `test_summary_task_flow.py` |
| 身份别名 | 匹配顺序：`platform:user_id`、`user_id`、用户名；总结前替换 | 保存规范化身份/作用域 | `core/memory_scope.py`、`main.py` | `test_summary_task_flow.py` |
| 记忆作用域 | `legacy/session/user/global`，`isolated_sessions` 优先；只影响新写入 | 按传入 session/persona 做类型化过滤 | `core/memory_scope.py` | `test_summary_task_flow.py` |
| 最近上下文年龄 | 仅拼接配置年龄范围内的历史消息 | 无 | `main.py` | `test_summary_task_flow.py` |
| 工具调用污染规避 | 不把工具中间响应或工具循环最终附加内容写入会话/总结 | 无 | `main.py::on_llm_response` | `test_summary_task_flow.py` |
| 指定 N 条总结 | 指定数量时忽略旧进度，取当前会话最近 N 条 | 提供会话范围读取和结构化写入 | `/prag summarize [message_count]` | `test_command_registration.py`、`test_summary_task_flow.py` |
| 按需原文 | 默认不返回；只有明确请求且命中记录有原文时读取 | 库级鉴权 source 接口 | `recall_prag_memory(include_source=false)` | `test_summary_task_flow.py` |
| 六类提示词 | 群聊、私聊、基础系统、人格系统、注入头、注入尾；占位符校验与恢复默认 | 无 | `core/prompt_manager.py`、Dashboard | `test_prompt_manager.py` |
| 工具式写入 | canonical 优先 key facts，缺失时回退 summary；persona 保留原始表达 | 类型化新增接口 | `memorize_prag_memory` 的结构化参数 | `test_summary_task_flow.py` |

## 兼容与不变式

1. 数据库版本继续是 8；不会自动创建 v9。
2. `memory_sources` 只在首次实际保存原文时惰性创建；普通详情读取和召回不建表。
3. 旧索引不会后台升级。历史 manifest 没有 `graph_vector_granularity` 时继续按 entry 读取；显式图重建才切换到 memory 粒度。
4. 归档记录仍存在于 documents 和 `memory_sources`，但不能从任何检索路径命中。
5. 导入、重新总结和内容编辑不复用旧向量；新派生索引完全按当前 Provider 和配置生成。
6. 作用域与别名只影响升级后的新写入，不自动迁移旧记忆。
7. PersonalityRAG 的 `psk-` 库密钥、任务队列、监听面 allowlist 和 WebUI 风格保持不变。
8. `beileite_test_old` 在真实切换后作为恢复副本保留；不得对它自动加载、迁移或重建索引。

## 验收状态

- [x] 版本与数据库结构审查
- [x] 核心与适配器主要代码覆盖
- [x] 新库 memory 粒度、旧 manifest entry 粒度兼容策略
- [x] 核心定向测试与适配器单元测试
- [x] 核心全量 pytest、WebUI 明暗主题/桌面/移动端隔离验收
- [x] 上游 2.5.0 隔离参考实例验证
- [x] 真实插件正式 update、配置保留、175 条文档和 DB v8 核验
- [x] `beileite_test_old` 可恢复副本与真实快照导入
- [x] bge-m3 → m3e-small → bge-m3 严格 A/B/A
- [x] 原始 Embedding 与完整 2.5.0 检索链的 ID、顺序、路由及 `1e-5` 分数阈值验收
- [x] 私有备份分支提交与推送

## 2026-07-31 实库验收结果

- AstrBot 通过正式插件管理 API 更新到 LivingMemory `2.5.0`，运行时源码与下载的 2.5.0 master 参考仓库一致；旧配置的 59 个显式叶子值全部保留，并补齐 15 个新配置项。
- 上游真实库和重新导入的 `beileite_test` 均为数据库版本 8：175 条 documents、1,682 个图节点、9,500 条图边、14,274 条图条目、73 个原子、42 个会话和 10,725 条消息。
- 显式图重建后图向量从 entry 粒度升级为 memory 粒度：175 个 active 源记忆对应 175 个图向量；原始 documents、会话和逻辑图数据的规范化哈希保持一致。
- `beileite_test_old` 保留为未自动迁移的恢复副本；原 `beileite_test` 删除操作已进入 PersonalityRAG trash，并另有更新前、图重建前和更新后的一致性快照。
- 真实库严格 A/B/A 报告位于 `data/reports/aba-recall-parity-20260731-034015-770500/`：A1、B、A2 的原始向量层和完整检索链在 Top-K 5/10/20 全部通过，误差阈值为 `1e-5`，A1/A2 完全复现且 B 与 A 确实不同。
- 自动化验证结果：核心 440 项 pytest 通过，记忆适配器 96 项 pytest 通过；Python compileall、前端 Node 语法检查、协议检查和 `git diff --check` 均通过。
