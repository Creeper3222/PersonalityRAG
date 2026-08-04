# LivingMemory 2.5.3 → 2.5.6 对齐与归属矩阵

## 结论

- 2.5.4、2.5.5、2.5.6 继续使用数据库版本 `8`；`livingmemory.db`、`memory_sources` 与 `conversations.db` 均无结构迁移，因此 PersonalityRAG 继续使用 `livingmemory_v8`，不创建 v9。
- 验收基准固定为官方 tag `v2.5.6`（提交 `bb76c7945a41f66bf83f036ebd184b409cd9590c`）。master 比该 tag 多出的内容仅为文档，不作为运行时基准。
- 历史记忆不会被静默改写；新总结、仅原文导入重新总结和手动写入从本次版本开始使用 2.5.6 的存储语义。

## 变化与归属

| 版本 | 上游变化 | PersonalityRAG 本体 | 记忆适配器 | 验收重点 |
|---|---|---|---|---|
| 2.5.4 | Assistant 消息使用 Bot 自身稳定身份；记忆页修复过期筛选结果覆盖，并加入选择、批量编辑、受限滚动与面板可访问性 | 记忆列表加入请求代次保护、安全批量重要性/归档/恢复/删除、受限滚动和详情面板 `inert`；不开放 `memory_type` 编辑 | 按 `get_self_id`、消息对象和平台回退顺序解析 Bot 身份，绝不继承触发用户 | 快速筛选不回跳、Bot 身份、批量派生数据一致性、键盘与面板状态 |
| 2.5.5 | FAISS 最低安全版本提升为 1.14.3 | `faiss-cpu>=1.14.3,<2`，共享 FAISS 运行时诊断继续覆盖所有库类型 | AstrBot live 环境同步验证 1.14.3 | Python 包/二进制绑定、非法指令和 Provider 错误分类 |
| 2.5.6 | 内置群聊/私聊提示词不再要求 `canonical_summary`；检索正文改为 `summary + 前 5 条 key_facts`，自定义提示词仍可提供独立 canonical；新增记忆页批量编辑 | `content`、`canonical_summary`、`persona_summary` 三通道独立传输、导入、导出和重新总结；旧 canonical-only 数据继续兼容 | 六份运行时默认提示词的 SHA-256 与官方文件一致；自动总结、Agent 写入和仅原文导入统一生成富检索正文 | 官方提示词哈希、三通道 round-trip、纯向量与完整召回严格 A/B/A |

## 不变边界

- 数据库 schema version 保持 `8`，不新增表、列或 SQL 索引。
- `content` 仍是 BM25/FAISS 检索正文；`persona_summary` 仍优先用于展示与注入；可选 `canonical_summary` 供图抽取等中性文本消费者使用。
- PersonalityRAG 的候选集、双路召回、过滤、近期补位、MMR、Rerank 与 Top-K 语义不因本次升级改变。
- LivingMemory v8 不持久化用户可编辑的 `memory_type`；批量状态操作必须调用归档/恢复接口，不能直接篡改状态元数据。
- `conversations.db` 的会话、消息顺序、`last_summarized_index`、`pending_summary` 和异常 pending 范围原样保留。

## 严格验收

- 上游 v2.5.6 全量测试、核心全量 pytest、适配器单测/编译、前端语法与交互契约验收全部通过。
- 使用同一对 SQLite Backup API 快照，按 bge-m3 → m3e-small → bge-m3 执行 A/B/A；原始 document-vector 与完整双路链分别比较 Top-K 5/10/20。
- ID、顺序、路由和过滤结果必须一致，分数绝对误差不超过 `1e-5`；A1/A2 必须复现。
- `livingmemory.db` 与 `conversations.db` 的非派生业务数据规范化哈希保持不变，仅允许预期的索引代次和派生图数据变化。
