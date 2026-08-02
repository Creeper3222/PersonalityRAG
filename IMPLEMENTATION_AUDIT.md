# PersonalityRAG v0.1.0 多记忆库与模型提供商计划审查

审查日期：2026-06-24

结论：执行计划已完整落盘，未发现仍以单记忆库或单 Provider 方式运行的核心链路。

## 架构落盘

- 全局控制库：`data/personalityrag_system.db`
- 默认记忆库：`data/libraries/beileite`
- 记忆库运行时由 `LibraryManager` 懒加载，数据库、FTS、FAISS、检索缓存、写入锁、维护任务和设置互相隔离。
- Recall 与衰减、清理、备份设置按库保存在系统库，不写入 LivingMemory 兼容表。
- Provider 配置使用不可变 revision；修改配置不会改变正在运行的索引绑定。
- 重建期间查询继续使用旧 snapshot；候选 generation 完成数量、ID、维度、向量及抽样召回验证后才切换。
- 旧 generation 与旧 Provider 客户端保留到安全释放点，失败不会改变当前 `CURRENT`。

## Provider 与 WebUI

- 已实现 OpenAI Embedding、Ollama Embedding、vLLM Embedding。
- vLLM 会解析 `served-model-name`、不发送 `dimensions`，本地和私网地址绕过系统代理。
- Provider API Key 仅返回掩码；支持留空保留、显式清除和服务端安全复制。
- 使用中的 Provider 禁止停用或删除，卡片显示占用库、revision、可用/不可用/待重建状态。
- 记忆库卡片显示六类统计、Provider、模型、维度、generation、索引健康和默认人格。
- 浏览器实测三类 Provider 选择、12 个编辑字段、固定 ID、密钥掩码、明暗主题与中英俄切换。

## 数据与兼容

- 正式数据：175 条记忆、1209 个节点、8566 条关系、10188 个图条目、101 个原子、10725 条消息、42 个会话。
- LivingMemory 非派生表和会话库的主键、UUID、metadata 与逐行哈希保持一致。
- 当前 generation：`gen-20260624-235520-22843939`
- 文档与图谱 FAISS ID 集合均与 SQLite 完全一致。
- 96 条匿名查询与原索引的 Top-10 集合重合率：
  - 文档：99.375%
  - 图谱：99.583%
- 旧无 `library_id` API 始终代理默认库 `beileite`，不受浏览器当前选择影响。

## 验证

- Python 全量编译：通过
- JavaScript 语法检查：通过
- 自动化测试：20 项通过
- SQLite integrity：`ok`
- 外键违规：0
- 本机 vLLM：`http://127.0.0.1:8001/v1`
- 解析模型：`bge-m3`
- 向量维度：1024
- 正式完整重建：通过，失败向量 0
- Edge WebUI 运行时异常：0

详细证据位于 `data/libraries/beileite/reports`。

> Current runtime layout note: v0.1.1 migrates the historical
> `data/libraries/beileite` root to
> `data/databases/memory_stores/livingmemory_v8/beileite`; the historical
> paths above describe the original v0.1.0 audit artifacts.
