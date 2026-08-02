# LivingMemory 2.5.0 → 2.5.3 对齐与归属矩阵

更新时间：2026-08-01

## 结论

- 2.5.0、2.5.1、2.5.2、2.5.3 的数据库版本均为 `8`；`memory_sources` 仍是 v8 运行时扩展表。因此 PersonalityRAG 继续使用 `livingmemory_v8`，不创建 v9。
- 2.5.3 release 与 `astrbot_plugin_livingmemory-master-v2.5.3` 的 Python 运行时代码一致；验收以 release 源码为准。
- `conversation_store.py` 在 2.5.0 到 2.5.3 之间没有结构迁移；`conversations.db` 必须原样导入并保留待总结游标、pending 范围、参与者与消息顺序。
- PersonalityRAG 拥有数据库、索引、图谱、检索、维护状态和任务；AstrBot 记忆适配器拥有事件身份提取、提示词、白名单/作用域和上下文注入。

## 版本差异与实现归属

| 版本 | 上游变化 | PersonalityRAG 本体 | 记忆适配器 | 主要验收 |
|---|---|---|---|---|
| 2.5.1 | 稳定参与者身份、昵称历史、人物别名去主题化、persona 优先展示、canonical/persona 职责分离、归档恢复兼容、响应式图谱工具栏 | `graph.py` 生成 `person:account:<platform>:<sender_id>`；人物→事实使用 `mentioned_in`，主题→事实使用 `describes`；列表/详情 persona 优先；图谱工具栏响应式 | 从实际总结窗口提取并合并 `participant_identities`；更新缺失的默认提示词，保留自定义模板 | 稳定账号、别名主题抑制、旧记录回退、桌面/移动端图谱 |
| 2.5.2 | 后台索引检查、影子重建、原子切换、并发增量重放、功能指纹、active-only、维护进度 | `service.py` 使用 SQLite Backup API 构建影子派生数据和 FAISS；双阶段增量重放与短激活屏障；持久化激活日志；`indexes.py` 记录 Provider 功能指纹；所有显式重建只处理 active | Dashboard 与 `/prag status` 展示维护阶段、进度、可召回状态和错误分类，不再把长重建当作长期离线 | 并发增删改/归档/恢复、崩溃回滚、同维度功能变化、无有效代次 |
| 2.5.3 | FAISS 包/绑定不匹配诊断、Provider 与运行时错误分类、非法指令兼容回退、排除 1.14.2 | 共享 `faiss_runtime.py` 预检加载；记忆库和 `text_media_v1` 共用；依赖范围为 `faiss-cpu>=1.12.0,!=1.14.2,<2` | 展示本体返回的安全错误分类，不在适配器侧绕过或吞掉依赖错误 | 绑定不匹配、非法指令、Provider 错误分类、核心与适配器健康状态 |

## 数据和 API 契约

1. `livingmemory_v8` 写入与原子替换接口可选接收 `participant_identities`；旧客户端和缺少身份元数据的旧记录继续使用参与者名称回退。
2. `canonical_summary` 始终是客观、中性的检索正文；`persona_summary` 是第一人称人格记忆，只用于展示和注入。
3. 归档记录保留业务数据和原文，但不得重新进入 FTS、文档向量、图谱、图向量或近期补位。
4. 库详情、心跳和任务详情新增兼容的 `maintenance` 字段；没有有效代次时记忆写入与会话写入仍可用，召回返回结构化 `index_not_ready`。
5. 自动与手动索引/图重建均使用连续服务模式；最终原子切换期间只等待短激活屏障。
6. `conversations.db` 导入不修改 `message_count`、`last_summarized_index` 或 `pending_summary`，也不静默修复超出会话消息数的源 pending 范围。

## 维护窗口冻结基线

维护前初步审查值如下；正式操作时必须重新通过 SQLite Backup API 冻结并以新快照为准：

| 项目 | 初步值 |
|---|---:|
| 记忆文档 | 175 |
| 数据库版本 | 8 |
| 会话 | 42 |
| 消息 | 10,725 |
| 待总结消息 | 4,441 |

已知源数据警告：会话 `rocketchat onebot:GroupMessage:2000000003` 的一个 pending 结束位置超过当前会话计数。导入校验应报告并原样保留，不能自动修复。

## 严格验收门槛

- 上游 2.5.3 隔离测试、核心全量 pytest、适配器测试/编译、Node 语法和 `git diff --check` 全部通过。
- A1(bge-m3) → B(m3e-small) → A2(bge-m3) 每轮均从同一对 `livingmemory.db`、`conversations.db` 快照重建。
- 原始向量层与完整双路召回在 Top-K 5/10/20 的 ID、顺序和路由完全一致；所有对应分数误差不超过 `1e-5`。
- A1/A2 完全复现；B 必须由功能指纹触发真实重建并与 A 至少有一项向量校验和或排序不同。
- `conversations.db` 的 sessions、messages、消息顺序、participants、metadata、待总结游标和 pending 范围规范化哈希三轮不变。
- 任一 parity 或完整性门槛失败时保留报告和快照、恢复 Provider A 与最后有效代次，但不得宣告完成。

## 当前实施状态

- [x] 2.5.0→2.5.3 源码与数据库版本审查
- [x] 核心稳定身份、连续重建、active-only、功能指纹和 FAISS 诊断实现
- [x] 适配器身份提取、提示词、维护状态和明色 Dashboard 实现
- [x] 双数据库导入验证与 A/B/A 工具升级到 2.5.3
- [x] 核心全量测试：445 项通过
- [x] 上游 2.5.3 全量测试：707 项通过；核心与适配器隔离视觉验收通过
- [x] 正式插件 update、双数据库冻结和 `beileite_test_250` 安全副本
- [x] 同 ID 实库导入与 conversations 规范化哈希验收
- [x] 严格 A/B/A 报告
- [x] 核心与适配器私有备份分支提交和推送

## 2026-08-01 实库验收结果

- 上游 LivingMemory 已通过 AstrBot 正式插件更新流程从 2.5.0 升级到 2.5.3；live Python 运行时代码与 2.5.3 release 哈希一致，数据库版本仍为 8，FAISS 1.13.2 加载健康。
- 维护备份根目录为 `D:\git_test\RAG\living memory release compare\live-backups\lm253-maintenance-20260801-183709`。维护前后均使用 SQLite Backup API 冻结 `livingmemory.db` 与 `conversations.db`，并保留插件源码、配置、Provider 和索引代次清单。
- `beileite_test_250` 已通过本体复制流程保存并保持离线；`beileite_test_old` 未改动。当前 `beileite_test` 以同 ID 重建并完成双数据库原子导入，任务 ID 为 `058e213d12e748faa8d9cf3aa27afcf9`。
- 当前实库为 175 条 active 记忆、1,682 个节点、9,500 条关系、14,274 条图条目、73 个原子；文档向量 175，memory 粒度图向量 175，活动代次 `gen-20260801-190933-b127a28a`。
- `conversations.db` 导入前后均为 42 个会话、10,725 条消息、4,441 条待总结消息和 4 个 pending summary；sessions、messages、消息顺序、participants、metadata、summary state 六组规范化哈希逐项相同。源数据中 1 个 pending 范围越界仅告警并原样保留。
- 严格 A/B/A 报告位于 `aba-evidence\aba-recall-parity-20260801-200708-568300\aba-recall-parity.json`。A1(bge-m3)、B(m3e-small)、A2(bge-m3) 的 Top-K 5/10/20 ID 与顺序全部一致；原始向量分数最大误差为 0，完整双路最大数值误差分别为 `5.515174865688977e-7`、`5.455436706824912e-7`、`5.515174865688977e-7`，均低于 `1e-5`。A1/A2 完全复现，B 的功能指纹、维度和结果确实发生变化。
- 上游参考实例在一次图影子切换后清理旧 generation 时报告找不到已废弃的 176–350 图向量 ID；新 generation 已完整激活，计数、完整性和严格 parity 均通过。该非致命警告保留在原始验收证据中，没有被隐藏或改写。
- 最终自动化结果：上游 707 项、PersonalityRAG 核心 445 项、记忆适配器 98 项全部通过；Python compileall、前端 `node --check`、`git diff --check` 均通过。桌面/移动端、明暗主题本体页面及适配器明色 Dashboard 截图验收通过。
- 维护结束后记忆适配器和知识库适配器均已通过正式 Pages 重连入口恢复连接；当前库维护状态为 `ready`，Provider 已恢复 A（bge-m3），没有遗留运行中任务。
