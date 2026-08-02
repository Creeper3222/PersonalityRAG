# PersonalityRAG v0.1.0 交付记录

## 当前状态

- Windows 独立服务已运行于 `http://127.0.0.1:8765`
- API Key 保存在本机 `config/config.json`，本文不记录明文
- AstrBot 已在正式迁移后恢复运行
- LivingMemory 源目录及其数据库未被修改

## 正式迁移

- 迁移运行 ID：`20260624-141847-f3180af1`
- 记忆：175
- 图节点：1209
- 图关系：8566
- 图条目：10188
- 记忆原子：101
- 历史消息：10725
- 会话：42
- 当前索引维度：1024
- 当前模型：`bge-m3`

正式迁移后的默认记忆库 ID 为 `beileite`，显示名称为“贝雷特”。归档位于：

`data/libraries/beileite/imports/20260624-141847-f3180af1/source_archive`

归档中的源文件已设为只读。迁移报告、源目标逐表哈希对账、索引兼容报告和
WebUI 验收截图位于 `data/libraries/beileite/reports`。

## 多记忆库与 Provider 管理

- 全局控制库：`data/personalityrag_system.db`
- 默认库：`data/libraries/beileite`
- 默认 Provider：`vllm_embedding` revision 1
- 当前 generation：`gen-20260624-235520-22843939`
- 旧 generation 保留，可即时回滚
- 新 generation：文档 175、图条目 10188、维度 1024
- 向量范数：最小 0.99999988、最大 1.00000012、均值 1.0
- 每个库的 Recall 与衰减、清理、备份设置已独立写入全局控制库
- 多库隔离、懒加载、Provider revision、密钥掩码、占用保护及重启恢复均已实测

> Current runtime layout note: v0.1.1 migrates the historical
> `data/libraries/beileite` root to
> `data/databases/memory_stores/livingmemory_v8/beileite`; the historical
> paths above describe the original v0.1.0 delivery artifacts.

## 校验结论

- `livingmemory.db` 与源快照的非派生表数量、主键、UUID、metadata 和行哈希一致。
- `conversations.db` 的 10725 条消息和 42 个会话行哈希一致。
- SQLite `integrity_check` 为 `ok`，外键违规为 0。
- 文档向量 ID 集合为 175/175，图向量 ID 集合为 10188/10188。
- 正式重建失败数为 0，文档索引与图索引通过同一个 `CURRENT` 指针原子切换。
- 新 generation 在切换前通过文档与图谱抽样召回验证；本次完整重建耗时 53.34 秒。
- Edge 实测记忆库、Provider 和原有四个 WebUI 页面均正常，控制台无错误。
- Provider 类型选择器仅显示 OpenAI、Ollama、vLLM 三类；编辑字段、密钥掩码、明暗主题和中英俄切换均通过浏览器验收。
- Python/Provider/多库/回滚测试共 20 项全部通过，Python 编译与 JavaScript 语法检查通过。

## 关于旧索引顺序

旧索引与当前重建索引的 ID 集合完全一致。96 条匿名查询的原始 FAISS Top-10
集合重合率为：

- 文档：99.375%
- 图谱：99.583%

本机 bge-m3 服务对同一文本的单条请求可逐位复现，但批量请求会出现约
`1e-3` 量级的浮点差异，因此近分结果的严格顺序不能逐项完全复现。该现象
不会造成源数据、ID 或索引条目丢失；报告中不包含私人记忆正文。

## 启动

双击 `launcher.bat`。首次部署时脚本会创建虚拟环境、安装依赖、生成随机密钥
并打开 WebUI。已存在的配置、数据库与索引 generation 会被保留。
