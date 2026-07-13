# 可暂停任务与安全断点恢复验收

日期：2026-07-13
版本：v0.1.0 Windows live

## 自动化结果

- 全量测试：211 passed。
- 前端 JavaScript 语法、Python 编译和 `git diff --check` 通过。
- 单元测试覆盖同一任务 ID 暂停/继续、队首阻塞、关机恢复、异常中断、停止回滚、排队取消、损坏或冲突断点拒绝继续。

## 实机链路

- LivingMemory 源数据：175 documents / 10188 graph entries。
- `beileite_test2` 不暂停导入完成，形成 A1 对照。
- `beileite_test` 在 175 documents + 3000 graph entries 后手动暂停。
- 服务重启后，原任务保持 `paused / manual`，断点未变化；后续排队重建任务变为 `cancelled / shutdown`。
- 继续后在 4000 graph entries 停止 bge-m3 服务。配置重试耗尽后，任务变为 `interrupted / provider_unavailable`，仍保持 4000 条断点。
- 恢复 bge-m3 后继续同一 job ID，最终完成 175 + 10188 条索引。
- 独立临时库验证运行中停止：停止请求在当前批次安全边界生效，随后数据库恢复 0/0、进度归零、原 generation 状态恢复。
- 独立排队任务验证取消：任务进入 `cancelled` 历史，未执行 handler。

## 数据与索引一致性

- documents 的文本与非访问元数据、graph nodes/edges/entries、entry-node 映射和 memory atoms 与源库保护内容哈希一致。
- 召回会按既有设计更新 documents metadata 中的 `last_access_time` 和 `access_count`；这两个运行时字段不属于源内容漂移。
- 中断库与不中断对照库的 FAISS document/graph ID 顺序完全一致。
- vLLM 两次独立推理的 float32 字节不完全相同；最大绝对差为 0.0007764371。该差异在容器重启前构建的批次中已存在，因此不是断点拼接造成。
- 12 条固定查询的 A1 对照：Top-5、Top-10、Top-20 ID 顺序全部一致，最大分差 0.001239。
- LivingMemory 原索引 64 条查询集合重合率：所有 Top-5/10/20、document/graph 组合均不低于 0.99375。

## A/B/A

- B 使用 m3e-small：12 条查询在 Top-5、Top-10、Top-20 的顺序全部发生变化。
- A2 切回 bge-m3：Top-5 与 Top-10 为 12/12 完全恢复；Top-20 为 11/12 顺序完全恢复。
- Top-20 的唯一差异是同一 20 条结果中第 12、13 位近同分项互换，集合重合率 1.0，最大分差 0.002654。
- A2 后源数据保护哈希仍与 LivingMemory 源库一致。

## 结论

暂停、重启、Provider 故障、继续、停止回滚和排队取消链路均通过。没有发现数据丢失、ID 漂移、generation 绑定漂移或断点批次重复写入。实机存在的亚毫量级向量字节波动来自外部 vLLM 独立推理的数值非确定性，召回集合与既定 0.02 分数容差均通过。
