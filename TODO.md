# RAGEval — 进度与待办

> 更新时间：2026-10-28

## 已完成

| 项 | 说明 |
|---|---|
| P0-2 密钥配置化 | 全部改为 `Settings.from_env()`，缺少 Key 时给出明确报错；新增 `.env.example` 与 `.gitignore` |
| P0-3 并发上限 | `EvaluationEngine` 默认 `Semaphore(5)`，可用 `MAX_CONCURRENCY` / `--max-concurrency` 调整 |
| P0-4 重试与超时 | 429 / 5xx / 网络超时指数退避重试，默认 3 次；失败样本记 `score=None`，不再给默认分 |
| P0-5 断点续跑 | 每条样本落盘 `results/.in_progress/<dataset_hash>.jsonl`，重跑自动跳过已完成样本 |
| P1-1 结构化输出 | 请求 `response_format=json_object`，解析失败自动重试一次；仍失败记录原始输出 |
| P1-2 Answer Relevancy 死代码 | 已删除未使用的 `similarity_prompt`，调用次数由 4 次降为 2 次 |
| P1-3 位置加权精确度 | Context Precision 改为 Average Precision（相关段落越靠前分越高） |
| P1-4 Embedding 相似度 | 配置 `EMBEDDING_MODEL` 时相关性走余弦相似度，接口异常自动回退 LLM 判定 |
| P2-1 日志 | 全部替换为 `logging`，同时输出控制台与 `results/eval_<ts>.log` |
| P2-2 成本统计 | 报告记录 token、调用次数、缓存命中、重试、预估费用 |
| P2-3 响应缓存 | SQLite 缓存 `(model, messages, temperature)` 哈希 |
| P2-4 HTML 报告 | 自包含单文件：雷达图、分布、最差样本 Top 10 |
| P2-5 A/B 对比 | `python -m ragas_eval compare a.json b.json`，输出指标差值与胜/平/负 |
| P3-1 CLI | `run` / `compare` 两个子命令，不再需要改源码 |
| P3-2 单元测试 | `pytest` 30 项，全部离线 |
| P3-3 CI | GitHub Actions：push/PR 跑离线测试；配置 Key secret 后跑 normal 层小样本评测，`--fail-under` 门禁并上传报告 |
| P3-4 版本元数据 | 报告记录数据集指纹、judge 模型、被测系统版本 |
| P3-6 分层评测 | 数据集支持 `tags`（normal / boundary / adversarial），报告按层聚合；新增 8 条边界/对抗样本，合并为 `data/layered_dataset.json` |

## 待完成

- **P0-1 吊销历史 Key**：源码中的硬编码 Key 已清除，但仍需到火山引擎控制台**吊销已泄露的 Key**（控制台操作，无法在代码里完成）
- **P1-5 多次采样**：每个指标默认跑 3 次并输出均值与标准差
- **P1-6 人工金标校准**：抽 20–30 条人工打分，计算 Spearman / Cohen's Kappa
- **P2-6 接入真实 RAG API**：`RAGClient` 抽象，只给问题列表即可评测
- **P3-5 本地 judge 验证**：Ollama / vLLM 的 OpenAI 兼容接口已可用，尚未实测
- **数据集扩充**：主数据集 20 条（HotpotQA），分层集 13 条；目标主数据集扩到 100–200 条并记录版本
