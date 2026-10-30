# RAGEval — 进度与待办

> 更新时间：2026-10-31

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
| P3-2 单元测试 | `pytest` 46 项，全部离线 |
| P3-3 CI | GitHub Actions：push/PR 跑离线测试；配置 Key secret 后跑 normal 层小样本评测，`--fail-under` 门禁并上传报告 |
| P3-4 版本元数据 | 报告记录数据集指纹、judge 模型、被测系统版本 |
| P3-6 分层评测 | 数据集支持难度 + 领域标签，报告按标签聚合；主集扩至 41 条中英双语（25 normal / 8 boundary / 8 adversarial），含边界、注入、虚假前提等场景 |
| P2-6 接入真实 RAG API | `scripts/run_dify_system.py` 按标签路由到 Dify Chat App，采集回答与 `retriever_resources`；已完成 Dify 1.18 + BGE-M3 的 41 条端到端评测 |
| 检索深度诊断 | 零成本 hit-testing 扫描 top_k=4/6/8/10：policy 覆盖率 0.652/0.735/0.752/0.802（k=6 后增幅收窄），hotpotqa 0.405/0.565/0.715/0.800；结合证据排名复核拒答原因（生成侧 1、k 受限 2、召回不到 1、上下文噪声 1） |
| top_k 消融 | hotpotqa 20 条补跑 k=8/10：k=8 最佳（F 0.780→0.900，CR 0.925→0.975，严格拒答 4→2），k=10 的 F/CR 持平但 AR 回落；合并调优口径 F 0.947 / AR 0.675 / CP 0.821 / CR 0.963 |
| 数据修复 | 修正 2 条坏参考答（Random House、Brown State 的 country/county）、1 条无依据表述（实习生答案中的「外包人员」）、3 条对抗参考答（年假满 10 年、报销 15 个工作日、病假天数）；参考答基线 F 由 0.867 升至 0.984 |
| 数据去重与对抗改造 | 3 组同题 normal/adversarial 改为同主题「直接提问 vs 虚假前提」对照；新增题目唯一性与 normal 参考答非拒答测试 |
| yes/no GT 修复 | 5 道 HotpotQA yes/no 题的 GT 由单字扩写为事实句，同步派生系统数据；只重判受影响 CR，基线 0.870→0.967、系统 0.817→0.939、hotpotqa k=8 0.775→0.975；新增 GT 非单字测试（共 35 项） |
| P1-5 多次采样 | `--samples N`：首轮 T=0、其余按 `SAMPLE_TEMPERATURE`（默认 0.7），样本取均值与总体σ，汇总输出 `sampling_std`；非 0 温度绕过缓存保证独立采样；单次有效样本 σ 记 N/A，汇总记录可估样本与缺次；hotpotqa 20 条实测均值 F 0.813 / AR 0.503 / CP 0.749 / CR 0.925，sampling_std ≤0.055 |

## 待完成

- **P0-1 吊销历史 Key**：源码中的硬编码 Key 已清除，但仍需到火山引擎控制台**吊销已泄露的 Key**（控制台操作，无法在代码里完成）
- **P1-6 人工金标校准**：抽 20–30 条人工打分，计算 Spearman / Cohen's Kappa
- **P3-5 本地 judge 验证**：Ollama / vLLM 的 OpenAI 兼容接口已可用，尚未实测
- **检索消融后续**：k=8 已验证为 hotpotqa 最佳点；如需单次端到端口径，可把 hotpotqa 默认 top_k 改为 8 后重跑全量（judge 约 270k tokens），否则 README 中的调优数字按两次运行合并引用
- **数据集扩充**：主评测集 41 条（25 normal / 8 boundary / 8 adversarial）；英文侧继续扩到 100–200 条并记录版本
