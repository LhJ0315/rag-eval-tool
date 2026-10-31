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
| P3-2 单元测试 | `pytest` 73 项，全部离线 |
| P3-3 CI | GitHub Actions：push/PR 跑离线测试；配置 Key secret 后跑 normal 层小样本评测，`--fail-under` 门禁并上传报告 |
| P3-4 版本元数据 | 报告记录数据集指纹、judge 模型、被测系统版本 |
| P3-6 分层评测 | 数据集支持难度 + 领域标签，报告按标签聚合；主集扩至 41 条中英双语（25 normal / 8 boundary / 8 adversarial），含边界、注入、虚假前提等场景 |
| P2-6 接入真实 RAG API | `scripts/run_dify_system.py` 按标签路由到 Dify Chat App，采集回答与 `retriever_resources`；已完成 Dify 1.18 + BGE-M3 的 41 条端到端评测 |
| 检索深度诊断 | 零成本 hit-testing 扫描 top_k=4/6/8/10：policy 覆盖率 0.652/0.735/0.752/0.802（k=6 后增幅收窄），hotpotqa 0.405/0.565/0.715/0.800；结合证据排名复核拒答原因（生成侧 1、k 受限 2、召回不到 1、上下文噪声 1） |
| top_k 消融 | hotpotqa 20 条补跑 k=8/10：按 `relation-v2` 重判后 CR 为 k=4 0.700 / k=8 0.850 / k=10 0.850；k=8 相对 k=4 +0.150（3 胜 0 负 17 平，Animorphs、Vermont、Henry Roth 恢复），严格拒答 4→2、AR +0.058；k=10 的 CR 与 k=8 持平但构成不同（GNR 0→1、Aladin 1→0），AR 回落，2014 S/S 退化；Big Stone Gap 与 2014 S/S 在三个深度均为 0。k=10 才召回《Oh My God》单曲页并使 GNR 判 1.0，k=8 的 v1 CR=1.0 是关系级假阳性 |
| 单次端到端 k=8 全量 | 2026-10-31 以 policy k=4 + hotpotqa k=8 一次跑完 41 条：F 1.000（37 有效/4 跳过）/ AR 0.656 / CP 0.821 / CR 0.866（relation-v2），相对参考答基线 +0.017 / +0.154 / +0.210 / -0.101；hotpotqa 子集（F 1.000、AR 0.576、CP 0.733、CR 0.850）与 k=8 消融一致；报告 `results/system_dify_v1_k8hot/eval_report_20261031_154814_final.json`，CR 重判报告 `results/rejudge/k8_full/eval_report_20261031_184923.json`。v1 的 CR 为 0.963（hotpotqa 0.975）；v2 下 GNR 1→0、2014 S/S 1→0、Big Stone Gap 0.5→0、携带宠物 1→0、实习医保 1→0.5，Beckham 与 Vermont 保持 1.0 |
| 数据修复 | 修正 2 条坏参考答（Random House、Brown State 的 country/county）、1 条无依据表述（实习生答案中的「外包人员」）、3 条对抗参考答（年假满 10 年、报销 15 个工作日、病假天数）；参考答基线 F 由 0.867 升至 0.984（拒答口径重判后 0.983，1 条纯拒答跳过） |
| 数据去重与对抗改造 | 3 组同题 normal/adversarial 改为同主题「直接提问 vs 虚假前提」对照；新增题目唯一性与 normal 参考答非拒答测试 |
| yes/no GT 修复 | 5 道 HotpotQA yes/no 题的 GT 由单字扩写为事实句，同步派生系统数据；只重判受影响 CR（v1 口径：基线 0.870→0.967、系统 0.817→0.939、hotpotqa k=8 0.775→0.975）；新增 GT 非单字测试（共 35 项） |
| P1-5 多次采样 | `--samples N`：首轮 T=0、其余按 `SAMPLE_TEMPERATURE`（默认 0.7），样本取均值与总体σ，汇总输出 `sampling_std`；非 0 温度绕过缓存保证独立采样；单次有效样本 σ 记 N/A，汇总记录可估样本与缺次；hotpotqa 20 条实测均值 F 0.974（16 有效/4 跳过）/ AR 0.503 / CP 0.749 / CR 0.925，sampling_std ≤0.055 |
| 拒答口径与采样可诊断性 | 新增句级拒答识别：纯拒答的 Faithfulness 记 `skipped=True`，排除均值与 `failed_count`；部分拒答先过滤元陈述再验证剩余事实；聚合层记录 `sample_reasons`、`skipped_count`、`skipped_draws`；HTML 报告可逐次展开采样理由。重判 5 份报告：基线 F 0.983（40/1 跳过）、旧 k=4 系统 0.988（35/6）、hotpotqa k4 0.984（16/4）、k8/k10 1.000（18/2）、采样 F 0.974/σ 0.015；F 采样方差由 0.050 降至 0.015，此前系统 F 低 0.095 的结论被证伪 |
| P1-6 人工校准工具 | 新增 `ragas_eval/calibration.py` 与 `scripts/calibrate_judge.py`：25 条目的抽样（标签配额 + 全部纯拒答 + 分箱覆盖）、离线盲评 HTML（不显示 judge 分、localStorage 自动保存、导出带 `sheet_id`）、Spearman 置换检验与 Cohen's Kappa；计分时校验 `sheet_id` 防止新旧抽样表混用。后续新增按指标人工阈值（F/AR ≥3、CP/CR ≥4）与 ≥3/≥4 阈值敏感性表，支持 `--human-pass` 全局覆盖、`--human-pass-map` 按指标覆盖 |
| P1-6 首轮人工校准 | 用户完成 25 条盲评（sheet_id=`d8f98cae5326`），报告 `results/calibration/calibration_report.md`。Context Precision 排序得到支持（ρ 0.783/p 0.001；按默认 ≥4 阈值 κ 0.689、一致率 0.880、假阳性/假阴性 1/2；全局 ≥3 时 κ 0.519；可作答子集 ≥4 κ 0.829、≥3 κ 0.774）；Context Recall 在可作答样本上饱和（整体 ρ 0.461/p 0.055；≥4 时 κ 0.432、4 条假阳性，≥3 时 0.457），默认 ≥4 后 C06/C15/C18 等“部分覆盖”被判不通过；`C19`（Guns N' Roses）judge CR=1.0、人工 1 在关系级标准下是假阳性——标准答案只有 “1999”，上下文出现 1999 就判覆盖，但没有任何一段说明 GNR 为该片做宣传；AI 第二评分员给 3，旧全局 ≥3 通过、当前默认 ≥4 失败。该发现推翻了“k=8 已召回证据、生成不稳定”的旧结论；Answer Relevancy 与人工不符（ρ -0.079、9 条假阴性，其中 5 条在 0.50–0.583 阈值边缘，短答与拒答被 judge 系统性压低）；Faithfulness 因 judge 全 1.0 无法校准（21 条 100% 一致、0 假阳性）。4 条纯拒答的人工 F/AR 评分偏离 rubric，说明该子集按“回答是否得体”打分，AR/F 结论需谨慎。CR 已用同一批 25 条人工评分按 v2 重算（只替换 judge 分数）：ρ 0.461→0.484（p 0.055→0.015），默认 ≥4 的 κ 0.432→0.419、一致率 0.84→0.80，≥3 的 κ 0.457→0.516；C19 假阳性消除，C01（否定式 GT）与 C07（隐含否定）成为新假阴性 |
| P1-6 AI 第二评分员 | 独立子代理只读盲评表与 rubric，不看 judge 分数、人工评分或评测报告，完成 25 条。AI–judge：AR ρ 0.201/κ 0.111、CP ρ 0.747/κ 0.519、CR ρ 0.723/κ 1.000（23 条）；AI–人工：AR ρ 0.289/κ 0.000、CP ρ 0.487/κ -0.136（≥4 时 0.464）、CR ρ 0.557/κ 0.646。独立确认 AR 短答失真与人工拒答评分偏宽；CP 排序支持但阈值敏感；CR 仍饱和。AI 是 LLM，不是第二个真人，不能替代双人信度。按 v2 重算 CR：AI–judge ρ 0.834、κ(≥3) 0.465、κ(≥4) 0.777（n=23），AI–人工不变；C19 假阳性消除，C07 成为新假阴性 |
| P1-6 CR 关系级 v2 | Context Recall 改为关系级判定：每个要点必须由同一段/相邻句明确陈述实体、关系与结论，`covered=true` 必须给出可引用证据；只分散出现关键词不算覆盖。指标版本 `relation-v2` 写入报告 `metadata.metric_versions`。2026-10-31 已用 `rejudge` 重判基线（0.967，41 条与 v1 全同）、k8 全量（0.963→0.866）、旧 k4（0.939→0.793）、k8/k10 消融（0.975→0.850）：C19 1.0→0.0、2014 S/S 1.0→0.0、k10 的 GNR 1.0（《Oh My God》页明确写了 promo）、Beckham 跨段证据链 1.0。同一批 25 条人工评分重算：ρ 0.484/p 0.015、κ(≥4) 0.419、κ(≥3) 0.516，C19 假阳性修复但 C01/C07 成为新假阴性。重判消耗约 517k judge tokens。历史 v1 报告与 v2 不可直接横比 |

## 待完成

- **P0-1 吊销历史 Key**：源码中的硬编码 Key 已清除，但仍需到火山引擎控制台**吊销已泄露的 Key**（控制台操作，无法在代码里完成）
- **下一轮人工校准**：当前仅单评分人、25 条目的抽样，且 4 条纯拒答的评分偏离 rubric。下一轮建议两名评分人先对齐标准（纯拒答的 F=N/A、AR 低分），再计算评分者间信度；同时补入非饱和样本以校准 Faithfulness / Context Recall。CR v2 还要先定义两类边界：否定式 GT（C01）是否跳过 CR，隐含否定（C07，仅列正式员工适用）是否算覆盖
- **指标改进（由首轮校准暴露）**：Answer Relevancy 对短答与拒答系统性偏低，考虑为“完整句子短答”放宽反推问题匹配，或单独报告拒答率；可作答样本上 judge 与人工双方饱和，还需要部分覆盖样本或更细的 GT 要点拆分。按指标阈值与阈值敏感性已实现，CR 关系级 v2 已实现并完成同批人工评分重算；剩余是 judge 侧的 0.6 阈值未校准，以及 v2 的独立人工校准与否定式 GT 处理
- **CR v2 的独立人工校准**：重判已完成，但现有 v2 校准只是同一批 25 条人工评分的重算，不是新一轮独立抽样；下一步用新报告重新抽样，最好双人盲评，并覆盖否定式 GT 与隐含否定
- **P3-5 本地 judge 验证**：Ollama / vLLM 的 OpenAI 兼容接口已可用，尚未实测
- **生成稳定性复测**：`--samples N` 只重复 judge 判分，不重新生成回答，不能用来量化生成波动。已有 3 道证据明确的边界题（David Beckham、2014 S/S、Random House）显示回答固定而 AR 判分波动（σ 0.104–0.313），说明当前波动主要来自 judge；若要测生成稳定性，需要给 Dify 运行器加“同题生成 N 次”模式，再逐份判分。Guns N' Roses 在 k=8 两次检索都没有关系证据，v2 判 0.0（v1 的 1.0 是假阳性），k=10 判 1.0，不能算生成不稳定。Big Stone Gap 仍是召回漏检，需换 embedding/rerank 或补语料后再验证
- **数据集扩充**：主评测集 41 条（25 normal / 8 boundary / 8 adversarial）；英文侧继续扩到 100–200 条并记录版本
