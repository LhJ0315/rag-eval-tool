# -*- coding: utf-8 -*-
"""上下文召回率（Context Recall，关系级）：标准答案的要点是否被上下文完整支持。"""
from .base import BaseMetric


class ContextRecall(BaseMetric):
    name = "context_recall"
    version = "relation-v2"

    async def score(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
        *,
        temperature: float = 0.0,
    ) -> dict:
        if not ground_truth:
            return self.failure("未提供标准答案，召回率不参与统计")

        context_text = "\n\n".join(
            f"[{index}] {context}" for index, context in enumerate(contexts, start=1)
        )
        prompt = f"""你是检索召回评估器。请把标准答案拆分为若干独立要点，再逐点判断检索上下文是否完整支持该要点。

判定规则（关系级覆盖）：
- 要点必须来自标准答案本身，不要把上下文或常识中的其他事实拆成额外要点。标准答案只有一个数值或实体时，把它还原为问题所问的完整关系即可（例如问题问“哪一年 GNR 为电影做宣传”，要点写“GNR 在 1999 年为该电影做宣传”），不要拆成多个子事实。
- 先根据问题与标准答案识别需要验证的关系链（实体、动作/身份、时间或结论）。
- 每个关系环节都必须由某段上下文明确陈述；同一段或相邻句能直接读出的关系，优先作为证据。
- 允许跨段组合，但每段都必须明确支持自己承担的环节，且链条能完整连接；不能因为关键词、实体或数字分别出现，就把它们拼成一个关系。
- 不要用常识补全；covered=true 必须给出能直接引用的原文证据。
- 每个要点输出一条记录；covered=false 时 evidence 留空。

问题：
{question}

标准答案：
{ground_truth}

检索到的上下文：
{context_text}

请输出 JSON 对象，格式：
{{"points": [{{"point": "要点原文", "covered": true, "evidence": "支持该要点的原文片段（不超过 80 字）"}}, {{"point": "另一个要点", "covered": false, "evidence": "", "reason": "缺少什么关系或信息"}}]}}
只输出 JSON，不要输出解释。"""
        data, raw = await self._call_llm_json(prompt, temperature=temperature)
        if not isinstance(data, dict):
            return self.failure("召回率分析结果不是合法 JSON", raw=raw[:500])

        points = data.get("points")
        if not isinstance(points, list) or not points:
            return self.failure("召回率分析结果缺少 points", raw=raw[:500])

        covered_points = []
        missing_points = []
        records = []
        for entry in points:
            if not isinstance(entry, dict):
                return self.failure("召回率分析结果的 points 条目不是对象", raw=raw[:500])
            point = str(entry.get("point") or "").strip()
            if not point:
                continue
            covered = entry.get("covered")
            if not isinstance(covered, bool):
                return self.failure("召回率分析结果的 covered 必须是布尔值", raw=raw[:500])
            evidence = str(entry.get("evidence") or "").strip()
            reason = str(entry.get("reason") or "").strip()
            if covered and not evidence:
                covered = False
                reason = reason or "标记为覆盖但未给出可引用的原文证据"
            record = {"point": point, "covered": covered, "evidence": evidence}
            if reason:
                record["reason"] = reason
            records.append(record)
            if covered:
                covered_points.append(point)
            else:
                missing_points.append(point)

        total_points = len(covered_points) + len(missing_points)
        if total_points == 0:
            return self.failure("未能从标准答案中拆分出要点", raw=raw[:500])

        recall = len(covered_points) / total_points
        return {
            "score": round(recall, 3),
            "metric_version": self.version,
            "reason": (
                f"标准答案共 {total_points} 个要点，关系级覆盖 {len(covered_points)} 个，"
                f"缺失 {len(missing_points)} 个"
            ),
            "covered_points": covered_points,
            "missing_points": missing_points,
            "points": records,
        }
