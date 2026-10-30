# -*- coding: utf-8 -*-
"""上下文召回率（Context Recall）：标准答案的要点有多少被检索上下文覆盖。"""
from .base import BaseMetric


class ContextRecall(BaseMetric):
    name = "context_recall"

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

        context_text = "\n\n".join(contexts)
        prompt = f"""请把标准答案拆分为若干独立要点，并判断每个要点是否能在检索到的上下文中找到依据。

标准答案：
{ground_truth}

检索到的上下文：
{context_text}

请输出 JSON 对象，格式：
{{"covered_points": ["被覆盖的要点"], "missing_points": ["未被覆盖的要点"]}}
只输出 JSON，不要输出解释。"""
        data, raw = await self._call_llm_json(prompt, temperature=temperature)
        if not isinstance(data, dict):
            return self.failure("召回率分析结果不是合法 JSON", raw=raw[:500])

        covered = data.get("covered_points")
        missing = data.get("missing_points")
        if not isinstance(covered, list) or not isinstance(missing, list):
            return self.failure("召回率分析结果缺少 covered_points / missing_points", raw=raw[:500])

        covered = [str(item).strip() for item in covered if str(item).strip()]
        missing = [str(item).strip() for item in missing if str(item).strip()]
        total_points = len(covered) + len(missing)
        if total_points == 0:
            return self.failure("未能从标准答案中拆分出要点", raw=raw[:500])

        recall = len(covered) / total_points
        return {
            "score": round(recall, 3),
            "reason": (
                f"标准答案共 {total_points} 个要点，上下文覆盖 {len(covered)} 个，"
                f"缺失 {len(missing)} 个"
            ),
            "covered_points": covered,
            "missing_points": missing,
        }
