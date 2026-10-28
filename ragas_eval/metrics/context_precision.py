# -*- coding: utf-8 -*-
"""上下文精确度（Context Precision）：按相关上下文出现的位置加权。

采用与 RAGAS 一致的 Average Precision 口径：
对每个"相关"位置 k 计算 precision@k（前 k 段中相关段占比），
再对全部相关位置取平均。相关段落排得越靠前，分数越高。
"""
from .base import BaseMetric


class ContextPrecision(BaseMetric):
    name = "context_precision"

    async def score(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> dict:
        if not contexts:
            return self.failure("没有检索到上下文，无法计算精确度")

        details = []
        for index, context in enumerate(contexts):
            prompt = f"""请判断以下上下文片段对回答用户问题是否有帮助。

用户问题：{question}

上下文片段：
{context}

判断标准：
- Relevant：该上下文包含回答问题所需的信息
- Irrelevant：该上下文与问题无关，不包含有用信息

请只输出 Relevant 或 Irrelevant，不要解释。"""
            result = await self._call_llm(prompt)
            normalized = result.strip().lower()
            if normalized.startswith("irrelevant"):
                is_relevant = False
            elif normalized.startswith("relevant"):
                is_relevant = True
            else:
                return self.failure(
                    f"第 {index + 1} 段上下文的判定无法解析", raw=result[:200]
                )
            details.append(
                {
                    "index": index,
                    "is_relevant": is_relevant,
                    "preview": context[:100] + "..." if len(context) > 100 else context,
                }
            )

        flags = [item["is_relevant"] for item in details]
        hits = sum(flags)
        if hits == 0:
            return {
                "score": 0.0,
                "reason": f"共 {len(contexts)} 段上下文，均与问题无关",
                "total_contexts": len(contexts),
                "relevant_contexts": 0,
                "details": details,
            }

        precisions = []
        running_hits = 0
        for rank, is_relevant in enumerate(flags, start=1):
            if is_relevant:
                running_hits += 1
                precisions.append(running_hits / rank)
        score = sum(precisions) / len(precisions)

        return {
            "score": round(score, 3),
            "reason": (
                f"共 {len(contexts)} 段上下文，{hits} 段相关；"
                f"按位置加权的平均精确度为 {score:.3f}"
            ),
            "total_contexts": len(contexts),
            "relevant_contexts": hits,
            "details": details,
        }
