# -*- coding: utf-8 -*-
"""答案相关性（Answer Relevancy）：反推问题后计算与原问题的语义相似度。"""
import logging
import math

from .base import BaseMetric

logger = logging.getLogger(__name__)


class AnswerRelevancy(BaseMetric):
    name = "answer_relevancy"

    async def score(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
        *,
        temperature: float = 0.0,
    ) -> dict:
        reverse_prompt = f"""根据以下回答，推测用户可能提出的 3 个问题。

回答内容：
{answer}

请输出 JSON 对象，格式：{{"questions": ["问题1", "问题2", "问题3"]}}。
只输出 JSON，不要输出解释。"""
        data, raw = await self._call_llm_json(reverse_prompt, temperature=temperature)
        questions = data.get("questions") if isinstance(data, dict) else None
        if not isinstance(questions, list) or not questions:
            return self.failure("反推问题解析失败", raw=raw[:500])
        questions = [str(item).strip() for item in questions[:3] if str(item).strip()]
        if not questions:
            return self.failure("反推问题为空", raw=raw[:500])

        method = "llm"
        if self.llm.settings.embedding_model:
            try:
                vectors = await self.llm.embed([question, *questions])
                similarities = [_cosine(vectors[0], vector) for vector in vectors[1:]]
                method = "embedding"
            except Exception as exc:  # embedding 接口不可用时回退到 LLM 判定
                logger.warning("embedding 相似度计算失败，回退 LLM 判定：%s", exc)
                similarities = await self._llm_similarities(question, questions, temperature)
        else:
            similarities = await self._llm_similarities(question, questions, temperature)

        if similarities is None:
            return self.failure("相似度判定解析失败", generated_questions=questions)

        similarities = [min(1.0, max(0.0, value)) for value in similarities]
        average = sum(similarities) / len(similarities)
        return {
            "score": round(average, 3),
            "reason": (
                f"反推问题与原始问题的平均语义相似度为 {average:.2f}（{method}）"
            ),
            "method": method,
            "generated_questions": questions,
            "similarities": [round(value, 3) for value in similarities],
        }

    async def _llm_similarities(
        self, question: str, questions: list[str], temperature: float = 0.0
    ) -> list[float] | None:
        listing = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(questions))
        prompt = f"""请判断下列每个反推问题与原始问题的语义相似度。

原始问题：{question}
反推问题：
{listing}

请逐个给出 0 到 1 之间的分数（1 表示语义完全一致，0 表示完全无关）。
输出 JSON 对象，格式：{{"similarities": [0.9, 0.6, 0.3]}}，数组长度必须为 {len(questions)}。
只输出 JSON，不要输出解释。"""
        data, _ = await self._call_llm_json(prompt, temperature=temperature)
        values = data.get("similarities") if isinstance(data, dict) else None
        if not isinstance(values, list) or len(values) != len(questions):
            return None
        try:
            return [float(value) for value in values]
        except (TypeError, ValueError):
            return None


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0 or norm_right == 0:
        return 0.0
    return dot / (norm_left * norm_right)
