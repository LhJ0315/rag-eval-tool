# -*- coding: utf-8 -*-
"""忠实度（Faithfulness）：衡量回答是否基于上下文，有无幻觉。"""
from .base import BaseMetric


class Faithfulness(BaseMetric):
    name = "faithfulness"

    async def score(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
        *,
        temperature: float = 0.0,
    ) -> dict:
        context_text = "\n\n".join(contexts)

        extract_prompt = f"""请从以下回答中提取所有可独立验证的陈述性事实语句（Claim）。

回答内容：
{answer}

请输出 JSON 对象，格式：{{"claims": ["陈述1", "陈述2"]}}。
只输出 JSON，不要输出解释。"""
        data, raw = await self._call_llm_json(extract_prompt, temperature=temperature)
        claims = data.get("claims") if isinstance(data, dict) else None
        if not isinstance(claims, list):
            return self.failure("陈述提取结果不是合法 JSON", raw=raw[:500])

        claims = [str(claim).strip() for claim in claims if str(claim).strip()]
        if not claims:
            return {"score": 1.0, "reason": "回答中没有可验证的陈述", "claims": []}

        numbered = "\n".join(f"{i + 1}. {claim}" for i, claim in enumerate(claims))
        verify_prompt = f"""请逐条判断下列陈述是否能被给定上下文支持。

上下文：
{context_text}

待验证陈述：
{numbered}

请输出 JSON 对象，格式：{{"supported": [true, false, ...]}}。
数组长度必须为 {len(claims)}，顺序与陈述编号一致；true 表示被上下文支持，false 表示不被支持或属于编造。
只输出 JSON，不要输出解释。"""
        verdict, raw = await self._call_llm_json(verify_prompt, temperature=temperature)
        supported = verdict.get("supported") if isinstance(verdict, dict) else None
        if not isinstance(supported, list) or len(supported) != len(claims):
            return self.failure(
                "逐条验证结果格式不正确", raw=raw[:500], claims=claims
            )

        flags = [bool(value) for value in supported]
        unsupported_claims = [claim for claim, ok in zip(claims, flags) if not ok]
        score = sum(flags) / len(flags)
        reason = (
            f"共提取 {len(claims)} 条陈述，{sum(flags)} 条被上下文支持，"
            f"{len(unsupported_claims)} 条不被支持"
        )
        if unsupported_claims:
            reason += "。不支持的陈述：" + "；".join(unsupported_claims[:3])

        return {
            "score": round(score, 3),
            "reason": reason,
            "total_claims": len(claims),
            "supported_claims": sum(flags),
            "unsupported_claims": unsupported_claims,
        }
