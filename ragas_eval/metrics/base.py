# -*- coding: utf-8 -*-
"""评测指标基类"""
from abc import ABC, abstractmethod

from ..llm import LLMClient


class BaseMetric(ABC):
    name: str = "base_metric"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def _call_llm(
        self, prompt: str, system: str | None = None, temperature: float = 0.0
    ) -> str:
        return await self.llm.chat(prompt, system=system, temperature=temperature)

    async def _call_llm_json(
        self, prompt: str, system: str | None = None, temperature: float = 0.0
    ):
        return await self.llm.chat_json(prompt, system=system, temperature=temperature)

    @staticmethod
    def failure(reason: str, **extra) -> dict:
        """评测失败时返回 score=None，避免默认分污染均值。"""
        return {"score": None, "reason": reason, **extra}

    @abstractmethod
    async def score(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> dict:
        """返回包含 score (0-1, 失败为 None) 与 reason 的字典。"""
