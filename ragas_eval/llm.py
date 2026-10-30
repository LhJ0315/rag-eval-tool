# -*- coding: utf-8 -*-
"""LLM 客户端：统一处理缓存、超时、指数退避重试、结构化输出与用量统计。"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import re
from dataclasses import asdict, dataclass

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    RateLimitError,
)

from .cache import ResponseCache
from .config import Settings

logger = logging.getLogger(__name__)


def parse_json(text: str):
    """从模型输出中提取第一个合法 JSON 对象/数组，容忍代码块、前后解释与尾逗号。"""
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    candidates = [stripped]
    balanced = _extract_balanced(stripped)
    if balanced and balanced != stripped:
        candidates.append(balanced)
    for candidate in candidates:
        for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
            try:
                return json.loads(attempt)
            except (json.JSONDecodeError, TypeError):
                continue
    return None


def _extract_balanced(text: str) -> str | None:
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        return None
    start = min(starts)
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


@dataclass
class UsageStats:
    calls: int = 0
    cache_hits: int = 0
    retries: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict:
        data = asdict(self)
        data["total_tokens"] = self.total_tokens
        return data


class LLMClient:
    def __init__(self, settings: Settings, cache: ResponseCache | None = None):
        self.settings = settings
        self.cache = cache
        self.stats = UsageStats()
        self._json_mode_enabled = settings.json_mode
        self._client = AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.request_timeout,
        )

    async def chat(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.0,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        key = self._cache_key(messages, json_mode, temperature)
        # 采样温度下不落缓存，否则重复采样会命中同一响应，方差失真
        use_cache = self.cache is not None and temperature == 0.0
        if use_cache:
            cached = self.cache.get(key)
            if cached is not None:
                self.stats.cache_hits += 1
                return cached

        response = await self._request(messages, json_mode, temperature)
        text = (response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.stats.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.stats.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        self.stats.calls += 1
        if use_cache:
            self.cache.set(key, self.settings.model, text)
        return text

    async def chat_json(
        self, prompt: str, system: str | None = None, temperature: float = 0.0
    ):
        """返回 (解析后的对象, 原始文本)；解析失败时对象为 None。"""
        raw = await self.chat(prompt, system=system, json_mode=True, temperature=temperature)
        data = parse_json(raw)
        if data is None:
            repair = (
                "你上一次的输出不是合法 JSON。请重新完成下面的任务，"
                "只输出一个合法 JSON 对象，不要输出解释或 Markdown 代码块。\n\n"
                + prompt
            )
            raw = await self.chat(repair, system=system, json_mode=True, temperature=temperature)
            data = parse_json(raw)
        if data is None:
            self.stats.failures += 1
            logger.warning("JSON 解析失败，原始输出前 200 字：%s", raw[:200])
        return data, raw

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.settings.embedding_model:
            raise RuntimeError("未配置 EMBEDDING_MODEL")
        for attempt in range(1, self.settings.max_retries + 1):
            try:
                response = await self._client.embeddings.create(
                    model=self.settings.embedding_model, input=texts
                )
                self.stats.calls += 1
                return [item.embedding for item in response.data]
            except Exception as exc:
                if not self._retryable(exc) or attempt == self.settings.max_retries:
                    raise
                self.stats.retries += 1
                await asyncio.sleep(min(2 ** (attempt - 1) + random.random(), 10))
        raise RuntimeError("unreachable")

    def estimated_cost(self) -> float | None:
        price_in = self.settings.price_input_per_1m
        price_out = self.settings.price_output_per_1m
        if price_in is None and price_out is None:
            return None
        cost = (
            self.stats.prompt_tokens * (price_in or 0)
            + self.stats.completion_tokens * (price_out or 0)
        ) / 1_000_000
        return round(cost, 4)

    async def aclose(self) -> None:
        await self._client.close()

    async def _request(self, messages: list[dict], json_mode: bool, temperature: float):
        for attempt in range(1, self.settings.max_retries + 1):
            try:
                return await self._create(messages, json_mode, temperature)
            except Exception as exc:
                if not self._retryable(exc) or attempt == self.settings.max_retries:
                    raise
                self.stats.retries += 1
                delay = min(2 ** (attempt - 1) + random.random() * 0.5, 10)
                logger.warning(
                    "LLM 调用失败（第 %d/%d 次），%.1fs 后重试：%s",
                    attempt,
                    self.settings.max_retries,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
        raise RuntimeError("unreachable")

    async def _create(self, messages: list[dict], json_mode: bool, temperature: float):
        kwargs = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode and self._json_mode_enabled:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            return await self._client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            if "response_format" not in str(exc).lower():
                raise
            logger.warning("当前模型不支持 response_format，降级为普通输出")
            self._json_mode_enabled = False
            kwargs.pop("response_format", None)
            return await self._client.chat.completions.create(**kwargs)

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError, asyncio.TimeoutError)):
            return True
        return isinstance(exc, APIStatusError) and exc.status_code >= 500

    def _cache_key(self, messages: list[dict], json_mode: bool, temperature: float) -> str:
        payload = json.dumps(
            {
                "model": self.settings.model,
                "messages": messages,
                "temperature": temperature,
                "json_mode": json_mode,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
