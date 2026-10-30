# -*- coding: utf-8 -*-
import asyncio
from types import SimpleNamespace

from ragas_eval.cache import ResponseCache
from ragas_eval.config import Settings
from ragas_eval.llm import LLMClient


def _response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


def _client(tmp_path):
    client = LLMClient(
        Settings(api_key="test"), cache=ResponseCache(tmp_path / "cache.sqlite")
    )
    calls = []

    async def fake_request(messages, json_mode, temperature):
        calls.append(temperature)
        return _response(f"response-{len(calls)}")

    client._request = fake_request
    return client, calls


def _chat_twice(client, **kwargs):
    try:
        return (
            asyncio.run(client.chat("p", **kwargs)),
            asyncio.run(client.chat("p", **kwargs)),
        )
    finally:
        asyncio.run(client.aclose())
        client.cache.close()


def test_zero_temperature_reuses_cache(tmp_path):
    client, calls = _client(tmp_path)
    first, second = _chat_twice(client)
    assert first == second == "response-1"
    assert len(calls) == 1
    assert client.stats.cache_hits == 1


def test_sampling_temperature_bypasses_cache(tmp_path):
    client, calls = _client(tmp_path)
    first, second = _chat_twice(client, temperature=0.7)
    assert first == "response-1" and second == "response-2"
    assert calls == [0.7, 0.7]
    assert client.stats.cache_hits == 0
