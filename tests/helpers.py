# -*- coding: utf-8 -*-
from ragas_eval.config import Settings
from ragas_eval.llm import UsageStats


class FakeLLM:
    """按顺序返回预设结果的假客户端，用于离线测试。"""

    def __init__(self, chat_responses=None, json_responses=None, embeddings=None, settings=None):
        self.settings = settings or Settings(api_key="test")
        self.chat_responses = list(chat_responses or [])
        self.json_responses = list(json_responses or [])
        self.embeddings = list(embeddings or [])
        self.chat_calls = []
        self.json_calls = []
        self.embed_calls = []
        self.chat_temperatures = []
        self.json_temperatures = []
        self.stats = UsageStats()

    async def chat(self, prompt, system=None, json_mode=False, temperature=0.0):
        self.chat_calls.append(prompt)
        self.chat_temperatures.append(temperature)
        if not self.chat_responses:
            raise AssertionError(f"未预期的 chat 调用：{prompt[:80]}")
        return self.chat_responses.pop(0)

    async def chat_json(self, prompt, system=None, temperature=0.0):
        self.json_calls.append(prompt)
        self.json_temperatures.append(temperature)
        if not self.json_responses:
            raise AssertionError(f"未预期的 chat_json 调用：{prompt[:80]}")
        return self.json_responses.pop(0)

    async def embed(self, texts):
        self.embed_calls.append(texts)
        if not self.embeddings:
            raise AssertionError("未预期的 embed 调用")
        return self.embeddings.pop(0)

    def estimated_cost(self):
        return None

    async def aclose(self):
        pass


class CountingMetric:
    def __init__(self, score_value=1.0, scores=None):
        self.calls = 0
        self.score_value = score_value
        self.scores = list(scores or [])
        self.temperatures = []

    async def score(
        self, question, answer, contexts, ground_truth=None, *, temperature=0.0
    ):
        self.calls += 1
        self.temperatures.append(temperature)
        value = self.scores.pop(0) if self.scores else self.score_value
        return {"score": value, "reason": "ok"}
