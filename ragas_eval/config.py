# -*- coding: utf-8 -*-
"""运行时配置：统一从环境变量 / .env 读取，缺少关键项时尽早失败。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigError(RuntimeError):
    """配置缺失或格式错误。"""


def load_dotenv(path: str | Path = ".env") -> None:
    """极简 .env 读取：仅支持 KEY=VALUE 与 # 注释，不覆盖已有环境变量。"""
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def _as_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} 必须是整数，当前为 {raw!r}") from exc


def _as_float(name: str, default: float | None) -> float | None:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} 必须是数字，当前为 {raw!r}") from exc


def _as_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str | None = None
    model: str = "gpt-4o-mini"
    embedding_model: str | None = None
    max_concurrency: int = 5
    request_timeout: float = 60.0
    max_retries: int = 3
    json_mode: bool = True
    cache_path: str = "results/cache.sqlite"
    price_input_per_1m: float | None = None
    price_output_per_1m: float | None = None

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ConfigError("api_key 不能为空")
        if self.max_concurrency < 1:
            raise ConfigError("max_concurrency 必须 >= 1")
        if self.max_retries < 1:
            raise ConfigError("max_retries 必须 >= 1")
        if self.request_timeout <= 0:
            raise ConfigError("request_timeout 必须 > 0")

    @classmethod
    def from_env(cls, env_file: str = ".env") -> "Settings":
        load_dotenv(env_file)
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ARK_API_KEY")
        if not api_key:
            raise ConfigError(
                "缺少 API Key：请执行 `cp .env.example .env` 并填写 OPENAI_API_KEY"
            )
        return cls(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            model=os.getenv("JUDGE_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini",
            embedding_model=os.getenv("EMBEDDING_MODEL") or None,
            max_concurrency=_as_int("MAX_CONCURRENCY", 5),
            request_timeout=_as_float("REQUEST_TIMEOUT", 60.0) or 60.0,
            max_retries=_as_int("MAX_RETRIES", 3),
            json_mode=_as_bool("JSON_MODE", True),
            cache_path=os.getenv("CACHE_PATH") or "results/cache.sqlite",
            price_input_per_1m=_as_float("PRICE_INPUT_PER_1M", None),
            price_output_per_1m=_as_float("PRICE_OUTPUT_PER_1M", None),
        )
