# -*- coding: utf-8 -*-
"""LLM 响应缓存：以 (model, messages, temperature) 的哈希为 key 落盘 SQLite。"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path


class ResponseCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS responses (
                key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def get(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT response FROM responses WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def set(self, key: str, model: str, response: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO responses (key, model, response, created_at) VALUES (?, ?, ?, ?)",
            (key, model, response, datetime.now().isoformat(timespec="seconds")),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
