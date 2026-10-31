# -*- coding: utf-8 -*-
import asyncio
import json
from types import SimpleNamespace

import pytest

from scripts import run_dify_system as runner


def _apps_file(tmp_path):
    path = tmp_path / "apps.json"
    path.write_text(
        json.dumps(
            {
                "base_url": "http://test",
                "apps": {
                    "policy": {"app_id": "p", "api_key": "k", "dataset_id": "d"},
                    "hotpotqa": {"app_id": "h", "api_key": "k", "dataset_id": "d"},
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_load_questions_ignores_blank_and_comment_lines(tmp_path):
    path = tmp_path / "questions.txt"
    path.write_text("# 注释\n  Q1  \n\nQ2\n", encoding="utf-8")
    assert runner.load_questions(path) == ["Q1", "Q2"]


def test_select_samples_preserves_order_and_rejects_missing():
    dataset = [{"question": "Q1"}, {"question": "Q2"}]
    assert runner.select_samples(dataset, ["Q2"]) == [dataset[1]]
    assert runner.select_samples(dataset, None) == dataset
    with pytest.raises(ValueError):
        runner.select_samples(dataset, ["Q3"])


def test_checkpoint_key_includes_generation():
    assert runner.checkpoint_key({"question": "Q1"}) == ("Q1", 1)
    assert runner.checkpoint_key({"question": "Q1", "system": {"generation": 3}}) == ("Q1", 3)


def test_run_generates_each_question_multiple_times(tmp_path, monkeypatch):
    dataset = [
        {"question": "Q1", "ground_truth": "G1", "tags": ["policy"]},
        {"question": "Q2", "ground_truth": "G2", "tags": ["policy"]},
    ]
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")

    calls = []

    async def fake_ask(client, base_url, app, question, user):
        calls.append(question)
        return {
            "answer": f"{question}-{len(calls)}",
            "contexts": ["ctx"],
            "resources": [],
            "message_id": "m",
            "usage": {},
            "latency_ms": 1,
        }

    monkeypatch.setattr(runner, "ask", fake_ask)
    output = tmp_path / "nested" / "out.json"
    args = SimpleNamespace(
        dataset=str(dataset_path),
        output=str(output),
        questions_file=None,
        generations=3,
        apps_file=str(_apps_file(tmp_path)),
        concurrency=2,
        timeout=5,
        resume=False,
    )

    assert asyncio.run(runner.run(args)) == 0
    records = json.loads(output.read_text(encoding="utf-8"))
    assert len(records) == 6
    assert len(calls) == 6
    for question in ("Q1", "Q2"):
        generations = sorted(
            record["system"]["generation"] for record in records if record["question"] == question
        )
        assert generations == [1, 2, 3]
    assert len({record["answer"] for record in records}) == 6

    meta = json.loads(output.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["samples"] == 2
    assert meta["records"] == 6
    assert meta["generations"] == 3
    assert meta["failures"] == []
    assert not output.with_suffix(".jsonl").exists()


def test_run_filters_questions(tmp_path, monkeypatch):
    dataset = [
        {"question": "Q1", "ground_truth": "G1", "tags": ["policy"]},
        {"question": "Q2", "ground_truth": "G2", "tags": ["policy"]},
    ]
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
    questions_path = tmp_path / "questions.txt"
    questions_path.write_text("Q2\n", encoding="utf-8")

    async def fake_ask(client, base_url, app, question, user):
        return {
            "answer": question,
            "contexts": ["ctx"],
            "resources": [],
            "message_id": "m",
            "usage": {},
            "latency_ms": 1,
        }

    monkeypatch.setattr(runner, "ask", fake_ask)
    output = tmp_path / "out.json"
    args = SimpleNamespace(
        dataset=str(dataset_path),
        output=str(output),
        questions_file=str(questions_path),
        generations=2,
        apps_file=str(_apps_file(tmp_path)),
        concurrency=2,
        timeout=5,
        resume=False,
    )

    assert asyncio.run(runner.run(args)) == 0
    records = json.loads(output.read_text(encoding="utf-8"))
    assert [record["question"] for record in records] == ["Q2", "Q2"]
