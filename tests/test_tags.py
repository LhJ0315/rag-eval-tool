# -*- coding: utf-8 -*-
import asyncio
import json

from ragas_eval.config import Settings
from ragas_eval.dataset import filter_by_tags, load_dataset
from ragas_eval.engine import EvaluationEngine, quality_gate
from tests.helpers import CountingMetric, FakeLLM


def _write(tmp_path, data):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_dataset_normalizes_tags(tmp_path):
    path = _write(
        tmp_path,
        [
            {
                "question": "q",
                "answer": "a",
                "contexts": ["c"],
                "tags": [" normal ", "normal", "boundary"],
            },
            {"question": "q2", "answer": "a2", "contexts": []},
        ],
    )
    dataset = load_dataset(path)
    assert dataset[0]["tags"] == ["normal", "boundary"]
    assert dataset[1]["tags"] == []


def test_load_dataset_rejects_invalid_tags(tmp_path):
    path = _write(
        tmp_path,
        [{"question": "q", "answer": "a", "contexts": ["c"], "tags": [""]}],
    )
    try:
        load_dataset(path)
    except ValueError as exc:
        assert "tags" in str(exc)
    else:
        raise AssertionError("非法 tags 应被拒绝")


def test_filter_by_tags():
    dataset = [
        {"question": "q1", "tags": ["normal"]},
        {"question": "q2", "tags": ["adversarial"]},
        {"question": "q3", "tags": ["adversarial", "boundary"]},
    ]
    assert [item["question"] for item in filter_by_tags(dataset, ["adversarial"])] == [
        "q2",
        "q3",
    ]
    try:
        filter_by_tags(dataset, ["missing"])
    except ValueError as exc:
        assert "标签" in str(exc)
    else:
        raise AssertionError("无匹配标签时应报错")


def _engine(metric=None, checkpoint_dir=".unused"):
    settings = Settings(api_key="test")
    return EvaluationEngine(
        settings,
        client=FakeLLM(settings=settings),
        metrics={"counting": metric or CountingMetric()},
        checkpoint_dir=checkpoint_dir,
    )


def test_tag_summary_groups_and_untagged():
    engine = _engine()
    summary = engine._calculate_tag_summary(
        [
            {"tags": ["normal"], "metrics": {"counting": {"score": 1.0}}},
            {"tags": ["normal"], "metrics": {"counting": {"score": 0.5}}},
            {"tags": ["adversarial"], "metrics": {"counting": {"score": 0.0}}},
            {"tags": [], "metrics": {"counting": {"score": None}}},
        ]
    )
    assert list(summary) == ["normal", "adversarial", "untagged"]
    assert summary["normal"]["counting"]["avg_score"] == 0.75
    assert summary["adversarial"]["counting"]["avg_score"] == 0.0
    assert summary["untagged"]["counting"]["failed_count"] == 1


def test_quality_gate():
    summary = {"a": {"avg_score": 0.8}, "b": {"avg_score": 0.6}}
    assert quality_gate(summary, 0.5) == (True, 0.6)
    assert quality_gate(summary, 0.7) == (False, 0.6)
    assert quality_gate({"a": {"avg_score": None}}, 0.5) == (False, None)


def test_evaluate_dataset_reports_tag_summary(tmp_path):
    engine = _engine(checkpoint_dir=tmp_path / "checkpoint")
    dataset = [
        {"question": "q1", "answer": "a1", "contexts": ["c1"], "tags": ["normal"]},
        {"question": "q2", "answer": "a2", "contexts": ["c2"], "tags": ["adversarial"]},
    ]
    report = asyncio.run(
        engine.evaluate_dataset(dataset, output_dir=tmp_path / "out", html=False)
    )
    assert set(report["tag_summary"]) == {"normal", "adversarial"}
    assert report["metadata"]["tag_distribution"] == {"normal": 1, "adversarial": 1}
