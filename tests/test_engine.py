# -*- coding: utf-8 -*-
import json

from ragas_eval.config import ConfigError, Settings
from ragas_eval.engine import EvaluationEngine, dataset_fingerprint, sample_id
from tests.helpers import CountingMetric, FakeLLM

DATASET = [
    {"question": "q1", "answer": "a1", "contexts": ["c1"], "ground_truth": "g1"},
    {"question": "q2", "answer": "a2", "contexts": ["c2"], "ground_truth": "g2"},
]


def _engine(metric, checkpoint_dir, max_concurrency=2):
    settings = Settings(api_key="test", max_concurrency=max_concurrency)
    engine = EvaluationEngine(
        settings,
        client=FakeLLM(settings=settings),
        metrics={"counting": metric},
        checkpoint_dir=checkpoint_dir,
    )
    engine.client.estimated_cost = lambda: None
    return engine


def test_summary_reports_std_and_failures():
    engine = _engine(CountingMetric(), ".unused")
    summary = engine._calculate_summary(
        [
            {"metrics": {"counting": {"score": 1.0}}},
            {"metrics": {"counting": {"score": 0.5}}},
            {"metrics": {"counting": {"score": None}}},
        ]
    )
    assert summary["counting"]["avg_score"] == 0.75
    assert summary["counting"]["std_dev"] == 0.25
    assert summary["counting"]["failed_count"] == 1


def test_resume_skips_completed_samples(tmp_path):
    metric = CountingMetric()
    engine = _engine(metric, tmp_path / "checkpoint")

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(parents=True)
    resumed = {"sample_id": sample_id(DATASET[0]), "record": {"question": "q1", "metrics": {}}}
    path = checkpoint / f"{dataset_fingerprint(DATASET)}.jsonl"
    path.write_text(json.dumps(resumed, ensure_ascii=False) + "\n", encoding="utf-8")

    report = __import__("asyncio").run(
        engine.evaluate_dataset(DATASET, output_dir=tmp_path / "out", html=False)
    )
    assert metric.calls == 1
    assert report["metadata"]["resumed_samples"] == 1
    assert not path.exists()


def test_no_resume_reruns_everything(tmp_path):
    metric = CountingMetric()
    engine = _engine(metric, tmp_path / "checkpoint")
    __import__("asyncio").run(
        engine.evaluate_dataset(
            DATASET, output_dir=tmp_path / "out", html=False, resume=False
        )
    )
    assert metric.calls == 2


def test_report_contains_metadata_and_summary(tmp_path):
    metric = CountingMetric(score_value=0.4)
    engine = _engine(metric, tmp_path / "checkpoint")
    report = __import__("asyncio").run(
        engine.evaluate_dataset(
            DATASET,
            output_dir=tmp_path / "out",
            dataset_name="unit.json",
            system_version="v1",
            html=False,
        )
    )
    assert report["metadata"]["judge_model"] == "gpt-4o-mini"
    assert report["metadata"]["samples"] == 2
    assert report["summary"]["counting"]["avg_score"] == 0.4


def test_settings_requires_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    try:
        Settings.from_env(str(tmp_path / "missing.env"))
    except ConfigError as exc:
        assert "API Key" in str(exc)
    else:
        raise AssertionError("缺少 API Key 时应抛出 ConfigError")
