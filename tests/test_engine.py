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


def test_multi_sampling_reports_mean_and_std(tmp_path):
    metric = CountingMetric(scores=[1.0, 0.5, 0.0, 0.8, 0.6, 0.4])
    settings = Settings(
        api_key="test", max_concurrency=1, samples=3, sample_temperature=0.7
    )
    engine = EvaluationEngine(
        settings,
        client=FakeLLM(settings=settings),
        metrics={"counting": metric},
        checkpoint_dir=tmp_path / "checkpoint",
    )
    engine.client.estimated_cost = lambda: None
    report = __import__("asyncio").run(
        engine.evaluate_dataset(
            DATASET, output_dir=tmp_path / "out", html=False, resume=False
        )
    )

    first = report["detailed_results"][0]["metrics"]["counting"]
    assert first["score"] == 0.5
    assert first["sample_scores"] == [1.0, 0.5, 0.0]
    assert first["score_std"] == 0.408
    assert first["samples_ok"] == 3
    assert "均值 0.500" in first["reason"]
    assert report["metadata"]["sampling"]["runs_per_metric"] == 3
    assert metric.temperatures.count(0.0) == 2
    assert metric.temperatures.count(0.7) == 4
    assert report["summary"]["counting"]["sampling_std"] == 0.285
    assert report["summary"]["counting"]["sampling_std_samples"] == 2
    assert report["summary"]["counting"]["partial_failures"] == 0
    assert report["summary"]["counting"]["failed_draws"] == 0


def test_single_success_draw_has_no_std():
    engine = EvaluationEngine.__new__(EvaluationEngine)
    result = engine._aggregate_samples(
        [
            {"score": 0.4, "reason": "ok"},
            {"score": None, "reason": "timeout"},
            {"score": None, "reason": "timeout"},
        ]
    )
    assert result["score"] == 0.4
    assert result["score_std"] is None
    assert result["samples_ok"] == 1
    assert result["samples_failed"] == 2
    assert "无法估计 σ" in result["reason"]


def test_summary_excludes_unestimable_sampling_std(tmp_path):
    settings = Settings(api_key="test", samples=2)
    engine = EvaluationEngine(
        settings,
        client=FakeLLM(settings=settings),
        metrics={"counting": CountingMetric()},
        checkpoint_dir=tmp_path / "checkpoint",
    )
    records = [
        {
            "metrics": {
                "counting": {
                    "score": 0.2,
                    "score_std": None,
                    "samples_ok": 1,
                    "samples_failed": 1,
                }
            }
        },
        {
            "metrics": {
                "counting": {
                    "score": 0.5,
                    "score_std": 0.3,
                    "samples_ok": 2,
                    "samples_failed": 0,
                }
            }
        },
    ]
    summary = engine._calculate_summary(records)["counting"]
    assert summary["sampling_std"] == 0.3
    assert summary["sampling_std_samples"] == 1
    assert summary["partial_failures"] == 1
    assert summary["failed_draws"] == 1
    assert summary["failed_count"] == 0


def test_single_sample_keeps_raw_metric_result(tmp_path):
    metric = CountingMetric()
    engine = _engine(metric, tmp_path / "checkpoint")
    report = __import__("asyncio").run(
        engine.evaluate_dataset(
            DATASET, output_dir=tmp_path / "out", html=False, resume=False
        )
    )
    assert report["detailed_results"][0]["metrics"]["counting"] == {
        "score": 1.0,
        "reason": "ok",
    }
    assert "sampling_std" not in report["summary"]["counting"]


def test_settings_reject_invalid_sampling():
    try:
        Settings(api_key="test", samples=0)
    except ConfigError as exc:
        assert "samples" in str(exc)
    else:
        raise AssertionError("samples=0 应抛出 ConfigError")
    try:
        Settings(api_key="test", sample_temperature=3.0)
    except ConfigError as exc:
        assert "sample_temperature" in str(exc)
    else:
        raise AssertionError("sample_temperature 越界应抛出 ConfigError")


def test_settings_read_sampling_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("SAMPLES", "3")
    monkeypatch.setenv("SAMPLE_TEMPERATURE", "0")
    settings = Settings.from_env(str(tmp_path / "missing.env"))
    assert settings.samples == 3
    assert settings.sample_temperature == 0.0


def test_settings_requires_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    try:
        Settings.from_env(str(tmp_path / "missing.env"))
    except ConfigError as exc:
        assert "API Key" in str(exc)
    else:
        raise AssertionError("缺少 API Key 时应抛出 ConfigError")
