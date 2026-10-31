# -*- coding: utf-8 -*-
from ragas_eval.cache import ResponseCache
from ragas_eval.compare import compare_reports, format_table
from ragas_eval.report import render_html


def test_cache_roundtrip(tmp_path):
    cache = ResponseCache(tmp_path / "cache.sqlite")
    assert cache.get("missing") is None
    cache.set("k", "model", "value")
    assert cache.get("k") == "value"
    cache.close()


def _report(scores: list[float]) -> dict:
    return {
        "metadata": {"samples": len(scores), "judge_model": "test"},
        "summary": {
            "faithfulness": {
                "avg_score": sum(scores) / len(scores),
                "min_score": min(scores),
                "max_score": max(scores),
                "std_dev": 0,
                "sample_count": len(scores),
                "failed_count": 0,
            }
        },
        "detailed_results": [
            {
                "question": f"q{index}",
                "answer": "a",
                "contexts": ["c"],
                "metrics": {"faithfulness": {"score": score, "reason": "r"}},
            }
            for index, score in enumerate(scores)
        ],
    }


def test_compare_reports_delta_and_wins():
    result = compare_reports(_report([0.5, 0.5]), _report([0.8, 0.5]))
    row = result["rows"][0]
    assert row["delta"] == 0.15
    assert result["paired"]["faithfulness"]["wins"] == 1
    assert result["paired"]["faithfulness"]["ties"] == 1
    assert "差值" in format_table(result)


def test_html_report_contains_radar_and_worst_samples():
    report = _report([0.2, 0.9])
    report["summary"]["answer_relevancy"] = dict(report["summary"]["faithfulness"])
    report["summary"]["context_precision"] = dict(report["summary"]["faithfulness"])
    html = render_html(report)
    assert "<svg" in html
    assert "忠实度" in html
    assert "最差样本 Top 10" in html
    assert "q0" in html


def _metric_block(avg):
    return {
        "avg_score": avg,
        "min_score": avg,
        "max_score": avg,
        "std_dev": 0,
        "sample_count": 1,
        "failed_count": 0,
    }


def test_html_shows_tag_section():
    report = {
        "metadata": {"samples": 2, "judge_model": "test"},
        "summary": {"faithfulness": _metric_block(0.5)},
        "tag_summary": {
            "normal": {"faithfulness": _metric_block(1.0)},
            "adversarial": {"faithfulness": _metric_block(0.0)},
        },
        "detailed_results": [],
    }
    html = render_html(report)
    assert "分层结果（按标签）" in html
    assert "adversarial" in html


def test_html_shows_sampling_config_and_std():
    report = _report([0.5, 0.5])
    report["metadata"]["sampling"] = {
        "runs_per_metric": 3,
        "temperatures": [0.0, 0.7, 0.7],
    }
    report["summary"]["faithfulness"]["sampling_std"] = 0.12
    report["summary"]["faithfulness"].update(
        {"sampling_std_samples": 2, "partial_failures": 1, "failed_draws": 1}
    )
    html = render_html(report)
    assert "每条采样 3 次" in html
    assert "采样σ 0.12" in html
    assert "2 条可估" in html
    assert "采样缺次 1 条/1 次" in html


def test_html_shows_sampling_reasons_and_skipped():
    report = {
        "metadata": {
            "samples": 1,
            "judge_model": "test",
            "sampling": {"runs_per_metric": 2},
        },
        "summary": {
            "faithfulness": {
                **_metric_block(0.5),
                "skipped_count": 1,
            },
            "context_recall": {**_metric_block(None), "skipped_count": 1},
        },
        "detailed_results": [
            {
                "question": "q1",
                "answer": "a",
                "contexts": [],
                "metrics": {
                    "faithfulness": {
                        "score": 0.5,
                        "reason": "first",
                        "sample_scores": [1.0, 0.0],
                        "sample_reasons": ["first", "second"],
                        "score_std": 0.5,
                    },
                    "context_recall": {
                        "score": None,
                        "skipped": True,
                        "reason": "纯拒答",
                    },
                },
            }
        ],
    }
    page = render_html(report)
    assert "采样 [1.0, 0.0] · σ 0.5" in page
    assert "每次采样理由" in page
    assert "second" in page
    assert "跳过" in page


def test_html_hides_tag_section_when_untagged_only():
    report = {
        "metadata": {"samples": 1},
        "summary": {"faithfulness": _metric_block(None)},
        "tag_summary": {"untagged": {"faithfulness": _metric_block(None)}},
        "detailed_results": [],
    }
    assert "分层结果（按标签）" not in render_html(report)
