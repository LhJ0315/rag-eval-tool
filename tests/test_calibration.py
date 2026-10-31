# -*- coding: utf-8 -*-
from collections import Counter

from ragas_eval.calibration import (
    METRICS,
    blind_payload,
    build_calibration,
    cohen_kappa,
    render_calibration_markdown,
    render_sheet_html,
    score_calibration,
    select_calibration_items,
    spearman,
)


def _synthetic_report() -> dict:
    items = []
    for index in range(41):
        if index < 5:
            tags = ["normal", "policy"]
        elif index < 25:
            tags = ["normal", "hotpotqa"]
        elif index < 33:
            tags = ["boundary", "policy"]
        else:
            tags = ["adversarial", "policy"]
        skipped = index in {6, 7, 25, 33}
        metrics = {
            "faithfulness": {
                "score": None if skipped else 1.0,
                "skipped": skipped,
                "reason": "纯拒答" if skipped else "ok",
            },
            "answer_relevancy": {"score": ((index * 7) % 11) / 10},
            "context_precision": {"score": ((index * 3) % 5) / 4},
            "context_recall": {"score": ((index * 5) % 7) / 6},
        }
        items.append(
            {
                "question": f"q{index}",
                "answer": f"a{index}",
                "contexts": [f"c{index}"],
                "ground_truth": f"gt{index}",
                "tags": tags,
                "metrics": metrics,
            }
        )
    return {
        "metadata": {
            "dataset": "synthetic.json",
            "dataset_fingerprint": "abc123",
            "judge_model": "test-judge",
            "system_version": "test-system",
        },
        "detailed_results": items,
    }


def test_spearman_perfect_ties_and_constant():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40], permutations=200)["rho"] == 1.0
    assert spearman([1, 2, 2, 3], [1, 2, 2, 3], permutations=200)["rho"] == 1.0
    constant = spearman([1, 1, 1, 1], [1, 2, 3, 4], permutations=200)
    assert constant["rho"] is None
    assert "无方差" in constant["reason"]


def test_cohen_kappa_known_value():
    result = cohen_kappa([1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 1, 0, 0, 0])
    assert result["observed_agreement"] == 0.75
    assert result["kappa"] == 0.5
    assert result["counts"]["judge_pass_human_fail"] == 1
    assert result["counts"]["judge_fail_human_pass"] == 1


def test_selection_covers_quotas_and_all_refusals():
    report = _synthetic_report()
    items, quotas = select_calibration_items(report, n=25, seed=7)
    assert len(items) == 25
    assert len({item["id"] for item in items}) == 25
    assert len({item["report_index"] for item in items}) == 25
    assert quotas == {"normal_policy": 5, "normal_hotpotqa": 8, "boundary": 6, "adversarial": 6}
    assert Counter(item["group"] for item in items) == Counter(quotas)
    assert {6, 7, 25, 33}.issubset({item["report_index"] for item in items})
    again, _ = select_calibration_items(report, n=25, seed=7)
    assert [item["report_index"] for item in again] == [item["report_index"] for item in items]


def test_blind_payload_and_html_hide_judge_scores():
    calibration = build_calibration(_synthetic_report(), n=8, seed=1)
    blind = blind_payload(calibration)
    assert all("judge" not in item for item in blind["items"])
    assert all("tags" not in item for item in blind["items"])
    page = render_sheet_html(blind)
    assert "<script id=\"items-data\"" in page
    assert '"judge"' not in page
    assert '"score"' not in page
    assert "__N__" not in page
    assert "8 条盲评" in page
    assert "q0" in page
    assert blind["metadata"]["sheet_id"] in page


def _rating_item(sample_id: str, metric_ratings: dict) -> dict:
    return {"id": sample_id, "ratings": metric_ratings, "note": ""}


def test_score_calibration_aligned_ratings():
    items = []
    for index, (judge_score, human_rating) in enumerate(
        [(0.2, 1), (0.4, 2), (0.8, 4), (1.0, 5)], start=1
    ):
        items.append(
            {
                "id": f"C{index:02d}",
                "question": f"q{index}",
                "tags": ["normal"],
                "judge": {
                    metric: {"score": judge_score, "skipped": False, "reason": "ok"}
                    for metric in METRICS
                },
            }
        )
    calibration = {"metadata": {"n": 4}, "items": items}
    ratings = {
        "rater": "tester",
        "completed_at": "2026-10-31T00:00:00",
        "items": [
            _rating_item(item["id"], {metric: rating for metric in METRICS})
            for item, rating in zip(items, [1, 2, 4, 5])
        ],
    }
    result = score_calibration(ratings, calibration)
    assert result["rater"] == "tester"
    for metric in METRICS:
        block = result["metrics"][metric]
        assert block["n"] == 4
        assert block["spearman"]["rho"] == 1.0
        assert block["kappa"]["kappa"] == 1.0
        assert block["mismatches"] == []


def _threshold_calibration() -> tuple[dict, dict]:
    items = [
        {
            "id": f"C{index:02d}",
            "question": f"q{index}",
            "tags": ["normal"],
            "judge": {
                metric: {"score": 1.0, "skipped": False, "reason": "ok"}
                for metric in METRICS
            },
        }
        for index in (1, 2)
    ]
    ratings = {
        "items": [
            _rating_item("C01", {metric: 3 for metric in METRICS}),
            _rating_item("C02", {metric: 4 for metric in METRICS}),
        ]
    }
    return items, ratings


def test_score_calibration_uses_metric_specific_thresholds():
    items, ratings = _threshold_calibration()
    result = score_calibration(ratings, {"metadata": {"n": 2}, "items": items})
    assert result["human_pass_thresholds"] == {
        "faithfulness": 3,
        "answer_relevancy": 3,
        "context_precision": 4,
        "context_recall": 4,
    }
    assert result["metrics"]["answer_relevancy"]["mismatches"] == []
    cp_mismatches = result["metrics"]["context_precision"]["mismatches"]
    assert [item["id"] for item in cp_mismatches] == ["C01"]
    assert cp_mismatches[0]["human_pass"] is False


def test_score_calibration_global_override_and_sensitivity():
    items, ratings = _threshold_calibration()
    result = score_calibration(ratings, {"metadata": {"n": 2}, "items": items}, human_pass_threshold=3)
    assert set(result["human_pass_thresholds"].values()) == {3}
    assert result["metrics"]["context_precision"]["mismatches"] == []
    sensitivity = result["threshold_sensitivity"]["context_precision"]
    assert [row["human_pass_threshold"] for row in sensitivity] == [3, 4]
    assert sensitivity[0]["observed_agreement"] == 1.0
    assert sensitivity[1]["observed_agreement"] == 0.5
    assert sensitivity[0]["human_pass_threshold"] == 3
    assert sensitivity[1]["human_pass_threshold"] == 4
    markdown = render_calibration_markdown(result)
    assert "阈值敏感性" in markdown
    assert "人工通过线按指标配置" in markdown


def test_score_calibration_rejects_conflicting_or_invalid_thresholds():
    items, ratings = _threshold_calibration()
    calibration = {"metadata": {"n": 2}, "items": items}
    for kwargs in (
        {"human_pass_threshold": 3, "human_pass_by_metric": {"context_precision": 4}},
        {"human_pass_by_metric": {"unknown_metric": 4}},
        {"human_pass_by_metric": {"context_precision": 6}},
    ):
        try:
            score_calibration(ratings, calibration, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"应拒绝非法阈值配置：{kwargs}")


def test_score_calibration_rejects_stale_sheet():
    items = [
        {
            "id": "C01",
            "question": "q1",
            "judge": {
                metric: {"score": 1.0, "skipped": False}
                for metric in METRICS
            },
        }
    ]
    calibration = {"metadata": {"n": 1, "sheet_id": "fresh123"}, "items": items}
    ratings = {
        "sheet_id": "stale456",
        "items": [_rating_item("C01", {metric: 5 for metric in METRICS})],
    }
    try:
        score_calibration(ratings, calibration)
    except ValueError as exc:
        assert "sheet_id" in str(exc)
    else:
        raise AssertionError("sheet_id 不匹配时应拒绝计分")
    ratings["sheet_id"] = "fresh123"
    assert score_calibration(ratings, calibration)["metrics"]["faithfulness"]["n"] == 1


def test_score_calibration_excludes_skipped_and_flags_constant():
    items = [
        {
            "id": "C01",
            "question": "q1",
            "tags": ["normal"],
            "judge": {
                "faithfulness": {"score": None, "skipped": True, "reason": "纯拒答"},
                "answer_relevancy": {"score": 1.0, "skipped": False},
                "context_precision": {"score": 1.0, "skipped": False},
                "context_recall": {"score": 1.0, "skipped": False},
            },
        },
        {
            "id": "C02",
            "question": "q2",
            "tags": ["normal"],
            "judge": {
                "faithfulness": {"score": 1.0, "skipped": False},
                "answer_relevancy": {"score": 1.0, "skipped": False},
                "context_precision": {"score": 1.0, "skipped": False},
                "context_recall": {"score": 1.0, "skipped": False},
            },
        },
    ]
    ratings = {
        "rater": "",
        "items": [
            _rating_item("C01", {metric: "na" for metric in METRICS}),
            _rating_item("C02", {metric: 5 for metric in METRICS}),
        ],
    }
    result = score_calibration(ratings, {"metadata": {"n": 2}, "items": items})
    assert result["metrics"]["faithfulness"]["n"] == 1
    assert result["metrics"]["faithfulness"]["spearman"]["rho"] is None
    assert result["metrics"]["answer_relevancy"]["kappa"]["degenerate"] is True
    markdown = render_calibration_markdown(result)
    assert "Judge 人工校准报告" in markdown
    assert "不可估" in markdown
    assert "未知原因" not in markdown
    assert "一方判定无方差（退化）" in markdown
