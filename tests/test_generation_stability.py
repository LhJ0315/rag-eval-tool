# -*- coding: utf-8 -*-
import pytest

from scripts import analyze_generation_stability as stability


def _record(question, generation, answer, contexts):
    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "system": {"generation": generation},
    }


def _detail(question, answer, faithfulness, relevancy):
    return {
        "question": question,
        "answer": answer,
        "metrics": {
            "faithfulness": {"score": faithfulness},
            "answer_relevancy": {"score": relevancy},
        },
    }


def _fixture():
    records = [
        _record("Q1", 1, "same", ["a", "b"]),
        _record("Q1", 2, "same", ["a", "b"]),
        _record("Q1", 3, "same", ["a", "b"]),
        _record("Q2", 1, "one", ["a", "b"]),
        _record("Q2", 2, "two", ["a"]),
        _record("Q2", 3, "three", ["a", "c"]),
    ]
    details = [
        _detail("Q1", "same", 1.0, 0.9),
        _detail("Q1", "same", 1.0, 0.9),
        _detail("Q1", "same", 1.0, 0.9),
        _detail("Q2", "one", 1.0, 0.5),
        _detail("Q2", "two", 0.0, 0.5),
        _detail("Q2", "three", None, None),
    ]
    report = {
        "metadata": {
            "dataset": "system.json",
            "judge_model": "judge",
            "metrics": ["faithfulness", "answer_relevancy"],
            "metric_versions": {},
        },
        "detailed_results": details,
    }
    return records, report


def test_analyze_summarizes_stability_and_metric_variation():
    records, report = _fixture()
    result = stability.analyze(records, report)
    summary = result["summary"]

    assert summary["questions"] == 2
    assert summary["records"] == 6
    assert summary["generations"] == 3
    assert summary["answer_stable_questions"] == 1
    assert summary["context_stable_questions"] == 1
    assert summary["context_order_stable_questions"] == 1

    q1, q2 = result["questions"]
    assert q1["answer_variants"] == 1
    assert q1["answer_similarity_mean"] == 1.0
    assert q1["context_jaccard_mean"] == 1.0
    assert q2["answer_variants"] == 3
    assert q2["answer_stable"] is False
    assert q2["context_set_variants"] == 3
    assert q2["context_jaccard_mean"] == 0.444
    assert q2["metrics"]["faithfulness"]["std"] == 0.5
    assert q2["metrics"]["faithfulness"]["skipped"] == 1
    assert q2["metrics"]["answer_relevancy"]["std"] == 0.0

    assert summary["metric_variation"]["faithfulness"]["questions_with_variation"] == 1
    assert summary["metric_variation"]["faithfulness"]["mean_within_question_std"] == 0.25
    assert summary["metric_variation"]["faithfulness"]["max_within_question_std"] == 0.5
    assert summary["metric_variation"]["answer_relevancy"]["questions_with_variation"] == 0
    assert summary["by_generation"]["1"]["faithfulness"] == 1.0
    assert summary["by_generation"]["3"]["faithfulness"] == 1.0
    assert summary["by_generation"]["3"]["answer_relevancy"] == 0.9


def test_analyze_rejects_misaligned_report():
    records, report = _fixture()
    report["detailed_results"][1]["answer"] = "other"
    with pytest.raises(ValueError):
        stability.analyze(records, report)


def test_analyze_counts_pure_refusals():
    record = _record("Q3", 1, "I cannot find it in the knowledge base.", [])
    detail = _detail("Q3", record["answer"], None, 0.0)
    result = stability.analyze([record], {"metadata": {"metrics": ["faithfulness", "answer_relevancy"]}, "detailed_results": [detail]})
    assert result["questions"][0]["refusal_count"] == 1


def test_render_markdown_contains_metrics_and_counts():
    records, report = _fixture()
    text = stability.render_markdown(stability.analyze(records, report))
    assert "答案完全一致：1/2" in text
    assert "上下文集合一致：1/2" in text
    assert "faithfulness" in text
    assert "Q2" in text
