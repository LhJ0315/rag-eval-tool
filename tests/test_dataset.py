# -*- coding: utf-8 -*-
from collections import Counter
from pathlib import Path

from ragas_eval.dataset import load_dataset, samples_from_report

DATASET = Path(__file__).resolve().parents[1] / "data" / "eval_dataset.json"
DIFFICULTY_TAGS = {"normal", "boundary", "adversarial"}


def test_master_dataset_layers():
    dataset = load_dataset(DATASET)
    assert len(dataset) == 41
    tags = Counter(tag for sample in dataset for tag in sample["tags"])
    assert dict(tags) == {
        "normal": 25,
        "boundary": 8,
        "adversarial": 8,
        "hotpotqa": 20,
        "policy": 21,
    }
    for sample in dataset:
        assert sample["ground_truth"].strip()
        assert len(DIFFICULTY_TAGS.intersection(sample["tags"])) == 1
        if {"boundary", "adversarial"} & set(sample["tags"]):
            assert sample["note"].strip()


def test_questions_are_unique():
    dataset = load_dataset(DATASET)
    questions = [sample["question"] for sample in dataset]
    assert len(questions) == len(set(questions))


def test_normal_reference_answers_are_answers():
    dataset = load_dataset(DATASET)
    for sample in dataset:
        if "normal" in sample["tags"]:
            assert "无法回答" not in sample["answer"]


def test_ground_truths_are_not_bare_yes_no():
    dataset = load_dataset(DATASET)
    for sample in dataset:
        assert sample["ground_truth"].strip().lower() not in {"yes", "no"}


def test_samples_from_report_preserves_rerun_inputs():
    report = {
        "detailed_results": [
            {
                "question": "q1",
                "answer": "a1",
                "contexts": ["c1", "c2"],
                "ground_truth": "g1",
                "tags": ["normal", "policy"],
            }
        ]
    }
    samples = samples_from_report(report)
    assert samples == [
        {
            "question": "q1",
            "answer": "a1",
            "contexts": ["c1", "c2"],
            "ground_truth": "g1",
            "tags": ["normal", "policy"],
        }
    ]
    try:
        samples_from_report({"detailed_results": []})
    except ValueError as exc:
        assert "detailed_results" in str(exc)
    else:
        raise AssertionError("空报告应拒绝重判")
