# -*- coding: utf-8 -*-
from collections import Counter
from pathlib import Path

from ragas_eval.dataset import load_dataset

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
