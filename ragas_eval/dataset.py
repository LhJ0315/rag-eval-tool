# -*- coding: utf-8 -*-
"""数据集加载与格式校验"""
import json
import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


def _normalize_tags(sample: dict, index: int) -> list[str]:
    raw = sample.get("tags", [])
    if raw is None:
        raw = []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raise ValueError(f"第 {index} 条样本的 tags 必须是字符串数组")
    tags: list[str] = []
    for tag in raw:
        if not isinstance(tag, str) or not tag.strip():
            raise ValueError(f"第 {index} 条样本的 tags 含非法标签：{tag!r}")
        tag = tag.strip()
        if tag not in tags:
            tags.append(tag)
    return tags


def load_dataset(file_path: str | Path) -> list[dict]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"数据集文件不存在: {file_path}")

    dataset = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(dataset, list) or not dataset:
        raise ValueError("数据集必须是非空 JSON 数组")

    for index, sample in enumerate(dataset):
        if not isinstance(sample, dict):
            raise ValueError(f"第 {index} 条样本必须是 JSON 对象")
        for field in ("question", "answer", "contexts"):
            if field not in sample:
                raise ValueError(f"第 {index} 条样本缺少必填字段: {field}")
        if not isinstance(sample["question"], str) or not sample["question"].strip():
            raise ValueError(f"第 {index} 条样本的 question 必须是非空字符串")
        if not isinstance(sample["answer"], str):
            raise ValueError(f"第 {index} 条样本的 answer 必须是字符串")
        if not isinstance(sample["contexts"], list):
            raise ValueError(f"第 {index} 条样本的 contexts 必须是数组")
        if "ground_truth" in sample and not isinstance(sample["ground_truth"], str):
            raise ValueError(f"第 {index} 条样本的 ground_truth 必须是字符串")
        sample["tags"] = _normalize_tags(sample, index)

    empty_answers = sum(1 for sample in dataset if not sample["answer"].strip())
    logger.info("加载数据集成功，共 %d 条样本", len(dataset))
    if empty_answers:
        logger.warning("有 %d 条样本回答为空，忠实度/相关性分数会偏低", empty_answers)
    tag_counts = Counter(tag for sample in dataset for tag in sample["tags"])
    if tag_counts:
        logger.info("分层标签：%s", dict(tag_counts.most_common()))
    return dataset


def filter_by_tags(dataset: list[dict], tags: list[str] | set[str]) -> list[dict]:
    wanted = {tag.strip() for tag in tags if tag.strip()}
    if not wanted:
        return dataset
    selected = [
        sample for sample in dataset if wanted.intersection(sample.get("tags") or [])
    ]
    if not selected:
        raise ValueError(f"没有样本匹配标签：{', '.join(sorted(wanted))}")
    logger.info(
        "按标签过滤：%s -> %d/%d 条样本",
        ",".join(sorted(wanted)),
        len(selected),
        len(dataset),
    )
    return selected
