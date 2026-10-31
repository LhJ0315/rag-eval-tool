# -*- coding: utf-8 -*-
"""汇总同题多次生成的答案、检索上下文与指标波动。"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragas_eval.refusal import filter_refusal_sentences


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _pairwise(values: list, fn) -> list[float]:
    return [fn(values[i], values[j]) for i in range(len(values)) for j in range(i + 1, len(values))]


def _answer_ratio(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, _normalize(left), _normalize(right)).ratio()


def _jaccard(left: list[str], right: list[str]) -> float:
    a = {item.strip() for item in left if item.strip()}
    b = {item.strip() for item in right if item.strip()}
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _metric_stats(values: list[float | None]) -> dict:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return {"mean": None, "std": None, "min": None, "max": None, "valid": 0, "skipped": len(values)}
    return {
        "mean": round(statistics.fmean(numeric), 3),
        "std": round(statistics.pstdev(numeric), 3) if len(numeric) > 1 else None,
        "min": round(min(numeric), 3),
        "max": round(max(numeric), 3),
        "valid": len(numeric),
        "skipped": len(values) - len(numeric),
    }


def _validate(records: list[dict], report: dict) -> list[dict]:
    details = report.get("detailed_results") or []
    if len(records) != len(details):
        raise ValueError(f"系统输出 {len(records)} 条与报告 {len(details)} 条不一致")
    for index, (record, detail) in enumerate(zip(records, details)):
        if record.get("question") != detail.get("question") or record.get("answer") != detail.get("answer"):
            raise ValueError(f"第 {index} 条系统输出与报告未对齐")
    return details


def _question_summary(question: str, items: list[tuple[int, dict, dict]], metrics: list[str]) -> dict:
    items = sorted(items, key=lambda item: item[0])
    answers = [record.get("answer") or "" for _, record, _ in items]
    context_lists = [list(record.get("contexts") or []) for _, record, _ in items]
    answer_ratios = _pairwise(answers, _answer_ratio)
    context_overlaps = _pairwise(context_lists, _jaccard)
    refusals = 0
    for answer in answers:
        kept, removed = filter_refusal_sentences(answer)
        if removed and not kept:
            refusals += 1

    metric_stats = {}
    for metric in metrics:
        values = [
            ((detail.get("metrics") or {}).get(metric) or {}).get("score")
            for _, _, detail in items
        ]
        metric_stats[metric] = _metric_stats(values)

    return {
        "question": question,
        "generations": [generation for generation, _, _ in items],
        "answer_variants": len({_normalize(answer) for answer in answers}),
        "answer_stable": len({_normalize(answer) for answer in answers}) == 1,
        "answer_similarity_mean": round(statistics.fmean(answer_ratios), 3) if answer_ratios else 1.0,
        "answer_similarity_min": round(min(answer_ratios), 3) if answer_ratios else 1.0,
        "context_variants": len({tuple(contexts) for contexts in context_lists}),
        "context_set_variants": len({frozenset(contexts) for contexts in context_lists}),
        "context_stable": len({frozenset(contexts) for contexts in context_lists}) == 1,
        "context_order_stable": len({tuple(contexts) for contexts in context_lists}) == 1,
        "context_jaccard_mean": round(statistics.fmean(context_overlaps), 3) if context_overlaps else 1.0,
        "refusal_count": refusals,
        "metrics": metric_stats,
    }


def analyze(records: list[dict], report: dict) -> dict:
    details = _validate(records, report)
    metrics = list((report.get("metadata") or {}).get("metrics") or [])
    if not metrics:
        metrics = sorted({metric for detail in details for metric in (detail.get("metrics") or {})})

    grouped: OrderedDict[str, list[tuple[int, dict, dict]]] = OrderedDict()
    for record, detail in zip(records, details):
        generation = int((record.get("system") or {}).get("generation", 1))
        grouped.setdefault(record["question"], []).append((generation, record, detail))

    questions = [
        _question_summary(question, items, metrics) for question, items in grouped.items()
    ]
    if not questions:
        raise ValueError("系统输出为空，无法分析生成稳定性")
    generation_counts = {len(item["generations"]) for item in questions}
    if len(generation_counts) > 1:
        raise ValueError(f"各题的生成次数不一致：{sorted(generation_counts)}")

    metric_variation = {}
    for metric in metrics:
        stds = [
            item["metrics"][metric]["std"]
            for item in questions
            if item["metrics"][metric]["std"] is not None
        ]
        varied = 0
        for item in questions:
            stats = item["metrics"][metric]
            if stats["valid"] > 1 and stats["min"] != stats["max"]:
                varied += 1
        metric_variation[metric] = {
            "questions_with_variation": varied,
            "mean_within_question_std": round(statistics.fmean(stds), 3) if stds else None,
            "max_within_question_std": round(max(stds), 3) if stds else None,
        }

    by_generation = {}
    for generation in sorted({generation for items in grouped.values() for generation, _, _ in items}):
        by_generation[str(generation)] = {}
        for metric in metrics:
            values = []
            for items in grouped.values():
                for item_generation, _, detail in items:
                    if item_generation != generation:
                        continue
                    score = ((detail.get("metrics") or {}).get(metric) or {}).get("score")
                    if score is not None:
                        values.append(float(score))
            by_generation[str(generation)][metric] = (
                round(statistics.fmean(values), 3) if values else None
            )

    return {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "dataset": (report.get("metadata") or {}).get("dataset"),
            "judge_model": (report.get("metadata") or {}).get("judge_model"),
            "system_version": (report.get("metadata") or {}).get("system_version"),
            "metric_versions": (report.get("metadata") or {}).get("metric_versions"),
            "metrics": metrics,
        },
        "summary": {
            "questions": len(questions),
            "records": len(records),
            "generations": generation_counts.pop(),
            "answer_stable_questions": sum(1 for item in questions if item["answer_stable"]),
            "context_stable_questions": sum(1 for item in questions if item["context_stable"]),
            "context_order_stable_questions": sum(
                1 for item in questions if item["context_order_stable"]
            ),
            "mean_answer_similarity": round(
                statistics.fmean(item["answer_similarity_mean"] for item in questions), 3
            ),
            "mean_context_jaccard": round(
                statistics.fmean(item["context_jaccard_mean"] for item in questions), 3
            ),
            "metric_variation": metric_variation,
            "by_generation": by_generation,
        },
        "questions": questions,
    }


def render_markdown(result: dict) -> str:
    summary = result["summary"]
    metrics = result["metadata"]["metrics"]
    lines = [
        "# 生成稳定性分析",
        "",
        f"- 题目数：{summary['questions']}，每条生成 {summary['generations']} 次，共 {summary['records']} 条",
        f"- 答案完全一致：{summary['answer_stable_questions']}/{summary['questions']}",
        f"- 上下文集合一致：{summary['context_stable_questions']}/{summary['questions']}",
        f"- 平均答案相似度：{summary['mean_answer_similarity']:.3f}",
        f"- 平均上下文 Jaccard：{summary['mean_context_jaccard']:.3f}",
        "",
        "| 题目 | 答案变体 | 上下文变体 | 拒答次数 | "
        + " | ".join(f"{metric} 均值±σ" for metric in metrics)
        + " |",
        "|---|---|---|---|" + "---|" * len(metrics),
    ]
    for item in result["questions"]:
        cells = [
            item["question"].replace("|", "\\|"),
            str(item["answer_variants"]),
            str(item["context_set_variants"]),
            str(item["refusal_count"]),
        ]
        for metric in metrics:
            stats = item["metrics"][metric]
            if stats["mean"] is None:
                cells.append("N/A")
            elif stats["std"] is None:
                cells.append(f"{stats['mean']:.3f}")
            else:
                cells.append(f"{stats['mean']:.3f}±{stats['std']:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "## 指标跨生成波动", "", "| 指标 | 有波动题数 | 平均组内σ | 最大组内σ |", "|---|---|---|---|"])
    for metric in metrics:
        stats = summary["metric_variation"][metric]
        mean_std = "N/A" if stats["mean_within_question_std"] is None else f"{stats['mean_within_question_std']:.3f}"
        max_std = "N/A" if stats["max_within_question_std"] is None else f"{stats['max_within_question_std']:.3f}"
        lines.append(
            f"| {metric} | {stats['questions_with_variation']}/{summary['questions']} | {mean_std} | {max_std} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="分析同题多次生成的稳定性")
    parser.add_argument("--system", required=True, help="run_dify_system.py 的多代输出 JSON")
    parser.add_argument("--report", required=True, help="对应评测报告 JSON")
    parser.add_argument("--output-dir", default="results/stability")
    parser.add_argument("--label", default=None, help="输出文件名后缀")
    args = parser.parse_args()

    records = json.loads(Path(args.system).read_text(encoding="utf-8"))
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    result = analyze(records, report)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"generation_stability_{label}.json"
    md_path = output_dir / f"generation_stability_{label}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    print(f"报告：{json_path}")
    print(f"      {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
