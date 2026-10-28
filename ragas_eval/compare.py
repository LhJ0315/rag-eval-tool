# -*- coding: utf-8 -*-
"""A/B 对比：两份评测报告的指标差值与逐样本胜负。"""
from __future__ import annotations

import json
from pathlib import Path

from .report import METRIC_LABELS


def load_report(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_reports(report_a: dict, report_b: dict) -> dict:
    summary_a = report_a.get("summary", {})
    summary_b = report_b.get("summary", {})
    metrics = sorted(set(summary_a) | set(summary_b))

    rows = []
    for metric in metrics:
        avg_a = summary_a.get(metric, {}).get("avg_score")
        avg_b = summary_b.get(metric, {}).get("avg_score")
        delta = None
        if avg_a is not None and avg_b is not None:
            delta = round(avg_b - avg_a, 3)
        rows.append(
            {
                "metric": metric,
                "label": METRIC_LABELS.get(metric, metric),
                "a": avg_a,
                "b": avg_b,
                "delta": delta,
            }
        )

    paired = _paired_stats(report_a, report_b, metrics)
    return {"rows": rows, "paired": paired}


def _paired_stats(report_a: dict, report_b: dict, metrics: list[str]) -> dict:
    index_a = {
        result.get("question"): result
        for result in report_a.get("detailed_results", [])
    }
    index_b = {
        result.get("question"): result
        for result in report_b.get("detailed_results", [])
    }
    common = [question for question in index_a if question in index_b]

    paired = {}
    for metric in metrics:
        deltas = []
        for question in common:
            score_a = index_a[question].get("metrics", {}).get(metric, {}).get("score")
            score_b = index_b[question].get("metrics", {}).get(metric, {}).get("score")
            if isinstance(score_a, (int, float)) and isinstance(score_b, (int, float)):
                deltas.append(score_b - score_a)
        wins = sum(1 for delta in deltas if delta > 1e-6)
        losses = sum(1 for delta in deltas if delta < -1e-6)
        paired[metric] = {
            "common_samples": len(deltas),
            "wins": wins,
            "ties": len(deltas) - wins - losses,
            "losses": losses,
            "mean_delta": round(sum(deltas) / len(deltas), 3) if deltas else None,
        }
    return paired


def format_table(result: dict) -> str:
    header = f"{'指标':<24}{'A':>8}{'B':>8}{'差值':>8}"
    lines = [header, "-" * len(header)]
    for row in result["rows"]:
        flag = "" if row["delta"] is None else "↑" if row["delta"] > 0 else "↓" if row["delta"] < 0 else "="
        lines.append(
            f"{row['label']:<22}{_fmt(row['a']):>8}{_fmt(row['b']):>8}{_fmt(row['delta']):>8} {flag}"
        )
    lines.append("")
    lines.append("逐样本对比（B 相对 A）：")
    for metric, stats in result["paired"].items():
        lines.append(
            f"  {METRIC_LABELS.get(metric, metric)}：{stats['wins']} 胜 / "
            f"{stats['ties']} 平 / {stats['losses']} 负，"
            f"平均变化 {_fmt(stats['mean_delta'])}（{stats['common_samples']} 条共同样本）"
        )
    return "\n".join(lines)


def _fmt(value) -> str:
    return "N/A" if value is None else f"{value:+.3f}" if isinstance(value, float) else str(value)
