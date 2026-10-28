# -*- coding: utf-8 -*-
"""自包含 HTML 评测报告：雷达图、分数分布、最差样本钻取。"""
from __future__ import annotations

import html
import math

METRIC_LABELS = {
    "faithfulness": "忠实度",
    "answer_relevancy": "答案相关性",
    "context_precision": "上下文精确度",
    "context_recall": "上下文召回率",
}


def render_html(report: dict) -> str:
    summary = report.get("summary", {})
    results = report.get("detailed_results", [])
    metadata = report.get("metadata", {})
    scored = [name for name, data in summary.items() if data.get("avg_score") is not None]
    radar = _render_radar(scored, summary) if len(scored) >= 3 else ""
    cards = "".join(_render_card(name, summary[name]) for name in summary)
    distributions = "".join(_render_distribution(name, results) for name in scored)
    worst = _render_worst(results, limit=10)
    tag_summary = report.get("tag_summary", {})
    tag_table = (
        _render_tags(tag_summary)
        if any(tag != "untagged" for tag in tag_summary)
        else ""
    )
    tag_section = (
        f'<section><h2>分层结果（按标签）</h2>{tag_table}</section>' if tag_table else ""
    )
    usage = metadata.get("usage", {})
    cost = usage.get("estimated_cost")

    meta_bits = [
        f"评测时间 {html.escape(str(metadata.get('timestamp', '')))}",
        f"Judge 模型 {html.escape(str(metadata.get('judge_model', '')))}",
        f"样本 {metadata.get('samples', 0)} 条（续跑 {metadata.get('resumed_samples', 0)}）",
        f"并发 {metadata.get('max_concurrency', '')}",
        f"耗时 {metadata.get('duration_seconds', '')}s",
        f"数据集指纹 {html.escape(str(metadata.get('dataset_fingerprint', '')))}",
        f"被测系统 {html.escape(str(metadata.get('system_version', '')))}",
    ]
    usage_bits = [
        f"LLM 调用 {usage.get('calls', 0)} 次",
        f"缓存命中 {usage.get('cache_hits', 0)} 次",
        f"重试 {usage.get('retries', 0)} 次",
        f"解析失败 {usage.get('failures', 0)} 次",
        f"token {usage.get('total_tokens', 0)}",
        f"预估成本 {cost if cost is not None else '未配置单价'}",
    ]

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG 评测报告</title>
<style>
:root {{ color-scheme: light; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; padding: 32px; font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
       background: #f5f7fb; color: #1f2937; }}
.wrap {{ max-width: 1080px; margin: 0 auto; }}
header h1 {{ margin: 0 0 8px; font-size: 26px; }}
.meta {{ margin: 2px 0; color: #6b7280; font-size: 13px; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 24px; }}
section, .card {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px 24px; }}
section {{ margin-top: 20px; }}
h2 {{ margin: 0 0 16px; font-size: 17px; }}
.metric {{ display: flex; justify-content: space-between; align-items: baseline; padding: 10px 0;
           border-bottom: 1px dashed #e5e7eb; }}
.metric:last-child {{ border-bottom: 0; }}
.metric .name {{ font-weight: 600; }}
.metric .sub {{ color: #6b7280; font-size: 12px; }}
.metric .score {{ font-size: 22px; font-weight: 700; color: #2563eb; }}
.dist {{ margin-bottom: 18px; }}
.dist h3 {{ margin: 0 0 8px; font-size: 14px; color: #374151; }}
.bars {{ display: flex; align-items: flex-end; gap: 4px; height: 90px; padding: 6px 8px;
         background: #f9fafb; border-radius: 8px; }}
.bar {{ flex: 1; background: #93c5fd; border-radius: 3px 3px 0 0; min-height: 2px; }}
.bar:hover {{ background: #2563eb; }}
.ticks {{ display: flex; justify-content: space-between; color: #9ca3af; font-size: 11px; margin-top: 4px; }}
details {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px 16px; margin-bottom: 10px; }}
summary {{ cursor: pointer; font-weight: 600; }}
summary .rank {{ color: #dc2626; margin-right: 8px; }}
.detail {{ margin-top: 12px; font-size: 13px; color: #374151; }}
.detail pre {{ white-space: pre-wrap; background: #f9fafb; padding: 10px; border-radius: 8px; }}
.detail table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
.detail td, .detail th {{ border-bottom: 1px solid #f3f4f6; padding: 6px 8px; text-align: left; vertical-align: top; }}
table.tagtable {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
table.tagtable th, table.tagtable td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #f3f4f6; }}
table.tagtable th {{ color: #6b7280; font-weight: 600; }}
footer {{ margin-top: 28px; color: #9ca3af; font-size: 12px; text-align: center; }}
@media (max-width: 800px) {{ .grid {{ grid-template-columns: 1fr; }} body {{ padding: 16px; }} }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>RAG 评测报告</h1>
    <p class="meta">{' ｜ '.join(meta_bits)}</p>
    <p class="meta">{' ｜ '.join(usage_bits)}</p>
  </header>
  <div class="grid">
    <section class="card"><h2>指标雷达</h2>{radar or '<p class="meta">有效指标不足 3 个，未绘制雷达图</p>'}</section>
    <section class="card"><h2>指标概览</h2>{cards}</section>
  </div>
  <section><h2>分数分布</h2>{distributions or '<p class="meta">暂无有效分数</p>'}</section>
  {tag_section}
  <section><h2>最差样本 Top 10</h2>{worst or '<p class="meta">没有低分样本</p>'}</section>
  <footer>由 RAGEval 生成 · 单文件报告，可直接分享或打印</footer>
</div>
</body>
</html>
"""


def _render_radar(metrics: list[str], summary: dict) -> str:
    size, center, radius = 360, 180, 125
    count = len(metrics)
    angles = [-math.pi / 2 + 2 * math.pi * index / count for index in range(count)]

    def point(angle: float, distance: float) -> tuple[float, float]:
        return center + distance * math.cos(angle), center + distance * math.sin(angle)

    parts = ['<svg viewBox="0 0 360 360" width="100%" role="img" aria-label="指标雷达图">']
    for level in (0.25, 0.5, 0.75, 1.0):
        points = " ".join(
            f"{x:.1f},{y:.1f}" for x, y in (point(angle, radius * level) for angle in angles)
        )
        parts.append(
            f'<polygon points="{points}" fill="none" stroke="#e5e7eb" stroke-width="1"/>'
        )
    for angle in angles:
        x, y = point(angle, radius)
        parts.append(
            f'<line x1="{center}" y1="{center}" x2="{x:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>'
        )

    values = [summary[name]["avg_score"] for name in metrics]
    points = " ".join(
        f"{x:.1f},{y:.1f}"
        for x, y in (point(angle, radius * value) for angle, value in zip(angles, values))
    )
    parts.append(
        f'<polygon points="{points}" fill="rgba(37,99,235,0.22)" stroke="#2563eb" stroke-width="2"/>'
    )
    for angle, value in zip(angles, values):
        x, y = point(angle, radius * value)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="#2563eb"/>')

    for angle, name in zip(angles, metrics):
        x, y = point(angle, radius + 24)
        anchor = "start" if x > center + 5 else "end" if x < center - 5 else "middle"
        label = f"{METRIC_LABELS.get(name, name)} {summary[name]['avg_score']:.2f}"
        parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="12" fill="#374151">'
            f"{html.escape(label)}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _render_card(name: str, data: dict) -> str:
    if data.get("avg_score") is None:
        return (
            f'<div class="metric"><span class="name">{html.escape(METRIC_LABELS.get(name, name))}</span>'
            f'<span class="score" style="color:#9ca3af">N/A</span></div>'
        )
    sub = (
        f"min {data.get('min_score')} · max {data.get('max_score')} · "
        f"std {data.get('std_dev')} · 有效 {data.get('sample_count')} · 失败 {data.get('failed_count')}"
    )
    return (
        f'<div class="metric"><div><div class="name">{html.escape(METRIC_LABELS.get(name, name))}</div>'
        f'<div class="sub">{sub}</div></div>'
        f'<div class="score">{data["avg_score"]:.2f}</div></div>'
    )


def _render_tags(tag_summary: dict) -> str:
    metrics = sorted({name for summary in tag_summary.values() for name in summary})
    header = "".join(
        f"<th>{html.escape(METRIC_LABELS.get(name, name))}</th>" for name in metrics
    )
    rows = []
    for tag, summary in tag_summary.items():
        count = 0
        cells = []
        for name in metrics:
            data = summary.get(name, {})
            count = max(count, data.get("sample_count", 0) + data.get("failed_count", 0))
            avg = data.get("avg_score")
            cells.append(f"<td>{'—' if avg is None else f'{avg:.2f}'}</td>")
        rows.append(
            f"<tr><td>{html.escape(tag)}</td><td>{count}</td>{''.join(cells)}</tr>"
        )
    return (
        '<table class="tagtable"><tr><th>标签</th><th>样本数</th>'
        + header
        + "</tr>"
        + "".join(rows)
        + "</table>"
    )


def _render_distribution(metric: str, results: list[dict]) -> str:
    counts = [0] * 10
    for result in results:
        score = result.get("metrics", {}).get(metric, {}).get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            counts[min(9, max(0, int(float(score) * 10)))] += 1
    peak = max(counts) or 1
    bars = "".join(
        f'<div class="bar" style="height:{count / peak * 100:.1f}%" '
        f'title="{index / 10:.1f}-{(index + 1) / 10:.1f}: {count} 条"></div>'
        for index, count in enumerate(counts)
    )
    return (
        f'<div class="dist"><h3>{html.escape(METRIC_LABELS.get(metric, metric))}</h3>'
        f'<div class="bars">{bars}</div>'
        f'<div class="ticks"><span>0</span><span>0.5</span><span>1.0</span></div></div>'
    )


def _render_worst(results: list[dict], limit: int = 10) -> str:
    ranked = []
    for result in results:
        scores = [
            value
            for value in (
                item.get("score") for item in result.get("metrics", {}).values()
            )
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if scores:
            ranked.append((sum(scores) / len(scores), result))
    ranked.sort(key=lambda item: item[0])

    blocks = []
    for rank, (average, result) in enumerate(ranked[:limit], start=1):
        rows = "".join(
            f"<tr><td>{html.escape(METRIC_LABELS.get(name, name))}</td>"
            f"<td>{item.get('score')}</td>"
            f"<td>{html.escape(str(item.get('reason', '')))}</td></tr>"
            for name, item in result.get("metrics", {}).items()
        )
        contexts = "".join(
            f"<li>{html.escape(str(context)[:400])}</li>"
            for context in result.get("contexts", [])
        )
        tag_text = " ".join(
            f"[{html.escape(tag)}]" for tag in result.get("tags") or []
        )
        prefix = f"{tag_text} " if tag_text else ""
        blocks.append(
            f"<details><summary><span class='rank'>#{rank}</span>"
            f"均分 {average:.2f} ｜ {prefix}{html.escape(str(result.get('question', ''))[:120])}</summary>"
            f"<div class='detail'>"
            f"<p><strong>回答</strong></p><pre>{html.escape(str(result.get('answer', '')))}</pre>"
            f"<p><strong>检索上下文</strong></p><ol>{contexts}</ol>"
            f"<table><tr><th>指标</th><th>分数</th><th>理由</th></tr>{rows}</table>"
            f"</div></details>"
        )
    return "".join(blocks)
