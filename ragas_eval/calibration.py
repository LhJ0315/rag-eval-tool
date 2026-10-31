# -*- coding: utf-8 -*-
"""人工校准：分层抽样、盲评页面与 Spearman / Cohen's Kappa 统计。"""
from __future__ import annotations

import json
import math
import random
from datetime import datetime
from hashlib import sha256
from pathlib import Path

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
METRIC_LABELS = {
    "faithfulness": "Faithfulness 忠实度",
    "answer_relevancy": "Answer Relevancy 相关性",
    "context_precision": "Context Precision 检索精度",
    "context_recall": "Context Recall 检索召回",
}
METRIC_HINTS = {
    "faithfulness": "回答中的事实性陈述是否都能被检索上下文支持；按上下文判定，不看标准答案。",
    "answer_relevancy": "回答是否直接、完整地回应问题；按切题程度判定，不看事实正确性。",
    "context_precision": "检索片段是否与问题相关，相关片段是否排在前面。",
    "context_recall": "检索上下文是否覆盖标准答案的全部要点。",
}
DEFAULT_QUOTAS = {"normal_policy": 5, "normal_hotpotqa": 8, "boundary": 6, "adversarial": 6}
JUDGE_PASS = 0.6
HUMAN_PASS = 3
SHEET_SCHEMA_VERSION = 1


def _metric_result(item: dict, metric: str) -> dict:
    data = (item.get("metrics") or {}).get(metric) or {}
    skipped = bool(data.get("skipped"))
    raw = data.get("score")
    score = None if skipped or raw is None else float(raw)
    return {
        "score": score,
        "skipped": skipped,
        "reason": data.get("reason"),
    }


def _judge_score(item: dict, metric: str) -> float | None:
    data = (item.get("judge") or {}).get(metric) or {}
    if data.get("skipped"):
        return None
    score = data.get("score")
    return None if score is None else float(score)


def _group(tags: list[str] | None) -> str:
    tags = set(tags or [])
    if "normal" in tags:
        return "normal_policy" if "policy" in tags else "normal_hotpotqa"
    if "boundary" in tags:
        return "boundary"
    if "adversarial" in tags:
        return "adversarial"
    return "untagged"


def _quotas_for(n: int) -> dict[str, int]:
    if n == sum(DEFAULT_QUOTAS.values()):
        return dict(DEFAULT_QUOTAS)
    weights = {
        "normal_policy": DEFAULT_QUOTAS["normal_policy"] / 25,
        "normal_hotpotqa": DEFAULT_QUOTAS["normal_hotpotqa"] / 25,
        "boundary": DEFAULT_QUOTAS["boundary"] / 25,
        "adversarial": DEFAULT_QUOTAS["adversarial"] / 25,
    }
    quotas = {key: int(n * weight) for key, weight in weights.items()}
    remainder = n - sum(quotas.values())
    ranked = sorted(weights, key=lambda key: n * weights[key] - quotas[key], reverse=True)
    for key in ranked[:remainder]:
        quotas[key] += 1
    return quotas


def _coverage(item: dict) -> set[tuple[str, str]]:
    covered: set[tuple[str, str]] = set()
    for metric in ("answer_relevancy", "context_precision", "context_recall"):
        score = _judge_score(item, metric)
        if score is None:
            continue
        if score < 0.5:
            bucket = "low"
        elif score < 0.85:
            bucket = "mid"
        else:
            bucket = "high"
        covered.add((metric, bucket))
    return covered


def _min_score(item: dict) -> float:
    scores = [
        _judge_score(item, metric)
        for metric in ("answer_relevancy", "context_precision", "context_recall")
    ]
    available = [score for score in scores if score is not None]
    return min(available) if available else 0.0


def select_calibration_items(report: dict, n: int = 25, seed: int = 20261031) -> tuple[list[dict], dict]:
    """按标签配额和分数分箱做目的抽样，优先覆盖纯拒答与低分边界。"""
    detailed = report.get("detailed_results") or []
    if len(detailed) < n:
        raise ValueError(f"报告只有 {len(detailed)} 条样本，无法抽取 {n} 条")

    candidates = []
    for index, item in enumerate(detailed):
        judge = {metric: _metric_result(item, metric) for metric in METRICS}
        candidates.append(
            {
                "report_index": index,
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "contexts": list(item.get("contexts") or []),
                "ground_truth": item.get("ground_truth", ""),
                "tags": list(item.get("tags") or []),
                "group": _group(item.get("tags")),
                "judge": judge,
                "f_skipped": judge["faithfulness"]["skipped"],
            }
        )

    quotas = _quotas_for(n)
    chosen: list[dict] = []
    for group, quota in quotas.items():
        pool = [item for item in candidates if item["group"] == group]
        selected: list[dict] = []
        for item in pool:
            if item["f_skipped"] and len(selected) < quota:
                selected.append(item)
        covered: set[tuple[str, str]] = set()
        for item in selected:
            covered |= _coverage(item)
        while len(selected) < quota:
            best = None
            for item in pool:
                if item in selected:
                    continue
                gain = len(_coverage(item) - covered)
                key = (gain, -_min_score(item), -item["report_index"])
                if best is None or key > best[0]:
                    best = (key, item)
            if best is None:
                break
            selected.append(best[1])
            covered |= _coverage(best[1])
        chosen.extend(selected)

    if len(chosen) < n:
        remaining = [item for item in candidates if item not in chosen]
        while len(chosen) < n and remaining:
            covered = set()
            for item in chosen:
                covered |= _coverage(item)
            best = max(
                remaining,
                key=lambda item: (
                    len(_coverage(item) - covered),
                    -_min_score(item),
                    -item["report_index"],
                ),
            )
            chosen.append(best)
            remaining.remove(best)

    if len(chosen) != n:
        raise ValueError(f"只选出 {len(chosen)} 条，无法满足 {n} 条配额")
    if len({item["report_index"] for item in chosen}) != n:
        raise ValueError("抽样出现重复样本")

    random.Random(seed).shuffle(chosen)
    for number, item in enumerate(chosen, start=1):
        item["id"] = f"C{number:02d}"
    return chosen, quotas


def build_calibration(report: dict, n: int = 25, seed: int = 20261031) -> dict:
    items, quotas = select_calibration_items(report, n=n, seed=seed)
    metadata = report.get("metadata") or {}
    sheet_id = sha256(
        json.dumps(
            [(item["id"], item["report_index"], item["question"]) for item in items],
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()[:12]
    return {
        "metadata": {
            "source_dataset": metadata.get("dataset"),
            "dataset_fingerprint": metadata.get("dataset_fingerprint"),
            "judge_model": metadata.get("judge_model"),
            "system_version": metadata.get("system_version"),
            "n": len(items),
            "seed": seed,
            "sheet_id": sheet_id,
            "quotas": quotas,
            "selection": "标签配额 + 全部纯拒答 + AR/CP/CR 低/中/高分箱覆盖",
            "judge_pass_threshold": JUDGE_PASS,
            "human_pass_threshold": HUMAN_PASS,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        "items": items,
    }


def blind_payload(calibration: dict) -> dict:
    metadata = calibration["metadata"]
    return {
        "metadata": {
            "n": metadata["n"],
            "seed": metadata["seed"],
            "sheet_id": metadata["sheet_id"],
            "created_at": metadata["created_at"],
            "human_pass_threshold": metadata["human_pass_threshold"],
        },
        "items": [
            {
                "id": item["id"],
                "question": item["question"],
                "answer": item["answer"],
                "contexts": item["contexts"],
                "ground_truth": item["ground_truth"],
            }
            for item in calibration["items"]
        ],
    }


def ratings_template(calibration: dict) -> dict:
    return {
        "schema_version": SHEET_SCHEMA_VERSION,
        "sheet_id": calibration["metadata"]["sheet_id"],
        "rater": "",
        "completed_at": None,
        "items": [
            {
                "id": item["id"],
                "ratings": {metric: None for metric in METRICS},
                "note": "",
            }
            for item in calibration["items"]
        ],
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_rubric_markdown() -> str:
    return """# 人工校准评分标准

对每条样本独立评分，不要参考 LLM judge 的分数。四个指标都按 1–5 分评；如果该指标对这条样本不适用，选“N/A”。

| 分数 | 含义 |
|---|---|
| 5 | 完全满足指标定义，没有需要修改的地方 |
| 4 | 有轻微问题，但不影响可用性 |
| 3 | 部分满足，存在明确缺口 |
| 2 | 明显不足，可能误导使用者 |
| 1 | 基本不满足指标定义 |
| N/A | 该指标不适用（例如纯拒答没有可验证的事实性陈述） |

## Faithfulness（忠实度）

只看“回答中的事实性陈述能否被检索上下文支持”，不看回答是否符合标准答案。纯拒答没有事实性陈述，记 N/A；部分拒答只评有事实内容的句子。

## Answer Relevancy（相关性）

只看回答是否直接、完整地回应问题，不看事实是否正确。按指标定义，纯拒答视为没有回答问题，相关性低分；拒答本身是否合适另行记录在备注里。

## Context Precision（检索精度）

只看检索片段与问题的相关性及排序：相关片段越靠前分越高，无关片段越多分越低。不要求上下文覆盖全部答案要点（那属于 Context Recall）。

## Context Recall（检索召回）

只看检索上下文是否覆盖标准答案的全部要点，不看最终回答。标准答案本身是“无法回答/无相关信息”时，如果无法判断覆盖情况可选 N/A。

## 备注

- 评分时不要看 `calibration_key.json`，也不要看原始评测报告中的 judge 分数。
- 建议先读问题、标准答案和上下文，再评 Context Precision / Context Recall；最后读系统回答，评 Faithfulness / Answer Relevancy。
- 25 条是目的抽样，低分边界样本被刻意多抽，不能用来估计总体均值。
"""


def _json_for_script(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")


def render_sheet_html(blind: dict) -> str:
    data = _json_for_script(blind)
    template = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAGEval 人工校准</title>
<style>
:root { --bg:#f4f6fb; --panel:#fff; --ink:#1f2937; --muted:#6b7280; --line:#e5e7eb; --accent:#2563eb; --accent-soft:#eff6ff; --ok:#059669; --warn:#d97706; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif; }
header { position:sticky; top:0; z-index:10; background:rgba(255,255,255,.96); border-bottom:1px solid var(--line); backdrop-filter:blur(8px); }
.header-inner { max-width:1240px; margin:0 auto; padding:14px 20px; display:flex; gap:18px; align-items:center; flex-wrap:wrap; }
h1 { font-size:18px; margin:0; white-space:nowrap; }
.sub { color:var(--muted); font-size:12px; }
.progress-wrap { flex:1; min-width:220px; }
.progress { height:8px; background:#e5e7eb; border-radius:999px; overflow:hidden; }
.progress > i { display:block; height:100%; width:0; background:var(--accent); transition:width .2s; }
.header-actions { display:flex; gap:8px; align-items:center; }
input[type=text] { border:1px solid var(--line); border-radius:8px; padding:8px 10px; font:inherit; color:var(--ink); background:#fff; }
button { border:1px solid var(--line); background:#fff; color:var(--ink); border-radius:8px; padding:8px 12px; font:inherit; cursor:pointer; }
button:hover { border-color:#bfdbfe; background:var(--accent-soft); }
button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
button.primary:hover { background:#1d4ed8; }
button.danger { color:#b91c1c; }
.layout { max-width:1240px; margin:0 auto; padding:20px; display:grid; grid-template-columns:190px minmax(0,1fr); gap:20px; align-items:start; }
aside { position:sticky; top:86px; background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:10px; max-height:calc(100vh - 110px); overflow:auto; }
.list-title { font-size:12px; color:var(--muted); padding:4px 8px 8px; }
.item-btn { width:100%; display:flex; justify-content:space-between; align-items:center; gap:8px; border:0; background:transparent; padding:7px 8px; border-radius:8px; text-align:left; }
.item-btn:hover { background:#f8fafc; }
.item-btn.active { background:var(--accent-soft); color:#1d4ed8; font-weight:600; }
.dot { width:8px; height:8px; border-radius:50%; background:#d1d5db; flex:none; }
.dot.partial { background:var(--warn); }
.dot.done { background:var(--ok); }
main { min-width:0; display:grid; gap:14px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:18px 20px; }
.kicker { color:var(--muted); font-size:12px; display:flex; justify-content:space-between; gap:12px; }
h2 { margin:4px 0 0; font-size:22px; line-height:1.4; }
h3 { margin:0 0 8px; font-size:14px; }
.answer, .gt { white-space:pre-wrap; overflow-wrap:anywhere; }
.answer { background:#f8fafc; border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
.gt { background:#fffbeb; border:1px solid #fde68a; border-radius:10px; padding:12px 14px; }
.contexts { display:grid; gap:8px; }
details.ctx { border:1px solid var(--line); border-radius:10px; padding:8px 12px; background:#fff; }
details.ctx summary { cursor:pointer; color:#374151; font-weight:600; }
.ctx-body { white-space:pre-wrap; overflow-wrap:anywhere; color:#374151; padding-top:8px; max-height:260px; overflow:auto; }
.context-tools { display:flex; gap:8px; }
.rubric summary { cursor:pointer; font-weight:600; color:#1d4ed8; }
.rubric table { border-collapse:collapse; width:100%; margin:10px 0; font-size:13px; }
.rubric th, .rubric td { border:1px solid var(--line); padding:6px 8px; text-align:left; vertical-align:top; }
.rubric th { background:#f8fafc; }
.metric { border:1px solid var(--line); border-radius:12px; padding:12px 14px; margin:0 0 12px; }
.metric legend { font-weight:600; padding:0 6px; }
.hint { color:var(--muted); font-size:12px; margin:0 0 10px; }
.options { display:flex; gap:6px; flex-wrap:wrap; }
.options label { border:1px solid var(--line); border-radius:8px; padding:6px 9px; cursor:pointer; background:#fff; font-size:13px; }
.options label:has(input:checked) { border-color:var(--accent); background:var(--accent-soft); color:#1d4ed8; font-weight:600; }
.options input { margin-right:4px; }
.options label.na { color:#6b7280; }
textarea { width:100%; min-height:64px; border:1px solid var(--line); border-radius:10px; padding:10px 12px; font:inherit; resize:vertical; }
.nav { display:flex; justify-content:space-between; gap:10px; }
.footer-note { color:var(--muted); font-size:12px; text-align:center; padding:0 0 28px; }
@media (max-width:820px) {
  .layout { grid-template-columns:1fr; padding:12px; }
  aside { position:static; max-height:none; display:flex; gap:6px; overflow-x:auto; padding:8px; }
  .list-title { display:none; }
  .item-btn { width:auto; flex:none; }
  h2 { font-size:18px; }
  .header-inner { padding:10px 12px; }
}
</style>
</head>
<body>
<header>
  <div class="header-inner">
    <div>
      <h1>RAGEval 人工校准</h1>
      <div class="sub">__N__ 条盲评 · 不显示 LLM judge 分数</div>
    </div>
    <div class="progress-wrap">
      <div class="progress"><i id="progress-bar"></i></div>
      <div class="sub" id="progress-text">0 / __N__ 已完成</div>
    </div>
    <div class="header-actions">
      <input id="rater" type="text" placeholder="评分人（可选）" aria-label="评分人">
      <button class="primary" id="export">导出评分 JSON</button>
      <button class="danger" id="reset">重置</button>
    </div>
  </div>
</header>
<div class="layout">
  <aside aria-label="样本列表"><div class="list-title">样本</div><div id="list"></div></aside>
  <main id="panel"></main>
</div>
<div class="footer-note">评分自动保存在本机浏览器；完成后导出 JSON，交回评测脚本计算一致性。</div>
<script id="items-data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('items-data').textContent);
const METRICS = [
  {key:'faithfulness', label:'Faithfulness 忠实度', hint:'回答中的事实性陈述是否都能被检索上下文支持；按上下文判定，不看标准答案。'},
  {key:'answer_relevancy', label:'Answer Relevancy 相关性', hint:'回答是否直接、完整地回应问题；按切题程度判定，不看事实正确性。'},
  {key:'context_precision', label:'Context Precision 检索精度', hint:'检索片段是否与问题相关，相关片段是否排在前面。'},
  {key:'context_recall', label:'Context Recall 检索召回', hint:'检索上下文是否覆盖标准答案的全部要点。'}
];
const STORAGE_KEY = 'rageval-calibration-v1:' + (DATA.metadata.sheet_id || 'default');
let state = loadState();
let current = 0;

function blankState() {
  return {rater:'', ratings:{}};
}
function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return blankState();
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object' || !parsed.ratings) return blankState();
    return parsed;
  } catch (err) {
    return blankState();
  }
}
function saveState() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch (err) {}
}
function ensure(id) {
  if (!state.ratings[id]) {
    state.ratings[id] = {faithfulness:null, answer_relevancy:null, context_precision:null, context_recall:null, note:''};
  }
  if (!state.ratings[id].note) state.ratings[id].note = '';
  return state.ratings[id];
}
function isComplete(id) {
  const r = state.ratings[id];
  if (!r) return false;
  return METRICS.every(m => r[m.key] !== null && r[m.key] !== undefined && r[m.key] !== '');
}
function completedCount() {
  return DATA.items.filter(item => isComplete(item.id)).length;
}
function updateProgress() {
  const done = completedCount();
  document.getElementById('progress-text').textContent = done + ' / ' + DATA.items.length + ' 已完成';
  document.getElementById('progress-bar').style.width = (done / DATA.items.length * 100) + '%';
}
function renderList() {
  const list = document.getElementById('list');
  list.textContent = '';
  DATA.items.forEach((item, index) => {
    const button = document.createElement('button');
    button.className = 'item-btn' + (index === current ? ' active' : '');
    button.type = 'button';
    button.setAttribute('aria-current', index === current ? 'true' : 'false');
    const label = document.createElement('span');
    label.textContent = item.id;
    const dot = document.createElement('span');
    dot.className = 'dot ' + (isComplete(item.id) ? 'done' : (state.ratings[item.id] ? 'partial' : ''));
    button.append(label, dot);
    button.addEventListener('click', () => { current = index; renderAll(); });
    list.appendChild(button);
  });
}
function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}
function addCard(title, body) {
  const card = node('section', 'card');
  const heading = node('h3', '', title);
  card.append(heading, body);
  return card;
}
function contextCard(item) {
  const card = node('section', 'card');
  const top = node('div', 'kicker');
  const title = node('h3', '', '检索上下文（' + item.contexts.length + ' 段）');
  const tools = node('div', 'context-tools');
  const expand = node('button', '', '展开全部');
  const collapse = node('button', '', '收起全部');
  expand.type = 'button'; collapse.type = 'button';
  expand.addEventListener('click', () => card.querySelectorAll('details').forEach(d => d.open = true));
  collapse.addEventListener('click', () => card.querySelectorAll('details').forEach(d => d.open = false));
  tools.append(expand, collapse);
  top.append(title, tools);
  card.appendChild(top);
  const contexts = node('div', 'contexts');
  if (!item.contexts.length) {
    contexts.appendChild(node('div', 'sub', '（无检索上下文）'));
  }
  item.contexts.forEach((text, index) => {
    const details = node('details', 'ctx');
    const summary = node('summary', '', '片段 ' + (index + 1) + ' · ' + text.length + ' 字');
    const body = node('div', 'ctx-body', text);
    details.append(summary, body);
    contexts.appendChild(details);
  });
  card.appendChild(contexts);
  return card;
}
function rubricCard() {
  const details = node('details', 'rubric card');
  details.appendChild(node('summary', '', '评分标准（先读；1–5 分，可标 N/A）'));
  const intro = node('p', 'sub', '按指标定义评分，不要参考 LLM judge 分数。纯拒答没有事实性陈述，Faithfulness 可记 N/A。');
  details.appendChild(intro);
  const table = node('table');
  const head = node('tr');
  ['分数', '含义'].forEach(text => { const th = node('th', '', text); head.appendChild(th); });
  table.appendChild(head);
  [['5', '完全满足指标定义，无需修改'], ['4', '轻微问题，不影响可用性'], ['3', '部分满足，有明确缺口'], ['2', '明显不足，可能误导'], ['1', '基本不满足'], ['N/A', '该指标不适用']].forEach(row => {
    const tr = node('tr');
    row.forEach(text => tr.appendChild(node('td', '', text)));
    table.appendChild(tr);
  });
  details.appendChild(table);
  METRICS.forEach(metric => {
    const p = node('p');
    p.appendChild(node('strong', '', metric.label + '：'));
    p.appendChild(node('span', '', metric.hint));
    details.appendChild(p);
  });
  const note = node('p', 'sub', 'Context Precision 只看检索片段的相关性与排序；Context Recall 只看上下文是否覆盖标准答案要点，不看最终回答。');
  details.appendChild(note);
  return details;
}
function ratingBlock(item) {
  const card = node('section', 'card');
  card.appendChild(node('h3', '', '评分'));
  METRICS.forEach(metric => {
    const fieldset = node('fieldset', 'metric');
    fieldset.appendChild(node('legend', '', metric.label));
    fieldset.appendChild(node('p', 'hint', metric.hint));
    const options = node('div', 'options');
    [['5','完全满足'], ['4','轻微问题'], ['3','部分满足'], ['2','明显不足'], ['1','不满足'], ['na','N/A 不适用']].forEach(([value, label]) => {
      const wrapper = node('label', value === 'na' ? 'na' : '');
      const input = document.createElement('input');
      input.type = 'radio';
      input.name = metric.key;
      input.value = value;
      const rating = ensure(item.id)[metric.key];
      if (String(rating) === value) input.checked = true;
      input.addEventListener('change', () => {
        ensure(item.id)[metric.key] = value === 'na' ? 'na' : Number(value);
        saveState(); renderList(); updateProgress();
      });
      wrapper.append(input, document.createTextNode((value === 'na' ? '' : value + ' ') + label));
      options.appendChild(wrapper);
    });
    fieldset.appendChild(options);
    card.appendChild(fieldset);
  });
  const noteLabel = node('label', 'sub', '备注（可选）');
  const note = document.createElement('textarea');
  note.placeholder = '记录判断依据、疑问或无法判断的原因';
  note.value = ensure(item.id).note || '';
  note.addEventListener('input', () => { ensure(item.id).note = note.value; saveState(); });
  card.append(noteLabel, note);
  return card;
}
function renderPanel() {
  const panel = document.getElementById('panel');
  panel.textContent = '';
  const item = DATA.items[current];
  ensure(item.id);
  const head = node('section', 'card');
  const kicker = node('div', 'kicker');
  kicker.appendChild(node('span', '', '第 ' + (current + 1) + ' / ' + DATA.items.length + ' 条 · 盲评样本 ' + item.id));
  kicker.appendChild(node('span', '', isComplete(item.id) ? '已完成' : '未完成'));
  head.append(kicker, node('h2', '', item.question));
  panel.appendChild(head);
  panel.appendChild(addCard('系统回答', node('div', 'answer', item.answer || '（空回答）')));
  panel.appendChild(contextCard(item));
  panel.appendChild(addCard('标准答案', node('div', 'gt', item.ground_truth || '（无标准答案）')));
  panel.appendChild(rubricCard());
  panel.appendChild(ratingBlock(item));
  const nav = node('div', 'nav');
  const prev = node('button', '', '← 上一条'); prev.type = 'button';
  const next = node('button', 'primary', '下一条 →'); next.type = 'button';
  prev.disabled = current === 0;
  next.disabled = current === DATA.items.length - 1;
  prev.addEventListener('click', () => { if (current > 0) { current -= 1; renderAll(); window.scrollTo({top:0,behavior:'smooth'}); } });
  next.addEventListener('click', () => { if (current < DATA.items.length - 1) { current += 1; renderAll(); window.scrollTo({top:0,behavior:'smooth'}); } });
  nav.append(prev, next);
  panel.appendChild(nav);
}
function renderAll() {
  renderList();
  updateProgress();
  renderPanel();
}
function exportRatings() {
  const incomplete = DATA.items.filter(item => !isComplete(item.id));
  if (incomplete.length && !confirm('还有 ' + incomplete.length + ' 条未完成。仍要导出吗？')) return;
  const payload = {
    schema_version: 1,
    sheet_id: DATA.metadata.sheet_id,
    rater: (state.rater || '').trim(),
    completed_at: new Date().toISOString(),
    sheet: {n: DATA.items.length, seed: DATA.metadata.seed, sheet_id: DATA.metadata.sheet_id, created_at: DATA.metadata.created_at},
    items: DATA.items.map(item => ({
      id: item.id,
      ratings: {
        faithfulness: ensure(item.id).faithfulness,
        answer_relevancy: ensure(item.id).answer_relevancy,
        context_precision: ensure(item.id).context_precision,
        context_recall: ensure(item.id).context_recall
      },
      note: ensure(item.id).note || ''
    }))
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], {type: 'application/json'});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  const stamp = new Date().toISOString().slice(0, 19).replace(/[T:]/g, '-');
  link.download = 'calibration_ratings_' + stamp + '.json';
  link.click();
  URL.revokeObjectURL(link.href);
}
document.getElementById('rater').value = state.rater || '';
document.getElementById('rater').addEventListener('input', event => { state.rater = event.target.value; saveState(); });
document.getElementById('export').addEventListener('click', exportRatings);
document.getElementById('reset').addEventListener('click', () => {
  if (!confirm('清空本机保存的全部评分？此操作不可撤销。')) return;
  try { localStorage.removeItem(STORAGE_KEY); } catch (err) {}
  state = blankState(); current = 0;
  document.getElementById('rater').value = '';
  renderAll();
});
renderAll();
</script>
</body>
</html>
"""
    return template.replace("__N__", str(blind["metadata"]["n"])).replace("__DATA__", data)


def write_calibration_files(calibration: dict, output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    blind = blind_payload(calibration)
    paths = {
        "blind": output / "calibration_blind.json",
        "key": output / "calibration_key.json",
        "sheet": output / "calibration_sheet.html",
        "rubric": output / "calibration_rubric.md",
        "template": output / "calibration_ratings_template.json",
    }
    _write_json(paths["blind"], blind)
    _write_json(paths["key"], calibration)
    _write_json(paths["template"], ratings_template(calibration))
    paths["sheet"].write_text(render_sheet_html(blind), encoding="utf-8")
    paths["rubric"].write_text(render_rubric_markdown(), encoding="utf-8")
    return {key: str(value) for key, value in paths.items()}


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
            end += 1
        rank = (index + end) / 2 + 1
        for position in range(index, end + 1):
            ranks[order[position]] = rank
        index = end + 1
    return ranks


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    var_x = sum((a - mean_x) ** 2 for a in x)
    var_y = sum((b - mean_y) ** 2 for b in y)
    if var_x == 0 or var_y == 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def spearman(x: list[float], y: list[float], permutations: int = 2000, seed: int = 0) -> dict:
    """Spearman 秩相关；零方差时 rho 为 None，p 值用置换检验近似。"""
    if len(x) != len(y):
        raise ValueError("Spearman 两组数据长度不一致")
    if len(x) < 3:
        return {"rho": None, "p_value": None, "n": len(x), "reason": "样本少于 3 条"}
    rho = _pearson(_ranks(x), _ranks(y))
    if rho is None:
        return {"rho": None, "p_value": None, "n": len(x), "reason": "至少一组分数无方差"}
    observed = abs(rho)
    rng = random.Random(seed)
    shuffled = list(y)
    hits = 0
    for _ in range(permutations):
        rng.shuffle(shuffled)
        candidate = _pearson(_ranks(x), _ranks(shuffled))
        if candidate is not None and abs(candidate) >= observed:
            hits += 1
    return {
        "rho": round(rho, 4),
        "p_value": round((hits + 1) / (permutations + 1), 4),
        "n": len(x),
        "permutations": permutations,
    }


def cohen_kappa(a: list[int], b: list[int]) -> dict:
    """二分类 Cohen's Kappa；某一方全为同一类时标记 degenerate。"""
    if len(a) != len(b):
        raise ValueError("Kappa 两组标签长度不一致")
    if not a:
        return {"kappa": None, "n": 0, "reason": "没有可配对样本"}
    n = len(a)
    observed = sum(1 for left, right in zip(a, b) if left == right) / n
    pa = sum(a) / n
    pb = sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    kappa = None if expected == 1 else (observed - expected) / (1 - expected)
    counts = {
        "both_pass": sum(1 for left, right in zip(a, b) if left == 1 and right == 1),
        "judge_pass_human_fail": sum(1 for left, right in zip(a, b) if left == 1 and right == 0),
        "judge_fail_human_pass": sum(1 for left, right in zip(a, b) if left == 0 and right == 1),
        "both_fail": sum(1 for left, right in zip(a, b) if left == 0 and right == 0),
    }
    return {
        "kappa": None if kappa is None else round(kappa, 4),
        "observed_agreement": round(observed, 4),
        "expected_agreement": round(expected, 4),
        "n": n,
        "counts": counts,
        "degenerate": len(set(a)) == 1 or len(set(b)) == 1,
    }


def _human_rating(value) -> int | None:
    if value in (None, "", "na", "NA", "N/A"):
        return None
    try:
        rating = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"人工评分必须是 1–5 或 N/A，当前为 {value!r}") from exc
    if not 1 <= rating <= 5:
        raise ValueError(f"人工评分必须在 1–5 之间，当前为 {rating}")
    return rating


def score_calibration(
    ratings: dict,
    calibration: dict,
    judge_threshold: float = JUDGE_PASS,
    human_pass_threshold: int = HUMAN_PASS,
) -> dict:
    """按指标把人工 1–5 分与 judge 0–1 分配对，计算 Spearman 与二分类 Kappa。"""
    expected_sheet = (calibration.get("metadata") or {}).get("sheet_id")
    actual_sheet = ratings.get("sheet_id")
    if expected_sheet and actual_sheet and actual_sheet != expected_sheet:
        raise ValueError(
            f"评分文件与抽样表不匹配：sheet_id {actual_sheet} ≠ {expected_sheet}，请确认导出的评分文件对应本次抽样"
        )
    key_items = {item["id"]: item for item in calibration.get("items", [])}
    rating_items = {item["id"]: item for item in ratings.get("items", [])}
    missing = sorted(set(key_items) - set(rating_items))
    unknown = sorted(set(rating_items) - set(key_items))
    if missing:
        raise ValueError(f"评分文件缺少样本：{', '.join(missing)}")
    if unknown:
        raise ValueError(f"评分文件包含未知样本：{', '.join(unknown)}")

    result = {
        "rater": ratings.get("rater") or "anonymous",
        "completed_at": ratings.get("completed_at"),
        "metadata": calibration.get("metadata", {}),
        "judge_threshold": judge_threshold,
        "human_pass_threshold": human_pass_threshold,
        "metrics": {},
    }
    for metric in METRICS:
        pairs = []
        mismatches = []
        for sample_id, key_item in key_items.items():
            judge = key_item["judge"][metric]
            judge_score = judge.get("score")
            human = _human_rating((rating_items[sample_id].get("ratings") or {}).get(metric))
            if judge.get("skipped") or judge_score is None or human is None:
                continue
            judge_score = float(judge_score)
            pairs.append((judge_score, human))
            judge_pass = judge_score >= judge_threshold
            human_pass = human >= human_pass_threshold
            if judge_pass != human_pass:
                mismatches.append(
                    {
                        "id": sample_id,
                        "question": key_item["question"],
                        "judge_score": judge_score,
                        "human_rating": human,
                        "judge_pass": judge_pass,
                        "human_pass": human_pass,
                        "tags": key_item.get("tags", []),
                    }
                )
        x = [pair[0] for pair in pairs]
        y = [pair[1] for pair in pairs]
        rho = spearman(x, y) if pairs else {"rho": None, "p_value": None, "n": 0, "reason": "没有可配对样本"}
        a = [1 if pair[0] >= judge_threshold else 0 for pair in pairs]
        b = [1 if pair[1] >= human_pass_threshold else 0 for pair in pairs]
        kappa = cohen_kappa(a, b) if pairs else {"kappa": None, "n": 0, "reason": "没有可配对样本"}
        result["metrics"][metric] = {
            "label": METRIC_LABELS[metric],
            "n": len(pairs),
            "judge_mean": round(sum(x) / len(x), 4) if x else None,
            "human_mean": round(sum(y) / len(y), 4) if y else None,
            "spearman": rho,
            "kappa": kappa,
            "judge_pass_rate": round(sum(a) / len(a), 4) if a else None,
            "human_pass_rate": round(sum(b) / len(b), 4) if b else None,
            "mismatches": mismatches,
        }
    return result


def _format(value, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_calibration_markdown(result: dict) -> str:
    metadata = result.get("metadata") or {}
    lines = [
        "# Judge 人工校准报告",
        "",
        f"- 评分人：{result.get('rater') or 'anonymous'}",
        f"- 完成时间：{result.get('completed_at') or '未记录'}",
        f"- 样本数：{metadata.get('n', 'N/A')}（{metadata.get('selection', '目的抽样')}）",
        f"- 数据集指纹：`{metadata.get('dataset_fingerprint') or 'N/A'}`",
        f"- 被测系统：`{metadata.get('system_version') or 'N/A'}`",
        f"- 判定阈值：judge ≥ {result.get('judge_threshold')} 为通过；人工 ≥ {result.get('human_pass_threshold')} 为通过",
        "",
        "## 汇总",
        "",
        "| 指标 | n | Spearman ρ | p | Cohen's κ | 一致率 | judge 通过率 | 人工通过率 | 假阳性/假阴性 | 备注 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for metric in METRICS:
        data = result["metrics"][metric]
        rho = data["spearman"].get("rho")
        kappa = data["kappa"].get("kappa")
        kappa_note = data["kappa"].get("reason") or ("退化（一方无方差）" if data["kappa"].get("degenerate") else "")
        if rho is None:
            rho_note = data["spearman"].get("reason", "")
            kappa_note = kappa_note or rho_note
        counts = data["kappa"].get("counts") or {}
        fp = counts.get("judge_pass_human_fail", 0)
        fn = counts.get("judge_fail_human_pass", 0)
        lines.append(
            "| {label} | {n} | {rho} | {p} | {kappa} | {agree} | {jp} | {hp} | {fp}/{fn} | {note} |".format(
                label=data["label"],
                n=data["n"],
                rho=_format(rho),
                p=_format(data["spearman"].get("p_value")),
                kappa=_format(kappa),
                agree=_format(data["kappa"].get("observed_agreement")) if data["kappa"].get("observed_agreement") is not None else "N/A",
                jp=_format(data["judge_pass_rate"]),
                hp=_format(data["human_pass_rate"]),
                fp=fp,
                fn=fn,
                note=kappa_note,
            )
        )
    lines.extend(["", "## 逐指标明细", ""])
    for metric in METRICS:
        data = result["metrics"][metric]
        lines.append(f"### {data['label']}")
        lines.append("")
        lines.append(
            f"- 可配对样本：{data['n']}；judge 均值 {_format(data['judge_mean'])}；人工均值 {_format(data['human_mean'])}"
        )
        if data["spearman"].get("rho") is None:
            lines.append(f"- Spearman：不可估（{data['spearman'].get('reason', '未知原因')}）")
        else:
            lines.append(
                f"- Spearman ρ = {_format(data['spearman']['rho'])}，置换检验 p = {_format(data['spearman'].get('p_value'))}（n={data['n']}）"
            )
        if data["kappa"].get("kappa") is None:
            reason = data["kappa"].get("reason") or (
                "一方判定无方差（退化）" if data["kappa"].get("degenerate") else "未知原因"
            )
            lines.append(f"- Cohen's κ：不可估（{reason}）")
        else:
            lines.append(
                f"- Cohen's κ = {_format(data['kappa']['kappa'])}，一致率 {_format(data['kappa'].get('observed_agreement'))}"
            )
            if data["kappa"].get("degenerate"):
                lines.append("- 注意：有一方判定全为同一类，κ 的数值不可作强度解读。")
        if data.get("human_pass_rate") == 1.0 and (data.get("judge_pass_rate") or 0) < 1.0:
            lines.append(
                "- 注意：人工判定全部通过而 judge 有未通过样本，Kappa 退化；所有不一致都是 judge 不通过/人工通过（假阴性），说明 judge 在该指标上更严格。"
            )
        if data["mismatches"]:
            lines.append("- 判定不一致样本：")
            for item in data["mismatches"]:
                direction = "judge 通过/人工不通过" if item["judge_pass"] else "judge 不通过/人工通过"
                lines.append(
                    f"  - `{item['id']}` {direction} · judge {item['judge_score']:.3f} · 人工 {item['human_rating']} · {item['question'][:80]}"
                )
        else:
            lines.append("- 判定不一致样本：无")
        lines.append("")
    lines.extend(
        [
            "## 解读与局限",
            "",
            "- 这是单评分人的一致性检查，不是人工金标；没有第二评分人，无法估计评分者间信度。",
            "- 25 条为目的抽样，低分边界样本被刻意多抽，不能用来估计总体均值或通过率。",
            "- 人工 1–5 分与 judge 0–1 分是不同量纲，Spearman 只衡量排序一致性；Kappa 衡量按阈值二分后的一致性。",
            "- 某一方分数/判定无方差时，Spearman 或 Kappa 不可估；报告会显式标出，不能用 0 代替。",
            "- 若 Faithfulness 的 judge 分数全部为 1.0，该指标的 Spearman 不可估，只能看一致率和假阳性样本。",
        ]
    )
    return "\n".join(lines) + "\n"
