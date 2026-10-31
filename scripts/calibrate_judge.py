# -*- coding: utf-8 -*-
"""生成人工盲评表，并把人工评分与 judge 分数做一致性分析。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragas_eval.calibration import (
    DEFAULT_QUOTAS,
    HUMAN_PASS_BY_METRIC,
    JUDGE_PASS,
    build_calibration,
    render_calibration_markdown,
    score_calibration,
    write_calibration_files,
)


def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_command(args: argparse.Namespace) -> int:
    report = _load_json(args.report)
    calibration = build_calibration(report, n=args.n, seed=args.seed)
    paths = write_calibration_files(calibration, args.output_dir)
    metadata = calibration["metadata"]
    print(f"已抽取 {metadata['n']} 条盲评样本（seed={metadata['seed']}，sheet_id={metadata['sheet_id']}）")
    print(f"配额：{metadata['quotas']}（目标总配额 {sum(DEFAULT_QUOTAS.values())} 条时为默认值）")
    print("样本编号：" + ", ".join(item["id"] for item in calibration["items"]))
    print("输出：")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print("\n下一步：打开 calibration_sheet.html 逐条评分并导出 JSON。")
    print("calibration_key.json 含 judge 分数，评分完成前不要查看。")
    return 0


def _parse_pass_map(value: str) -> dict[str, int]:
    result = {}
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"--human-pass-map 需要 metric=score 格式，当前为 {chunk!r}")
        metric, raw = (part.strip() for part in chunk.split("=", 1))
        try:
            result[metric] = int(raw)
        except ValueError as exc:
            raise ValueError(f"--human-pass-map 的分数必须是整数，{metric} 当前为 {raw!r}") from exc
    return result


def score_command(args: argparse.Namespace) -> int:
    ratings = _load_json(args.ratings)
    calibration = _load_json(args.key)
    result = score_calibration(
        ratings,
        calibration,
        judge_threshold=args.judge_threshold,
        human_pass_threshold=args.human_pass,
        human_pass_by_metric=_parse_pass_map(args.human_pass_map) if args.human_pass_map else None,
    )
    markdown = render_calibration_markdown(result)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    json_output = output.with_suffix(".json")
    _write_json(json_output, result)
    print(markdown)
    print(f"报告：{output}")
    print(f"明细：{json_output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM judge 人工校准工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    make = subparsers.add_parser("make", help="从评测报告生成分层盲评表")
    make.add_argument("--report", required=True, help="评测报告 JSON")
    make.add_argument("--output-dir", default="results/calibration", help="输出目录")
    make.add_argument("--n", type=int, default=25, help="抽样条数（默认 25）")
    make.add_argument("--seed", type=int, default=20261031, help="抽样随机种子")
    make.set_defaults(handler=make_command)

    score = subparsers.add_parser("score", help="计算人工评分与 judge 的一致性")
    score.add_argument("--ratings", required=True, help="盲评页面导出的 JSON")
    score.add_argument("--key", required=True, help="make 生成的 calibration_key.json")
    score.add_argument("--output", default="results/calibration/calibration_report.md", help="Markdown 报告路径")
    score.add_argument("--judge-threshold", type=float, default=JUDGE_PASS, help="judge 二分类通过阈值")
    score.add_argument(
        "--human-pass",
        type=int,
        default=None,
        help="人工二分类通过阈值（1–5），指定后覆盖所有指标的默认值",
    )
    score.add_argument(
        "--human-pass-map",
        default=None,
        help="按指标指定人工通过阈值，如 context_precision=4,context_recall=4；未指定的指标用默认值",
    )
    score.set_defaults(handler=score_command)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.handler(args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
