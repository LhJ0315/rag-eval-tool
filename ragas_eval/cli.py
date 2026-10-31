# -*- coding: utf-8 -*-
"""命令行入口：run / compare。"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from .cache import ResponseCache
from .compare import compare_reports, format_table, load_report
from .config import ConfigError, Settings
from .dataset import filter_by_tags, load_dataset, samples_from_report
from .engine import METRIC_CLASSES, EvaluationEngine, quality_gate


def _setup_logging(log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-eval", description="RAG 系统自动化评测工具")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="运行评测")
    run.add_argument("--dataset", required=True, help="评测数据集 JSON 路径")
    run.add_argument("--output-dir", default="results", help="报告输出目录")
    run.add_argument(
        "--metrics",
        default=",".join(METRIC_CLASSES),
        help=f"逗号分隔的指标名，可选：{', '.join(METRIC_CLASSES)}",
    )
    run.add_argument("--max-concurrency", type=int, default=None, help="并发样本数")
    run.add_argument(
        "--samples",
        type=int,
        default=None,
        help="每个指标重复采样次数（默认读 SAMPLES，1 为单次；稳定性分析建议 3）",
    )
    run.add_argument("--threshold", type=float, default=0.6, help="低分样本阈值")
    run.add_argument("--system-version", default=None, help="被测 RAG 系统版本号")
    run.add_argument(
        "--tags",
        default=None,
        help="只评测包含指定标签的样本，逗号分隔，如 normal,adversarial",
    )
    run.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只评测过滤后的前 N 条样本，适合 CI 冒烟",
    )
    run.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="所有指标均分低于该值时退出码为 1（CI 质量门禁）",
    )
    run.add_argument("--no-resume", action="store_true", help="忽略断点，从头评测")
    run.add_argument("--no-html", action="store_true", help="不生成 HTML 报告")
    run.add_argument("--no-cache", action="store_true", help="禁用响应缓存")
    run.add_argument("--env-file", default=".env", help=".env 文件路径")

    rejudge = sub.add_parser("rejudge", help="用当前指标版本重判已有报告中的指定指标")
    rejudge.add_argument("--report", required=True, help="已有评测报告 JSON")
    rejudge.add_argument("--output-dir", default="results/rejudge", help="报告输出目录")
    rejudge.add_argument(
        "--metrics", default="context_recall", help="逗号分隔的指标名（默认重判 context_recall）"
    )
    rejudge.add_argument("--max-concurrency", type=int, default=None, help="并发样本数")
    rejudge.add_argument("--samples", type=int, default=None, help="每个指标重复采样次数")
    rejudge.add_argument("--threshold", type=float, default=0.6, help="低分样本阈值")
    rejudge.add_argument("--system-version", default=None, help="覆盖报告中的被测系统版本")
    rejudge.add_argument("--no-html", action="store_true", help="不生成 HTML 报告")
    rejudge.add_argument("--no-cache", action="store_true", help="禁用响应缓存")
    rejudge.add_argument("--env-file", default=".env", help=".env 文件路径")

    compare = sub.add_parser("compare", help="对比两份评测报告")
    compare.add_argument("report_a", help="基线报告 JSON")
    compare.add_argument("report_b", help="新版本报告 JSON")
    compare.add_argument("--json", action="store_true", help="以 JSON 输出对比结果")
    return parser


def _prepare_dataset(dataset: list[dict], tags: str | None, limit: int | None) -> list[dict]:
    if tags:
        dataset = filter_by_tags(
            dataset, [tag for tag in tags.split(",") if tag.strip()]
        )
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit 必须为正整数")
        dataset = dataset[:limit]
    return dataset


async def _evaluate(args, settings: Settings, cache: ResponseCache | None) -> int:
    metrics = [name.strip() for name in args.metrics.split(",") if name.strip()]
    dataset = _prepare_dataset(load_dataset(args.dataset), args.tags, args.limit)
    engine = EvaluationEngine(
        settings,
        cache=cache,
        metrics=metrics,
        max_concurrency=args.max_concurrency,
        samples=args.samples,
    )
    try:
        report = await engine.evaluate_dataset(
            dataset,
            output_dir=args.output_dir,
            resume=not args.no_resume,
            dataset_name=Path(args.dataset).name,
            system_version=args.system_version,
            html=not args.no_html,
            threshold=args.threshold,
        )
    finally:
        await engine.client.aclose()
        if cache is not None:
            cache.close()

    if args.fail_under is not None:
        passed, lowest = quality_gate(report["summary"], args.fail_under)
        if lowest is None:
            print("质量门禁未通过：没有有效指标分数", file=sys.stderr)
            return 1
        if not passed:
            print(
                f"质量门禁未通过：最低指标均分 {lowest:.3f} < {args.fail_under}",
                file=sys.stderr,
            )
            return 1
        print(f"质量门禁通过：最低指标均分 {lowest:.3f} >= {args.fail_under}")
    return 0


async def _rejudge(args, settings: Settings, cache: ResponseCache | None) -> int:
    source = load_report(args.report)
    metadata = source.get("metadata") or {}
    dataset = samples_from_report(source)
    metrics = [name.strip() for name in args.metrics.split(",") if name.strip()]
    engine = EvaluationEngine(
        settings,
        cache=cache,
        metrics=metrics,
        max_concurrency=args.max_concurrency,
        samples=args.samples,
    )
    try:
        await engine.evaluate_dataset(
            dataset,
            output_dir=args.output_dir,
            dataset_name=metadata.get("dataset"),
            system_version=args.system_version or metadata.get("system_version"),
            html=not args.no_html,
            threshold=args.threshold,
            metadata_extra={
                "rejudged_from": str(Path(args.report)),
                "rejudged_metrics": metrics,
                "source_metric_versions": metadata.get("metric_versions"),
            },
        )
    finally:
        await engine.client.aclose()
        if cache is not None:
            cache.close()
    return 0


def _cmd_run(args) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    settings = Settings.from_env(args.env_file)
    cache = None if args.no_cache else ResponseCache(settings.cache_path)
    return asyncio.run(_evaluate(args, settings, cache))


def _cmd_rejudge(args) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_dir / f"rejudge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    settings = Settings.from_env(args.env_file)
    cache = None if args.no_cache else ResponseCache(settings.cache_path)
    return asyncio.run(_rejudge(args, settings, cache))


def _cmd_compare(args) -> int:
    result = compare_reports(load_report(args.report_a), load_report(args.report_b))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_table(result))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "rejudge":
            return _cmd_rejudge(args)
        return _cmd_compare(args)
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"参数错误：{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("已中断，断点已保留", file=sys.stderr)
        return 130
