# -*- coding: utf-8 -*-
"""评测引擎：并发调度、断点续跑、结果汇总与报告输出。"""
from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
from collections import Counter
from datetime import datetime
from hashlib import sha256
from pathlib import Path

from .cache import ResponseCache
from .config import Settings
from .llm import LLMClient
from .metrics.answer_relevancy import AnswerRelevancy
from .metrics.context_precision import ContextPrecision
from .metrics.context_recall import ContextRecall
from .metrics.faithfulness import Faithfulness
from .report import render_html

logger = logging.getLogger(__name__)

METRIC_CLASSES = {
    "faithfulness": Faithfulness,
    "answer_relevancy": AnswerRelevancy,
    "context_precision": ContextPrecision,
    "context_recall": ContextRecall,
}

TAG_ORDER = {"normal": 0, "boundary": 1, "adversarial": 2}


def _tag_sort_key(tag: str) -> tuple[int, str]:
    if tag == "untagged":
        return (99, tag)
    return (TAG_ORDER.get(tag, 50), tag)


def quality_gate(summary: dict, minimum: float) -> tuple[bool, float | None]:
    """所有指标的有效均分都不低于 minimum 时通过。"""
    scores = [
        data["avg_score"]
        for data in summary.values()
        if isinstance(data.get("avg_score"), (int, float))
    ]
    if not scores:
        return False, None
    lowest = min(scores)
    return lowest >= minimum, lowest


def sample_id(sample: dict) -> str:
    payload = json.dumps(
        {key: sample.get(key) for key in ("question", "answer", "contexts", "ground_truth")},
        ensure_ascii=False,
        sort_keys=True,
    )
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


def dataset_fingerprint(dataset: list[dict]) -> str:
    payload = json.dumps(dataset, ensure_ascii=False, sort_keys=True)
    return sha256(payload.encode("utf-8")).hexdigest()[:12]


class EvaluationEngine:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: LLMClient | None = None,
        metrics: dict | list[str] | None = None,
        cache: ResponseCache | None = None,
        max_concurrency: int | None = None,
        samples: int | None = None,
        checkpoint_dir: str | Path = "results/.in_progress",
    ):
        if settings is not None:
            self.settings = settings
        elif client is not None:
            self.settings = Settings(api_key="unused")
        else:
            self.settings = Settings.from_env()

        self.client = client or LLMClient(self.settings, cache=cache)
        self.metrics = self._build_metrics(metrics)
        self.max_concurrency = max_concurrency or self.settings.max_concurrency
        self.samples = samples if samples is not None else self.settings.samples
        if self.samples < 1:
            raise ValueError("samples 必须 >= 1")
        self.semaphore = asyncio.Semaphore(self.max_concurrency)
        self.checkpoint_dir = Path(checkpoint_dir)

    def _build_metrics(self, metrics) -> dict:
        if isinstance(metrics, dict):
            return dict(metrics)
        names = list(metrics) if metrics else list(METRIC_CLASSES)
        unknown = [name for name in names if name not in METRIC_CLASSES]
        if unknown:
            raise ValueError(f"未知指标：{unknown}，可选：{list(METRIC_CLASSES)}")
        return {name: METRIC_CLASSES[name](self.client) for name in names}

    def _sample_temperatures(self) -> list[float]:
        """第 1 次保持与历史单次口径一致的 temperature=0，其余用采样温度。"""
        if self.samples == 1:
            return [0.0]
        return [0.0] + [self.settings.sample_temperature] * (self.samples - 1)

    def _aggregate_samples(self, attempts: list[dict]) -> dict:
        if len(attempts) == 1:
            return attempts[0]

        sample_scores = []
        numeric = []
        base = None
        for attempt in attempts:
            score = attempt.get("score")
            valid = isinstance(score, (int, float)) and not isinstance(score, bool)
            sample_scores.append(round(float(score), 3) if valid else None)
            if valid:
                numeric.append(float(score))
                if base is None:
                    base = attempt

        if not numeric:
            reasons = "；".join(str(item.get("reason", "未知错误")) for item in attempts)
            return {
                "score": None,
                "reason": f"{len(attempts)} 次采样全部失败：{reasons}"[:500],
                "sample_scores": sample_scores,
                "score_std": None,
                "samples_ok": 0,
                "samples_failed": len(attempts),
            }

        mean = statistics.fmean(numeric)
        std = statistics.pstdev(numeric) if len(numeric) > 1 else None
        std_text = (
            f"σ {std:.3f}" if std is not None else "仅 1 次有效，无法估计 σ"
        )
        aggregated = dict(base)
        aggregated.update(
            {
                "score": round(mean, 3),
                "sample_scores": sample_scores,
                "score_std": round(std, 3) if std is not None else None,
                "samples_ok": len(numeric),
                "samples_failed": len(attempts) - len(numeric),
                "reason": (
                    f"{base.get('reason', '')}"
                    f"（{len(numeric)}/{len(attempts)} 次采样成功，均值 {mean:.3f}，{std_text}）"
                ),
            }
        )
        return aggregated

    async def evaluate_single(self, sample: dict) -> dict:
        results = {}
        temperatures = self._sample_temperatures()
        for name, metric in self.metrics.items():
            attempts = []
            for index, temperature in enumerate(temperatures, start=1):
                try:
                    attempts.append(
                        await metric.score(
                            sample["question"],
                            sample["answer"],
                            sample["contexts"],
                            sample.get("ground_truth", ""),
                            temperature=temperature,
                        )
                    )
                except Exception as exc:  # 单次采样异常不影响其他次
                    logger.warning("指标 %s 第 %d 次采样异常：%s", name, index, exc)
                    attempts.append({"score": None, "reason": f"评测异常: {exc}"})
            results[name] = self._aggregate_samples(attempts)

        return {
            "question": sample["question"],
            "answer": sample["answer"],
            "contexts": sample["contexts"],
            "ground_truth": sample.get("ground_truth", ""),
            "contexts_count": len(sample["contexts"]),
            "tags": list(sample.get("tags") or []),
            "metrics": results,
        }

    async def evaluate_dataset(
        self,
        dataset: list[dict],
        output_dir: str | Path = "results",
        *,
        resume: bool = True,
        dataset_name: str | None = None,
        system_version: str | None = None,
        html: bool = True,
        threshold: float = 0.6,
        checkpoint_dir: str | Path | None = None,
    ) -> dict:
        if not dataset:
            raise ValueError("数据集为空")

        checkpoint_dir = Path(checkpoint_dir or self.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = dataset_fingerprint(dataset)
        checkpoint_name = (
            f"{fingerprint}.jsonl" if self.samples == 1 else f"{fingerprint}_s{self.samples}.jsonl"
        )
        checkpoint_path = checkpoint_dir / checkpoint_name

        current_ids = {sample_id(sample) for sample in dataset}
        completed = {}
        if resume:
            completed = {
                key: value
                for key, value in self._load_checkpoint(checkpoint_path).items()
                if key in current_ids
            }
            if completed:
                logger.info("断点续跑：已完成 %d/%d 条样本", len(completed), len(dataset))

        started = time.perf_counter()
        done_count = 0
        write_lock = asyncio.Lock()
        mode = "a" if resume else "w"

        with open(checkpoint_path, mode, encoding="utf-8") as checkpoint_file:

            async def run_one(sample: dict) -> dict:
                nonlocal done_count
                sid = sample_id(sample)
                if sid in completed:
                    record = completed[sid]
                    if not record.get("tags"):
                        record["tags"] = list(sample.get("tags") or [])
                    return record
                async with self.semaphore:
                    record = await self._evaluate_single_safe(sample)
                async with write_lock:
                    checkpoint_file.write(
                        json.dumps({"sample_id": sid, "record": record}, ensure_ascii=False)
                        + "\n"
                    )
                    checkpoint_file.flush()
                    done_count += 1
                    logger.info(
                        "[%d/%d] %s",
                        len(completed) + done_count,
                        len(dataset),
                        sample["question"][:40],
                    )
                return record

            try:
                all_results = await asyncio.gather(*(run_one(sample) for sample in dataset))
            except (KeyboardInterrupt, asyncio.CancelledError):
                logger.warning("评测中断，断点已保存：%s", checkpoint_path)
                raise

        duration = time.perf_counter() - started
        summary = self._calculate_summary(all_results)
        tag_summary = self._calculate_tag_summary(all_results)
        problematic = self._find_problematic(all_results, threshold=threshold)

        metadata = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": dataset_name or "",
            "dataset_fingerprint": fingerprint,
            "judge_model": self.settings.model,
            "base_url": self.settings.base_url or "",
            "system_version": system_version or "unspecified",
            "samples": len(dataset),
            "resumed_samples": len(completed),
            "metrics": list(self.metrics),
            "max_concurrency": self.max_concurrency,
            "sampling": {
                "runs_per_metric": self.samples,
                "temperatures": self._sample_temperatures(),
            },
            "tag_distribution": dict(
                Counter(
                    tag
                    for result in all_results
                    for tag in (result.get("tags") or ["untagged"])
                )
            ),
            "duration_seconds": round(duration, 1),
            "usage": {
                **self.client.stats.as_dict(),
                "estimated_cost": self.client.estimated_cost(),
            },
        }
        report = {
            "metadata": metadata,
            "summary": summary,
            "tag_summary": tag_summary,
            "problematic_samples": problematic,
            "detailed_results": all_results,
        }

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = output_path / f"eval_report_{timestamp}.json"
        json_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("JSON 报告：%s", json_path)

        if html:
            html_path = output_path / f"eval_report_{timestamp}.html"
            html_path.write_text(render_html(report), encoding="utf-8")
            logger.info("HTML 报告：%s", html_path)

        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("断点文件已清理：%s", checkpoint_path)

        self._log_summary(summary)
        self._log_tag_summary(tag_summary)
        return report

    async def _evaluate_single_safe(self, sample: dict) -> dict:
        try:
            return await self.evaluate_single(sample)
        except Exception as exc:
            logger.exception("样本评测失败：%s", sample.get("question", "")[:40])
            return {
                "question": sample.get("question", ""),
                "answer": sample.get("answer", ""),
                "contexts": sample.get("contexts", []),
                "ground_truth": sample.get("ground_truth", ""),
                "contexts_count": len(sample.get("contexts") or []),
                "tags": list(sample.get("tags") or []),
                "metrics": {},
                "error": str(exc),
            }

    def _load_checkpoint(self, path: Path) -> dict:
        records = {}
        if not path.exists():
            return records
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and item.get("sample_id") and isinstance(item.get("record"), dict):
                records[item["sample_id"]] = item["record"]
        return records

    def _calculate_summary(self, results: list[dict]) -> dict:
        summary = {}
        for metric in self.metrics:
            scores = []
            per_sample_stds = []
            partial_failures = 0
            failed_draws = 0
            for result in results:
                cell = result.get("metrics", {}).get(metric, {})
                score = cell.get("score")
                if isinstance(score, (int, float)) and not isinstance(score, bool):
                    scores.append(float(score))
                    std = cell.get("score_std")
                    if isinstance(std, (int, float)) and not isinstance(std, bool):
                        per_sample_stds.append(float(std))
                ok = cell.get("samples_ok")
                failed = cell.get("samples_failed")
                if isinstance(failed, int) and failed > 0:
                    failed_draws += failed
                    if isinstance(ok, int) and ok > 0:
                        partial_failures += 1
            if scores:
                summary[metric] = {
                    "avg_score": round(statistics.fmean(scores), 3),
                    "min_score": round(min(scores), 3),
                    "max_score": round(max(scores), 3),
                    "std_dev": round(statistics.pstdev(scores), 3),
                    "sample_count": len(scores),
                    "failed_count": len(results) - len(scores),
                }
            else:
                summary[metric] = {
                    "avg_score": None,
                    "sample_count": 0,
                    "failed_count": len(results),
                }
            if self.samples > 1:
                summary[metric].update(
                    {
                        "sampling_std": (
                            round(statistics.fmean(per_sample_stds), 3)
                            if per_sample_stds
                            else None
                        ),
                        "sampling_std_samples": len(per_sample_stds),
                        "partial_failures": partial_failures,
                        "failed_draws": failed_draws,
                    }
                )
        return summary

    def _calculate_tag_summary(self, results: list[dict]) -> dict:
        groups: dict[str, list[dict]] = {}
        for result in results:
            for tag in result.get("tags") or ["untagged"]:
                groups.setdefault(tag, []).append(result)
        return {
            tag: self._calculate_summary(groups[tag])
            for tag in sorted(groups, key=_tag_sort_key)
        }

    def _find_problematic(self, results: list[dict], threshold: float = 0.6, limit: int = 20) -> list[dict]:
        entries = []
        for result in results:
            issues = []
            lowest = 1.0
            for metric_name, metric_result in result.get("metrics", {}).items():
                score = metric_result.get("score")
                if score is not None:
                    lowest = min(lowest, score)
                if score is not None and score < threshold:
                    issues.append(
                        f"{metric_name}: {score} ({metric_result.get('reason', '')})"
                    )
            if issues:
                entries.append(
                    (
                        lowest,
                        {
                            "question": result.get("question", ""),
                            "issues": issues,
                        },
                    )
                )
        entries.sort(key=lambda item: item[0])
        return [entry for _, entry in entries[:limit]]

    def _log_summary(self, summary: dict) -> None:
        logger.info("=" * 60)
        for metric, data in summary.items():
            sampling = ""
            if data.get("sampling_std") is not None:
                sampling = (
                    f" 采样σ={data['sampling_std']}"
                    f"/{data.get('sampling_std_samples', '?')}条"
                )
            if data.get("failed_draws"):
                sampling += f" 采样缺次={data['failed_draws']}"
            logger.info(
                "%-20s avg=%s std=%s%s (样本 %d，失败 %d)",
                metric,
                data.get("avg_score", "N/A"),
                data.get("std_dev", "N/A"),
                sampling,
                data.get("sample_count", 0),
                data.get("failed_count", 0),
            )
        logger.info("=" * 60)

    def _log_tag_summary(self, tag_summary: dict) -> None:
        if not tag_summary or set(tag_summary) == {"untagged"}:
            return
        logger.info("-" * 60)
        for tag, summary in tag_summary.items():
            parts = [
                f"{metric}={data.get('avg_score', 'N/A')}"
                for metric, data in summary.items()
            ]
            logger.info("分层 %-12s %s", tag, " ".join(parts))
        logger.info("-" * 60)
