# -*- coding: utf-8 -*-
"""调用 Dify Chat App 生成系统回答与检索上下文，产出评测用数据集。

每条样本按 tags 路由到对应的 Dify 应用（policy / hotpotqa），
回答与 contexts 全部来自被测系统，而不是数据集里的参考答案。
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import httpx

logger = logging.getLogger("run_dify_system")

DEFAULT_APPS_FILE = Path(os.environ.get("DIFY_APPS_FILE", r"F:\agent项目\dify\rageval-apps.json"))
DOMAINS = ("policy", "hotpotqa")


def route_domain(sample: dict) -> str:
    tags = set(sample.get("tags") or [])
    matched = [d for d in DOMAINS if d in tags]
    if len(matched) != 1:
        raise ValueError(f"样本标签无法唯一确定应用：{sample.get('question', '')[:40]} tags={sorted(tags)}")
    return matched[0]


def load_apps(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    apps = config.get("apps") or {}
    missing = [d for d in DOMAINS if d not in apps]
    if missing:
        raise ValueError(f"应用配置缺少：{missing}")
    for domain, app in apps.items():
        if not app.get("api_key"):
            raise ValueError(f"{domain} 应用缺少 api_key")
    return {"base_url": config.get("base_url", "http://localhost:8080"), "apps": apps}


async def ask(client: httpx.AsyncClient, base_url: str, app: dict, question: str, user: str) -> dict:
    payload = {
        "inputs": {},
        "query": question,
        "response_mode": "blocking",
        "user": user,
        "auto_generate_name": False,
    }
    headers = {"Authorization": f"Bearer {app['api_key']}"}
    last_error = ""
    for attempt in range(3):
        started = time.perf_counter()
        try:
            response = await client.post(
                f"{base_url}/v1/chat-messages", json=payload, headers=headers
            )
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            latency_ms = round((time.perf_counter() - started) * 1000)
            if response.status_code == 200:
                body = response.json()
                metadata = body.get("metadata") or {}
                resources = metadata.get("retriever_resources") or []
                return {
                    "answer": (body.get("answer") or "").strip(),
                    "contexts": [r.get("content", "") for r in resources],
                    "resources": [
                        {
                            "score": r.get("score"),
                            "document_name": r.get("document_name"),
                            "segment_id": r.get("segment_id"),
                        }
                        for r in resources
                    ],
                    "message_id": body.get("message_id"),
                    "usage": body.get("metadata", {}).get("usage"),
                    "latency_ms": latency_ms,
                }
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            if response.status_code < 500:
                break
        logger.warning("第 %d 次调用失败（%s），重试中", attempt + 1, last_error)
        await asyncio.sleep(2 * (attempt + 1))
    raise RuntimeError(last_error)


async def run(args) -> int:
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    config = load_apps(Path(args.apps_file))
    base_url = config["base_url"]
    apps = config["apps"]
    semaphore = asyncio.Semaphore(args.concurrency)
    checkpoint = Path(args.output).with_suffix(".jsonl")
    completed: dict[str, dict] = {}
    if args.resume and checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            completed[item["question"]] = item
        logger.info("断点续跑：已完成 %d/%d 条", len(completed), len(dataset))

    started_at = datetime.now().isoformat(timespec="seconds")
    failures: list[dict] = []
    lock = asyncio.Lock()

    async with httpx.AsyncClient(timeout=args.timeout) as client:

        async def process(sample: dict) -> dict:
            question = sample["question"]
            if question in completed:
                return completed[question]
            domain = route_domain(sample)
            async with semaphore:
                try:
                    result = await ask(
                        client, base_url, apps[domain], question,
                        user="rageval-" + hashlib.sha1(question.encode()).hexdigest()[:12],
                    )
                except Exception as exc:
                    logger.error("失败：%s（%s）", question[:40], exc)
                    async with lock:
                        failures.append({"question": question, "domain": domain, "error": str(exc)})
                    return {
                        "question": question,
                        "answer": "",
                        "contexts": [],
                        "ground_truth": sample.get("ground_truth", ""),
                        "tags": list(sample.get("tags") or []),
                        "note": sample.get("note", ""),
                        "system": {"domain": domain, "error": str(exc)},
                    }
            record = {
                "question": question,
                "answer": result["answer"],
                "contexts": result["contexts"],
                "ground_truth": sample.get("ground_truth", ""),
                "tags": list(sample.get("tags") or []),
                "note": sample.get("note", ""),
                "system": {
                    "domain": domain,
                    "app_id": apps[domain]["app_id"],
                    "message_id": result["message_id"],
                    "latency_ms": result["latency_ms"],
                    "context_count": len(result["contexts"]),
                    "resources": result["resources"],
                    "usage": result["usage"],
                },
            }
            async with lock:
                with checkpoint.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info("完成：%s", question[:40])
            return record

        records = await asyncio.gather(*(process(sample) for sample in dataset))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "timestamp": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(args.dataset),
        "base_url": base_url,
        "apps": {domain: {"app_id": app["app_id"], "dataset_id": app.get("dataset_id")}
                 for domain, app in apps.items()},
        "samples": len(records),
        "failures": failures,
        "empty_retrieval": sum(1 for r in records if not r["contexts"]),
        "empty_answers": sum(1 for r in records if not r["answer"]),
        "avg_latency_ms": round(
            sum((r["system"].get("latency_ms") or 0) for r in records) / max(len(records), 1)
        ),
        "domain_counts": {d: sum(1 for r in records if r["system"].get("domain") == d) for d in DOMAINS},
    }
    meta_path = output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("系统输出：%s（失败 %d，空检索 %d，空回答 %d）",
                output, len(failures), meta["empty_retrieval"], meta["empty_answers"])
    if checkpoint.exists():
        checkpoint.unlink()
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="通过 Dify Chat App 生成系统回答")
    parser.add_argument("--dataset", default="data/eval_dataset.json")
    parser.add_argument("--output", default="data/system_dify_v1.json")
    parser.add_argument("--apps-file", default=str(DEFAULT_APPS_FILE))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
