# -*- coding: utf-8 -*-
"""为数据集中 answer 为空的样本生成 RAG 回答。"""
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragas_eval import Settings
from ragas_eval.llm import LLMClient

logger = logging.getLogger("generate_answers")

PROMPT = """请根据以下上下文回答用户问题。
要求：
1. 只根据上下文回答，不要编造信息
2. 回答简洁明了
3. 如果上下文里没有答案，就说"根据现有信息无法回答"

上下文：
{contexts}

用户问题：{question}

回答："""


async def generate_answer(client: LLMClient, sample: dict) -> str:
    prompt = PROMPT.format(
        contexts="\n\n".join(sample["contexts"]), question=sample["question"]
    )
    return await client.chat(prompt)


async def run(args) -> int:
    path = Path(args.dataset)
    dataset = json.loads(path.read_text(encoding="utf-8"))
    settings = Settings.from_env(args.env_file)
    client = LLMClient(settings)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def process(sample: dict) -> str:
        if sample.get("answer") and not args.overwrite:
            return sample["answer"]
        async with semaphore:
            try:
                return await generate_answer(client, sample)
            except Exception as exc:
                logger.error("生成失败：%s（%s）", sample.get("question", "")[:40], exc)
                return ""

    try:
        answers = await asyncio.gather(*(process(sample) for sample in dataset))
    finally:
        await client.aclose()

    for sample, answer in zip(dataset, answers):
        sample["answer"] = answer
    path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")

    succeeded = sum(1 for answer in answers if answer)
    logger.info("完成：成功生成 %d/%d 条回答，已写回 %s", succeeded, len(dataset), path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="为评测数据集生成 RAG 回答")
    parser.add_argument("--dataset", default="data/eval_dataset.json")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有回答")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
