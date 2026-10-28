# -*- coding: utf-8 -*-
"""程序化调用示例：python examples/run_eval.py --dataset data/eval_dataset.json"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragas_eval import EvaluationEngine, Settings, load_dataset


async def main(dataset_path: str, output_dir: str) -> None:
    settings = Settings.from_env()
    engine = EvaluationEngine(settings)
    try:
        await engine.evaluate_dataset(
            load_dataset(dataset_path),
            output_dir=output_dir,
            system_version="example",
        )
    finally:
        await engine.client.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/eval_dataset.json")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    asyncio.run(main(args.dataset, args.output_dir))
