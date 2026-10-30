# -*- coding: utf-8 -*-
"""
数据集转换脚本：把 Hugging Face 公开数据集转成项目需要的评测格式

支持的数据集：
1. HotpotQA - 多跳推理英文问答数据集
2. RAGBench - 专门的 RAG 评测数据集
3. 自定义 JSON 格式转换

使用方法：
    python scripts/convert_dataset.py --dataset hotpotqa --output data/hotpotqa_dataset.json --num 200
"""
import json
import argparse
import os
import sys


def convert_hotpotqa(output_path: str, num_samples: int = 200):
    """
    转换 HotpotQA 数据集
    
    HotpotQA 原始字段：
    - question: 问题
    - answer: 标准答案
    - context: 上下文（title + sentences 列表）
    - supporting_facts: 支撑证据
    
    转换为我们的格式：
    - question: 用户问题
    - answer: （留空，等你的RAG系统填）
    - contexts: 检索到的上下文片段
    - ground_truth: 标准答案
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("需要先安装 datasets 库：")
        print("  pip install datasets")
        return
    
    print("正在从 Hugging Face 加载 HotpotQA 数据集...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation[:{}]".format(num_samples))
    
    converted = []
    for item in dataset:
        # 提取上下文：把每篇文章的句子拼成一段
        contexts = []
        titles = item["context"]["title"]
        sentences_list = item["context"]["sentences"]
        
        for title, sents in zip(titles, sentences_list):
            context_text = "".join(sents)
            contexts.append(f"【{title}】{context_text}")
        
        converted.append({
            "question": item["question"],
            "answer": "",  # 留空，你的RAG系统跑完后填这里
            "contexts": contexts,
            "ground_truth": item["answer"],
        })
    
    _save_json(converted, output_path)
    print(f"转换完成！共 {len(converted)} 条样本，保存到: {output_path}")


def convert_ragbench(output_path: str, num_samples: int = 200):
    """
    转换 RAGBench 数据集（专门测 RAG 的）
    
    RAGBench 原始字段：
    - question: 问题
    - answer: 标准答案
    - contexts: 检索到的上下文
    - generated_answer: 模型生成的回答
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("需要先安装 datasets 库：")
        print("  pip install datasets")
        return
    
    print("正在从 Hugging Face 加载 RAGBench 数据集...")
    # RAGBench 有多个子集，选最常用的 deepformers 子集
    dataset = load_dataset("rungalileo/ragbench", "deepformers", split="test[:{}]".format(num_samples))
    
    converted = []
    for item in dataset:
        contexts = item.get("contexts", [])
        if isinstance(contexts, str):
            contexts = [contexts]
        
        converted.append({
            "question": item["question"],
            "answer": item.get("generated_answer", ""),  # 数据集里自带模型回答
            "contexts": contexts,
            "ground_truth": item["answer"],
        })
    
    _save_json(converted, output_path)
    print(f"转换完成！共 {len(converted)} 条样本，保存到: {output_path}")


def convert_local_json(input_path: str, output_path: str, question_field: str = "question",
                       answer_field: str = "answer", contexts_field: str = "contexts",
                       ground_truth_field: str = "ground_truth"):
    """
    通用本地 JSON 转换：自定义字段名映射
    
    适合你有自己格式的 JSON 文件，需要转换成标准格式
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    converted = []
    for item in data:
        converted.append({
            "question": item.get(question_field, ""),
            "answer": item.get(answer_field, ""),
            "contexts": item.get(contexts_field, []),
            "ground_truth": item.get(ground_truth_field, ""),
        })
    
    _save_json(converted, output_path)
    print(f"转换完成！共 {len(converted)} 条样本，保存到: {output_path}")


def _save_json(data: list, path: str):
    """保存 JSON 文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="RAG 评测数据集转换工具")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["hotpotqa", "ragbench", "local"],
                        help="数据集名称")
    parser.add_argument("--output", type=str, default="data/hotpotqa_dataset.json",
                        help="输出文件路径")
    parser.add_argument("--num", type=int, default=200,
                        help="转换样本数量（默认200条）")
    
    # 本地转换的额外参数
    parser.add_argument("--input", type=str, help="本地输入文件路径（dataset=local时用）")
    parser.add_argument("--question-field", type=str, default="question", help="问题字段名")
    parser.add_argument("--answer-field", type=str, default="answer", help="答案字段名")
    parser.add_argument("--contexts-field", type=str, default="contexts", help="上下文字段名")
    parser.add_argument("--ground-truth-field", type=str, default="ground_truth", help="标准答案字段名")
    
    args = parser.parse_args()
    
    if args.dataset == "hotpotqa":
        convert_hotpotqa(args.output, args.num)
    elif args.dataset == "ragbench":
        convert_ragbench(args.output, args.num)
    elif args.dataset == "local":
        if not args.input:
            print("错误：本地转换需要指定 --input 参数")
            return
        convert_local_json(
            args.input, args.output,
            args.question_field, args.answer_field,
            args.contexts_field, args.ground_truth_field
        )


if __name__ == "__main__":
    main()
