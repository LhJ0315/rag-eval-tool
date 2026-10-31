# -*- coding: utf-8 -*-
"""识别拒答与“知识库没有提到”这类元陈述。

Faithfulness 只该验证关于世界的事实 claim。拒答本身不是事实陈述，
“上下文没有提供 X”也不是对世界的断言；把它们当 claim 会让同一句
拒答在 0 与 1 之间随机跳变。这里用句级规则把它们过滤掉。
"""
import re

_SENTENCE_SPLIT = re.compile(
    r"(?<=[。！？；!?;])\s*|\n+|(?<=\.)\s+(?=[\"“'A-Z0-9])"
)

_PATTERNS = [
    re.compile(
        r"\b(?:i|we)\s+(?:cannot|can't|could not|couldn't|am unable to|are unable to|"
        r"do not|don't|did not|didn't)\s+"
        r"(?:find|answer|determine|locate|know|confirm|tell|provide)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:the\s+)?(?:knowledge base|retrieved context|provided context|"
        r"provided information|available information|the context)\b[^.!?。！？]{0,80}"
        r"\b(?:does not|doesn't|did not|didn't|has no|have no|contains no|"
        r"is not|isn't)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:there is|there's)\s+no\s+(?:information|mention|evidence|data)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:is|are|was|were)\s+not\s+(?:explicitly\s+)?"
        r"(?:stated|mentioned|specified|provided|included)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bnot (?:explicitly )?(?:stated|mentioned|provided|specified)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:知识库|上下文|提供的?(?:资料|信息|内容)|现有(?:信息|资料)|"
        r"以上(?:资料|信息)|检索结果)[^。！？；\n]{0,40}"
        r"(?:没有|未|无法|不能|找不到|查不到|不含|不包含)"
    ),
    re.compile(r"(?:无法|不能)(?:据此|根据[^。！？；\n]{0,20})?(?:回答|确认|确定|判断)"),
    re.compile(
        r"(?:没有|未能|无法)(?:找到|提供|提及|提到|说明|给出)"
        r"[^。！？；\n]{0,30}(?:信息|内容|规定|资料|答案)"
    ),
    re.compile(r"^(?:抱歉|对不起)?[，,\s]*(?:我)?(?:不知道|不清楚|不了解)"),
]


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT.split(text) if part and part.strip()]


def is_refusal_sentence(sentence: str) -> bool:
    return any(pattern.search(sentence) for pattern in _PATTERNS)


def filter_refusal_sentences(text: str) -> tuple[str, list[str]]:
    """返回（保留的事实句, 被过滤的拒答/元陈述句）。"""
    kept: list[str] = []
    removed: list[str] = []
    for sentence in split_sentences(text):
        (removed if is_refusal_sentence(sentence) else kept).append(sentence)
    return " ".join(kept), removed
