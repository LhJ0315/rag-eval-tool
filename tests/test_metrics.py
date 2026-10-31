# -*- coding: utf-8 -*-
import asyncio

from ragas_eval.config import Settings
from ragas_eval.metrics.answer_relevancy import AnswerRelevancy
from ragas_eval.metrics.context_precision import ContextPrecision
from ragas_eval.metrics.context_recall import ContextRecall
from ragas_eval.metrics.faithfulness import Faithfulness
from tests.helpers import FakeLLM


def test_faithfulness_scores_supported_claims():
    llm = FakeLLM(
        json_responses=[
            ({"claims": ["c1", "c2"]}, ""),
            ({"supported": [True, False]}, ""),
        ]
    )
    result = asyncio.run(Faithfulness(llm).score("q", "a", ["ctx"]))
    assert result["score"] == 0.5
    assert result["unsupported_claims"] == ["c2"]


def test_faithfulness_bad_shape_returns_none():
    llm = FakeLLM(
        json_responses=[({"claims": ["c1"]}, ""), ({"supported": [True, True]}, "")]
    )
    result = asyncio.run(Faithfulness(llm).score("q", "a", ["ctx"]))
    assert result["score"] is None


def test_faithfulness_empty_answer_is_failure_without_llm():
    result = asyncio.run(Faithfulness(FakeLLM()).score("q", "", ["ctx"]))
    assert result["score"] is None
    assert not result.get("skipped")
    assert "回答为空" in result["reason"]


def test_context_precision_is_position_weighted():
    # 相关段落出现在第 1、3 位：(1/1 + 2/3) / 2 = 0.833
    llm = FakeLLM(chat_responses=["Relevant", "Irrelevant", "Relevant"])
    result = asyncio.run(ContextPrecision(llm).score("q", "a", ["c1", "c2", "c3"]))
    assert result["score"] == 0.833
    assert result["relevant_contexts"] == 2


def test_context_precision_rewards_earlier_positions():
    late = FakeLLM(chat_responses=["Irrelevant", "Irrelevant", "Relevant"])
    early = FakeLLM(chat_responses=["Relevant", "Irrelevant", "Irrelevant"])
    late_score = asyncio.run(ContextPrecision(late).score("q", "a", ["c1", "c2", "c3"]))["score"]
    early_score = asyncio.run(ContextPrecision(early).score("q", "a", ["c1", "c2", "c3"]))["score"]
    assert early_score > late_score


def test_context_precision_unparseable_returns_none():
    llm = FakeLLM(chat_responses=["Maybe"])
    result = asyncio.run(ContextPrecision(llm).score("q", "a", ["c1"]))
    assert result["score"] is None


def test_answer_relevancy_uses_two_calls():
    llm = FakeLLM(
        json_responses=[
            ({"questions": ["q1", "q2", "q3"]}, ""),
            ({"similarities": [0.9, 0.5, 0.1]}, ""),
        ]
    )
    result = asyncio.run(AnswerRelevancy(llm).score("q", "a", ["ctx"]))
    assert result["score"] == 0.5
    assert result["method"] == "llm"
    assert len(llm.json_calls) == 2


def test_answer_relevancy_prefers_embeddings_when_configured():
    settings = Settings(api_key="test", embedding_model="text-embedding-test")
    llm = FakeLLM(
        json_responses=[({"questions": ["q1", "q2", "q3"]}, "")],
        embeddings=[[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]],
        settings=settings,
    )
    result = asyncio.run(AnswerRelevancy(llm).score("q", "a", ["ctx"]))
    assert result["method"] == "embedding"
    assert result["score"] == 0.333
    assert len(llm.embed_calls) == 1


def test_context_recall_scores_relation_level_coverage():
    llm = FakeLLM(
        json_responses=[
            (
                {
                    "points": [
                        {"point": "p1", "covered": True, "evidence": "ctx states p1"},
                        {"point": "p2", "covered": False, "evidence": "", "reason": "缺少关系"},
                    ]
                },
                "",
            )
        ]
    )
    result = asyncio.run(ContextRecall(llm).score("q", "a", ["ctx"], "truth"))
    assert result["score"] == 0.5
    assert result["metric_version"] == "relation-v2"
    assert result["missing_points"] == ["p2"]
    prompt = llm.json_calls[0]
    assert "关系级覆盖" in prompt
    assert "不要把上下文或常识中的其他事实拆成额外要点" in prompt
    assert "允许跨段组合" in prompt
    assert "不要用常识补全" in prompt
    assert "q" in prompt


def test_context_recall_requires_evidence_for_covered_points():
    llm = FakeLLM(
        json_responses=[
            ({"points": [{"point": "p1", "covered": True, "evidence": ""}]}, "")
        ]
    )
    result = asyncio.run(ContextRecall(llm).score("q", "a", ["ctx"], "truth"))
    assert result["score"] == 0.0
    assert result["points"][0]["reason"] == "标记为覆盖但未给出可引用的原文证据"


def test_context_recall_without_ground_truth_returns_none():
    result = asyncio.run(ContextRecall(FakeLLM()).score("q", "a", ["ctx"], ""))
    assert result["score"] is None


def test_faithfulness_skips_pure_refusal_without_llm():
    llm = FakeLLM()
    result = asyncio.run(
        Faithfulness(llm).score(
            "q", "I cannot find it in the knowledge base.", ["ctx"]
        )
    )
    assert result["score"] is None
    assert result["skipped"] is True
    assert result["refusal"] is True
    assert result["filtered_sentences"] == ["I cannot find it in the knowledge base."]
    assert llm.json_calls == []


def test_faithfulness_filters_meta_sentence_before_extraction():
    answer = (
        "The knowledge base does not explicitly state who formed the group; "
        "it identifies the album as by WINNER and released by YG Entertainment."
    )
    llm = FakeLLM(
        json_responses=[
            (
                {
                    "claims": [
                        "WINNER released the album through YG Entertainment.",
                        "The knowledge base does not explicitly state who formed the group.",
                    ]
                },
                "",
            ),
            ({"supported": [True]}, ""),
        ]
    )
    result = asyncio.run(Faithfulness(llm).score("q", answer, ["ctx"]))
    assert result["score"] == 1.0
    assert result["total_claims"] == 1
    assert len(result["filtered_sentences"]) == 1
    assert len(result["filtered_claims"]) == 1
    assert "WINNER" in llm.json_calls[0]
    assert "does not explicitly state" not in llm.json_calls[0]


def test_faithfulness_skips_when_only_meta_claims_remain():
    answer = (
        "The knowledge base does not state who formed the group; "
        "it identifies the album as by WINNER."
    )
    llm = FakeLLM(
        json_responses=[
            (
                {
                    "claims": [
                        "The knowledge base does not state who formed the group."
                    ]
                },
                "",
            )
        ]
    )
    result = asyncio.run(Faithfulness(llm).score("q", answer, ["ctx"]))
    assert result["score"] is None
    assert result["skipped"] is True
    assert result["refusal"] is True
    assert len(llm.json_calls) == 1


def test_faithfulness_keeps_negative_answer():
    answer = (
        "No, not both are described as used for real estate. "
        "Random House Tower is used as headquarters."
    )
    llm = FakeLLM(
        json_responses=[
            ({"claims": ["Random House Tower is used as headquarters."]}, ""),
            ({"supported": [True]}, ""),
        ]
    )
    result = asyncio.run(Faithfulness(llm).score("q", answer, ["ctx"]))
    assert result["score"] == 1.0
    assert "filtered_sentences" not in result


def test_faithfulness_passes_temperature_to_llm():
    llm = FakeLLM(
        json_responses=[
            ({"claims": ["c1"]}, ""),
            ({"supported": [True]}, ""),
        ]
    )
    result = asyncio.run(
        Faithfulness(llm).score("q", "a", ["ctx"], temperature=0.7)
    )
    assert result["score"] == 1.0
    assert llm.json_temperatures == [0.7, 0.7]


def test_context_precision_passes_temperature_to_llm():
    llm = FakeLLM(chat_responses=["Relevant"])
    asyncio.run(ContextPrecision(llm).score("q", "a", ["c1"], temperature=0.7))
    assert llm.chat_temperatures == [0.7]
