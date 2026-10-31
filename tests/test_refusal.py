# -*- coding: utf-8 -*-
from ragas_eval.refusal import filter_refusal_sentences, is_refusal_sentence


def test_english_pure_refusal_is_detected():
    assert is_refusal_sentence("I cannot find it in the knowledge base.")


def test_chinese_pure_refusals_are_detected():
    assert is_refusal_sentence("知识库中没有关于员工可否带宠物上班的规定，因此无法确认。")
    assert is_refusal_sentence("知识库中没有提供员工带薪病假天数的信息。")
    assert is_refusal_sentence("根据现有信息无法回答")


def test_meta_sentence_is_filtered_but_factual_part_is_kept():
    answer = (
        "The knowledge base does not explicitly state who formed the group; "
        "it identifies the album as by WINNER and released by YG Entertainment."
    )
    kept, removed = filter_refusal_sentences(answer)
    assert "WINNER" in kept
    assert "does not explicitly state" not in kept
    assert len(removed) == 1


def test_negative_answer_is_not_treated_as_refusal():
    answer = (
        "No, not both are described as used for real estate. "
        "Random House Tower is used as headquarters and as luxury apartments. "
        "The knowledge base does not state that it is used for real estate."
    )
    kept, removed = filter_refusal_sentences(answer)
    assert "No, not both are described" in kept
    assert "Random House Tower is used as headquarters" in kept
    assert len(removed) == 1
    assert "does not state" in removed[0]


def test_factual_negation_is_not_treated_as_refusal():
    assert not is_refusal_sentence("The film was not released in the United States.")
    assert not is_refusal_sentence("不可以。实习生不建立劳动关系，不能享受补充医疗保险。")


def test_empty_answer_has_no_removed_sentences():
    assert filter_refusal_sentences("") == ("", [])
