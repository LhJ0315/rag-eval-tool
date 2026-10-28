# -*- coding: utf-8 -*-
from ragas_eval.llm import parse_json


def test_parse_plain_object():
    assert parse_json('{"a": 1}') == {"a": 1}


def test_parse_code_fence():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_prose_and_trailing_comma():
    text = '结果如下：\n{"claims": ["a", "b",],}\n以上。'
    assert parse_json(text) == {"claims": ["a", "b"]}


def test_parse_array():
    assert parse_json('```\n[1, 2, 3]\n```') == [1, 2, 3]


def test_parse_invalid_returns_none():
    assert parse_json("这不是 JSON") is None
    assert parse_json("") is None
