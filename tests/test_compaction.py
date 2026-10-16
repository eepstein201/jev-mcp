"""Chunking and per-chunk scoring behind jev_compact_context."""

from unittest.mock import MagicMock

import pytest

from jev_mcp.compaction import (
    CHUNK_MAX_CHARS,
    GENERIC_QUESTION,
    chunk_state,
    score_chunk,
)
from jev_mcp.provider import EngineUnavailableError


def test_chunk_state_makes_one_chunk_per_top_level_key():
    chunks = chunk_state({"a": "alpha", "b": {"n": 1}})
    assert chunks == ["a: alpha", 'b: {"n": 1}']


def test_chunk_state_splits_text_on_blank_lines_then_single_newlines():
    assert chunk_state({"text": "one\n\ntwo"}) == ["one", "two"]
    assert chunk_state({"text": "one\ntwo"}) == ["one\ntwo"]  # no blank line: one chunk


def test_chunk_state_slices_long_chunks_verbatim():
    text = "x" * (CHUNK_MAX_CHARS + 5)
    chunks = chunk_state({"text": text})
    assert chunks == [text[:CHUNK_MAX_CHARS], text[CHUNK_MAX_CHARS:]]


def test_chunk_state_returns_nothing_for_blank_text():
    assert chunk_state({"text": "  \n "}) == []


def test_chunk_state_json_encodes_non_dict_state():
    assert chunk_state(["a", "b"]) == ['["a", "b"]']


def test_generic_question_names_the_goal():
    prompt = GENERIC_QUESTION.format(goal="ship it")
    assert "directly relevant" in prompt and "'ship it'" in prompt


def test_score_chunk_returns_the_fast_tiers_noul_score():
    provider = MagicMock()
    provider.fast_provider.evaluate_batch.return_value = {"chunk_0": {"noul": 0.4}}
    assert score_chunk(provider, 0, "text", "Q?") == 0.4
    asked = provider.fast_provider.evaluate_batch.call_args[0]
    assert asked[0] == {"text": "text"} and asked[1][0].prompt == "Q?"


def test_score_chunk_uses_the_plain_provider_without_a_fast_tier():
    provider = MagicMock(spec=["evaluate_batch"])
    provider.evaluate_batch.return_value = {"chunk_3": {}}
    assert score_chunk(provider, 3, "t", "Q?") == 0.0


def test_score_chunk_raises_when_no_engine_answered():
    provider = MagicMock()
    provider.fast_provider.evaluate_batch.return_value = {}
    with pytest.raises(EngineUnavailableError):
        score_chunk(provider, 0, "t", "Q?")
