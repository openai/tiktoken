"""Regression tests for the GPT-6 model family encoding lookup.

These verify local tiktoken name resolution only, not API-side token usage.
"""

from unittest.mock import patch

import pytest

import tiktoken
from tiktoken.model import encoding_name_for_model


@pytest.mark.parametrize(
    "model",
    [
        "gpt-6",
        "gpt-6-astra",
        "gpt-6-sol",
        "gpt-6-luna",
        "gpt-6.1-sol",
        "gpt-6-astra-2026-10-09",
    ],
)
def test_gpt6_family_uses_o200k_base_name(model: str) -> None:
    assert encoding_name_for_model(model) == "o200k_base"


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5", "o200k_base"),
        ("gpt-5.6-sol", "o200k_base"),
        ("gpt-4o", "o200k_base"),
        ("gpt-4o-2024-05-13", "o200k_base"),
        ("gpt-oss-120b", "o200k_harmony"),
        ("gpt-4", "cl100k_base"),
        ("gpt2", "gpt2"),
    ],
)
def test_existing_model_encoding_mappings_are_unchanged(
    model: str, expected: str
) -> None:
    assert encoding_name_for_model(model) == expected


def test_encoding_for_model_uses_the_lookup_without_new_encodings() -> None:
    expected = object()
    with patch("tiktoken.model.get_encoding", return_value=expected) as get_encoding:
        assert tiktoken.encoding_for_model("gpt-6-astra") is expected
        get_encoding.assert_called_once_with("o200k_base")


def test_unknown_future_family_is_still_rejected() -> None:
    with pytest.raises(KeyError, match="Could not automatically map"):
        encoding_name_for_model("gpt-7-astra")


def test_gpt6_lookup_uses_the_existing_local_encoder() -> None:
    new = tiktoken.encoding_for_model("gpt-6-astra")
    old = tiktoken.encoding_for_model("gpt-5")
    assert new is old
    for text in (
        "hello world",
        "Hello, I want to run the tests.",
        "日本語の文章を解析します。",
        "Пример проверки токенов.",
    ):
        encoded = new.encode(text)
        assert encoded == old.encode(text)
        assert new.decode(encoded) == text
