import pytest

import tiktoken


def test_list_encoding_names_includes_builtin_encodings():
    names = tiktoken.list_encoding_names()
    for name in ["gpt2", "r50k_base", "p50k_base", "p50k_edit", "cl100k_base", "o200k_base"]:
        assert name in names


def test_get_encoding_unknown_name():
    with pytest.raises(ValueError, match="Unknown encoding no_such_encoding"):
        tiktoken.get_encoding("no_such_encoding")


@pytest.mark.parametrize("bad_name", [None, 123, b"cl100k_base"])
def test_get_encoding_requires_str(bad_name):
    with pytest.raises(ValueError, match="Expected a string"):
        tiktoken.get_encoding(bad_name)  # type: ignore[arg-type]


def test_get_encoding_returns_cached_instance():
    assert tiktoken.get_encoding("cl100k_base") is tiktoken.get_encoding("cl100k_base")


@pytest.mark.parametrize(
    "model_name, encoding_name",
    [
        ("o1-mini", "o200k_base"),
        ("gpt-4o-2024-05-13", "o200k_base"),
        ("gpt-4-32k", "cl100k_base"),
        ("gpt-3.5-turbo-0301", "cl100k_base"),
        ("gpt-35-turbo-16k", "cl100k_base"),
        ("ft:gpt-4o-mini:my-org::abc123", "o200k_base"),
        ("ft:gpt-3.5-turbo:my-org::abc123", "cl100k_base"),
        ("text-embedding-3-small", "cl100k_base"),
    ],
)
def test_encoding_name_for_model_known(model_name, encoding_name):
    assert tiktoken.encoding_name_for_model(model_name) == encoding_name


@pytest.mark.parametrize("model_name", ["", "not-a-model", "claude-3-opus"])
def test_encoding_name_for_model_unknown(model_name):
    with pytest.raises(KeyError, match="Could not automatically map"):
        tiktoken.encoding_name_for_model(model_name)
    with pytest.raises(KeyError, match="Could not automatically map"):
        tiktoken.encoding_for_model(model_name)
