import pytest

import tiktoken
import tiktoken_ext.openai_public
from tiktoken.load import (
    _load_tiktoken_bpe_python,
    _load_tiktoken_bpe_core,
    load_tiktoken_bpe,
)


def test_load_tiktoken_bpe_rust_matches_python(tmp_path):
    bpe_file = tmp_path / "tiny.tiktoken"
    bpe_file.write_bytes(b"IQ== 0\nIg== 1\n4pyT 2\n")

    assert load_tiktoken_bpe(str(bpe_file)) == _load_tiktoken_bpe_python(str(bpe_file))


def test_load_tiktoken_bpe_core(tmp_path):
    bpe_file = tmp_path / "tiny.tiktoken"
    bpe_file.write_bytes(b"IQ== 0\nIg== 1\n4pyT 2\n")

    core_bpe, mergeable_ranks_len, mergeable_ranks_max_token_value = _load_tiktoken_bpe_core(
        str(bpe_file),
        special_tokens={"<|special|>": 3},
        pat_str=r""".+""",
    )

    assert mergeable_ranks_len == 3
    assert mergeable_ranks_max_token_value == 2
    assert core_bpe.encode_single_token(b"!") == 0
    assert core_bpe.encode_single_token("<|special|>".encode()) == 3


def test_load_tiktoken_bpe_parse_error_includes_source(tmp_path):
    bpe_file = tmp_path / "bad.tiktoken"
    bpe_file.write_bytes(b"IQ== 0 extra\n")

    with pytest.raises(ValueError, match="bad.tiktoken"):
        load_tiktoken_bpe(str(bpe_file))


def test_public_encoding_mergeable_ranks_materialize_lazily():
    enc = tiktoken.get_encoding("cl100k_base")

    assert enc.__dict__["_mergeable_ranks"] is None
    assert enc._mergeable_ranks[b"!"] == 0
    assert isinstance(enc.__dict__["_mergeable_ranks"], dict)


def test_extending_public_encoding_after_lazy_construction():
    base = tiktoken.get_encoding("cl100k_base")
    enc = tiktoken.Encoding(
        name="cl100k_test",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens={
            **base._special_tokens,
            "<|test|>": 100264,
        },
    )

    assert enc.encode("hello <|test|>", allowed_special="all") == [15339, 220, 100264]


def test_openai_public_constructor_private_core_path_still_constructs():
    constructor_args = tiktoken_ext.openai_public.cl100k_base()
    assert constructor_args["mergeable_ranks"][b"!"] == 0
    assert dict(constructor_args["mergeable_ranks"])[b"!"] == 0

    enc = tiktoken.Encoding(**tiktoken_ext.openai_public.cl100k_base())

    assert enc.encode("hello world") == [15339, 1917]
    assert enc.__dict__["_mergeable_ranks"] is None
