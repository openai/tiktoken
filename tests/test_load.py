import pytest

from tiktoken.load import (
    _load_tiktoken_bpe_python,
    load_tiktoken_bpe,
    load_tiktoken_bpe_with_core,
)


def test_load_tiktoken_bpe_rust_matches_python(tmp_path):
    bpe_file = tmp_path / "tiny.tiktoken"
    bpe_file.write_bytes(b"IQ== 0\nIg== 1\n4pyT 2\n")

    assert load_tiktoken_bpe(str(bpe_file)) == _load_tiktoken_bpe_python(str(bpe_file))


def test_load_tiktoken_bpe_with_core(tmp_path):
    bpe_file = tmp_path / "tiny.tiktoken"
    bpe_file.write_bytes(b"IQ== 0\nIg== 1\n4pyT 2\n")

    mergeable_ranks, core_bpe = load_tiktoken_bpe_with_core(
        str(bpe_file),
        special_tokens={"<|special|>": 3},
        pat_str=r""".+""",
    )

    assert mergeable_ranks == {b"!": 0, b'"': 1, "✓".encode(): 2}
    assert core_bpe.encode_single_token(b"!") == 0
    assert core_bpe.encode_single_token("<|special|>".encode()) == 3


def test_load_tiktoken_bpe_parse_error_includes_source(tmp_path):
    bpe_file = tmp_path / "bad.tiktoken"
    bpe_file.write_bytes(b"IQ== 0 extra\n")

    with pytest.raises(ValueError, match="bad.tiktoken"):
        load_tiktoken_bpe(str(bpe_file))
