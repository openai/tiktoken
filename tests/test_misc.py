import subprocess
import sys

import pytest

import tiktoken


def test_encoding_for_model():
    enc = tiktoken.encoding_for_model("gpt2")
    assert enc.name == "gpt2"
    enc = tiktoken.encoding_for_model("text-davinci-003")
    assert enc.name == "p50k_base"
    enc = tiktoken.encoding_for_model("text-davinci-edit-001")
    assert enc.name == "p50k_edit"
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo-0301")
    assert enc.name == "cl100k_base"
    enc = tiktoken.encoding_for_model("gpt-4")
    assert enc.name == "cl100k_base"
    enc = tiktoken.encoding_for_model("gpt-4o")
    assert enc.name == "o200k_base"
    enc = tiktoken.encoding_for_model("gpt-oss-120b")
    assert enc.name == "o200k_harmony"


def test_optional_blobfile_dependency():
    prog = """
import tiktoken
import sys
assert "blobfile" not in sys.modules
"""
    subprocess.check_call([sys.executable, "-c", prog])


def test_empty_special_token_rejected():
    ranks = {bytes([i]): i for i in range(256)}
    with pytest.raises(ValueError, match="non-empty"):
        tiktoken.Encoding(
            name="empty_special_token",
            pat_str=r"\S+|\s+",
            mergeable_ranks=ranks,
            special_tokens={"": 256},
        )


def test_encode_with_unknown_allowed_special_token():
    ranks = {bytes([i]): i for i in range(256)}
    enc = tiktoken.Encoding(
        name="no_special_tokens", pat_str=r"\S+|\s+", mergeable_ranks=ranks, special_tokens={}
    )
    assert enc.encode("hi", allowed_special={""}) == enc.encode_ordinary("hi")
