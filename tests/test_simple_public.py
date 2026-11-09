import subprocess
import sys

import pytest

import tiktoken


def test_simple():
    # Note that there are more actual tests, they're just not currently public :-)
    enc = tiktoken.get_encoding("gpt2")
    assert enc.encode("hello world") == [31373, 995]
    assert enc.decode([31373, 995]) == "hello world"
    assert enc.encode("hello <|endoftext|>", allowed_special="all") == [31373, 220, 50256]

    enc = tiktoken.get_encoding("cl100k_base")
    assert enc.encode("hello world") == [15339, 1917]
    assert enc.decode([15339, 1917]) == "hello world"
    assert enc.encode("hello <|endoftext|>", allowed_special="all") == [15339, 220, 100257]

    for enc_name in tiktoken.list_encoding_names():
        enc = tiktoken.get_encoding(enc_name)
        for token in range(10_000):
            assert enc.encode_single_token(enc.decode_single_token_bytes(token)) == token


def test_encoding_for_model():
    enc = tiktoken.encoding_for_model("gpt2")
    assert enc.name == "gpt2"
    enc = tiktoken.encoding_for_model("text-davinci-003")
    assert enc.name == "p50k_base"
    enc = tiktoken.encoding_for_model("text-davinci-edit-001")
    assert enc.name == "p50k_edit"
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo-0301")
    assert enc.name == "cl100k_base"


def test_optional_blobfile_dependency():
    prog = """
import tiktoken
import sys
assert "blobfile" not in sys.modules
"""
    subprocess.check_call([sys.executable, "-c", prog])


def test_decode_invalid_tokens():
    enc = tiktoken.get_encoding("cl100k_base")
    for bad in [-1, 2**32, 2**64, 99_999_999]:
        msg = f"Invalid token for decoding: {bad}"
        for fn in [
            lambda t: enc.decode([t]),
            lambda t: enc.decode_bytes([t]),
            lambda t: enc.decode_single_token_bytes(t),
            lambda t: enc.decode_tokens_bytes([t]),
            lambda t: enc.decode_with_offsets([t]),
        ]:
            with pytest.raises(KeyError) as exc_info:
                fn(bad)
            assert exc_info.value.args == (msg,)

    with pytest.raises(KeyError) as exc_info:
        enc.decode([15339, -1])
    assert exc_info.value.args == ("Invalid token for decoding: -1",)

    huge = 10**5000
    try:
        huge_repr = str(huge)
    except ValueError:
        huge_repr = "<integer too large to display>"
    huge_msg = f"Invalid token for decoding: {huge_repr}"

    for fn in [
        lambda t: enc.decode([t]),
        lambda t: enc.decode_bytes([t]),
        lambda t: enc.decode_single_token_bytes(t),
        lambda t: enc.decode_tokens_bytes([t]),
        lambda t: enc.decode_with_offsets([t]),
    ]:
        with pytest.raises(KeyError) as exc_info:
            fn(huge)
        assert exc_info.value.args == (huge_msg,)
