# Note that there are more actual tests, they're just not currently public :-)

from typing import Callable

import hypothesis
import hypothesis.strategies as st
import pytest

import tiktoken

from .test_helpers import ENCODING_FACTORIES, MAX_EXAMPLES


def test_simple():
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
        for token in range(min(10_000, enc.max_token_value - 1)):
            assert enc.encode_single_token(enc.decode_single_token_bytes(token)) == token


def test_simple_repeated():
    enc = tiktoken.get_encoding("gpt2")
    assert enc.encode("0") == [15]
    assert enc.encode("00") == [405]
    assert enc.encode("000") == [830]
    assert enc.encode("0000") == [2388]
    assert enc.encode("00000") == [20483]
    assert enc.encode("000000") == [10535]
    assert enc.encode("0000000") == [24598]
    assert enc.encode("00000000") == [8269]
    assert enc.encode("000000000") == [10535, 830]
    assert enc.encode("0000000000") == [8269, 405]
    assert enc.encode("00000000000") == [8269, 830]
    assert enc.encode("000000000000") == [8269, 2388]
    assert enc.encode("0000000000000") == [8269, 20483]
    assert enc.encode("00000000000000") == [8269, 10535]
    assert enc.encode("000000000000000") == [8269, 24598]
    assert enc.encode("0000000000000000") == [25645]
    assert enc.encode("00000000000000000") == [8269, 10535, 830]


def test_simple_regex():
    enc = tiktoken.get_encoding("cl100k_base")
    assert enc.encode("rer") == [38149]
    assert enc.encode("'rer") == [2351, 81]
    assert enc.encode("today\n ") == [31213, 198, 220]
    assert enc.encode("today\n \n") == [31213, 27907]
    assert enc.encode("today\n  \n") == [31213, 14211]


def test_basic_encode():
    enc = tiktoken.get_encoding("r50k_base")
    assert enc.encode("hello world") == [31373, 995]

    enc = tiktoken.get_encoding("p50k_base")
    assert enc.encode("hello world") == [31373, 995]

    enc = tiktoken.get_encoding("cl100k_base")
    assert enc.encode("hello world") == [15339, 1917]
    assert enc.encode(" \x850") == [220, 126, 227, 15]


def test_encode_empty():
    enc = tiktoken.get_encoding("r50k_base")
    assert enc.encode("") == []


def test_encode_bytes():
    enc = tiktoken.get_encoding("cl100k_base")
    assert enc._encode_bytes(b" \xec\x8b\xa4\xed") == [62085]


def test_encode_surrogate_pairs():
    enc = tiktoken.get_encoding("cl100k_base")

    assert enc.encode("👍") == [9468, 239, 235]
    # surrogate pair gets converted to codepoint
    assert enc.encode("\ud83d\udc4d") == [9468, 239, 235]

    # lone surrogate just gets replaced
    assert enc.encode("\ud83d") == enc.encode("�")


@pytest.mark.parametrize("make_enc", ENCODING_FACTORIES)
def test_catastrophically_repetitive(make_enc: Callable[[], tiktoken.Encoding]):
    enc = make_enc()
    for c in ["^", "0", "a", "'s", " ", "\n"]:
        big_value = c * 10_000
        assert big_value == enc.decode(enc.encode(big_value))

        big_value = " " + big_value
        assert big_value == enc.decode(enc.encode(big_value))

        big_value = big_value + "\n"
        assert big_value == enc.decode(enc.encode(big_value))


# ====================
# Roundtrip
# ====================


@pytest.mark.parametrize("make_enc", ENCODING_FACTORIES)
def test_basic_roundtrip(make_enc):
    enc = make_enc()
    for value in (
        "hello",
        "hello ",
        "hello  ",
        " hello",
        " hello ",
        " hello  ",
        "hello world",
        "请考试我的软件！12345",
    ):
        assert value == enc.decode(enc.encode(value))
        assert value == enc.decode(enc.encode_ordinary(value))


@pytest.mark.parametrize("make_enc", ENCODING_FACTORIES)
@hypothesis.given(text=st.text())
@hypothesis.settings(deadline=None)
def test_hyp_roundtrip(make_enc: Callable[[], tiktoken.Encoding], text):
    enc = make_enc()

    assert text == enc.decode(enc.encode(text))


@pytest.mark.parametrize("make_enc", ENCODING_FACTORIES)
def test_single_token_roundtrip(make_enc: Callable[[], tiktoken.Encoding]):
    enc = make_enc()

    for token in range(enc.n_vocab):
        try:
            token_bytes = enc.decode_single_token_bytes(token)
        except KeyError:
            continue
        assert enc.encode_single_token(token_bytes) == token


# ====================
# Special tokens
# ====================


def test_special_token():
    enc = tiktoken.get_encoding("cl100k_base")

    eot = enc.encode_single_token("<|endoftext|>")
    assert eot == enc.eot_token
    fip = enc.encode_single_token("<|fim_prefix|>")
    fim = enc.encode_single_token("<|fim_middle|>")

    text = "<|endoftext|> hello <|fim_prefix|>"
    assert eot not in enc.encode(text, disallowed_special=())
    with pytest.raises(ValueError):
        enc.encode(text)
    with pytest.raises(ValueError):
        enc.encode(text, disallowed_special="all")
    with pytest.raises(ValueError):
        enc.encode(text, disallowed_special={"<|endoftext|>"})
    with pytest.raises(ValueError):
        enc.encode(text, disallowed_special={"<|fim_prefix|>"})

    text = "<|endoftext|> hello <|fim_prefix|> there <|fim_middle|>"
    tokens = enc.encode(text, disallowed_special=())
    assert eot not in tokens
    assert fip not in tokens
    assert fim not in tokens

    tokens = enc.encode(text, allowed_special="all", disallowed_special=())
    assert eot in tokens
    assert fip in tokens
    assert fim in tokens

    tokens = enc.encode(text, allowed_special="all", disallowed_special="all")
    assert eot in tokens
    assert fip in tokens
    assert fim in tokens

    tokens = enc.encode(text, allowed_special={"<|fim_prefix|>"}, disallowed_special=())
    assert eot not in tokens
    assert fip in tokens
    assert fim not in tokens

    tokens = enc.encode(text, allowed_special={"<|endoftext|>"}, disallowed_special=())
    assert eot in tokens
    assert fip not in tokens
    assert fim not in tokens

    tokens = enc.encode(text, allowed_special={"<|fim_middle|>"}, disallowed_special=())
    assert eot not in tokens
    assert fip not in tokens
    assert fim in tokens


@pytest.mark.parametrize("make_enc", ENCODING_FACTORIES)
@hypothesis.given(text=st.text())
@hypothesis.settings(deadline=None, max_examples=MAX_EXAMPLES)
def test_hyp_special_ordinary(make_enc, text: str):
    enc = make_enc()
    assert enc.encode_ordinary(text) == enc.encode(text, disallowed_special=())


# ====================
# Batch encoding
# ====================


@pytest.mark.parametrize("make_enc", ENCODING_FACTORIES)
def test_batch_encode(make_enc: Callable[[], tiktoken.Encoding]):
    enc = make_enc()
    text1 = "hello world"
    text2 = "goodbye world"

    assert enc.encode_batch([text1]) == [enc.encode(text1)]
    assert enc.encode_batch([text1, text2]) == [enc.encode(text1), enc.encode(text2)]

    assert enc.encode_ordinary_batch([text1]) == [enc.encode_ordinary(text1)]
    assert enc.encode_ordinary_batch([text1, text2]) == [
        enc.encode_ordinary(text1),
        enc.encode_ordinary(text2),
    ]


@pytest.mark.parametrize("make_enc", ENCODING_FACTORIES)
@hypothesis.given(batch=st.lists(st.text()))
@hypothesis.settings(deadline=None)
def test_hyp_batch_roundtrip(make_enc: Callable[[], tiktoken.Encoding], batch):
    enc = make_enc()

    encoded = enc.encode_batch(batch)
    assert encoded == [enc.encode(t) for t in batch]
    decoded = enc.decode_batch(encoded)
    assert decoded == batch
