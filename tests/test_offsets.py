from typing import Callable

import hypothesis
import pytest
from hypothesis import strategies as st

import tiktoken

from .test_helpers import MAX_EXAMPLES, SOME_ENCODING_FACTORIES


def _common_prefix_len(a, b):
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return i


def _token_offsets_reference(enc, tokens):
    text = enc.decode(tokens, errors="strict")
    res = []
    for i in range(len(tokens)):
        prefix = enc.decode(tokens[:i], errors="ignore")
        res.append(_common_prefix_len(text, prefix))
    return res


@pytest.mark.parametrize("make_enc", SOME_ENCODING_FACTORIES)
@hypothesis.given(data=st.data())
@hypothesis.settings(deadline=None, max_examples=MAX_EXAMPLES)
def test_hyp_offsets(make_enc: Callable[[], tiktoken.Encoding], data):
    enc = make_enc()

    tokens_st = st.lists(
        st.integers(0, enc.n_vocab - 1).filter(
            lambda x: x in enc._special_tokens.values() or x in enc._mergeable_ranks.values()
        ),
        min_size=1,
        max_size=20,
    )
    tokens = data.draw(tokens_st)

    # This is a dumb hack to make sure that our tokens are a valid UTF-8 string
    # We could potentially drop this, see the TODO in decode_with_offsets
    tokens = enc.encode(enc.decode(tokens, errors="ignore"), allowed_special="all")
    assert enc.decode_with_offsets(tokens)[1] == _token_offsets_reference(enc, tokens)


def test_basic_offsets():
    enc = tiktoken.get_encoding("cl100k_base")

    prompt = "hello world"
    p, o = enc.decode_with_offsets(enc.encode(prompt))
    assert p == prompt
    assert o == [0, 5]

    prompt = "hello world<|endoftext|> green cow"
    p, o = enc.decode_with_offsets(enc.encode(prompt, allowed_special="all"))
    assert p == prompt
    assert o == [0, 5, 11, 24, 30]

    prompt = "我非常渴望与人工智能一起工作"
    p, o = enc.decode_with_offsets(enc.encode(prompt))
    assert p == prompt
    assert o == [0, 1, 2, 3, 3, 4, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 13]

    # contains the interesting tokens b'\xe0\xae\xbf\xe0\xae' and b'\xe0\xaf\x8d\xe0\xae'
    # in which \xe0 is the start of a 3-byte UTF-8 character
    prompt = "நடிகர் சூர்யா"
    p, o = enc.decode_with_offsets(enc.encode(prompt))
    assert p == prompt
    assert o == [0, 0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 8, 8, 9, 9, 10, 11, 12, 12]

    # contains the interesting token b'\xa0\xe9\x99\xa4'
    # in which \xe9 is the start of a 3-byte UTF-8 character and \xa0 is a continuation byte
    prompt = " Ġ除"
    p, o = enc.decode_with_offsets(enc.encode(prompt))
    assert p == prompt
    assert o == [0, 1]
