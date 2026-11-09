"""Ensure Python BPE helpers use complete regex matches, not capture groups."""

import pytest

import tiktoken
from tiktoken._educational import SimpleBytePairEncoding, bpe_train

PATTERNS = [
    (r" ?([a-z]+)|\s+|[^a-z\s]+", r" ?[a-z]+|\s+|[^a-z\s]+"),
    (r"(\w)(\w*)|\W+", r"\w\w*|\W+"),
    (r"((\w)+)|\W+", r"(?:\w)+|\W+"),
    (r"\w+|\W+", r"\w+|\W+"),
]


@pytest.mark.parametrize("pattern,reference", PATTERNS)
@pytest.mark.parametrize("text", ["ababa words 123!", " café 東 \n", "", "  !  "])
def test_python_encoding_matches_native_with_captures(pattern, reference, text):
    ranks = {bytes([byte]): byte for byte in range(256)}
    ranks.update({b"ab": 256, b"aba": 257, b" word": 258})
    native = tiktoken.Encoding(
        name="synthetic-captures", pat_str=pattern, mergeable_ranks=ranks, special_tokens={}
    )
    expected = native.encode(text)
    assert native.decode(expected) == text
    educational = SimpleBytePairEncoding.from_tiktoken(native)
    actual = educational.encode(text, visualise=None)
    assert actual == expected
    assert educational.decode(actual) == text
    assert native._encode_only_native_bpe(text) == expected


@pytest.mark.parametrize("pattern,reference", PATTERNS)
@pytest.mark.parametrize("vocab_size", [256, 260])
def test_training_uses_the_same_full_matches_as_non_capturing_pattern(
    pattern, reference, vocab_size
):
    data = "ababa words 123! ababa words 123! café 東 ababa"
    expected = bpe_train(data, vocab_size, reference, visualise=None)
    actual = bpe_train(data, vocab_size, pattern, visualise=None)
    assert actual == expected
    encoding = SimpleBytePairEncoding(pat_str=pattern, mergeable_ranks=actual)
    assert encoding.decode(encoding.encode(data, visualise=None)) == data
