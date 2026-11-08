import pytest

import tiktoken


def test_encode_with_unstable_surrogate_pairs():
    enc = tiktoken.Encoding(
        name="test",
        pat_str=r"(?s:.)",
        mergeable_ranks={bytes([i]): i for i in range(256)},
        special_tokens={},
    )

    for text in ["py\ud83d\udc4d", "py\ud83d"]:
        stable_tokens, _ = enc.encode_with_unstable(text)
        assert stable_tokens == enc.encode(text)[: len(stable_tokens)]


def test_encode_with_unstable_surrogate_pair_special_token():
    special = "\U0001f44d"
    surrogate_pair = "\ud83d\udc4d"
    enc = tiktoken.Encoding(
        name="test",
        pat_str=r"(?s:.)",
        mergeable_ranks={bytes([i]): i for i in range(256)},
        special_tokens={special: 256},
    )

    with pytest.raises(ValueError, match="disallowed special token"):
        enc.encode_with_unstable("py" + special)
    with pytest.raises(ValueError, match="disallowed special token"):
        enc.encode_with_unstable("py" + surrogate_pair)

    assert enc.encode_with_unstable(
        "py" + surrogate_pair, allowed_special={special}
    ) == enc.encode_with_unstable("py" + special, allowed_special={special})
