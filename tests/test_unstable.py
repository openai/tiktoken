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
