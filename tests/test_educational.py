from tiktoken._educational import bpe_train

# GPT-2 style split pattern (same family used by train_simple_encoding).
_PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def test_bpe_train_stops_when_corpus_is_exhausted():
    # A small corpus cannot supply vocab_size - 256 merges. Training must stop
    # gracefully with the ranks learned so far, not raise
    # "ValueError: max() iterable argument is empty".
    ranks = bpe_train(data="hello world", vocab_size=300, pat_str=_PAT, visualise=None)

    # 256 byte tokens are always present; a few merges are learnable from the data.
    assert len(ranks) >= 256
    assert len(ranks) < 300  # corpus exhausted before reaching vocab_size
    # byte tokens must still be intact
    assert all(ranks[bytes([i])] == i for i in range(256))
