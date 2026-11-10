import collections
import random

import pytest
import regex

from tiktoken._educational import bpe_train


def _reference_bpe_train(data: str, vocab_size: int, pat_str: str) -> dict[bytes, int]:
    ranks = {bytes([i]): i for i in range(256)}
    words = [
        [bytes([b]) for b in word.encode("utf-8")]
        for word in regex.findall(pat_str, data)
    ]
    while len(ranks) < vocab_size:
        stats = collections.Counter(
            pair for word in words for pair in zip(word[:-1], word[1:])
        )
        most_common_pair = max(stats, key=lambda pair: stats[pair])
        token_bytes = most_common_pair[0] + most_common_pair[1]
        ranks[token_bytes] = len(ranks)
        new_words = []
        for word in words:
            new_word = []
            i = 0
            while i < len(word) - 1:
                if (word[i], word[i + 1]) == most_common_pair:
                    new_word.append(token_bytes)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            if i == len(word) - 1:
                new_word.append(word[i])
            new_words.append(new_word)
        words = new_words
    return ranks


def test_bpe_train_preserves_merge_order():
    ranks = bpe_train("abab baba", vocab_size=260, pat_str=r"[^ ]+", visualise=None)

    assert [(token, ranks[token]) for token in (b"ab", b"abab", b"bab", b"baba")] == [
        (b"ab", 256),
        (b"abab", 257),
        (b"bab", 258),
        (b"baba", 259),
    ]


def test_bpe_train_merges_overlapping_pairs_left_to_right():
    ranks = bpe_train("aaa aaa", vocab_size=258, pat_str=r"[^ ]+", visualise=None)

    assert ranks[b"aa"] == 256
    assert ranks[b"aaa"] == 257


def test_bpe_train_breaks_ties_in_corpus_order():
    ranks = bpe_train("bc ab", vocab_size=257, pat_str=r"[^ ]+", visualise=None)

    assert ranks[b"bc"] == 256


def test_bpe_train_updates_first_occurrence_after_a_merge():
    # Merging ac + b removes the earliest c + b. The next tie is won by ac + c.
    ranks = bpe_train("acb accb", vocab_size=259, pat_str=r"[^ ]+", visualise=None)

    assert list(ranks.items())[256:] == [(b"ac", 256), (b"acb", 257), (b"acc", 258)]


@pytest.mark.parametrize(
    "data,vocab_size",
    [("aaaaaaa aaaaa aaaa", 259), ("abababa bababab ababababa", 262)],
)
def test_bpe_train_matches_reference_with_overlapping_runs(data: str, vocab_size: int):
    pat_str = r"[^ ]+"
    assert bpe_train(data, vocab_size, pat_str, visualise=None) == _reference_bpe_train(
        data, vocab_size, pat_str
    )


@pytest.mark.parametrize("alphabet", ["abcde", "aé你🙂"])
@pytest.mark.parametrize("seed", [0, 7, 42])
@pytest.mark.parametrize("vocab_size", [260, 272])
def test_bpe_train_matches_reference(alphabet: str, seed: int, vocab_size: int):
    rng = random.Random(seed)
    data = " ".join(
        "".join(rng.choice(alphabet) for _ in range(rng.randint(4, 12)))
        for _ in range(24)
    )
    pat_str = r"[^ ]+"

    assert bpe_train(data, vocab_size, pat_str, visualise=None) == _reference_bpe_train(
        data, vocab_size, pat_str
    )
