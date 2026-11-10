"""Compare trainers: python -m scripts.benchmark_educational --help."""

import argparse
import collections
import gc
import platform
import random
import statistics
import time
import tracemalloc

import regex

from tiktoken._educational import bpe_train


def reference_bpe_train(data: str, vocab_size: int, pat_str: str) -> dict[bytes, int]:
    ranks = {bytes([i]): i for i in range(256)}
    words = [
        [bytes([b]) for b in word.encode("utf-8")]
        for word in regex.findall(pat_str, data)
    ]
    while len(ranks) < vocab_size:
        stats = collections.Counter()
        for word in words:
            for pair in zip(word[:-1], word[1:]):
                stats[pair] += 1
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


def measure(
    trainer, data: str, vocab_size: int, pat_str: str, repeat: int
) -> tuple[float, int, dict[bytes, int]]:
    kwargs = {"visualise": None} if trainer is bpe_train else {}
    # Warm up the regex cache before collecting timings.
    ranks = trainer(data, vocab_size, pat_str, **kwargs)
    timings = []
    for _ in range(repeat):
        gc.collect()
        start = time.perf_counter()
        result = trainer(data, vocab_size, pat_str, **kwargs)
        timings.append(time.perf_counter() - start)
        assert result == ranks
        del result

    # Allocation tracing adds overhead; measure peak memory in a separate run.
    gc.collect()
    tracemalloc.start()
    try:
        result = trainer(data, vocab_size, pat_str, **kwargs)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result == ranks
    return statistics.median(timings), peak, ranks


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--words", type=positive_int, default=500)
    parser.add_argument("--word-length", type=positive_int, default=40)
    parser.add_argument("--vocab-size", type=positive_int, default=512)
    parser.add_argument("--repeat", type=positive_int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.vocab_size < 256:
        parser.error("--vocab-size must be at least 256")

    rng = random.Random(args.seed)
    data = " ".join(
        "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(args.word_length))
        for _ in range(args.words)
    )
    pat_str = r"[^ ]+"
    print(f"Python {platform.python_version()}, {platform.system()} {platform.machine()}")
    print(
        f"words={args.words}, word_length={args.word_length}, vocab_size={args.vocab_size}, "
        f"seed={args.seed}, repeat={args.repeat} (median); memory measured separately"
    )
    results = []
    for name, trainer in (("reference", reference_bpe_train), ("optimized", bpe_train)):
        elapsed, peak, ranks = measure(trainer, data, args.vocab_size, pat_str, args.repeat)
        results.append((elapsed, ranks))
        print(f"{name}: {elapsed:.3f}s, peak={peak / 1024 / 1024:.1f} MiB")
    assert results[0][1] == results[1][1], "trainers produced different merge ranks"
    print(f"speedup: {results[0][0] / results[1][0]:.2f}x; merge ranks match")


if __name__ == "__main__":
    main()
