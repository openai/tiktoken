"""
benchmark_tail_latency.py — tail-latency harness for encode_ordinary_batch.

Measures median and worst-of-N wall-clock time across multiple synthetic
corpora and thread counts. Reproduces the measurement methodology from
https://github.com/openai/tiktoken/issues/530

Usage:
    python scripts/benchmark_tail_latency.py [--runs N] [--batch-size N] [--encoding NAME]

Example:
    python scripts/benchmark_tail_latency.py --runs 10 --batch-size 256 --encoding o200k_base
"""

from __future__ import annotations

import argparse
import statistics
import time

import tiktoken

# ---------------------------------------------------------------------------
# Synthetic corpora
# Each entry is (name, generator_fn).
# We generate corpora at runtime so the script ships without data files.
# ---------------------------------------------------------------------------

def _english_prose(n_chars: int) -> str:
    words = (
        "the quick brown fox jumps over the lazy dog "
        "OpenAI language model tokenizer benchmark test "
    ) * (n_chars // 80 + 1)
    return words[:n_chars]


def _python_source(n_chars: int) -> str:
    snippet = (
        "def encode_batch(texts, enc):\n"
        "    return [enc.encode_ordinary(t) for t in texts]\n\n"
        "class Tokenizer:\n"
        "    def __init__(self, model):\n"
        "        self.enc = tiktoken.get_encoding(model)\n\n"
    ) * (n_chars // 150 + 1)
    return snippet[:n_chars]


def _multilingual(n_chars: int) -> str:
    chars = (
        "日本語テスト "
        "Héllo wörld "
        "Привет мир "
        "مرحبا بالعالم "
        "🌍🚀✨ "
        "中文测试 "
    ) * (n_chars // 50 + 1)
    return chars[:n_chars]


def _random_bytes_as_latin1(n_chars: int) -> str:
    # Reproducible pseudo-random latin-1 safe bytes
    import random
    rng = random.Random(42)
    chars = [chr(rng.randint(32, 126)) for _ in range(n_chars)]
    return "".join(chars)


CORPORA: list[tuple[str, int]] = [
    ("english prose",      40_000),
    ("python source",      80_000),
    ("multilingual+emoji", 90_000),
    ("random ascii",      120_000),
]

CORPUS_GENERATORS = {
    "english prose":      _english_prose,
    "python source":      _python_source,
    "multilingual+emoji": _multilingual,
    "random ascii":       _random_bytes_as_latin1,
}

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    encoding_name: str,
    batch_size: int,
    runs: int,
    thread_counts: list[int],
) -> None:
    print(f"\nencoding: {encoding_name}  |  batch_size: {batch_size}  |  runs: {runs}")

    enc = tiktoken.get_encoding(encoding_name)
    # Warm up the encoding (loads vocab, compiles regex, etc.)
    enc.encode_ordinary_batch(["warmup"] * 4, num_threads=1)

    for num_threads in thread_counts:
        print(f"\n── num_threads={num_threads} " + "─" * 50)
        print(
            f"{'corpus':<24} {'tokens/batch':>14} "
            f"{'median ms':>10} {'worst ms':>10} {'worst/med':>10}"
        )
        print("-" * 74)

        for corpus_name, n_chars in CORPORA:
            text = CORPUS_GENERATORS[corpus_name](n_chars)
            docs = [text] * batch_size

            # Count tokens once
            sample_tokens = enc.encode_ordinary(text)
            tokens_per_batch = len(sample_tokens) * batch_size

            # Collect timing samples
            samples: list[float] = []
            for _ in range(runs):
                t0 = time.perf_counter()
                enc.encode_ordinary_batch(docs, num_threads=num_threads)
                samples.append(time.perf_counter() - t0)

            samples.sort()
            median_ms = statistics.median(samples) * 1000
            worst_ms  = samples[-1] * 1000
            ratio     = samples[-1] / statistics.median(samples)

            print(
                f"{corpus_name:<24} {tokens_per_batch:>14,} "
                f"{median_ms:>10.0f} {worst_ms:>10.0f} {ratio:>9.1f}x"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tail-latency benchmark for tiktoken encode_ordinary_batch"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of timed runs per corpus (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of documents per batch (default: 64)",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="o200k_base",
        help="tiktoken encoding name (default: o200k_base)",
    )
    parser.add_argument(
        "--threads",
        type=str,
        default="1,4,8",
        help="Comma-separated thread counts to benchmark (default: 1,4,8)",
    )
    args = parser.parse_args()

    thread_counts = [int(t) for t in args.threads.split(",")]

    run_benchmark(
        encoding_name=args.encoding,
        batch_size=args.batch_size,
        runs=args.runs,
        thread_counts=thread_counts,
    )


if __name__ == "__main__":
    main()