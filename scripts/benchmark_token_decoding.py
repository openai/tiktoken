#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import tiktoken


def _tool_json(i: int) -> str:
    return json.dumps(
        {
            "tool": "lookup_order",
            "arguments": {
                "order_id": f"ord_{i:06d}",
                "urgent": i % 13 == 0,
                "destination": f"warehouse-{i % 17}",
            },
        },
        separators=(",", ":"),
    )


def make_workloads() -> dict[str, str]:
    chat = "\n".join(
        (
            f"user: customer {i} asked whether order ord_{i:06d} can be rerouted before "
            "the warehouse batch closes.\n"
            f"assistant: I checked the carrier window and found option {i % 5}."
        )
        for i in range(5_000)
    )
    unicode_notes = "\n".join(
        f"{i}: 我非常渴望与人工智能一起工作. நடிகர் சூர்யா. Ġ除."
        for i in range(5_000)
    )
    return {
        "tiny_lines_20k": "\n".join(f"hello world {i}" for i in range(20_000)),
        "chat_transcript_5k": chat,
        "tool_json_10k": "\n".join(_tool_json(i) for i in range(10_000)),
        "unicode_notes_5k": unicode_notes,
        "long_doc": ("The quick brown fox jumps over the lazy dog. " * 40_000),
    }


def _validate_decode_tokens_bytes(out: Any, tokens: list[int]) -> None:
    if len(out) != len(tokens):
        raise RuntimeError(f"expected {len(tokens)} token byte chunks, got {len(out)}")


def _validate_decode_with_offsets(out: Any, tokens: list[int]) -> None:
    text, offsets = out
    if not isinstance(text, str):
        raise RuntimeError("decode_with_offsets returned non-string text")
    if len(offsets) != len(tokens):
        raise RuntimeError(f"expected {len(tokens)} offsets, got {len(offsets)}")


def measure(
    name: str,
    fn: Callable[[list[int]], Any],
    tokens: list[int],
    reps: int,
    warmups: int,
) -> dict[str, Any]:
    for _ in range(warmups):
        fn(tokens)

    times = []
    for _ in range(reps):
        start = time.perf_counter()
        out = fn(tokens)
        elapsed = time.perf_counter() - start
        if name == "decode_tokens_bytes":
            _validate_decode_tokens_bytes(out, tokens)
        elif name == "decode_with_offsets":
            _validate_decode_with_offsets(out, tokens)
        times.append(elapsed)

    best = min(times)
    median = statistics.median(times)
    p95 = sorted(times)[max(0, int(len(times) * 0.95) - 1)]
    return {
        "best_s": best,
        "median_s": median,
        "p95_s": p95,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark tiktoken token decoding workloads.")
    parser.add_argument("--encoding", default="cl100k_base")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    enc = tiktoken.get_encoding(args.encoding)
    workloads = make_workloads()
    benchmarks: dict[str, Callable[[list[int]], Any]] = {
        "decode_tokens_bytes": enc.decode_tokens_bytes,
        "decode_with_offsets": enc.decode_with_offsets,
    }

    result: dict[str, Any] = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "tiktoken_file": tiktoken.__file__,
            "encoding": args.encoding,
            "reps": args.reps,
            "warmups": args.warmups,
        },
        "results": {},
    }

    for workload_name, text in workloads.items():
        tokens = enc.encode_ordinary(text)
        num_bytes = len(text.encode("utf-8"))
        workload_result: dict[str, Any] = {
            "bytes": num_bytes,
            "tokens": len(tokens),
            "benchmarks": {},
        }
        print(f"{workload_name}: bytes={num_bytes} tokens={len(tokens)}")
        for bench_name, bench_fn in benchmarks.items():
            metrics = measure(bench_name, bench_fn, tokens, args.reps, args.warmups)
            metrics["tokens_per_s"] = len(tokens) / metrics["best_s"]
            metrics["mb_per_s"] = num_bytes / 1_000_000 / metrics["best_s"]
            workload_result["benchmarks"][bench_name] = metrics
            print(
                f"  {bench_name}: best={metrics['best_s'] * 1000:.3f}ms "
                f"median={metrics['median_s'] * 1000:.3f}ms "
                f"tokens/s={metrics['tokens_per_s']:.0f} "
                f"MB/s={metrics['mb_per_s']:.2f}"
            )
        result["results"][workload_name] = workload_result

    if args.json_output is not None:
        args.json_output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
