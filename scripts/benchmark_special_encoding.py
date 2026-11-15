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


def make_single_workloads() -> dict[str, str]:
    return {
        "tiny": "hello world",
        "chat": "user: customer asked whether order can be rerouted before close.",
        "medium": "The quick brown fox jumps over the lazy dog. " * 100,
        "long": "The quick brown fox jumps over the lazy dog. " * 5_000,
    }


def make_batch_workloads() -> dict[str, list[str]]:
    return {
        "tiny_10k": [f"hello world {i}" for i in range(10_000)],
        "chat_10k": [
            f"user: customer {i} asked whether order ord_{i:06d} can be rerouted before close."
            for i in range(10_000)
        ],
        "tool_json_5k": [
            json.dumps(
                {
                    "tool": "lookup_order",
                    "arguments": {
                        "order_id": f"ord_{i:06d}",
                        "urgent": False,
                    },
                },
                separators=(",", ":"),
            )
            for i in range(5_000)
        ],
    }


def measure_single(
    fn: Callable[[str], list[int]], text: str, reps: int, warmups: int
) -> dict[str, Any]:
    for _ in range(warmups):
        fn(text)

    times = []
    out = []
    for _ in range(reps):
        start = time.perf_counter()
        out = fn(text)
        times.append(time.perf_counter() - start)

    return {
        "best_s": min(times),
        "median_s": statistics.median(times),
        "tokens": len(out),
    }


def measure_batch(
    fn: Callable[[list[str]], list[list[int]]], docs: list[str], reps: int, warmups: int
) -> dict[str, Any]:
    for _ in range(warmups):
        fn(docs[: min(len(docs), 128)])

    times = []
    out: list[list[int]] = []
    for _ in range(reps):
        start = time.perf_counter()
        out = fn(docs)
        times.append(time.perf_counter() - start)

    return {
        "best_s": min(times),
        "median_s": statistics.median(times),
        "tokens": sum(map(len, out)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark tiktoken special-token encode paths.")
    parser.add_argument("--encoding", default="o200k_harmony")
    parser.add_argument("--single-reps", type=int, default=2_000)
    parser.add_argument("--batch-reps", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    enc = tiktoken.get_encoding(args.encoding)
    result: dict[str, Any] = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "tiktoken_file": tiktoken.__file__,
            "encoding": args.encoding,
            "special_tokens": len(enc.special_tokens_set),
        },
        "single": {},
        "batch": {},
    }

    single_benchmarks: dict[str, Callable[[str], list[int]]] = {
        "encode": enc.encode,
        "encode_disallowed_special_empty": lambda text: enc.encode(text, disallowed_special=()),
        "encode_ordinary": enc.encode_ordinary,
    }
    for workload_name, text in make_single_workloads().items():
        print(f"{workload_name}: bytes={len(text.encode('utf-8'))}")
        workload_result = {}
        reps = args.single_reps if len(text) < 1_000 else max(100, args.single_reps // 20)
        for bench_name, bench_fn in single_benchmarks.items():
            metrics = measure_single(bench_fn, text, reps, args.warmups)
            workload_result[bench_name] = metrics
            print(
                f"  {bench_name}: best={metrics['best_s'] * 1_000_000:.3f}us "
                f"median={metrics['median_s'] * 1_000_000:.3f}us "
                f"tokens={metrics['tokens']}"
            )
        result["single"][workload_name] = workload_result

    batch_benchmarks: dict[str, Callable[[list[str]], list[list[int]]]] = {
        "encode_batch": enc.encode_batch,
        "encode_batch_disallowed_special_empty": lambda docs: enc.encode_batch(
            docs, disallowed_special=()
        ),
        "encode_ordinary_batch": enc.encode_ordinary_batch,
    }
    for workload_name, docs in make_batch_workloads().items():
        num_bytes = sum(len(doc.encode("utf-8")) for doc in docs)
        print(f"{workload_name}: docs={len(docs)} bytes={num_bytes}")
        workload_result = {}
        for bench_name, bench_fn in batch_benchmarks.items():
            metrics = measure_batch(bench_fn, docs, args.batch_reps, args.warmups)
            metrics["docs_per_s"] = len(docs) / metrics["best_s"]
            metrics["mb_per_s"] = num_bytes / 1_000_000 / metrics["best_s"]
            workload_result[bench_name] = metrics
            print(
                f"  {bench_name}: best={metrics['best_s'] * 1000:.3f}ms "
                f"median={metrics['median_s'] * 1000:.3f}ms "
                f"docs/s={metrics['docs_per_s']:.0f} "
                f"MB/s={metrics['mb_per_s']:.2f}"
            )
        result["batch"][workload_name] = workload_result

    if args.json_output is not None:
        args.json_output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
