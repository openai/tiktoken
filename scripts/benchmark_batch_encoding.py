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
                "include": ["status", "refunds", "shipments"],
                "urgent": i % 13 == 0,
            },
        },
        separators=(",", ":"),
    )


def _chat_message(i: int) -> str:
    roles = ["system", "user", "assistant", "tool"]
    return (
        f"{roles[i % len(roles)]}: customer {i} asked whether order ord_{i:06d} "
        "can be rerouted before the warehouse batch closes."
    )


def _rag_snippet(i: int) -> str:
    return (
        f"doc={i} title=Returns policy. Customers can request a return within 30 days "
        "when the item is unused, in original packaging, and accompanied by a receipt. "
        f"Region shard {i % 17}."
    )


def make_workloads() -> dict[str, list[str]]:
    return {
        "tiny_10k": [f"hello world {i}" for i in range(10_000)],
        "chat_messages_10k": [_chat_message(i) for i in range(10_000)],
        "tool_json_5k": [_tool_json(i) for i in range(5_000)],
        "rag_snippets_5k": [_rag_snippet(i) for i in range(5_000)],
        "medium_1k": [("The quick brown fox jumps over the lazy dog. " * 20) + str(i) for i in range(1_000)],
    }


def measure(
    name: str,
    fn: Callable[[list[str]], list[list[int]]],
    docs: list[str],
    reps: int,
    warmups: int,
) -> dict[str, Any]:
    for _ in range(warmups):
        fn(docs[: min(len(docs), 128)])

    times = []
    token_count = None
    for _ in range(reps):
        start = time.perf_counter()
        out = fn(docs)
        elapsed = time.perf_counter() - start
        if len(out) != len(docs):
            raise RuntimeError(f"{name}: expected {len(docs)} outputs, got {len(out)}")
        current_token_count = sum(map(len, out))
        if token_count is None:
            token_count = current_token_count
        elif current_token_count != token_count:
            raise RuntimeError(f"{name}: token count changed between runs")
        times.append(elapsed)

    best = min(times)
    median = statistics.median(times)
    p95 = sorted(times)[max(0, int(len(times) * 0.95) - 1)]
    return {
        "best_s": best,
        "median_s": median,
        "p95_s": p95,
        "tokens": token_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark tiktoken batch encoding workloads.")
    parser.add_argument("--encoding", default="cl100k_base")
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    enc = tiktoken.get_encoding(args.encoding)
    workloads = make_workloads()
    benchmarks: dict[str, Callable[[list[str]], list[list[int]]]] = {
        "encode_batch": lambda docs: enc.encode_batch(docs, num_threads=args.num_threads),
        "encode_ordinary_batch": lambda docs: enc.encode_ordinary_batch(
            docs, num_threads=args.num_threads
        ),
    }

    result: dict[str, Any] = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "tiktoken_file": tiktoken.__file__,
            "encoding": args.encoding,
            "num_threads": args.num_threads,
            "reps": args.reps,
            "warmups": args.warmups,
        },
        "results": {},
    }

    for workload_name, docs in workloads.items():
        num_bytes = sum(len(doc.encode("utf-8")) for doc in docs)
        workload_result: dict[str, Any] = {
            "documents": len(docs),
            "bytes": num_bytes,
            "avg_chars": sum(map(len, docs)) / len(docs),
            "benchmarks": {},
        }
        print(
            f"{workload_name}: docs={len(docs)} bytes={num_bytes} "
            f"avg_chars={workload_result['avg_chars']:.1f}"
        )
        for bench_name, bench_fn in benchmarks.items():
            metrics = measure(bench_name, bench_fn, docs, args.reps, args.warmups)
            metrics["docs_per_s"] = len(docs) / metrics["best_s"]
            metrics["mb_per_s"] = num_bytes / 1_000_000 / metrics["best_s"]
            workload_result["benchmarks"][bench_name] = metrics
            print(
                f"  {bench_name}: best={metrics['best_s'] * 1000:.3f}ms "
                f"median={metrics['median_s'] * 1000:.3f}ms "
                f"docs/s={metrics['docs_per_s']:.0f} "
                f"MB/s={metrics['mb_per_s']:.2f}"
            )
        result["results"][workload_name] = workload_result

    if args.json_output is not None:
        args.json_output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
