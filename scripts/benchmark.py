import base64
import functools
import gzip
import json
import os
import random
import time
from typing import Any, cast
import numpy as np

import blobfile

import tiktoken


def benchmark_batch(documents: list[str], num_threads: int) -> None:
    # num_threads = 1#int(os.environ["RAYON_NUM_THREADS"])
    num_bytes = sum(map(len, map(str.encode, documents)))
    # print(f"num_threads: {num_threads}, num_bytes: {num_bytes}")

    enc = tiktoken.get_encoding("gpt2")
    enc.encode("warmup")

    start = time.perf_counter_ns()
    enc.encode_ordinary_batch(documents, num_threads=num_threads)
    end = time.perf_counter_ns()
    # print(f"tiktoken \t{num_bytes / (end - start) * 1e9} bytes / s")

    return num_bytes / (end - start) * 1e9

    import transformers

    hf_enc = cast(Any, transformers).GPT2TokenizerFast.from_pretrained("gpt2")
    hf_enc.model_max_length = 1e30  # silence!
    hf_enc.encode("warmup")

    start = time.perf_counter_ns()
    hf_enc(documents)
    end = time.perf_counter_ns()
    print(f"huggingface \t{num_bytes / (end - start) * 1e9} bytes / s")

    return num_bytes / (end - start) * 1e9


import base64, random


def base64_noise_documents(n_docs: int):
    rand = random.Random(217)
    documents = []
    for _ in range(n_docs):
        documents.append(
            base64.b64encode(rand.randbytes(rand.randint(100, 10_000))).decode()
        )
    return documents


def run_benchmark(batch_size, num_threads):
    return np.mean(
        [
            benchmark_batch(base64_noise_documents(1000), num_threads=num_threads)
            for _ in range(batch_size)
        ]
    )


print("threads,bytes_per_second")
for thread_count in range(1, 12, 2):
    bytes_per_second = run_benchmark(10, thread_count)
    print(f"{thread_count},{bytes_per_second}")