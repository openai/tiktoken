import bisect
import functools
import os

import pytest

import tiktoken

MAX_EXAMPLES: int = int(os.environ.get("TIKTOKEN_MAX_EXAMPLES", "100"))

ENCODINGS = ["r50k_base", "cl100k_base"]
SOME_ENCODINGS = ["cl100k_base"]


ENCODING_FACTORIES = [
    pytest.param(functools.partial(tiktoken.get_encoding, name), id=name) for name in ENCODINGS
]
SOME_ENCODING_FACTORIES = [
    pytest.param(functools.partial(tiktoken.get_encoding, name), id=name) for name in SOME_ENCODINGS
]


