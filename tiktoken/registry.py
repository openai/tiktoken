from __future__ import annotations

import importlib
import pkgutil
import threading
import json
from typing import Any, Callable, Optional

from tiktoken.core import Encoding
from tiktoken.load import data_gym_to_mergeable_bpe_ranks, load_tiktoken_bpe

_lock = threading.RLock()
ENCODINGS: dict[str, Encoding] = {}
ENCODING_DEFS: dict[str, Any] = None

def _load_encoding_defs():
    global ENCODING_DEFS
    if not ENCODING_DEFS is None:
        return ENCODING_DEFS

    try:
        import importlib.resources as pkg_resources
    except ImportError:
        # Try backported to PY<37 `importlib_resources`.
        import importlib_resources as pkg_resources

    # read registry.json
    # note: was trying to place it into /data/registry.json but python packaging is always unhappy
    ENCODING_DEFS = json.loads(pkg_resources.read_text("tiktoken", "registry.json"))

    return ENCODING_DEFS

def get_encoding(encoding_name: str) -> Encoding:
    if encoding_name in ENCODINGS:
        return ENCODINGS[encoding_name]

    with _lock:
        if encoding_name in ENCODINGS:
            return ENCODINGS[encoding_name]

        _load_encoding_defs()
        if encoding_name not in ENCODING_DEFS:
            raise ValueError(f"Unknown encoding {encoding_name}")

        encoding_def = dict(ENCODING_DEFS[encoding_name])
        encoding_def["name"] = encoding_name	

        if "load_tiktoken_bpe" in encoding_def:
            encoding_def["mergeable_ranks"] = load_tiktoken_bpe(encoding_def["load_tiktoken_bpe"])
            del encoding_def["load_tiktoken_bpe"]
        elif "data_gym_to_mergeable_bpe_ranks" in encoding_def:
            encoding_def["mergeable_ranks"] = data_gym_to_mergeable_bpe_ranks(**encoding_def["data_gym_to_mergeable_bpe_ranks"])
            del encoding_def["data_gym_to_mergeable_bpe_ranks"]
        else:
            raise ValueError(f"Unknown loader {encoding_name}")
        enc = Encoding(**encoding_def)
        ENCODINGS[encoding_name] = enc
        return enc


def list_encoding_names() -> list[str]:
    with _lock:
        return list(_load_encoding_defs().keys())
