from __future__ import annotations

from .core import Encoding
from .registry import get_encoding
import json

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

# TODO: this will likely be replaced by an API endpoint
MODEL_TO_ENCODING: dict[str, str] = json.loads(pkg_resources.read_text("tiktoken", "model_to_encoding.json"))

def encoding_for_model(model_name: str) -> Encoding:
    try:
        encoding_name = MODEL_TO_ENCODING[model_name]
    except KeyError:
        raise KeyError(
            f"Could not automatically map {model_name} to a tokeniser. "
            "Please use `tiktok.get_encoding` to explicitly get the tokeniser you expect."
        ) from None
    return get_encoding(encoding_name)
