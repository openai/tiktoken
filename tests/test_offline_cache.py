import importlib.util
import os
from pathlib import Path
from unittest.mock import patch
import pytest

load_path = Path(__file__).parent.parent / "tiktoken" / "load.py"
spec = importlib.util.spec_from_file_location("tiktoken_load", str(load_path))
tiktoken_load = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tiktoken_load)
read_file_cached = tiktoken_load.read_file_cached


def test_read_file_cached_basename_fallback(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(cache_dir))

    url = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
    cached_file = cache_dir / "cl100k_base.tiktoken"
    cached_file.write_bytes(b"mock_token_data")

    # With network blocked, reading from cache by basename must succeed
    with patch.object(tiktoken_load, "read_file", side_effect=RuntimeError("Network request made in offline/cache mode")):
        data = read_file_cached(url)
        assert data == b"mock_token_data"


def test_read_file_cached_offline_error_when_missing(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("TIKTOKEN_OFFLINE", "1")

    url = "https://openaipublic.blob.core.windows.net/encodings/missing.tiktoken"

    with patch.object(tiktoken_load, "read_file", side_effect=RuntimeError("Network request made in offline mode")):
        with pytest.raises(FileNotFoundError, match="TIKTOKEN_OFFLINE is enabled"):
            read_file_cached(url)
