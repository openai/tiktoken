import hashlib
import importlib.util
from pathlib import Path

import pytest

_LOAD_SPEC = importlib.util.spec_from_file_location(
    "tiktoken_load_for_tests",
    Path(__file__).resolve().parents[1] / "tiktoken" / "load.py",
)
assert _LOAD_SPEC is not None and _LOAD_SPEC.loader is not None
_LOAD_MODULE = importlib.util.module_from_spec(_LOAD_SPEC)
_LOAD_SPEC.loader.exec_module(_LOAD_MODULE)
read_file_cached = _LOAD_MODULE.read_file_cached


def test_read_file_cached_offline_uses_legacy_filename_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    blobpath = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
    contents = b"hello from cache"
    expected_hash = hashlib.sha256(contents).hexdigest()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "cl100k_base.tiktoken").write_bytes(contents)

    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("TIKTOKEN_OFFLINE", "1")

    assert read_file_cached(blobpath, expected_hash) == contents


def test_read_file_cached_offline_cache_miss_does_not_hit_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    blobpath = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("TIKTOKEN_OFFLINE", "1")

    with pytest.raises(FileNotFoundError, match="offline mode enabled"):
        read_file_cached(blobpath)
