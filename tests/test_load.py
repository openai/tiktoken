import os
import tempfile

import pytest

import tiktoken.load


def _raise_oserror(*args, **kwargs):
    raise OSError("simulated filesystem failure")


def _make_blob(tmp_path, contents=b"hello world"):
    blobpath = tmp_path / "blob.bin"
    blobpath.write_bytes(contents)
    return str(blobpath)


def _tmp_files(cache_dir):
    return sorted(p.name for p in cache_dir.iterdir() if p.name.endswith(".tmp"))


def test_read_file_cached_round_trip(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(cache_dir))
    blobpath = _make_blob(tmp_path)

    assert tiktoken.load.read_file_cached(blobpath) == b"hello world"
    assert tiktoken.load.read_file_cached(blobpath) == b"hello world"
    assert _tmp_files(cache_dir) == []


def test_read_file_cached_cleans_up_tmp_file_on_failure(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(cache_dir))
    # e.g. we run out of disk space, or on Windows another process won the race
    # to create the cache file
    monkeypatch.setattr(os, "rename", _raise_oserror)
    blobpath = _make_blob(tmp_path)

    with pytest.raises(OSError):
        tiktoken.load.read_file_cached(blobpath)
    assert _tmp_files(cache_dir) == []


def test_read_file_cached_cleans_up_tmp_file_on_failure_default_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
    monkeypatch.delenv("DATA_GYM_CACHE_DIR", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(os, "rename", _raise_oserror)
    blobpath = _make_blob(tmp_path)

    # failures writing to the default cache are not raised, see issue #75
    assert tiktoken.load.read_file_cached(blobpath) == b"hello world"
    assert _tmp_files(tmp_path / "data-gym-cache") == []
