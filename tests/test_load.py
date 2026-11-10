import builtins
import hashlib
import os
import tempfile
from unittest.mock import MagicMock, Mock

import pytest

import tiktoken.load as load


def setup_cache(tmp_path, monkeypatch, cache_variable=None):
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
    monkeypatch.delenv("DATA_GYM_CACHE_DIR", raising=False)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    cache_dir = tmp_path / "data-gym-cache"
    cache_dir.mkdir()
    if cache_variable is not None:
        monkeypatch.setenv(cache_variable, str(cache_dir))

    blobpath = "https://example.invalid/encoding.tiktoken"
    contents = b"vocabulary bytes"
    cache_path = cache_dir / hashlib.sha1(blobpath.encode()).hexdigest()
    cache_path.write_bytes(contents)
    expected_hash = hashlib.sha256(contents).hexdigest()
    download = Mock(return_value=contents)
    monkeypatch.setattr(load, "read_file", download)
    return blobpath, cache_path, contents, expected_hash, download


def fail_cache_read(monkeypatch, cache_path, error_type, failure):
    def open_cache(path, mode="r", *args, **kwargs):
        if os.fspath(path) == str(cache_path) and mode == "rb":
            error = error_type("cache unavailable")
            if failure == "open":
                raise error
            reader = MagicMock()
            reader.__enter__.return_value.read.side_effect = error
            return reader
        return builtins.open(path, mode, *args, **kwargs)

    monkeypatch.setattr(load, "open", open_cache, raising=False)


@pytest.mark.parametrize("error_type", [PermissionError, FileNotFoundError, IsADirectoryError])
@pytest.mark.parametrize("failure", ["open", "read"])
def test_unreadable_default_cache_refetches(tmp_path, monkeypatch, error_type, failure):
    blobpath, cache_path, contents, expected_hash, download = setup_cache(tmp_path, monkeypatch)
    fail_cache_read(monkeypatch, cache_path, error_type, failure)

    assert load.read_file_cached(blobpath, expected_hash) == contents
    download.assert_called_once_with(blobpath)


@pytest.mark.parametrize("cache_variable", ["TIKTOKEN_CACHE_DIR", "DATA_GYM_CACHE_DIR"])
@pytest.mark.parametrize("error_type", [PermissionError, FileNotFoundError, IsADirectoryError])
@pytest.mark.parametrize("failure", ["open", "read"])
def test_unreadable_explicit_cache_raises(
    tmp_path, monkeypatch, cache_variable, error_type, failure
):
    blobpath, cache_path, _, expected_hash, download = setup_cache(
        tmp_path, monkeypatch, cache_variable
    )
    fail_cache_read(monkeypatch, cache_path, error_type, failure)

    with pytest.raises(error_type, match="cache unavailable"):
        load.read_file_cached(blobpath, expected_hash)
    download.assert_not_called()


def test_valid_default_cache_does_not_refetch(tmp_path, monkeypatch):
    blobpath, _, contents, expected_hash, download = setup_cache(tmp_path, monkeypatch)

    assert load.read_file_cached(blobpath, expected_hash) == contents
    download.assert_not_called()


def test_refetched_contents_still_require_expected_hash(tmp_path, monkeypatch):
    blobpath, cache_path, _, expected_hash, download = setup_cache(tmp_path, monkeypatch)
    fail_cache_read(monkeypatch, cache_path, PermissionError, "open")
    download.return_value = b"corrupted vocabulary"

    with pytest.raises(ValueError, match="Hash mismatch"):
        load.read_file_cached(blobpath, expected_hash)
