from __future__ import annotations

import hashlib
import os
import stat
import tempfile

import pytest

from tiktoken.load import _default_cache_dir, read_file_cached


@pytest.fixture()
def isolated_cache(tmp_path, monkeypatch):
    cache_dir = str(tmp_path / "tiktoken-cache")
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", cache_dir)
    return cache_dir


@pytest.fixture()
def default_cache(monkeypatch):
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
    monkeypatch.delenv("DATA_GYM_CACHE_DIR", raising=False)


class TestDefaultCacheDir:
    def test_respects_xdg_cache_home(self, monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", "/custom/cache")
        monkeypatch.delenv("LOCALAPPDATA", raising=False)
        assert _default_cache_dir() == "/custom/cache/tiktoken"

    def test_falls_back_to_home_dot_cache(self, monkeypatch):
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.delenv("LOCALAPPDATA", raising=False)
        result = _default_cache_dir()
        assert result.endswith(os.path.join(".cache", "tiktoken"))

    def test_not_in_tmp(self, monkeypatch):
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.delenv("LOCALAPPDATA", raising=False)
        result = _default_cache_dir()
        assert tempfile.gettempdir() not in result


class TestCacheDirPermissions:
    def test_cache_dir_created_with_restricted_permissions(self, tmp_path, monkeypatch):
        cache_dir = str(tmp_path / "new-cache")
        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", cache_dir)

        test_file = str(tmp_path / "test_data.txt")
        with open(test_file, "wb") as f:
            f.write(b"test data")

        read_file_cached(test_file, expected_hash=None)

        dir_stat = os.stat(cache_dir)
        mode = stat.S_IMODE(dir_stat.st_mode)
        assert mode & stat.S_IRWXG == 0, "Group should have no permissions"
        assert mode & stat.S_IRWXO == 0, "Others should have no permissions"


class TestHashVerification:
    def test_valid_hash_returns_data(self, isolated_cache, tmp_path):
        test_data = b"hello tiktoken"
        test_file = str(tmp_path / "source.txt")
        with open(test_file, "wb") as f:
            f.write(test_data)

        expected_hash = hashlib.sha256(test_data).hexdigest()
        result = read_file_cached(test_file, expected_hash=expected_hash)
        assert result == test_data

        result2 = read_file_cached(test_file, expected_hash=expected_hash)
        assert result2 == test_data

    def test_invalid_hash_rejects_download(self, isolated_cache, tmp_path):
        test_data = b"hello tiktoken"
        test_file = str(tmp_path / "source.txt")
        with open(test_file, "wb") as f:
            f.write(test_data)

        with pytest.raises(ValueError, match="Hash mismatch"):
            read_file_cached(test_file, expected_hash="badhash" * 8)

    def test_tampered_cache_is_refetched(self, isolated_cache, tmp_path):
        test_data = b"original data"
        test_file = str(tmp_path / "source.txt")
        with open(test_file, "wb") as f:
            f.write(test_data)

        expected_hash = hashlib.sha256(test_data).hexdigest()
        read_file_cached(test_file, expected_hash=expected_hash)

        cache_key = hashlib.sha1(test_file.encode()).hexdigest()
        cache_path = os.path.join(isolated_cache, cache_key)
        with open(cache_path, "wb") as f:
            f.write(b"tampered data")

        result = read_file_cached(test_file, expected_hash=expected_hash)
        assert result == test_data

    def test_no_hash_logs_warning(self, isolated_cache, tmp_path, caplog):
        import logging

        test_data = b"some data"
        test_file = str(tmp_path / "source.txt")
        with open(test_file, "wb") as f:
            f.write(test_data)

        read_file_cached(test_file, expected_hash=None)

        with caplog.at_level(logging.WARNING, logger="tiktoken"):
            read_file_cached(test_file, expected_hash=None)

        assert "without hash verification" in caplog.text
