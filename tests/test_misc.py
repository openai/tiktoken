import hashlib
import os
import stat
import subprocess
import sys

import tiktoken
import tiktoken.load


def test_encoding_for_model():
    enc = tiktoken.encoding_for_model("gpt2")
    assert enc.name == "gpt2"
    enc = tiktoken.encoding_for_model("text-davinci-003")
    assert enc.name == "p50k_base"
    enc = tiktoken.encoding_for_model("text-davinci-edit-001")
    assert enc.name == "p50k_edit"
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo-0301")
    assert enc.name == "cl100k_base"
    enc = tiktoken.encoding_for_model("gpt-4")
    assert enc.name == "cl100k_base"
    enc = tiktoken.encoding_for_model("gpt-4o")
    assert enc.name == "o200k_base"
    enc = tiktoken.encoding_for_model("gpt-oss-120b")
    assert enc.name == "o200k_harmony"


def test_optional_blobfile_dependency():
    prog = """
import tiktoken
import sys
assert "blobfile" not in sys.modules
"""
    subprocess.check_call([sys.executable, "-c", prog])


def test_default_cache_dir_is_user_specific(tmp_path, monkeypatch):
    data = b"token data"
    expected_hash = hashlib.sha256(data).hexdigest()
    blobpath = "https://openaipublic.blob.core.windows.net/encodings/example.tiktoken"
    cache_key = hashlib.sha1(blobpath.encode()).hexdigest()

    monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
    monkeypatch.delenv("DATA_GYM_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.setattr(tiktoken.load, "read_file", lambda _: data)

    assert tiktoken.load.read_file_cached(blobpath, expected_hash) == data

    cache_dir = tmp_path / "xdg-cache" / "tiktoken"
    assert (cache_dir / cache_key).read_bytes() == data
    assert not (tmp_path / "data-gym-cache").exists()

    def fail_read_file(_: str) -> bytes:
        raise AssertionError("cached file was not used")

    monkeypatch.setattr(tiktoken.load, "read_file", fail_read_file)
    assert tiktoken.load.read_file_cached(blobpath, expected_hash) == data

    if os.name != "nt":
        assert stat.S_IMODE(cache_dir.stat().st_mode) == 0o700
