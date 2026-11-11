import subprocess
import sys

import tiktoken
from tiktoken.load import load_tiktoken_bpe


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


def test_relative_bpe_paths_use_distinct_cache_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "cache"))
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    (first_dir / "vocab.tiktoken").write_bytes(b"YQ== 0\n")
    (second_dir / "vocab.tiktoken").write_bytes(b"Yg== 0\n")

    monkeypatch.chdir(first_dir)
    assert load_tiktoken_bpe("vocab.tiktoken") == {b"a": 0}
    monkeypatch.chdir(second_dir)
    assert load_tiktoken_bpe("vocab.tiktoken") == {b"b": 0}
