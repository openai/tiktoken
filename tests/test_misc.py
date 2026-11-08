import io
import subprocess
import sys
from types import SimpleNamespace

import tiktoken


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


def test_local_bpe_dump_without_blobfile(monkeypatch, tmp_path):
    from tiktoken.load import dump_tiktoken_bpe, load_tiktoken_bpe

    monkeypatch.setitem(sys.modules, "blobfile", None)
    path = tmp_path / "vocab.tiktoken"
    ranks = {b"a": 0, b"bc": 1}

    dump_tiktoken_bpe(ranks, str(path))

    assert load_tiktoken_bpe(str(path)) == ranks


def test_remote_bpe_dump_uses_blobfile(monkeypatch):
    from tiktoken.load import dump_tiktoken_bpe

    written = {}

    class BlobFile(io.BytesIO):
        def __init__(self, path, mode):
            assert (path, mode) == ("gs://bucket/vocab.tiktoken", "wb")
            super().__init__()

        def close(self):
            written["contents"] = self.getvalue()
            super().close()

    monkeypatch.setitem(sys.modules, "blobfile", SimpleNamespace(BlobFile=BlobFile))

    dump_tiktoken_bpe({b"a": 0}, "gs://bucket/vocab.tiktoken")

    assert written["contents"] == b"YQ== 0\n"
