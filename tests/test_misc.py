import subprocess
import sys

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


def test_encoding_name_for_fine_tuned_model():
    assert tiktoken.encoding_name_for_model("ft:gpt-4o-2024-08-06:org::abc123") == "o200k_base"
    assert tiktoken.encoding_name_for_model("ft:gpt-3.5-turbo-0125:org::abc123") == "cl100k_base"
    assert (
        tiktoken.encoding_name_for_model("ft:o4-mini-2025-04-16:org:custom-suffix:abc123")
        == "o200k_base"
    )


def test_optional_blobfile_dependency():
    prog = """
import tiktoken
import sys
assert "blobfile" not in sys.modules
"""
    subprocess.check_call([sys.executable, "-c", prog])
