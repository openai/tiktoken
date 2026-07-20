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
    # gpt-5.1 uses a dotted name and must not require a hyphenated prefix match
    assert tiktoken.model.encoding_name_for_model("gpt-5.1") == "o200k_base"
    assert tiktoken.model.encoding_name_for_model("gpt-5.1-2025-11-13") == "o200k_base"


def test_optional_blobfile_dependency():
    prog = """
import tiktoken
import sys
assert "blobfile" not in sys.modules
"""
    subprocess.check_call([sys.executable, "-c", prog])
