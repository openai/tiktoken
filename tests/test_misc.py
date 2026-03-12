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


def test_optional_blobfile_dependency():
    prog = """
import tiktoken
import sys
assert "blobfile" not in sys.modules
"""
    subprocess.check_call([sys.executable, "-c", prog])


def test_is_special_token():
    enc = tiktoken.get_encoding("gpt2")
    eot_token = enc.eot_token
    # The eot_token should be identified as a special token
    assert enc.is_special_token(eot_token) is True
    # Token 0 is a regular mergeable token, not special
    assert enc.is_special_token(0) is False


def test_max_threads_default():
    import os

    from tiktoken.core import _MAX_THREADS

    cpu_count = os.cpu_count() or 8
    assert _MAX_THREADS == min(cpu_count, 32)
    assert _MAX_THREADS >= 1


def test_list_encoding_names_optimized():
    """Test that list_encoding_names works even with assertions disabled (python -O)."""
    prog = """
import tiktoken
names = tiktoken.list_encoding_names()
assert len(names) > 0
assert "gpt2" in names
"""
    subprocess.check_call([sys.executable, "-O", "-c", prog])
