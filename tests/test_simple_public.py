import subprocess
import sys

import tiktoken


def test_simple_encoding():
    test_cases = [
        ("gpt2", [[31373, 995], [31373, 220, 50256], [15339, 1917], [15339, 220, 100257]]),
        ("cl100k_base", [[15339, 1917], [15339, 220, 100257]])
    ]

    for enc_name, cases in test_cases:
        enc = tiktoken.get_encoding(enc_name)

        assert enc.encode("hello world") == cases[0]
        assert enc.decode(cases[0]) == "hello world"
        assert enc.encode("hello ", allowed_special="all") == cases[1]

        for token in range(10_000):
            assert enc.encode_single_token(enc.decode_single_token_bytes(token)) == token


def test_encoding_for_model():
    test_cases = [
        ("gpt2", "gpt2"),
        ("text-davinci-003", "p50k_base"),
        ("text-davinci-edit-001", "p50k_edit"),
        ("gpt-3.5-turbo-0301", "cl100k_base")
    ]

    for model_name, expected_enc_name in test_cases:
        enc = tiktoken.encoding_for_model(model_name)
        assert enc.name == expected_enc_name


def test_optional_blobfile_dependency():
    prog = """
import tiktoken
import sys
assert "blobfile" not in sys.modules
"""
    subprocess.check_call([sys.executable, "-c", prog])


if __name__ == "__main__":
    test_simple_encoding()
    test_encoding_for_model()
    test_optional_blobfile_dependency()
