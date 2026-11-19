import base64

import pytest

from tiktoken.load import load_tiktoken_bpe


def test_load_tiktoken_bpe_rejects_invalid_base64(tmp_path) -> None:
    path = tmp_path / "invalid.tiktoken"
    path.write_bytes(b"!!! 0\n")

    with pytest.raises(ValueError, match=r"Error parsing line b'!!! 0'"):
        load_tiktoken_bpe(str(path))


def test_load_tiktoken_bpe_accepts_valid_base64(tmp_path) -> None:
    path = tmp_path / "valid.tiktoken"
    path.write_bytes(base64.b64encode(b"hello") + b" 42\n")

    assert load_tiktoken_bpe(str(path)) == {b"hello": 42}
