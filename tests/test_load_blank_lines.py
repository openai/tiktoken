import pytest

from tiktoken.load import load_tiktoken_bpe


@pytest.mark.parametrize("separator", [b"", b" ", b"\t \t"])
def test_load_tiktoken_bpe_ignores_blank_lines(tmp_path, monkeypatch, separator):
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", "")
    vocab = tmp_path / "test.tiktoken"
    vocab.write_bytes(b"YQ== 0\n" + separator + b"\nYg== 1\n")

    assert load_tiktoken_bpe(str(vocab)) == {b"a": 0, b"b": 1}


def test_load_tiktoken_bpe_still_rejects_incomplete_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", "")
    vocab = tmp_path / "test.tiktoken"
    vocab.write_bytes(b"YQ==\n")

    with pytest.raises(ValueError, match="Error parsing line"):
        load_tiktoken_bpe(str(vocab))
