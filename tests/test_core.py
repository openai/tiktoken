import pickle
import pytest
import regex
import tiktoken.registry as registry
from tiktoken import (
    _tiktoken,
)
from tiktoken.core import (
    Encoding,
    raise_disallowed_special_token,
)
from typing import (
    AbstractSet,
    Collection,
    Sequence,
)


class DummyCoreBPE:
    """Dummy implementation of CoreBPE to simulate token encoding and decoding."""

    def __init__(self, mergeable_ranks, special_tokens, pat_str):
        self.mergeable_ranks = mergeable_ranks
        self.special_tokens = special_tokens
        self.pat_str = pat_str

    def encode_ordinary(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def encode(self, text: str, allowed_special: set) -> list[int]:
        return [ord(ch) for ch in text]

    def encode_with_unstable(
        self, text: str, allowed_special: set
    ) -> tuple[list[int], list[list[int]]]:
        return ([ord(text[0])], [[ord(ch) for ch in text]])

    def encode_single_token(self, text_or_bytes: bytes) -> int:
        return sum(text_or_bytes)

    def decode_bytes(self, tokens: Sequence[int]) -> bytes:
        return bytes(tokens)

    def decode_single_token_bytes(self, token: int) -> bytes:
        return bytes([token % 256])

    def token_byte_values(self) -> list[bytes]:
        return list(self.mergeable_ranks.keys())

    def encode_single_piece(self, text_or_bytes: bytes) -> list[int]:
        return list(text_or_bytes)

    def _encode_bytes(self, text: bytes) -> list[int]:
        return list(text)


@pytest.fixture(autouse=True)
def use_dummy_core_bpe(monkeypatch):
    """
    Replaces _tiktoken.CoreBPE with DummyCoreBPE for testing purposes.
    """

    class DummyCoreBPEWrapper:
        """Dummy implementation of CoreBPE to simulate token encoding and decoding."""

        def __init__(self, mergeable_ranks, special_tokens, pat_str):
            self.mergeable_ranks = mergeable_ranks
            self.special_tokens = special_tokens
            self.pat_str = pat_str

        def encode_ordinary(self, text: str) -> list[int]:
            return [ord(ch) for ch in text]

        def encode(self, text: str, allowed_special: set) -> list[int]:
            return [ord(ch) for ch in text]

        def encode_with_unstable(
            self, text: str, allowed_special: set
        ) -> tuple[list[int], list[list[int]]]:
            return ([ord(text[0])], [[ord(ch) for ch in text]])

        def encode_single_token(self, text_or_bytes: bytes) -> int:
            return sum(text_or_bytes)

        def decode_bytes(self, tokens: Sequence[int]) -> bytes:
            return bytes(tokens)

        def decode_single_token_bytes(self, token: int) -> bytes:
            return bytes([token % 256])

        def token_byte_values(self) -> list[bytes]:
            return list(self.mergeable_ranks.keys())

        def encode_single_piece(self, text_or_bytes: bytes) -> list[int]:
            return list(text_or_bytes)

        def _encode_bytes(self, text: bytes) -> list[int]:
            return list(text)

    monkeypatch.setattr(_tiktoken, "CoreBPE", DummyCoreBPEWrapper)


def test_encoding_dummy_flow():
    """
    Test the basic operations of Encoding (encoding, decoding, batch processing, special token handling,
    and private methods) using a dummy CoreBPE implementation.
    """
    mergeable_ranks = {b"a": 1, b"b": 2}
    special_tokens = {"<|endoftext|>": 999}
    pat_str = ".+?"
    enc = Encoding(
        "dummy",
        pat_str=pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
    assert "dummy" in repr(enc)
    result = enc.encode_ordinary("ab")
    assert result == [97, 98]
    text_with_special = "hello <|endoftext|> world"
    with pytest.raises(ValueError) as excinfo:
        enc.encode(text_with_special)
    assert "<|endoftext|>" in str(excinfo.value)
    allowed = "all"
    tokens = enc.encode("<|endoftext|>", allowed_special=allowed)
    expected = [ord(c) for c in "<|endoftext|>"]
    assert tokens == expected
    batch_result = enc.encode_batch(["a", "b"])
    assert batch_result == [[97], [98]]
    stable, completions = enc.encode_with_unstable("abc", allowed_special=allowed)
    assert stable == [ord("a")]
    assert completions == [[ord("a"), ord("b"), ord("c")]]
    single_token = enc.encode_single_token("a")
    assert single_token == 97
    decoded_bytes = enc.decode_bytes([65, 66])
    assert decoded_bytes == b"AB"
    decoded_text = enc.decode([65, 66])
    assert decoded_text == "AB"
    token_byte = enc.decode_single_token_bytes(65)
    assert token_byte == b"A"
    tokens_bytes = enc.decode_tokens_bytes([65, 66])
    assert tokens_bytes == [b"A", b"B"]
    text_out, offsets = enc.decode_with_offsets([65, 66])
    assert text_out == "AB"
    assert offsets == [0, 1]
    decoded_batch = enc.decode_batch([[65, 66]])
    assert decoded_batch == ["AB"]
    decoded_bytes_batch = enc.decode_bytes_batch([[65, 66]])
    assert decoded_bytes_batch == [b"AB"]
    token_values = enc.token_byte_values()
    assert set(token_values) == set(mergeable_ranks.keys())
    state = pickle.dumps(enc)
    new_enc = pickle.loads(state)
    assert repr(new_enc) == repr(enc)
    assert new_enc.encode_ordinary("x") == [120]
    piece_tokens = enc._encode_single_piece("abc")
    assert piece_tokens == list(b"abc")
    bytes_tokens = enc._encode_bytes(b"abc")
    assert bytes_tokens == list(b"abc")


@pytest.fixture(autouse=True)
def dummy_registry(monkeypatch):
    """
    Provides a dummy tiktoken.registry module with an ENCODINGS dict and a get_encoding function.
    This ensures that when an Encoding is pickled, it uses the registry by reference if possible.
    """
    try:
        import tiktoken.registry as registry
    except ImportError:
        import types

        registry = types.ModuleType("tiktoken.registry")
    dummy_encodings = {}
    dummy_encodings["dummy"] = None
    monkeypatch.setattr(registry, "ENCODINGS", dummy_encodings)
    monkeypatch.setattr(registry, "get_encoding", lambda name: dummy_encodings[name])
    monkeypatch.setitem(dummy_encodings, "dummy", None)


def test_registered_encoding_pickle(monkeypatch):
    """
    Test that when an Encoding is registered in tiktoken.registry,
    the pickling process uses the registry branch of __getstate__ (returning the encoding name)
    and that the encoding is correctly restored via __setstate__.
    """
    import tiktoken.registry as registry

    mergeable_ranks = {b"x": 1}
    special_tokens = {"<|endoftext|>": 1}
    pat_str = ".+?"
    enc = Encoding(
        "registered_dummy",
        pat_str=pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
        explicit_n_vocab=2,
    )
    registry.ENCODINGS["registered_dummy"] = enc
    state = enc.__getstate__()
    assert (
        state == "registered_dummy"
    ), "Expected __getstate__ to return the encoding name when registered."
    monkeypatch.setattr(registry, "get_encoding", lambda name: registry.ENCODINGS[name])
    pickled = pickle.dumps(enc)
    unpickled = pickle.loads(pickled)
    assert unpickled.name == enc.name
    assert unpickled.encode_ordinary("x") == enc.encode_ordinary("x")


def test_encoding_explicit_n_vocab_and_setstate(monkeypatch):
    """
    Test that:
    1. An Encoding with an explicit_n_vocab that is inconsistent with the provided mergeable ranks
       and special tokens raises an AssertionError.
    2. __setstate__ properly re-initializes an Encoding instance when provided with a state dictionary.
    """
    mergeable_ranks_err = {b"a": 1}
    special_tokens_err = {"<|endoftext|>": 5}
    pat_str = ".+?"
    with pytest.raises(AssertionError):
        Encoding(
            "error_case",
            pat_str=pat_str,
            mergeable_ranks=mergeable_ranks_err,
            special_tokens=special_tokens_err,
            explicit_n_vocab=2,
        )
    mergeable_ranks_valid = {b"a": 0}
    special_tokens_valid = {"<|endoftext|>": 1}
    explicit_n_vocab = 2
    enc = Encoding(
        "valid_case",
        pat_str=pat_str,
        mergeable_ranks=mergeable_ranks_valid,
        special_tokens=special_tokens_valid,
        explicit_n_vocab=explicit_n_vocab,
    )
    assert enc.n_vocab == 2
    new_state = {
        "name": "state_case",
        "pat_str": pat_str,
        "mergeable_ranks": mergeable_ranks_valid,
        "special_tokens": special_tokens_valid,
    }
    enc2 = Encoding(
        "dummy",
        pat_str=pat_str,
        mergeable_ranks=mergeable_ranks_valid,
        special_tokens=special_tokens_valid,
    )
    enc2.__setstate__(new_state)
    assert enc2.name == "state_case"
    result = enc2.encode_ordinary("a")
    assert result == [ord("a")]


def test_encode_ordinary_fallback(monkeypatch):
    """
    Test that if encode_ordinary initially raises a UnicodeEncodeError,
    the fallback mechanism (encoding with utf-16 surrogatepass and decode with replace)
    is correctly applied and returns the expected tokens.
    """
    mergeable_ranks = {b"x": ord("x")}
    special_tokens = {"<|endoftext|>": 999}
    pat_str = ".+?"
    enc = Encoding(
        "fallback_test",
        pat_str=pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
    counter = [0]
    original_encode = enc._core_bpe.encode_ordinary

    def fake_encode_ordinary(text):
        if counter[0] == 0:
            counter[0] += 1
            raise UnicodeEncodeError("utf-8", text, 0, 1, "forced error")
        else:
            return original_encode(text)

    monkeypatch.setattr(enc._core_bpe, "encode_ordinary", fake_encode_ordinary)
    test_text = "test_text"
    fallback_text = test_text.encode("utf-16", "surrogatepass").decode(
        "utf-16", "replace"
    )
    expected = [ord(c) for c in fallback_text]
    result = enc.encode_ordinary(test_text)
    assert result == expected
