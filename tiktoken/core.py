from __future__ import annotations

import functools
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, AbstractSet, Collection, Iterator, Literal, NoReturn, Sequence

from tiktoken import _tiktoken

if TYPE_CHECKING:
    import re

    import numpy as np
    import numpy.typing as npt


class _LazyMergeableRanks(Mapping[bytes, int]):
    def __init__(
        self,
        core_bpe: _tiktoken.CoreBPE,
        n_mergeable_ranks: int,
        max_token_value: int,
    ):
        self._core_bpe = core_bpe
        self._n_mergeable_ranks = n_mergeable_ranks
        self._max_token_value = max_token_value
        self._mergeable_ranks: dict[bytes, int] | None = None

    @property
    def n_mergeable_ranks(self) -> int:
        return self._n_mergeable_ranks

    @property
    def max_token_value(self) -> int:
        return self._max_token_value

    @property
    def core_bpe(self) -> _tiktoken.CoreBPE:
        return self._core_bpe

    def _materialized(self) -> dict[bytes, int]:
        mergeable_ranks = self._mergeable_ranks
        if mergeable_ranks is None:
            mergeable_ranks = self._core_bpe.mergeable_ranks()
            self._mergeable_ranks = mergeable_ranks
        return mergeable_ranks

    def __getitem__(self, key: bytes) -> int:
        return self._materialized()[key]

    def __iter__(self) -> Iterator[bytes]:
        return iter(self._materialized())

    def __len__(self) -> int:
        return self._n_mergeable_ranks

    def copy(self) -> dict[bytes, int]:
        return self._materialized().copy()


class Encoding:
    def __init__(
        self,
        name: str,
        *,
        pat_str: str,
        mergeable_ranks: dict[bytes, int] | _LazyMergeableRanks | None,
        special_tokens: dict[str, int],
        explicit_n_vocab: int | None = None,
        _core_bpe: _tiktoken.CoreBPE | None = None,
        _mergeable_ranks_len: int | None = None,
        _mergeable_ranks_max_token_value: int | None = None,
    ):
        """Creates an Encoding object.

        See openai_public.py for examples of how to construct an Encoding object.

        Args:
            name: The name of the encoding. It should be clear from the name of the encoding
                what behaviour to expect, in particular, encodings with different special tokens
                should have different names.
            pat_str: A regex pattern string that is used to split the input text.
            mergeable_ranks: A dictionary mapping mergeable token bytes to their ranks. The ranks
                must correspond to merge priority.
            special_tokens: A dictionary mapping special token strings to their token values.
            explicit_n_vocab: The number of tokens in the vocabulary. If provided, it is checked
                that the number of mergeable tokens and special tokens is equal to this number.
        """
        self.name = name

        self._pat_str = pat_str
        if isinstance(mergeable_ranks, _LazyMergeableRanks):
            if _core_bpe is None:
                _core_bpe = mergeable_ranks.core_bpe
            if _mergeable_ranks_len is None:
                _mergeable_ranks_len = mergeable_ranks.n_mergeable_ranks
            if _mergeable_ranks_max_token_value is None:
                _mergeable_ranks_max_token_value = mergeable_ranks.max_token_value
            mergeable_ranks = None

        self._mergeable_ranks = mergeable_ranks
        self._special_tokens = special_tokens

        mergeable_ranks_len = (
            len(mergeable_ranks) if mergeable_ranks is not None else _mergeable_ranks_len
        )
        mergeable_ranks_max_token_value = (
            max(mergeable_ranks.values())
            if mergeable_ranks is not None
            else _mergeable_ranks_max_token_value
        )
        assert mergeable_ranks_len is not None
        assert mergeable_ranks_max_token_value is not None

        self.max_token_value = max(
            mergeable_ranks_max_token_value, max(special_tokens.values(), default=0)
        )
        if explicit_n_vocab:
            assert mergeable_ranks_len + len(special_tokens) == explicit_n_vocab
            assert self.max_token_value == explicit_n_vocab - 1

        # Contains on set is significantly faster than on dict_values
        self._special_token_values = set(self._special_tokens.values())
        self._special_tokens_set_frozen = frozenset(self._special_tokens)

        if _core_bpe is not None:
            self._core_bpe = _core_bpe
        else:
            assert mergeable_ranks is not None
            self._core_bpe = _tiktoken.CoreBPE(mergeable_ranks, special_tokens, pat_str)

    def __repr__(self) -> str:
        return f"<Encoding {self.name!r}>"

    @property
    def _mergeable_ranks(self) -> dict[bytes, int]:
        mergeable_ranks = self.__dict__.get("_mergeable_ranks")
        if mergeable_ranks is None:
            mergeable_ranks = self._core_bpe.mergeable_ranks()
            self.__dict__["_mergeable_ranks"] = mergeable_ranks
        return mergeable_ranks

    @_mergeable_ranks.setter
    def _mergeable_ranks(self, mergeable_ranks: dict[bytes, int] | None) -> None:
        self.__dict__["_mergeable_ranks"] = mergeable_ranks

    # ====================
    # Encoding
    # ====================

    def encode_ordinary(self, text: str) -> list[int]:
        """Encodes a string into tokens, ignoring special tokens.

        This is equivalent to `encode(text, disallowed_special=())` (but slightly faster).

        ```
        >>> enc.encode_ordinary("hello world")
        [31373, 995]
        """
        try:
            return self._core_bpe.encode_ordinary(text)
        except UnicodeEncodeError:
            # See comment in encode
            text = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
            return self._core_bpe.encode_ordinary(text)

    def encode(
        self,
        text: str,
        *,
        allowed_special: Literal["all"] | AbstractSet[str] = set(),  # noqa: B006
        disallowed_special: Literal["all"] | Collection[str] = "all",
    ) -> list[int]:
        """Encodes a string into tokens.

        Special tokens are artificial tokens used to unlock capabilities from a model,
        such as fill-in-the-middle. So we want to be careful about accidentally encoding special
        tokens, since they can be used to trick a model into doing something we don't want it to do.

        Hence, by default, encode will raise an error if it encounters text that corresponds
        to a special token. This can be controlled on a per-token level using the `allowed_special`
        and `disallowed_special` parameters. In particular:
        - Setting `disallowed_special` to () will prevent this function from raising errors and
          cause all text corresponding to special tokens to be encoded as natural text.
        - Setting `allowed_special` to "all" will cause this function to treat all text
          corresponding to special tokens to be encoded as special tokens.

        ```
        >>> enc.encode("hello world")
        [31373, 995]
        >>> enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
        [50256]
        >>> enc.encode("<|endoftext|>", allowed_special="all")
        [50256]
        >>> enc.encode("<|endoftext|>")
        # Raises ValueError
        >>> enc.encode("<|endoftext|>", disallowed_special=())
        [27, 91, 437, 1659, 5239, 91, 29]
        ```
        """
        if allowed_special == "all":
            allowed_special = self.special_tokens_set
        if disallowed_special == "all":
            disallowed_special = (
                self._special_tokens_set_frozen
                if not allowed_special
                else self._special_tokens_set_frozen - allowed_special
            )
        if disallowed_special:
            if not isinstance(disallowed_special, frozenset):
                disallowed_special = frozenset(disallowed_special)
            common_prefix = _special_token_common_prefix(disallowed_special)
            if not common_prefix or common_prefix in text:
                if match := _special_token_regex(disallowed_special).search(text):
                    raise_disallowed_special_token(match.group())

        try:
            if not allowed_special:
                return self._core_bpe.encode_ordinary(text)
            return self._core_bpe.encode(text, allowed_special)
        except UnicodeEncodeError:
            # BPE operates on bytes, but the regex operates on unicode. If we pass a str that is
            # invalid UTF-8 to Rust, it will rightfully complain. Here we do a quick and dirty
            # fixup for any surrogate pairs that may have sneaked their way into the text.
            # Technically, this introduces a place where encode + decode doesn't roundtrip a Python
            # string, but given that this is input we want to support, maybe that's okay.
            # Also we use errors="replace" to handle weird things like lone surrogates.
            text = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
            if not allowed_special:
                return self._core_bpe.encode_ordinary(text)
            return self._core_bpe.encode(text, allowed_special)

    def encode_to_numpy(
        self,
        text: str,
        *,
        allowed_special: Literal["all"] | AbstractSet[str] = set(),  # noqa: B006
        disallowed_special: Literal["all"] | Collection[str] = "all",
    ) -> npt.NDArray[np.uint32]:
        """Encodes a string into tokens, returning a numpy array.

        Avoids the overhead of copying the token buffer into a Python list.
        """
        if allowed_special == "all":
            allowed_special = self.special_tokens_set
        if disallowed_special == "all":
            disallowed_special = (
                self._special_tokens_set_frozen
                if not allowed_special
                else self._special_tokens_set_frozen - allowed_special
            )
        if disallowed_special:
            if not isinstance(disallowed_special, frozenset):
                disallowed_special = frozenset(disallowed_special)
            common_prefix = _special_token_common_prefix(disallowed_special)
            if not common_prefix or common_prefix in text:
                if match := _special_token_regex(disallowed_special).search(text):
                    raise_disallowed_special_token(match.group())

        import numpy as np

        buffer = self._core_bpe.encode_to_tiktoken_buffer(text, allowed_special)
        return np.frombuffer(buffer, dtype=np.uint32)

    def encode_ordinary_batch(self, text: list[str], *, num_threads: int = 8) -> list[list[int]]:
        """Encodes a list of strings into tokens, in parallel, ignoring special tokens.

        This is equivalent to `encode_batch(text, disallowed_special=())` (but slightly faster).

        ```
        >>> enc.encode_ordinary_batch(["hello world", "goodbye world"])
        [[31373, 995], [11274, 16390, 995]]
        ```
        """
        if num_threads <= 0:
            raise ValueError("max_workers must be greater than 0")

        try:
            batch_len = len(text)
        except TypeError:
            batch_len = None

        if batch_len == 0:
            return []

        if _use_native_batch(text, batch_len, num_threads):
            try:
                return self._core_bpe.encode_ordinary_batch(text)
            except (TypeError, UnicodeEncodeError):
                # Match encode_ordinary's surrogate fixup behavior by falling back to the
                # per-string path when any string cannot be passed to Rust as UTF-8.
                pass

        encoder = functools.partial(self.encode_ordinary)
        with ThreadPoolExecutor(num_threads) as e:
            return list(e.map(encoder, text))

    def encode_batch(
        self,
        text: list[str],
        *,
        num_threads: int = 8,
        allowed_special: Literal["all"] | AbstractSet[str] = set(),  # noqa: B006
        disallowed_special: Literal["all"] | Collection[str] = "all",
    ) -> list[list[int]]:
        """Encodes a list of strings into tokens, in parallel.

        See `encode` for more details on `allowed_special` and `disallowed_special`.

        ```
        >>> enc.encode_batch(["hello world", "goodbye world"])
        [[31373, 995], [11274, 16390, 995]]
        ```
        """
        if allowed_special == "all":
            allowed_special = self.special_tokens_set
        if disallowed_special == "all":
            disallowed_special = (
                self._special_tokens_set_frozen
                if not allowed_special
                else self._special_tokens_set_frozen - allowed_special
            )
        if not isinstance(disallowed_special, frozenset):
            disallowed_special = frozenset(disallowed_special)

        if num_threads <= 0:
            raise ValueError("max_workers must be greater than 0")

        try:
            batch_len = len(text)
        except TypeError:
            batch_len = None

        if batch_len == 0:
            return []

        if _use_native_batch(text, batch_len, num_threads):
            try:
                if disallowed_special:
                    common_prefix = _special_token_common_prefix(disallowed_special)
                    special_regex = None
                    for piece in text:
                        if common_prefix and common_prefix not in piece:
                            continue
                        if special_regex is None:
                            special_regex = _special_token_regex(disallowed_special)
                        if match := special_regex.search(piece):
                            raise_disallowed_special_token(match.group())
                if not allowed_special:
                    return self._core_bpe.encode_ordinary_batch(text)
                return self._core_bpe.encode_batch(text, allowed_special)
            except (TypeError, UnicodeEncodeError):
                pass

        encoder = functools.partial(
            self.encode, allowed_special=allowed_special, disallowed_special=disallowed_special
        )
        with ThreadPoolExecutor(num_threads) as e:
            return list(e.map(encoder, text))

    def encode_with_unstable(
        self,
        text: str,
        *,
        allowed_special: Literal["all"] | AbstractSet[str] = set(),  # noqa: B006
        disallowed_special: Literal["all"] | Collection[str] = "all",
    ) -> tuple[list[int], list[list[int]]]:
        """Encodes a string into stable tokens and possible completion sequences.

        Note that the stable tokens will only represent a substring of `text`.

        See `encode` for more details on `allowed_special` and `disallowed_special`.

        This API should itself be considered unstable.

        ```
        >>> enc.encode_with_unstable("hello fanta")
        ([31373], [(277, 4910), (5113, 265), ..., (8842,)])

        >>> text = "..."
        >>> stable_tokens, completions = enc.encode_with_unstable(text)
        >>> assert text.encode().startswith(enc.decode_bytes(stable_tokens))
        >>> assert all(enc.decode_bytes(stable_tokens + seq).startswith(text.encode()) for seq in completions)
        ```
        """
        if allowed_special == "all":
            allowed_special = self.special_tokens_set
        if disallowed_special == "all":
            disallowed_special = (
                self._special_tokens_set_frozen
                if not allowed_special
                else self._special_tokens_set_frozen - allowed_special
            )
        if disallowed_special:
            if not isinstance(disallowed_special, frozenset):
                disallowed_special = frozenset(disallowed_special)
            common_prefix = _special_token_common_prefix(disallowed_special)
            if not common_prefix or common_prefix in text:
                if match := _special_token_regex(disallowed_special).search(text):
                    raise_disallowed_special_token(match.group())

        return self._core_bpe.encode_with_unstable(text, allowed_special)

    def encode_single_token(self, text_or_bytes: str | bytes) -> int:
        """Encodes text corresponding to a single token to its token value.

        NOTE: this will encode all special tokens.

        Raises `KeyError` if the token is not in the vocabulary.

        ```
        >>> enc.encode_single_token("hello")
        31373
        ```
        """
        if isinstance(text_or_bytes, str):
            text_or_bytes = text_or_bytes.encode("utf-8")
        return self._core_bpe.encode_single_token(text_or_bytes)

    # ====================
    # Decoding
    # ====================

    def decode_bytes(self, tokens: Sequence[int]) -> bytes:
        """Decodes a list of tokens into bytes.

        ```
        >>> enc.decode_bytes([31373, 995])
        b'hello world'
        ```
        """
        return self._core_bpe.decode_bytes(tokens)

    def decode(self, tokens: Sequence[int], errors: str = "replace") -> str:
        """Decodes a list of tokens into a string.

        WARNING: the default behaviour of this function is lossy, since decoded bytes are not
        guaranteed to be valid UTF-8. You can control this behaviour using the `errors` parameter,
        for instance, setting `errors=strict`.

        ```
        >>> enc.decode([31373, 995])
        'hello world'
        ```
        """
        return self._core_bpe.decode_bytes(tokens).decode("utf-8", errors=errors)

    def decode_single_token_bytes(self, token: int) -> bytes:
        """Decodes a token into bytes.

        NOTE: this will decode all special tokens.

        Raises `KeyError` if the token is not in the vocabulary.

        ```
        >>> enc.decode_single_token_bytes(31373)
        b'hello'
        ```
        """
        return self._core_bpe.decode_single_token_bytes(token)

    def decode_tokens_bytes(self, tokens: Sequence[int]) -> list[bytes]:
        """Decodes a list of tokens into a list of bytes.

        Useful for visualising tokenisation.
        >>> enc.decode_tokens_bytes([31373, 995])
        [b'hello', b' world']
        """
        try:
            return self._core_bpe.decode_tokens_bytes(tokens)
        except TypeError:
            return [self.decode_single_token_bytes(token) for token in tokens]

    def decode_with_offsets(self, tokens: Sequence[int]) -> tuple[str, list[int]]:
        """Decodes a list of tokens into a string and a list of offsets.

        Each offset is the index into text corresponding to the start of each token.
        If UTF-8 character boundaries do not line up with token boundaries, the offset is the index
        of the first character that contains bytes from the token.

        This will currently raise if given tokens that decode to invalid UTF-8; this behaviour may
        change in the future to be more permissive.

        >>> enc.decode_with_offsets([31373, 995])
        ('hello world', [0, 5])
        """
        try:
            text_bytes, offsets = self._core_bpe.decode_with_offsets(tokens)
            text = text_bytes.decode("utf-8", errors="strict")
            return text, offsets
        except TypeError:
            token_bytes = self.decode_tokens_bytes(tokens)

            text_len = 0
            offsets = []
            for token in token_bytes:
                offsets.append(max(0, text_len - (0x80 <= token[0] < 0xC0)))
                text_len += sum(1 for c in token if not 0x80 <= c < 0xC0)

            text = b"".join(token_bytes).decode("utf-8", errors="strict")
            return text, offsets

    def decode_batch(
        self, batch: Sequence[Sequence[int]], *, errors: str = "replace", num_threads: int = 8
    ) -> list[str]:
        """Decodes a batch (list of lists of tokens) into a list of strings."""
        if num_threads <= 0:
            raise ValueError("max_workers must be greater than 0")

        try:
            batch_len = len(batch)
        except TypeError:
            batch_len = None

        if batch_len == 0:
            return []

        if _use_native_decode_batch(batch, batch_len, num_threads):
            try:
                return [
                    text.decode("utf-8", errors=errors)
                    for text in self._core_bpe.decode_bytes_batch(batch)
                ]
            except TypeError:
                pass

        decoder = functools.partial(self.decode, errors=errors)
        with ThreadPoolExecutor(num_threads) as e:
            return list(e.map(decoder, batch))

    def decode_bytes_batch(
        self, batch: Sequence[Sequence[int]], *, num_threads: int = 8
    ) -> list[bytes]:
        """Decodes a batch (list of lists of tokens) into a list of bytes."""
        if num_threads <= 0:
            raise ValueError("max_workers must be greater than 0")

        try:
            batch_len = len(batch)
        except TypeError:
            batch_len = None

        if batch_len == 0:
            return []

        if _use_native_decode_batch(batch, batch_len, num_threads):
            try:
                return self._core_bpe.decode_bytes_batch(batch)
            except TypeError:
                pass

        with ThreadPoolExecutor(num_threads) as e:
            return list(e.map(self.decode_bytes, batch))

    # ====================
    # Miscellaneous
    # ====================

    def token_byte_values(self) -> list[bytes]:
        """Returns the list of all token byte values."""
        return self._core_bpe.token_byte_values()

    @property
    def eot_token(self) -> int:
        return self._special_tokens["<|endoftext|>"]

    @functools.cached_property
    def special_tokens_set(self) -> set[str]:
        return set(self._special_tokens.keys())

    def is_special_token(self, token: int) -> bool:
        assert isinstance(token, int)
        return token in self._special_token_values

    @property
    def n_vocab(self) -> int:
        """For backwards compatibility. Prefer to use `enc.max_token_value + 1`."""
        return self.max_token_value + 1

    # ====================
    # Private
    # ====================

    def _encode_single_piece(self, text_or_bytes: str | bytes) -> list[int]:
        """Encodes text corresponding to bytes without a regex split.

        NOTE: this will not encode any special tokens.

        ```
        >>> enc.encode_single_piece("helloqqqq")
        [31373, 38227, 38227]
        ```
        """
        if isinstance(text_or_bytes, str):
            text_or_bytes = text_or_bytes.encode("utf-8")
        return self._core_bpe.encode_single_piece(text_or_bytes)

    def _encode_only_native_bpe(self, text: str) -> list[int]:
        """Encodes a string into tokens, but do regex splitting in Python."""
        # We need specifically `regex` in order to compile pat_str due to e.g. \p
        import regex

        _unused_pat = regex.compile(self._pat_str)
        ret = []
        for piece in regex.findall(_unused_pat, text):
            ret.extend(self._core_bpe.encode_single_piece(piece.encode("utf-8")))
        return ret

    def _encode_bytes(self, text: bytes) -> list[int]:
        return self._core_bpe._encode_bytes(text)

    def __getstate__(self) -> object:
        import tiktoken.registry

        # As an optimisation, pickle registered encodings by reference
        if self is tiktoken.registry.ENCODINGS.get(self.name):
            return self.name
        return {
            "name": self.name,
            "pat_str": self._pat_str,
            "mergeable_ranks": self._mergeable_ranks,
            "special_tokens": self._special_tokens,
        }

    def __setstate__(self, value: object) -> None:
        import tiktoken.registry

        if isinstance(value, str):
            self.__dict__ = tiktoken.registry.get_encoding(value).__dict__
            return
        self.__init__(**value)


@functools.lru_cache(maxsize=128)
def _special_token_regex(tokens: frozenset[str]) -> re.Pattern[str]:
    try:
        import regex as re
    except ImportError:
        import re
    inner = "|".join(re.escape(token) for token in tokens)
    return re.compile(f"({inner})")


@functools.lru_cache(maxsize=128)
def _special_token_common_prefix(tokens: frozenset[str]) -> str:
    if not tokens:
        return ""
    first = min(tokens)
    last = max(tokens)
    for i, char in enumerate(first):
        if i == len(last) or last[i] != char:
            return first[:i]
    return first


def _use_native_batch(text: list[str], batch_len: int | None, num_threads: int) -> bool:
    if batch_len is None:
        return False
    if num_threads == 1:
        return True

    try:
        head = min(batch_len, 32)
        sample_chars = 0
        sample_count = 0
        for i in range(head):
            sample_chars += len(text[i])
            sample_count += 1
        for i in range(max(head, batch_len - 32), batch_len):
            sample_chars += len(text[i])
            sample_count += 1
    except (IndexError, TypeError):
        return False

    return sample_chars <= sample_count * 256


def _use_native_decode_batch(
    batch: Sequence[Sequence[int]], batch_len: int | None, num_threads: int
) -> bool:
    if batch_len is None:
        return False
    if num_threads == 1:
        return True

    try:
        head = min(batch_len, 32)
        sample_tokens = 0
        sample_count = 0
        for i in range(head):
            sample_tokens += len(batch[i])
            sample_count += 1
        for i in range(max(head, batch_len - 32), batch_len):
            sample_tokens += len(batch[i])
            sample_count += 1
    except (IndexError, TypeError):
        return False

    if sample_tokens <= sample_count * 256:
        return True

    return batch_len >= 1000 and sample_tokens <= sample_count * 2048


def raise_disallowed_special_token(token: str) -> NoReturn:
    raise ValueError(
        f"Encountered text corresponding to disallowed special token {token!r}.\n"
        "If you want this text to be encoded as a special token, "
        f"pass it to `allowed_special`, e.g. `allowed_special={{{token!r}, ...}}`.\n"
        f"If you want this text to be encoded as normal text, disable the check for this token "
        f"by passing `disallowed_special=(enc.special_tokens_set - {{{token!r}}})`.\n"
        "To disable this check for all special tokens, pass `disallowed_special=()`.\n"
    )
