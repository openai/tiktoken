"""This is an educational implementation of the byte pair encoding algorithm."""

from __future__ import annotations

import heapq

import regex

import tiktoken


class SimpleBytePairEncoding:
    def __init__(self, *, pat_str: str, mergeable_ranks: dict[bytes, int]) -> None:
        """Creates an Encoding object."""
        # A regex pattern string that is used to split the input text
        self.pat_str = pat_str
        # A dictionary mapping token bytes to their ranks. The ranks correspond to merge priority
        self.mergeable_ranks = mergeable_ranks

        self._decoder = {
            token: token_bytes for token_bytes, token in mergeable_ranks.items()
        }
        self._pat = regex.compile(pat_str)

    def encode(self, text: str, visualise: str | None = "colour") -> list[int]:
        """Encodes a string into tokens.

        >>> enc.encode("hello world")
        [388, 372]
        """
        # Use the regex to split the text into (approximately) words
        words = self._pat.findall(text)
        tokens = []
        for word in words:
            # Turn each word into tokens, using the byte pair encoding algorithm
            word_bytes = word.encode("utf-8")
            word_tokens = bpe_encode(
                self.mergeable_ranks, word_bytes, visualise=visualise
            )
            tokens.extend(word_tokens)
        return tokens

    def decode_bytes(self, tokens: list[int]) -> bytes:
        """Decodes a list of tokens into bytes.

        >>> enc.decode_bytes([388, 372])
        b'hello world'
        """
        return b"".join(self._decoder[token] for token in tokens)

    def decode(self, tokens: list[int]) -> str:
        """Decodes a list of tokens into a string.

        Decoded bytes are not guaranteed to be valid UTF-8. In that case, we replace
        the invalid bytes with the replacement character "�".

        >>> enc.decode([388, 372])
        'hello world'
        """
        return self.decode_bytes(tokens).decode("utf-8", errors="replace")

    def decode_tokens_bytes(self, tokens: list[int]) -> list[bytes]:
        """Decodes a list of tokens into a list of bytes.

        Useful for visualising how a string is tokenised.

        >>> enc.decode_tokens_bytes([388, 372])
        [b'hello', b' world']
        """
        return [self._decoder[token] for token in tokens]

    @staticmethod
    def train(training_data: str, vocab_size: int, pat_str: str):
        """Train a BPE tokeniser on some data!"""
        mergeable_ranks = bpe_train(
            data=training_data, vocab_size=vocab_size, pat_str=pat_str
        )
        return SimpleBytePairEncoding(pat_str=pat_str, mergeable_ranks=mergeable_ranks)

    @staticmethod
    def from_tiktoken(encoding):
        if isinstance(encoding, str):
            encoding = tiktoken.get_encoding(encoding)
        return SimpleBytePairEncoding(
            pat_str=encoding._pat_str, mergeable_ranks=encoding._mergeable_ranks
        )


def bpe_encode(
    mergeable_ranks: dict[bytes, int], input: bytes, visualise: str | None = "colour"
) -> list[int]:
    parts = [bytes([b]) for b in input]
    while True:
        # See the intermediate merges play out!
        if visualise:
            if visualise in ["colour", "color"]:
                visualise_tokens(parts)
            elif visualise == "simple":
                print(parts)

        # Iterate over all pairs and find the pair we want to merge the most
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank

        # If there were no pairs we could merge, we're done!
        if min_rank is None:
            break
        assert min_idx is not None

        # Otherwise, merge that pair and leave the rest unchanged. Then repeat.
        parts = (
            parts[:min_idx]
            + [parts[min_idx] + parts[min_idx + 1]]
            + parts[min_idx + 2 :]
        )

    if visualise:
        print()

    tokens = [mergeable_ranks[part] for part in parts]
    return tokens


def bpe_train(
    data: str, vocab_size: int, pat_str: str, visualise: str | None = "colour"
) -> dict[bytes, int]:
    # First, add tokens for each individual byte value
    if vocab_size < 2**8:
        raise ValueError("vocab_size must be at least 256, so we can encode all bytes")
    ranks = {}
    for i in range(2**8):
        ranks[bytes([i])] = i

    # Keep each word as a linked list. A merge only changes the two pairs touching the
    # merged nodes, so training no longer rescans the whole corpus for every new token.
    tokens: list[bytes] = []
    previous: list[int] = []
    following: list[int] = []
    heads: list[int | None] = []
    for word in regex.findall(pat_str, data):
        word_bytes = word.encode("utf-8")
        word_ids = []
        for value in word_bytes:
            word_ids.append(len(tokens))
            tokens.append(bytes([value]))
            previous.append(word_ids[-2] if len(word_ids) > 1 else -1)
            following.append(-1)
        for left, right in zip(word_ids, word_ids[1:]):
            following[left] = right
        heads.append(word_ids[0] if word_ids else None)

    # Each entry stores the left node of an adjacent pair. Node ids preserve the original
    # word order, which gives the same first-seen tie break as the reference implementation.
    occurrences: dict[tuple[bytes, bytes], set[int]] = {}
    first_nodes: dict[tuple[bytes, bytes], list[int]] = {}
    candidates: list[tuple[int, int, tuple[bytes, bytes]]] = []

    def add_occurrence(pair: tuple[bytes, bytes], left: int) -> None:
        pair_occurrences = occurrences.setdefault(pair, set())
        if left not in pair_occurrences:
            pair_occurrences.add(left)
            first_nodes.setdefault(pair, [])
            heapq.heappush(first_nodes[pair], left)

    def remove_occurrence(pair: tuple[bytes, bytes], left: int) -> None:
        pair_occurrences = occurrences.get(pair)
        if pair_occurrences is not None:
            pair_occurrences.discard(left)

    def current_first(pair: tuple[bytes, bytes]) -> int:
        pair_occurrences = occurrences[pair]
        first = first_nodes[pair]
        while first and first[0] not in pair_occurrences:
            heapq.heappop(first)
        return first[0]

    def pair_at(left: int) -> tuple[bytes, bytes] | None:
        right = following[left]
        if right == -1:
            return None
        return tokens[left], tokens[right]

    for head in heads:
        left = head
        while left is not None and left != -1 and following[left] != -1:
            pair = pair_at(left)
            assert pair is not None
            add_occurrence(pair, left)
            left = following[left]

    for pair, pair_occurrences in occurrences.items():
        if pair_occurrences:
            heapq.heappush(
                candidates, (-len(pair_occurrences), current_first(pair), pair)
            )

    # Now, use our data to figure out which merges we should make
    while len(ranks) < vocab_size:
        # Discard stale heap entries left behind by earlier local updates.
        while candidates:
            neg_count, first, most_common_pair = heapq.heappop(candidates)
            pair_occurrences = occurrences[most_common_pair]
            if pair_occurrences and (-neg_count, first) == (
                len(pair_occurrences),
                current_first(most_common_pair),
            ):
                break
        else:
            raise ValueError("training data has no mergeable pairs")

        token_bytes = most_common_pair[0] + most_common_pair[1]
        token = len(ranks)
        # Add the new token!
        ranks[token_bytes] = token

        # Merge matching occurrences from left to right. This preserves the reference
        # implementation's non-overlapping merge rule (e.g. "aaa" merges as "aa" + "a").
        affected: set[tuple[bytes, bytes]] = {most_common_pair}
        for left in sorted(occurrences[most_common_pair]):
            right = following[left]
            if right == -1 or pair_at(left) != most_common_pair:
                continue

            before = previous[left]
            after = following[right]
            for node in (before, left, right):
                if node != -1:
                    pair = pair_at(node)
                    if pair is not None:
                        remove_occurrence(pair, node)
                        affected.add(pair)

            tokens[left] = token_bytes
            following[left] = after
            following[right] = -1
            if after != -1:
                previous[after] = left
            if before != -1:
                pair = pair_at(before)
                if pair is not None:
                    add_occurrence(pair, before)
                    affected.add(pair)
            pair = pair_at(left)
            if pair is not None:
                add_occurrence(pair, left)
                affected.add(pair)

        for pair in affected:
            pair_occurrences = occurrences[pair]
            if pair_occurrences:
                heapq.heappush(
                    candidates, (-len(pair_occurrences), current_first(pair), pair)
                )

        # See the intermediate merges play out!
        if visualise:
            print(
                f"The current most common pair is {most_common_pair[0]} + {most_common_pair[1]}"
            )
            print(f"So we made {token_bytes} our {len(ranks)}th token")
            if visualise in ["colour", "color"]:
                print("Now the first fifty words in our training data look like:")
                visualise_tokens(
                    [
                        tokens[node]
                        for head in heads[:50]
                        for node in _linked_nodes(head, following)
                    ]
                )
            elif visualise == "simple":
                print("Now the first twenty words in our training data look like:")
                for head in heads[:20]:
                    print([tokens[node] for node in _linked_nodes(head, following)])
            print("\n")

    return ranks


def _linked_nodes(head: int | None, following: list[int]):
    while head is not None and head != -1:
        yield head
        head = following[head]


def visualise_tokens(token_values: list[bytes]) -> None:
    background = [f"\u001b[48;5;{i}m" for i in [167, 179, 185, 77, 80, 68, 134]]
    # If token boundaries do not occur at unicode character boundaries, it's unclear how best to
    # visualise the token. Here, we'll just use the unicode replacement character to represent some
    # fraction of a character.
    unicode_token_values = [x.decode("utf-8", errors="replace") for x in token_values]

    running_length = 0
    last_color = None
    for token in unicode_token_values:
        color = background[running_length % len(background)]
        if color == last_color:
            color = background[(running_length + 1) % len(background)]
            assert color != last_color
        last_color = color
        running_length += len(token)
        print(color + token, end="")
    print("\u001b[0m")


def train_simple_encoding():
    gpt2_pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open(__file__) as f:
        data = f.read()

    enc = SimpleBytePairEncoding.train(data, vocab_size=600, pat_str=gpt2_pattern)

    print("This is the sequence of merges performed in order to encode 'hello world':")
    tokens = enc.encode("hello world")
    assert enc.decode(tokens) == "hello world"
    assert enc.decode_bytes(tokens) == b"hello world"
    assert enc.decode_tokens_bytes(tokens) == [b"hello", b" world"]

    return enc
