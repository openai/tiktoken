import pytest

import tiktoken


def test_duplicate_mergeable_ranks_raise_value_error():
    with pytest.raises(ValueError):
        tiktoken.Encoding(
            name="duplicate_ranks",
            pat_str=r".",
            mergeable_ranks={b"a": 0, b"b": 0, b"c": 1},
            special_tokens={},
        )
