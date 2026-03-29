# tests/test_token_ids_unique.py
# Checks that token IDs are unique. We don't check token "names" (dict keys are unique by definition).

import pytest
import tiktoken
from collections import defaultdict

ENCODING_NAMES = tiktoken.list_encoding_names()

@pytest.mark.parametrize("enc_name", ENCODING_NAMES)
def test_special_token_ids_are_unique(enc_name):
    """
    Special tokens: no two different names should share the same token id.
    """
    enc = tiktoken.get_encoding(enc_name)
    sp = getattr(enc, "_special_tokens", {})
    if not sp:
        pytest.skip(f"{enc_name}: no special tokens")

    id2names = defaultdict(list)
    for name, tid in sp.items():
        id2names[tid].append(name)

    dups = {tid: names for tid, names in id2names.items() if len(names) > 1}
    assert not dups, f"{enc_name}: duplicated special token ids: {dups}"

@pytest.mark.parametrize("enc_name", ENCODING_NAMES)
def test_mergeable_token_ids_are_unique(enc_name):
    """
    Mergeable (vocab) tokens: token ids should be unique.
    Note: some builds may not expose `_mergeable_ranks` on Python side; skip in that case.
    """
    enc = tiktoken.get_encoding(enc_name)
    mr = getattr(enc, "_mergeable_ranks", None)
    if not mr:
        pytest.skip(f"{enc_name}: mergeable ranks not exposed")

    ids = list(mr.values())
    assert len(ids) == len(set(ids)), f"{enc_name}: duplicated mergeable token ids"

