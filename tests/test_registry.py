import gc
import pickle
import weakref
from concurrent.futures import ThreadPoolExecutor
from threading import Event, RLock

import pytest

import tiktoken
import tiktoken.registry as registry


@pytest.fixture
def encoding_registry(monkeypatch):
    def constructor(name):
        return lambda: {
            "name": name,
            "pat_str": r"(?s).",
            "mergeable_ranks": {bytes([i]): i for i in range(256)},
            "special_tokens": {},
        }

    # Real Encoding objects with a small, offline vocabulary, isolated from other tests.
    monkeypatch.setattr(registry, "ENCODINGS", {})
    monkeypatch.setattr(
        registry,
        "ENCODING_CONSTRUCTORS",
        {name: constructor(name) for name in ("o200k_base", "test_encoding")},
    )


def test_clear_encoding_cache_releases_encodings(encoding_registry):
    enc = tiktoken.encoding_for_model("gpt-4o")
    assert enc is tiktoken.get_encoding("o200k_base")
    ref = weakref.ref(enc)
    del enc
    gc.collect()
    assert ref() is not None

    tiktoken.clear_encoding_cache()
    gc.collect()
    assert ref() is None


def test_clear_encoding_cache_preserves_live_references(encoding_registry):
    enc = tiktoken.get_encoding("o200k_base")
    encode = enc.encode
    expected = enc.encode("hello 🌍")
    ref = weakref.ref(enc)

    tiktoken.clear_encoding_cache()
    replacement = tiktoken.get_encoding("o200k_base")
    assert replacement is not enc
    assert replacement is tiktoken.encoding_for_model("gpt-4o")
    assert replacement.encode("hello 🌍") == expected
    assert enc.decode(encode("hello 🌍")) == "hello 🌍"

    del enc
    gc.collect()
    assert ref() is not None  # A bound method still owns its Encoding.
    del encode
    gc.collect()
    assert ref() is None


def test_clear_encoding_cache_keeps_constructors(encoding_registry):
    names = tiktoken.list_encoding_names()
    refs = [weakref.ref(tiktoken.get_encoding(name)) for name in names]
    tiktoken.clear_encoding_cache()
    tiktoken.clear_encoding_cache()  # Clearing an empty cache is safe.
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert tiktoken.list_encoding_names() == names
    for name in names:
        enc = tiktoken.get_encoding(name)
        assert enc.decode(enc.encode("hello")) == "hello"


def test_clear_encoding_cache_before_first_load(monkeypatch):
    monkeypatch.setattr(registry, "ENCODINGS", {})
    monkeypatch.setattr(registry, "ENCODING_CONSTRUCTORS", None)
    tiktoken.clear_encoding_cache()
    assert registry.ENCODING_CONSTRUCTORS is None


def test_encoding_pickle_after_cache_clear(encoding_registry):
    enc = tiktoken.get_encoding("test_encoding")
    cached_pickle = pickle.dumps(enc)
    tiktoken.clear_encoding_cache()
    uncached_pickle = pickle.dumps(enc)

    for data in (cached_pickle, uncached_pickle):
        restored = pickle.loads(data)
        assert restored.name == enc.name
        assert restored.encode("hello 🌍") == enc.encode("hello 🌍")


def test_clear_encoding_cache_during_construction(encoding_registry, monkeypatch):
    constructing = Event()
    finish_construction = Event()
    clearing = Event()
    lock = RLock()
    constructor = registry.ENCODING_CONSTRUCTORS["test_encoding"]

    class ObservedLock:
        def __enter__(self):
            if constructing.is_set():
                clearing.set()
            lock.acquire()

        def __exit__(self, *args):
            lock.release()

    def slow_constructor():
        constructing.set()
        assert finish_construction.wait(timeout=5)
        return constructor()

    monkeypatch.setattr(registry, "_lock", ObservedLock())
    monkeypatch.setitem(registry.ENCODING_CONSTRUCTORS, "test_encoding", slow_constructor)
    with ThreadPoolExecutor(max_workers=2) as pool:
        loading = pool.submit(tiktoken.get_encoding, "test_encoding")
        try:
            assert constructing.wait(timeout=5)
            cleared = pool.submit(tiktoken.clear_encoding_cache)
            assert clearing.wait(timeout=5)
        finally:
            finish_construction.set()
        original = loading.result(timeout=5)
        cleared.result(timeout=5)

    replacement = tiktoken.get_encoding("test_encoding")
    assert replacement is not original
    assert original.encode("hello") == replacement.encode("hello")


def test_clear_encoding_cache_during_cached_lookup(encoding_registry, monkeypatch):
    original = tiktoken.get_encoding("test_encoding")
    looked_up = Event()
    resume_lookup = Event()

    class PausingCache(dict):
        def __contains__(self, key):
            found = super().__contains__(key)
            looked_up.set()
            assert resume_lookup.wait(timeout=5)
            return found

        def get(self, key, default=None):
            value = super().get(key, default)
            looked_up.set()
            assert resume_lookup.wait(timeout=5)
            return value

    monkeypatch.setattr(registry, "ENCODINGS", PausingCache(registry.ENCODINGS))
    with ThreadPoolExecutor(max_workers=1) as pool:
        lookup = pool.submit(tiktoken.get_encoding, "test_encoding")
        try:
            # Clear after the reader's first dictionary operation, before it returns.
            assert looked_up.wait(timeout=5)
            tiktoken.clear_encoding_cache()
        finally:
            resume_lookup.set()
        retained = lookup.result(timeout=5)

    assert retained is original
    replacement = tiktoken.get_encoding("test_encoding")
    assert replacement is not original
    assert retained.encode("hello") == replacement.encode("hello")
