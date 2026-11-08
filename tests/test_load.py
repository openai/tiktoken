import hashlib

import tiktoken.load


def test_read_file_cached_after_another_process_filled_the_cache(tmp_path, monkeypatch):
    # Two processes filling an empty cache at the same time both finish with a rename
    # onto the same path. os.rename fails on Windows when the target exists, so the
    # slower process used to raise FileExistsError; os.replace is atomic everywhere.
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))
    blobpath = "https://example.invalid/encodings/test.tiktoken"
    contents = b"some bpe ranks"
    cache_path = tmp_path / hashlib.sha1(blobpath.encode()).hexdigest()

    def read_file_and_lose_the_race(path):
        assert path == blobpath
        cache_path.write_bytes(contents)  # the other process got there first
        return contents

    monkeypatch.setattr(tiktoken.load, "read_file", read_file_and_lose_the_race)

    assert tiktoken.load.read_file_cached(blobpath) == contents
    assert cache_path.read_bytes() == contents
    # and no leftover .tmp file next to it
    assert [p.name for p in tmp_path.iterdir()] == [cache_path.name]
