import hashlib
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

import tiktoken.load


def test_read_file_cached_concurrent_writers():
    contents = b"concurrent cache contents"
    expected_hash = hashlib.sha256(contents).hexdigest()
    blobpath = "https://example.invalid/encoding.tiktoken"
    downloaded = Barrier(2)

    def read_file(path):
        assert path == blobpath
        # Both readers must miss the cache before either can populate it.
        downloaded.wait(timeout=10)
        return contents

    with tempfile.TemporaryDirectory() as directory:
        cache_dir = Path(directory) / "cache"
        with patch.dict(os.environ, {"TIKTOKEN_CACHE_DIR": str(cache_dir)}):
            with patch.object(tiktoken.load, "read_file", read_file):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [
                        executor.submit(tiktoken.load.read_file_cached, blobpath, expected_hash)
                        for _ in range(2)
                    ]
                    assert [future.result(timeout=15) for future in futures] == [contents, contents]

            assert tiktoken.load.read_file_cached(blobpath, expected_hash) == contents
            assert len(list(cache_dir.iterdir())) == 1
