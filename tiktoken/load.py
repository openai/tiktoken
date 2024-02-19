from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import uuid
from typing import Optional

import requests


def read_file(blobpath: str) -> bytes:
    """
    Reads the contents of a file specified by the given blobpath.

    Parameters
    ----------
    blobpath : str
        The path or URL to the file to be read.

    Returns
    -------
    bytes
        The binary content of the file.
    """
    if not blobpath.startswith("http://") and not blobpath.startswith("https://"):
        try:
            import blobfile
        except ImportError as e:
            raise ImportError(
                "blobfile is not installed. Please install it by running `pip install blobfile`."
            ) from e
        with blobfile.BlobFile(blobpath, "rb") as f:
            return f.read()
    # avoiding blobfile for public files helps avoid auth issues, like MFA prompts
    resp = requests.get(blobpath)
    resp.raise_for_status()
    return resp.content


def check_hash(data: bytes, expected_hash: str) -> bool:
    """
    Checks if the hash of the given data matches the expected hash.

    Parameters
    ----------
    data : bytes
        The binary data to be hashed.

    expected_hash : str
        The expected hash value.

    Returns
    -------
    bool
        True if the actual hash matches the expected hash, False otherwise.
    """
    actual_hash = hashlib.sha256(data).hexdigest()
    return actual_hash == expected_hash


def read_file_cached(blobpath: str, expected_hash: Optional[str] = None) -> bytes:
    """
    Reads the contents of a file specified by the given blobpath from cache if available,
    otherwise fetches it from the source, caches it, and returns the content.

    Parameters
    ----------
    blobpath : str
        The path or URL to the file to be read.

    expected_hash : str, optional
        The expected hash value of the file content. Default is None.

    Returns
    -------
    bytes
        The binary content of the file.
    """
    user_specified_cache = True
    if "TIKTOKEN_CACHE_DIR" in os.environ:
        cache_dir = os.environ["TIKTOKEN_CACHE_DIR"]
    elif "DATA_GYM_CACHE_DIR" in os.environ:
        cache_dir = os.environ["DATA_GYM_CACHE_DIR"]
    else:
        cache_dir = os.path.join(tempfile.gettempdir(), "data-gym-cache")
        user_specified_cache = False

    if cache_dir == "":
        # disable caching
        return read_file(blobpath)

    cache_key = hashlib.sha1(blobpath.encode()).hexdigest()

    cache_path = os.path.join(cache_dir, cache_key)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            data = f.read()
        if expected_hash is None or check_hash(data, expected_hash):
            return data

        # the cached file does not match the hash, remove it and re-fetch
        try:
            os.remove(cache_path)
        except OSError:
            pass

    contents = read_file(blobpath)
    if expected_hash and not check_hash(contents, expected_hash):
        raise ValueError(
            f"Hash mismatch for data downloaded from {blobpath} (expected {expected_hash}). "
            f"This may indicate a corrupted download. Please try again."
        )

    try:
        os.makedirs(cache_dir, exist_ok=True)
        tmp_filename = cache_path + "." + str(uuid.uuid4()) + ".tmp"
        with open(tmp_filename, "wb") as f:
            f.write(contents)
        os.rename(tmp_filename, cache_path)
    except OSError:
        # don't raise if we can't write to the default cache, e.g. issue #75
        if user_specified_cache:
            raise

    return contents


def data_gym_to_mergeable_bpe_ranks(
    vocab_bpe_file: str,
    encoder_json_file: str,
    vocab_bpe_hash: Optional[str] = None,
    encoder_json_hash: Optional[str] = None,
) -> dict[bytes, int]:
    """
    Converts a vocab BPE file and an encoder JSON file into mergeable BPE ranks.

    Parameters
    ----------
    vocab_bpe_file : str
        The path to the vocabulary BPE file.

    encoder_json_file : str
        The path to the encoder JSON file.

    vocab_bpe_hash : str, optional
        The expected hash value of the vocabulary BPE file. Default is None.

    encoder_json_hash : str, optional
        The expected hash value of the encoder JSON file. Default is None.

    Returns
    -------
    dict[bytes, int]
        A dictionary mapping mergeable BPE tokens to their ranks.
    """
    # NB: do not add caching to this function
    rank_to_intbyte = [b for b in range(2**8) if chr(b).isprintable() and chr(b) != " "]

    data_gym_byte_to_byte = {chr(b): b for b in rank_to_intbyte}
    n = 0
    for b in range(2**8):
        if b not in rank_to_intbyte:
            rank_to_intbyte.append(b)
            data_gym_byte_to_byte[chr(2**8 + n)] = b
            n += 1
    assert len(rank_to_intbyte) == 2**8

    # vocab_bpe contains the merges along with associated ranks
    vocab_bpe_contents = read_file_cached(vocab_bpe_file, vocab_bpe_hash).decode()
    bpe_merges = [tuple(merge_str.split()) for merge_str in vocab_bpe_contents.split("\n")[1:-1]]

    def decode_data_gym(value: str) -> bytes:
        return bytes(data_gym_byte_to_byte[b] for b in value)

    # add the single byte tokens
    bpe_ranks = {bytes([b]): i for i, b in enumerate(rank_to_intbyte)}
    # add the merged tokens
    n = len(bpe_ranks)
    for first, second in bpe_merges:
        bpe_ranks[decode_data_gym(first) + decode_data_gym(second)] = n
        n += 1

    # check that the encoder file matches the merges file
    # this sanity check is important since tiktoken assumes that ranks are ordered the same
    # as merge priority
    encoder_json = json.loads(read_file_cached(encoder_json_file, encoder_json_hash))
    encoder_json_loaded = {decode_data_gym(k): v for k, v in encoder_json.items()}
    # drop these two special tokens if present, since they're not mergeable bpe tokens
    encoder_json_loaded.pop(b"<|endoftext|>", None)
    encoder_json_loaded.pop(b"<|startoftext|>", None)
    assert bpe_ranks == encoder_json_loaded

    return bpe_ranks


def dump_tiktoken_bpe(bpe_ranks: dict[bytes, int], tiktoken_bpe_file: str) -> None:
    """
    Dumps the mergeable BPE ranks to a TikToken BPE file.

    Parameters
    ----------
    bpe_ranks : dict[bytes, int]
        A dictionary mapping mergeable BPE tokens to their ranks.

    tiktoken_bpe_file : str
        The path to the TikToken BPE file.

    Returns
    -------
    None
    """
    try:
        import blobfile
    except ImportError as e:
        raise ImportError(
            "blobfile is not installed. Please install it by running `pip install blobfile`."
        ) from e
    with blobfile.BlobFile(tiktoken_bpe_file, "wb") as f:
        for token, rank in sorted(bpe_ranks.items(), key=lambda x: x[1]):
            f.write(base64.b64encode(token) + b" " + str(rank).encode() + b"\n")


def load_tiktoken_bpe(
    tiktoken_bpe_file: str, expected_hash: Optional[str] = None
) -> dict[bytes, int]:
    """
    Loads mergeable BPE ranks from a TikToken BPE file.

    Parameters
    ----------
    tiktoken_bpe_file : str
        The path to the TikToken BPE file.
        
    expected_hash : str, optional
        The expected hash value of the file content. Default is None.

    Returns
    -------
    dict[bytes, int]
        A dictionary mapping mergeable BPE tokens to their ranks.
    """
    # NB: do not add caching to this function
    contents = read_file_cached(tiktoken_bpe_file, expected_hash)
    return {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in contents.splitlines() if line)
    }
