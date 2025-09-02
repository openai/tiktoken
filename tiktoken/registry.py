from __future__ import annotations

import functools
import importlib
import pkgutil
import threading
from typing import Any, Callable, Sequence

import tiktoken_ext

import tiktoken
from tiktoken.core import Encoding

_lock = threading.RLock()
ENCODINGS: dict[str, Encoding] = {}
ENCODING_CONSTRUCTORS: dict[str, Callable[[], dict[str, Any]]] | None = None


@functools.lru_cache
def _available_plugin_modules() -> Sequence[str]:
    """Returns a sequence of available plugin modules.

    Args:
        None

    Returns:
        Sequence[str]: List of available plugin module names.

    Note:
        This function inspects tiktoken_ext namespace package for available plugin modules.
        Submodules inside tiktoken_ext will be checked for ENCODING_CONSTRUCTORS attributes.
        Uses namespace package pattern for faster pkgutil.iter_modules operation.

        tiktoken_ext is implemented as a separate top-level package because namespace
        subpackages of non-namespace packages don't work as expected with editable installs.
    """
    # tiktoken_ext is a namespace package
    # submodules inside tiktoken_ext will be inspected for ENCODING_CONSTRUCTORS attributes
    # - we use namespace package pattern so `pkgutil.iter_modules` is fast
    # - it's a separate top-level package because namespace subpackages of non-namespace
    #   packages don't quite do what you want with editable installs
    mods = []
    plugin_mods = pkgutil.iter_modules(tiktoken_ext.__path__, tiktoken_ext.__name__ + ".")
    for _, mod_name, _ in plugin_mods:
        mods.append(mod_name)
    return mods


def _find_constructors() -> None:
    """Searches for and registers encoding constructors from available plugin modules.

    This function populates the global ENCODING_CONSTRUCTORS dictionary by searching through
    available plugin modules to find encoding constructors. It ensures there are no duplicate
    encoding names across different plugins.

    Args:
        None

    Raises:
        ValueError: If either:
            - A plugin module does not define ENCODING_CONSTRUCTORS
            - There are duplicate encoding names across plugins

    Note:
        - Uses a lock to ensure thread safety when populating the global dictionary
        - If ENCODING_CONSTRUCTORS is already populated, returns early
        - In case of any exception, ENCODING_CONSTRUCTORS is reset to None before re-raising
    """
    global ENCODING_CONSTRUCTORS
    with _lock:
        if ENCODING_CONSTRUCTORS is not None:
            return
        ENCODING_CONSTRUCTORS = {}

        try:
            for mod_name in _available_plugin_modules():
                mod = importlib.import_module(mod_name)
                try:
                    constructors = mod.ENCODING_CONSTRUCTORS
                except AttributeError as e:
                    raise ValueError(
                        f"tiktoken plugin {mod_name} does not define ENCODING_CONSTRUCTORS"
                    ) from e
                for enc_name, constructor in constructors.items():
                    if enc_name in ENCODING_CONSTRUCTORS:
                        raise ValueError(
                            f"Duplicate encoding name {enc_name} in tiktoken plugin {mod_name}"
                        )
                    ENCODING_CONSTRUCTORS[enc_name] = constructor
        except Exception:
            # Ensure we idempotently raise errors
            ENCODING_CONSTRUCTORS = None
            raise




def get_encoding(encoding_name: str) -> Encoding:
    """Returns an Encoding object for the given encoding name.

    If the encoding has been previously loaded, returns the cached version.
    Otherwise, constructs a new Encoding object using the appropriate constructor.

    Args:
        encoding_name (str): The name of the encoding to retrieve.

    Returns:
        Encoding: An Encoding object for the specified encoding.

    Raises:
        ValueError: If either:
            - encoding_name is not a string
            - the specified encoding name is unknown

    Examples:
        >>> encoding = get_encoding("gpt2")
        >>> encoding
        <Encoding object at 0x...>
    """
    if not isinstance(encoding_name, str):
        raise ValueError(f"Expected a string in get_encoding, got {type(encoding_name)}")

    if encoding_name in ENCODINGS:
        return ENCODINGS[encoding_name]

    with _lock:
        if encoding_name in ENCODINGS:
            return ENCODINGS[encoding_name]

        if ENCODING_CONSTRUCTORS is None:
            _find_constructors()
            assert ENCODING_CONSTRUCTORS is not None

        if encoding_name not in ENCODING_CONSTRUCTORS:
            raise ValueError(
                f"Unknown encoding {encoding_name}.\n"
                f"Plugins found: {_available_plugin_modules()}\n"
                f"tiktoken version: {tiktoken.__version__} (are you on latest?)"
            )

        constructor = ENCODING_CONSTRUCTORS[encoding_name]
        enc = Encoding(**constructor())
        ENCODINGS[encoding_name] = enc
        return enc


def list_encoding_names() -> list[str]:
    """Lists all available encoding names that can be used with tiktoken.

    Args:
        None

    Returns:
        list[str]: List of available encoding names.

    Examples:
        >>> list_encoding_names()
        ['gpt2', 'r50k_base', 'p50k_base', ...]
    """
    with _lock:
        if ENCODING_CONSTRUCTORS is None:
            _find_constructors()
            assert ENCODING_CONSTRUCTORS is not None
        return list(ENCODING_CONSTRUCTORS)
