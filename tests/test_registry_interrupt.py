"""Registry recovery after an interrupted plugin import."""

from types import SimpleNamespace

import pytest

import tiktoken
import tiktoken.registry as registry


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit, RuntimeError])
@pytest.mark.parametrize("failed_module", ["first", "second"])
@pytest.mark.parametrize("entrypoint", ["get", "list"])
def test_interrupted_discovery_is_retried(monkeypatch, error_type, failed_module, entrypoint):
    monkeypatch.setattr(registry, "ENCODINGS", {})
    monkeypatch.setattr(registry, "ENCODING_CONSTRUCTORS", None)
    monkeypatch.setattr(registry, "_available_plugin_modules", lambda: ["first", "second"])
    failure = error_type("Synthetic interrupted plugin")
    should_fail = True
    imports = []

    def constructor(name):
        return lambda: dict(
            name=name,
            pat_str=r"(?s:.)",
            mergeable_ranks={bytes([value]): value for value in range(256)},
            special_tokens={},
        )

    modules = {
        "first": SimpleNamespace(
            ENCODING_CONSTRUCTORS={"first_encoding": constructor("first_encoding")}
        ),
        "second": SimpleNamespace(
            ENCODING_CONSTRUCTORS={"second_encoding": constructor("second_encoding")}
        ),
    }

    def import_module(name):
        nonlocal should_fail
        imports.append(name)
        if name == failed_module and should_fail:
            should_fail = False
            raise failure
        return modules[name]

    monkeypatch.setattr(registry.importlib, "import_module", import_module)
    invoke = (
        (lambda: registry.get_encoding("second_encoding"))
        if entrypoint == "get"
        else registry.list_encoding_names
    )
    with pytest.raises(error_type) as raised:
        invoke()
    assert raised.value is failure
    assert registry.ENCODING_CONSTRUCTORS is None
    assert registry.ENCODINGS == {}

    imports.clear()
    assert registry.list_encoding_names() == ["first_encoding", "second_encoding"]
    assert imports == ["first", "second"]
    encoding = registry.get_encoding("second_encoding")
    assert isinstance(encoding, tiktoken.Encoding)
    assert encoding.decode(encoding.encode("synthetic café")) == "synthetic café"
    assert registry.get_encoding("second_encoding") is encoding
    assert imports == ["first", "second"]


def test_repeated_interruptions_do_not_publish_partial_constructors(monkeypatch):
    monkeypatch.setattr(registry, "ENCODINGS", {})
    monkeypatch.setattr(registry, "ENCODING_CONSTRUCTORS", None)
    monkeypatch.setattr(registry, "_available_plugin_modules", lambda: ["first", "broken"])
    failure = KeyboardInterrupt("Synthetic repeated interruption")

    def import_module(name):
        if name == "broken":
            raise failure
        return SimpleNamespace(ENCODING_CONSTRUCTORS={"synthetic": lambda: {}})

    monkeypatch.setattr(registry.importlib, "import_module", import_module)
    for _ in range(3):
        with pytest.raises(KeyboardInterrupt) as raised:
            registry.list_encoding_names()
        assert raised.value is failure
        assert registry.ENCODING_CONSTRUCTORS is None
