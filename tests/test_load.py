"""Tests for ``tiktoken.load``."""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest

from tiktoken import load


class _FakeResponse:
    def __init__(self, content: bytes = b"") -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


def test_read_file_https_passes_default_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: ``read_file`` must pass a non-None timeout to ``requests.get``.

    Without it, ``requests.get`` blocks indefinitely on DNS/SYN/TCP-reset
    failures and silently hangs ``encoding_for_model`` on first use.
    """
    monkeypatch.delenv("TIKTOKEN_HTTP_TIMEOUT", raising=False)
    captured: dict[str, Any] = {}

    def fake_get(url: str, *, timeout: Any = None, **_: Any) -> _FakeResponse:
        captured["url"] = url
        captured["timeout"] = timeout
        return _FakeResponse(b"data")

    fake_requests = mock.Mock()
    fake_requests.get = fake_get

    with mock.patch.dict("sys.modules", {"requests": fake_requests}):
        result = load.read_file("https://example.invalid/path")

    assert result == b"data"
    assert captured["url"] == "https://example.invalid/path"
    assert captured["timeout"] == 60.0


def test_read_file_https_respects_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """``TIKTOKEN_HTTP_TIMEOUT`` overrides the default timeout."""
    monkeypatch.setenv("TIKTOKEN_HTTP_TIMEOUT", "5.5")
    captured: dict[str, Any] = {}

    def fake_get(url: str, *, timeout: Any = None, **_: Any) -> _FakeResponse:
        captured["timeout"] = timeout
        return _FakeResponse(b"")

    fake_requests = mock.Mock()
    fake_requests.get = fake_get

    with mock.patch.dict("sys.modules", {"requests": fake_requests}):
        load.read_file("http://example.invalid/path")

    assert captured["timeout"] == 5.5


def test_read_file_https_falls_back_on_invalid_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unparseable ``TIKTOKEN_HTTP_TIMEOUT`` falls back to the default rather than crashing."""
    monkeypatch.setenv("TIKTOKEN_HTTP_TIMEOUT", "not-a-number")
    captured: dict[str, Any] = {}

    def fake_get(url: str, *, timeout: Any = None, **_: Any) -> _FakeResponse:
        captured["timeout"] = timeout
        return _FakeResponse(b"")

    fake_requests = mock.Mock()
    fake_requests.get = fake_get

    with mock.patch.dict("sys.modules", {"requests": fake_requests}):
        load.read_file("http://example.invalid/path")

    assert captured["timeout"] == 60.0
