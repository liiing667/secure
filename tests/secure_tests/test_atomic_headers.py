import asyncio
from collections.abc import Iterator

import pytest

from secure import CustomHeader, Secure
from secure._internal.emit import set_headers_sync
from secure.secure import HeaderSetError


class _FailingHeadersMapping:
    """
    Mapping container whose ``__setitem__`` raises after a number of writes.

    The backing dict keeps every mutation, so tests can assert whether earlier
    writes were rolled back after a mid-group failure.
    """

    def __init__(self, *, fail_after: int, error: type[Exception] = ValueError) -> None:
        self._store: dict[str, str] = {}
        self._fail_after = fail_after
        self._error = error
        self._writes = 0

    def keys(self):
        return self._store.keys()

    def __getitem__(self, key: str) -> str:
        return self._store[key]

    def __delitem__(self, key: str) -> None:
        del self._store[key]

    def __setitem__(self, key: str, value: str) -> None:
        self._writes += 1
        if self._writes > self._fail_after:
            raise self._error("boom")
        self._store[key] = value


class _FailingMappingResponse:
    """Response exposing a mapping container that fails halfway through."""

    def __init__(self, *, fail_after: int, error: type[Exception] = ValueError) -> None:
        self.headers = _FailingHeadersMapping(fail_after=fail_after, error=error)


class _AsyncFailingHeadersMapping:
    """Async counterpart of :class:`_FailingHeadersMapping`."""

    def __init__(self, *, fail_after: int) -> None:
        self._store: dict[str, str] = {}
        self._fail_after = fail_after
        self._writes = 0

    def keys(self):
        return self._store.keys()

    def __getitem__(self, key: str) -> str:
        return self._store[key]

    async def __delitem__(self, key: str) -> None:
        del self._store[key]

    async def __setitem__(self, key: str, value: str) -> None:
        self._writes += 1
        if self._writes > self._fail_after:
            raise ValueError("boom")
        self._store[key] = value


class _AsyncFailingMappingResponse:
    def __init__(self, *, fail_after: int) -> None:
        self.headers = _AsyncFailingHeadersMapping(fail_after=fail_after)


class _FailingSetterResponse:
    """
    Response exposing both ``set_header`` and a mapping snapshot.

    ``set_header`` raises halfway through the group; ``headers`` provides the
    read/write surface used to roll back the already-applied subset.
    """

    def __init__(self, *, fail_after: int) -> None:
        self.headers: dict[str, str] = {}
        self._fail_after = fail_after
        self._calls = 0

    def set_header(self, key: str, value: str) -> None:
        self._calls += 1
        if self._calls > self._fail_after:
            raise ValueError("boom")
        self.headers[key] = value


class _AsyncFailingSetterResponse:
    """Async counterpart of :class:`_FailingSetterResponse`."""

    def __init__(self, *, fail_after: int) -> None:
        self.headers: dict[str, str] = {}
        self._fail_after = fail_after
        self._calls = 0

    async def set_header(self, key: str, value: str) -> None:
        self._calls += 1
        if self._calls > self._fail_after:
            raise ValueError("boom")
        self.headers[key] = value


class _FailingContainer:
    """Werkzeug-style headers container exposing ``set`` plus mapping access."""

    def __init__(self, *, fail_after: int) -> None:
        self._store: dict[str, str] = {}
        self._fail_after = fail_after
        self._calls = 0

    def keys(self):
        return self._store.keys()

    def __getitem__(self, key: str) -> str:
        return self._store[key]

    def __delitem__(self, key: str) -> None:
        del self._store[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def __setitem__(self, key: str, value: str) -> None:
        self._store[key] = value

    def set(self, key: str, value: str) -> None:
        self._calls += 1
        if self._calls > self._fail_after:
            raise ValueError("boom")
        self._store[key] = value


class _FailingContainerResponse:
    def __init__(self, *, fail_after: int) -> None:
        self.headers = _FailingContainer(fail_after=fail_after)


def _secure_with_headers() -> Secure:
    return Secure(
        custom=[
            CustomHeader("Strict-Transport-Security", "max-age=1"),
            CustomHeader("Content-Security-Policy", "default-src 'self'"),
            CustomHeader("X-Frame-Options", "DENY"),
            CustomHeader("X-Content-Type-Options", "nosniff"),
        ]
    )


def _expected_order() -> list[str]:
    return [name for name, _value in _secure_with_headers().header_items()]


def test_sync_mapping_mid_failure_rolls_back_everything() -> None:
    secure_headers = _secure_with_headers()
    response = _FailingMappingResponse(fail_after=2)

    with pytest.raises(HeaderSetError):
        secure_headers.set_headers(response)

    assert dict(response.headers) == {}


def test_sync_mapping_failure_restores_preexisting_headers() -> None:
    secure_headers = _secure_with_headers()
    response = _FailingMappingResponse(fail_after=2)
    response.headers._store["Referrer-Policy"] = "unsafe-url"

    with pytest.raises(HeaderSetError):
        secure_headers.set_headers(response)

    assert dict(response.headers) == {"Referrer-Policy": "unsafe-url"}


def test_async_mapping_mid_failure_rolls_back_everything() -> None:
    secure_headers = _secure_with_headers()
    response = _AsyncFailingMappingResponse(fail_after=1)

    with pytest.raises(HeaderSetError):
        asyncio.run(secure_headers.set_headers_async(response))

    assert dict(response.headers) == {}


def test_sync_setter_mid_failure_rolls_back_applied_subset() -> None:
    secure_headers = _secure_with_headers()
    response = _FailingSetterResponse(fail_after=2)

    with pytest.raises(HeaderSetError):
        secure_headers.set_headers(response)

    assert response.headers == {}


def test_async_setter_mid_failure_rolls_back_applied_subset() -> None:
    secure_headers = _secure_with_headers()
    response = _AsyncFailingSetterResponse(fail_after=2)

    with pytest.raises(HeaderSetError):
        asyncio.run(secure_headers.set_headers_async(response))

    assert response.headers == {}


def test_sync_container_set_mid_failure_rolls_back() -> None:
    secure_headers = _secure_with_headers()
    response = _FailingContainerResponse(fail_after=2)

    with pytest.raises(HeaderSetError):
        secure_headers.set_headers(response)

    assert dict(response.headers) == {}


def test_multi_valued_header_is_rolled_back_on_later_failure() -> None:
    secure_headers = Secure(
        custom=[
            CustomHeader("Content-Security-Policy", "default-src 'self'"),
            CustomHeader("Content-Security-Policy", "script-src 'none'"),
            CustomHeader("X-Frame-Options", "DENY"),
        ]
    )
    response = _FailingMappingResponse(fail_after=2)

    with pytest.raises(HeaderSetError):
        secure_headers.set_headers(response)

    assert dict(response.headers) == {}


def test_preflight_rejects_non_string_value_without_touching_response() -> None:
    response = _FailingMappingResponse(fail_after=1)

    with pytest.raises(HeaderSetError):
        secure_set_headers_with_bad_item(response)

    assert dict(response.headers) == {}


def secure_set_headers_with_bad_item(response: object) -> None:
    set_headers_sync(response, (("X-Good", "value"), ("X-Bad", 123)))  # type: ignore[arg-type]


def test_successful_delivery_preserves_order_and_last_write_wins() -> None:
    secure_headers = Secure(
        custom=[
            CustomHeader("X-Test", "first"),
            CustomHeader("X-Test", "second"),
            CustomHeader("X-Other", "value"),
        ]
    )

    class _PlainResponse:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def __setitem__(self, key: str, value: str) -> None:
            self.headers[key] = value

    response = _PlainResponse()
    secure_headers.set_headers(response)

    assert list(response.headers.keys()) == ["X-Test", "X-Other"]
    assert response.headers["X-Test"] == "second"


def test_successful_delivery_matches_expected_header_order() -> None:
    secure_headers = _secure_with_headers()

    class _PlainResponse:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def __setitem__(self, key: str, value: str) -> None:
            self.headers[key] = value

    response = _PlainResponse()
    secure_headers.set_headers(response)

    assert list(response.headers.keys()) == _expected_order()
