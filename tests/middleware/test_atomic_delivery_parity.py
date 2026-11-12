"""
Cross-checks between response-object header delivery and ASGI/WSGI middleware.

The middleware builds the complete header set in an isolated list before a
single hand-off, so an application never observes a partial subset. The
``Secure.set_headers`` / ``set_headers_async`` object paths now provide the
same all-or-nothing guarantee via preflight validation plus rollback.
"""

import asyncio

from tests.middleware.test_middleware import _http_app, _run_asgi, _run_wsgi_with_middleware

from secure import CustomHeader, Secure
from secure.middleware import SecureASGIMiddleware, SecureWSGIMiddleware
from secure.secure import HeaderSetError


class _FailingHeaders:
    """Mapping that rejects writes once ``fail_after`` successful writes land."""

    def __init__(self, *, fail_after: int) -> None:
        self._store: dict[str, str] = {}
        self._fail_after = fail_after
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
            raise ValueError("boom")
        self._store[key] = value


class _FailingResponse:
    def __init__(self, *, fail_after: int) -> None:
        self.headers = _FailingHeaders(fail_after=fail_after)


def _secure() -> Secure:
    return Secure(
        custom=[
            CustomHeader("Strict-Transport-Security", "max-age=1"),
            CustomHeader("Content-Security-Policy", "default-src 'self'"),
            CustomHeader("X-Frame-Options", "DENY"),
        ]
    )


def test_middleware_and_response_object_agree_on_full_header_set() -> None:
    secure_headers = _secure()
    expected = dict(secure_headers.headers)

    class _PlainResponse:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def __setitem__(self, key: str, value: str) -> None:
            self.headers[key] = value

    response = _PlainResponse()
    secure_headers.set_headers(response)

    def app(environ, start_response):
        start_response("200 OK", [])
        return [b"ok"]

    wsgi = _run_wsgi_with_middleware(SecureWSGIMiddleware(app, secure=secure_headers))
    wsgi_headers = dict(wsgi["headers"])

    asgi_messages = asyncio.run(_run_asgi({"type": "http"}, SecureASGIMiddleware(_http_app([]), secure=secure_headers)))
    start = next(message for message in asgi_messages if message["type"] == "http.response.start")
    asgi_headers = {key.decode(): value.decode() for key, value in start["headers"]}

    assert response.headers == expected
    assert wsgi_headers == expected
    assert asgi_headers == expected


def test_response_object_rolls_back_when_middleware_keeps_group_intact() -> None:
    """
    A mid-group failure on the object path leaves zero applied headers,
    mirroring the middleware's single-shot (never partially delivered) model.
    """
    secure_headers = _secure()
    response = _FailingResponse(fail_after=1)

    try:
        secure_headers.set_headers(response)
    except HeaderSetError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected HeaderSetError")

    assert dict(response.headers) == {}

    # The middleware itself never mutates app state incrementally: the full
    # group still lands in one http.response.start / start_response hand-off.
    def app(environ, start_response):
        start_response("200 OK", [("X-Keep", "1")])
        return [b"ok"]

    wsgi = _run_wsgi_with_middleware(SecureWSGIMiddleware(app, secure=secure_headers))
    assert dict(wsgi["headers"])["X-Keep"] == "1"
    assert len(wsgi["headers"]) == len(secure_headers.headers) + 1
