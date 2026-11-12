from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, MutableMapping

    from .types import HeaderItems, ResponseProtocol


class HeaderSetError(RuntimeError):
    """Raised when applying a header to a response fails."""


class _MappingSnapshot:
    """Immutable point-in-time copy of a mapping-like header container."""

    __slots__ = ("available", "pairs")

    def __init__(self, pairs: tuple[tuple[str, str], ...], *, available: bool) -> None:
        self.pairs = pairs
        self.available = available


def _preflight_header_items(items: HeaderItems) -> None:
    """
    Validate the entire group of header items before touching the response.

    This is the "validate first" half of atomic delivery: a malformed
    ``(name, value)`` pair is rejected before any setter is invoked, so the
    response can never end up with a partial subset of the configured headers.
    """
    for name, value in items:
        if not isinstance(name, str):
            raise HeaderSetError(f"Failed to set headers: header name must be a str, got {type(name).__name__!r}")
        if not isinstance(value, str):
            raise HeaderSetError(
                f"Failed to set headers: value for header {name!r} must be a str, got {type(value).__name__!r}"
            )


def _capture_mapping_snapshot(target: object) -> _MappingSnapshot:
    """
    Snapshot a mapping-like header container for later rollback.

    Returns an unavailable snapshot when the container does not expose the
    read interface needed to restore its previous state.
    """
    keys = getattr(target, "keys", None)
    get_item = getattr(target, "__getitem__", None)
    if not callable(keys) or not callable(get_item):
        return _MappingSnapshot((), available=False)

    try:
        pairs = tuple((cast("str", key), cast("str", target[key])) for key in keys())  # type: ignore[index]
    except Exception:  # pragma: no cover - defensive: unreadable containers cannot be rolled back
        return _MappingSnapshot((), available=False)

    return _MappingSnapshot(pairs, available=True)


def _restore_mapping_sync(mapping: MutableMapping[str, str], snapshot: _MappingSnapshot) -> None:
    """Best-effort synchronous rollback; never masks the original error."""
    if not snapshot.available:
        return

    try:
        original_keys = {name for name, _value in snapshot.pairs}
        for name in tuple(mapping.keys()):
            if name not in original_keys:
                del mapping[name]
        for name, value in snapshot.pairs:
            mapping[name] = value
    except Exception:  # pragma: no cover - rollback is best effort  # noqa: S110
        pass


async def _restore_mapping_async(mapping: object, snapshot: _MappingSnapshot) -> None:
    """Best-effort asynchronous rollback; never masks the original error."""
    if not snapshot.available:
        return

    try:
        original_keys = {name for name, _value in snapshot.pairs}
        for name in tuple(mapping.keys()):  # type: ignore[attr-defined]
            if name not in original_keys:
                del_item = getattr(mapping, "__delitem__", None)
                if callable(del_item):
                    result = del_item(name)
                    if inspect.isawaitable(result):
                        await result
        for name, value in snapshot.pairs:
            result = mapping.__setitem__(name, value)  # type: ignore[attr-defined]
            if inspect.isawaitable(result):
                await result
    except Exception:  # pragma: no cover - rollback is best effort  # noqa: S110
        pass


def _sync_rollback_for(target: object) -> Callable[[], None] | None:
    """Build a synchronous rollback callback for a mapping-like target."""
    snapshot = _capture_mapping_snapshot(target)
    if not snapshot.available:
        return None

    mapping = cast("MutableMapping[str, str]", target)

    def rollback() -> None:
        _restore_mapping_sync(mapping, snapshot)

    return rollback


def _async_rollback_for(target: object) -> Callable[[], Awaitable[None]] | None:
    """Build an asynchronous rollback callback for a mapping-like target."""
    snapshot = _capture_mapping_snapshot(target)
    if not snapshot.available:
        return None

    async def rollback() -> None:
        await _restore_mapping_async(target, snapshot)

    return rollback


def _run_rollback(rollback: Callable[[], None] | None) -> None:
    if rollback is not None:
        rollback()


async def _run_async_rollback(rollback: Callable[[], Awaitable[None]] | None) -> None:
    if rollback is not None:
        await rollback()


def set_headers_sync(response: ResponseProtocol, items: HeaderItems) -> None:
    """
    Apply header items to a sync response object.
    """
    _preflight_header_items(items)

    if hasattr(response, "set_header"):
        rollback = _sync_rollback_for(getattr(response, "headers", None))
        _apply_sync_setter(
            response.set_header,
            items,
            rollback=rollback,
            coroutine_function_error=(
                "Async 'set_header' detected in sync context. Use 'await set_headers_async(response)'."
            ),
            awaitable_error=(
                "Async 'set_header' returned awaitable in sync context. Use 'await set_headers_async(response)'."
            ),
        )
        return

    _set_headers_container_sync(_get_headers_container(response), items)


async def set_headers_async(response: ResponseProtocol, items: HeaderItems) -> None:
    """
    Apply header items to a sync or async response object from async code.
    """
    _preflight_header_items(items)

    if hasattr(response, "set_header"):
        rollback = _async_rollback_for(getattr(response, "headers", None))
        await _apply_async_setter(
            response.set_header,
            items,
            rollback=rollback,
        )
        return

    await _set_headers_container_async(_get_headers_container(response), items)


def _get_headers_container(response: ResponseProtocol) -> object:
    if hasattr(response, "headers"):
        return cast("object", response.headers)
    raise AttributeError("Response object does not support setting headers.")


def _set_headers_container_sync(headers: object, items: HeaderItems) -> None:
    set_fn = getattr(headers, "set", None)
    if callable(set_fn):
        rollback = _sync_rollback_for(headers)
        _apply_sync_setter(
            set_fn,
            items,
            rollback=rollback,
            coroutine_function_error=(
                "Async headers setter detected in sync context. Use 'await set_headers_async(response)'."
            ),
            awaitable_error=(
                "Async headers setter returned awaitable in sync context. Use 'await set_headers_async(response)'."
            ),
        )
        return

    _set_headers_mapping_sync(headers, items)


async def _set_headers_container_async(headers: object, items: HeaderItems) -> None:
    set_fn = getattr(headers, "set", None)
    if callable(set_fn):
        rollback = _async_rollback_for(headers)
        await _apply_async_setter(set_fn, items, rollback=rollback)
        return

    await _set_headers_mapping_async(headers, items)


def _apply_sync_setter(
    setter: object,
    items: HeaderItems,
    *,
    rollback: Callable[[], None] | None,
    coroutine_function_error: str,
    awaitable_error: str,
) -> None:
    if inspect.iscoroutinefunction(setter):
        raise RuntimeError(coroutine_function_error)

    typed_setter = cast("Callable[[str, str], object]", setter)

    try:
        for name, value in items:
            result = typed_setter(name, value)
            if inspect.isawaitable(result):
                raise RuntimeError(awaitable_error)  # noqa: TRY301 - rollback must run before propagation
    except (TypeError, ValueError, AttributeError) as error:
        _run_rollback(rollback)
        raise HeaderSetError(f"Failed to set headers: {error}") from error
    except RuntimeError:
        _run_rollback(rollback)
        raise


async def _apply_async_setter(
    setter: object,
    items: HeaderItems,
    *,
    rollback: Callable[[], Awaitable[None]] | None = None,
) -> None:
    typed_setter = cast("Callable[[str, str], object]", setter)

    try:
        for name, value in items:
            result = typed_setter(name, value)
            if inspect.isawaitable(result):
                await result
    except (TypeError, ValueError, AttributeError) as error:
        await _run_async_rollback(rollback)
        raise HeaderSetError(f"Failed to set headers: {error}") from error


def _set_headers_mapping_sync(headers: object, items: HeaderItems) -> None:
    setitem = getattr(headers, "__setitem__", None)
    if not callable(setitem):
        raise AttributeError(  # noqa: TRY004
            "Response object has .headers but it does not support setting header values."
        )

    if inspect.iscoroutinefunction(setitem):
        raise RuntimeError("Async headers mapping detected in sync context. Use 'await set_headers_async(response)'.")

    rollback = _sync_rollback_for(headers)  # captured before any mutation

    try:
        headers_map = cast("MutableMapping[str, str]", headers)
        for name, value in items:
            headers_map[name] = value
    except (TypeError, ValueError, AttributeError) as error:
        _run_rollback(rollback)
        raise HeaderSetError(f"Failed to set headers: {error}") from error


async def _set_headers_mapping_async(headers: object, items: HeaderItems) -> None:
    setitem = getattr(headers, "__setitem__", None)
    if not callable(setitem):
        raise AttributeError(  # noqa: TRY004
            "Response object has .headers but it does not support setting header values."
        )

    rollback = _async_rollback_for(headers)  # captured before any mutation
    await _apply_async_setter(setitem, items, rollback=rollback)
