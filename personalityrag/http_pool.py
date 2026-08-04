from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Mapping

import httpx

from .resource_limits import configured_http_limits
from .resource_quotas import provider_slot


@dataclass(slots=True)
class _PoolEntry:
    client: httpx.AsyncClient
    references: int = 0


_entries: dict[tuple[str, str, bool], _PoolEntry] = {}
_lock = threading.Lock()


def _pool_key(base_url: str, proxy: str | None, trust_env: bool) -> tuple[str, str, bool]:
    return base_url.rstrip("/"), str(proxy or ""), bool(trust_env)


class PooledAsyncClient:
    """A request-header lease over a shared HTTPX connection pool."""

    def __init__(
        self,
        *,
        key: tuple[str, str, bool],
        entry: _PoolEntry,
        headers: Mapping[str, str] | None,
        timeout: float,
    ):
        self._key = key
        self._entry = entry
        self._headers = dict(headers or {})
        self._timeout = float(timeout)
        self._closed = False

    def _request_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        result = dict(kwargs)
        request_headers = dict(self._headers)
        request_headers.update(dict(result.pop("headers", None) or {}))
        if request_headers:
            result["headers"] = request_headers
        result.setdefault("timeout", self._timeout)
        return result

    async def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        if self._closed:
            raise RuntimeError("pooled HTTP client is closed")
        async with provider_slot():
            return await self._entry.client.request(
                method,
                url,
                **self._request_kwargs(kwargs),
            )

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return await self.request("POST", url, **kwargs)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        close_client: httpx.AsyncClient | None = None
        with _lock:
            current = _entries.get(self._key)
            if current is self._entry:
                current.references = max(0, current.references - 1)
                if current.references == 0:
                    _entries.pop(self._key, None)
                    close_client = current.client
        if close_client is not None:
            await close_client.aclose()


def acquire_http_client(
    *,
    base_url: str,
    timeout: float,
    headers: Mapping[str, str] | None = None,
    proxy: str | None = None,
    trust_env: bool = False,
) -> PooledAsyncClient:
    key = _pool_key(base_url, proxy, trust_env)
    with _lock:
        entry = _entries.get(key)
        if entry is None:
            maximum, keepalive = configured_http_limits()
            kwargs: dict[str, Any] = {
                "base_url": key[0],
                "timeout": None,
                "trust_env": bool(trust_env),
                "follow_redirects": False,
                "limits": httpx.Limits(
                    max_connections=maximum,
                    max_keepalive_connections=keepalive,
                    keepalive_expiry=30.0,
                ),
            }
            if proxy:
                kwargs["proxy"] = proxy
            entry = _PoolEntry(httpx.AsyncClient(**kwargs))
            _entries[key] = entry
        entry.references += 1
    return PooledAsyncClient(
        key=key,
        entry=entry,
        headers=headers,
        timeout=timeout,
    )


def http_pool_status() -> dict[str, int]:
    with _lock:
        return {
            "transport_pool_count": len(_entries),
            "transport_lease_count": sum(
                entry.references for entry in _entries.values()
            ),
        }


async def close_http_pools() -> None:
    """Close all remaining shared transports during application shutdown."""

    with _lock:
        entries = list(_entries.values())
        _entries.clear()
    for entry in entries:
        if not entry.client.is_closed:
            await entry.client.aclose()
