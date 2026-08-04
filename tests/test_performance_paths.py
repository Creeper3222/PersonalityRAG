from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from personalityrag.compression import SelectiveGZipMiddleware
from personalityrag import http_pool, resource_quotas
from personalityrag.libraries import DatabaseManager


@pytest.mark.asyncio
async def test_background_quota_wait_cancellation_releases_reserved_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for slot_name, total_name, background_name in (
        ("io_slot", "io_total", "io_background"),
        ("provider_slot", "provider_total", "provider_background"),
    ):
        quotas = SimpleNamespace(
            io_total=asyncio.Semaphore(0),
            io_background=asyncio.Semaphore(1),
            provider_total=asyncio.Semaphore(0),
            provider_background=asyncio.Semaphore(1),
        )
        monkeypatch.setattr(resource_quotas, "_loop_quotas", lambda: quotas)

        async def wait_for_slot() -> None:
            with resource_quotas.resource_lane("task"):
                async with getattr(resource_quotas, slot_name)():
                    raise AssertionError("blocked slot should not be entered")

        waiting = asyncio.create_task(wait_for_slot())
        for _ in range(20):
            await asyncio.sleep(0)
            if getattr(quotas, background_name).locked():
                break
        assert getattr(quotas, background_name).locked()
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert getattr(quotas, background_name)._value == 1

        getattr(quotas, total_name).release()
        with resource_quotas.resource_lane("task"):
            async with getattr(resource_quotas, slot_name)():
                pass
        assert getattr(quotas, background_name)._value == 1



@pytest.mark.asyncio
async def test_shared_http_transport_keeps_authorization_per_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await http_pool.close_http_pools()
    created = []

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.requests = []
            self.is_closed = False
            created.append(self)

        async def request(self, method, url, **kwargs):
            self.requests.append((method, url, dict(kwargs.get("headers") or {})))
            return SimpleNamespace(status_code=200)

        async def aclose(self):
            self.is_closed = True

    monkeypatch.setattr(http_pool.httpx, "AsyncClient", FakeAsyncClient)
    first = http_pool.acquire_http_client(
        base_url="https://provider.example/v1",
        timeout=10,
        headers={"Authorization": "Bearer first"},
    )
    second = http_pool.acquire_http_client(
        base_url="https://provider.example/v1/",
        timeout=20,
        headers={"Authorization": "Bearer second"},
    )

    await first.get("/models")
    await second.get("/models", headers={"X-Request-ID": "second-request"})

    assert len(created) == 1
    assert created[0].requests[0][2] == {"Authorization": "Bearer first"}
    assert created[0].requests[1][2] == {
        "Authorization": "Bearer second",
        "X-Request-ID": "second-request",
    }
    assert http_pool.http_pool_status() == {
        "transport_pool_count": 1,
        "transport_lease_count": 2,
    }
    await first.aclose()
    assert created[0].is_closed is False
    await second.aclose()
    assert created[0].is_closed is True
    assert http_pool.http_pool_status() == {
        "transport_pool_count": 0,
        "transport_lease_count": 0,
    }


@pytest.mark.asyncio
async def test_selective_compression_skips_media_streams_ranges_and_attachments() -> None:
    body = b"x" * 4096

    async def json_endpoint(_request):
        return JSONResponse({"payload": "x" * 4096})

    async def image_endpoint(_request):
        return Response(body, media_type="image/png")

    async def range_endpoint(_request):
        return Response(
            body,
            media_type="application/octet-stream",
            headers={"Content-Range": "bytes 0-4095/4096"},
            status_code=206,
        )

    async def attachment_endpoint(_request):
        return Response(
            body,
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="export.json"'},
        )

    async def sse_endpoint(_request):
        async def chunks():
            yield b"data: " + body + b"\n\n"

        return StreamingResponse(chunks(), media_type="text/event-stream")

    app = Starlette(
        routes=[
            Route("/json", json_endpoint),
            Route("/image", image_endpoint),
            Route("/range", range_endpoint),
            Route("/attachment", attachment_endpoint),
            Route("/sse", sse_endpoint),
        ]
    )
    app.add_middleware(SelectiveGZipMiddleware, minimum_size=1024, compresslevel=5)
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Accept-Encoding": "gzip"},
    ) as client:
        responses = {
            path: await client.get(path)
            for path in ("/json", "/image", "/range", "/attachment", "/sse")
        }

    assert responses["/json"].headers["content-encoding"] == "gzip"
    for path in ("/image", "/range", "/attachment", "/sse"):
        assert "content-encoding" not in responses[path].headers


@pytest.mark.asyncio
async def test_database_summary_coalesces_only_overlapping_requests() -> None:
    release = asyncio.Event()

    class FakeManager:
        def __init__(self, database_type: str):
            self.database_type = database_type
            self.calls = 0

        async def list_libraries(self, *, stats_mode: str):
            assert stats_mode == "summary"
            self.calls += 1
            await release.wait()
            return [
                {
                    "id": self.database_type,
                    "database_type": self.database_type,
                    "is_default": False,
                    "created_at": 1,
                }
            ]

    first = FakeManager("first")
    second = FakeManager("second")
    manager = DatabaseManager.__new__(DatabaseManager)
    manager._managers = {"first": first, "second": second}
    manager._summary_list_task = None

    requests = [
        asyncio.create_task(manager.list_libraries(stats_mode="summary"))
        for _ in range(8)
    ]
    for _ in range(20):
        await asyncio.sleep(0)
        if first.calls and second.calls:
            break
    assert (first.calls, second.calls) == (1, 1)
    release.set()
    results = await asyncio.gather(*requests)
    assert all(result == results[0] for result in results)

    await manager.list_libraries(stats_mode="summary")
    assert (first.calls, second.calls) == (2, 2)
