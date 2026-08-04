from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from personalityrag.application import create_app
from personalityrag.application_context import ApplicationContext
from personalityrag.auth import hash_password
from personalityrag.config import AppConfig
from personalityrag.library_types.livingmemory_v8.driver import LivingMemoryV8Driver
from personalityrag.listener_surface import (
    ADAPTER_ACCESS_SURFACE,
    WEBUI_SURFACE,
    is_adapter_access_request_allowed,
    listener_app,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _context(tmp_path: Path) -> ApplicationContext:
    return ApplicationContext.create(
        source_root=REPO_ROOT,
        state_root=tmp_path / "state",
        config=AppConfig(
            api_key="fixture-admin-key",
            session_secret="fixture-session",
            library_psk_secret="fixture-database-secret",
            webui_password_hash=hash_password("fixture-password"),
        ),
        configure_logs=False,
    )


def test_adapter_access_allowlist_matches_only_runtime_protocol() -> None:
    memory_base = "/api/v1/memory-libraries/livingmemory_v8/demo"
    knowledge_base = "/api/v1/knowledge-libraries/text_media_v1/demo"
    allowed = {
        ("GET", "/api/v1/health"),
        ("GET", memory_base),
        ("PATCH", memory_base),
        ("GET", f"{memory_base}/stats"),
        ("POST", f"{memory_base}/adapters/heartbeat"),
        ("POST", f"{memory_base}/recall"),
        ("POST", f"{memory_base}/memories"),
        ("DELETE", f"{memory_base}/memories/42"),
        ("GET", f"{memory_base}/memories/42/source"),
        ("POST", f"{memory_base}/memories/42/archive"),
        ("POST", f"{memory_base}/memories/42/restore"),
        ("POST", f"{memory_base}/memories/42/resummary"),
        ("GET", f"{memory_base}/transfers/export"),
        ("POST", f"{memory_base}/transfers/imports/preview"),
        ("POST", f"{memory_base}/transfers/imports/preview-123/commit"),
        ("POST", f"{memory_base}/indexes/rebuild"),
        ("POST", f"{memory_base}/graph/rebuild"),
        ("POST", f"{memory_base}/conversations/messages"),
        ("GET", f"{memory_base}/conversations/aiocqhttp:GroupMessage:1"),
        (
            "GET",
            f"{memory_base}/conversations/aiocqhttp:GroupMessage:1/messages",
        ),
        (
            "PATCH",
            f"{memory_base}/conversations/aiocqhttp:GroupMessage:1/metadata",
        ),
        (
            "POST",
            f"{memory_base}/conversations/aiocqhttp:GroupMessage:1/clear",
        ),
        (
            "POST",
            f"{memory_base}/conversations/aiocqhttp:GroupMessage:1/trim",
        ),
        ("GET", knowledge_base),
        ("POST", f"{knowledge_base}/adapters/heartbeat"),
        ("POST", f"{knowledge_base}/search"),
        ("POST", f"{knowledge_base}/assets/asset_1/signed-url"),
        ("GET", f"{knowledge_base}/assets/asset_1/content"),
        ("GET", f"{knowledge_base}/assets/asset_1/thumbnail"),
    }
    rejected = {
        ("GET", "/"),
        ("GET", "/static/app.js"),
        ("GET", "/docs"),
        ("GET", "/redoc"),
        ("GET", "/openapi.json"),
        ("GET", "/api/v1/auth/status"),
        ("POST", "/api/v1/auth/login"),
        ("GET", "/api/v1/settings"),
        ("GET", "/api/v1/jobs"),
        ("GET", "/api/v1/providers"),
        ("GET", "/api/v1/files"),
        ("GET", f"{memory_base}/indexes"),
        ("GET", f"{memory_base}/memories"),
        ("PATCH", f"{memory_base}/memories/42"),
        ("PUT", f"{memory_base}/memories/42/source"),
        ("GET", f"{memory_base}/memories/42/archive"),
        ("POST", f"{memory_base}/transfers/imports/preview-123/delete"),
        ("DELETE", memory_base),
        ("POST", f"{knowledge_base}/indexes/rebuild"),
        ("GET", f"{knowledge_base}/documents"),
        ("OPTIONS", f"{knowledge_base}/search"),
    }

    assert all(is_adapter_access_request_allowed(*item) for item in allowed)
    assert not any(is_adapter_access_request_allowed(*item) for item in rejected)


@pytest.mark.asyncio
async def test_listener_identity_survives_reverse_proxy_host_and_separates_auth(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    await context.manager.initialize()
    try:
        application = create_app(context)
        webui = listener_app(application, WEBUI_SURFACE)
        access = listener_app(application, ADAPTER_ACCESS_SURFACE)
        access_key = LivingMemoryV8Driver().derive_access_key(
            context.config.library_psk_secret,
            "Default",
        )
        proxy_headers = {
            "Host": "memory.example.test:443",
            "X-Forwarded-Host": "memory.example.test",
            "X-Forwarded-Port": "443",
            "X-Forwarded-Proto": "https",
        }

        async with (
            AsyncClient(
                transport=ASGITransport(app=webui),
                base_url="https://memory.example.test",
            ) as webui_client,
            AsyncClient(
                transport=ASGITransport(app=access),
                base_url="https://memory.example.test",
            ) as access_client,
        ):
            login = await webui_client.post(
                "/api/v1/auth/login",
                json={"credential": "fixture-password"},
            )
            webui_settings = await webui_client.get(
                "/api/v1/settings",
                headers={"Authorization": "Bearer fixture-admin-key"},
            )
            typed = await access_client.get(
                "/api/v1/memory-libraries/livingmemory_v8/Default",
                headers={
                    **proxy_headers,
                    "Authorization": f"Bearer {access_key}",
                },
            )
            global_key = await access_client.get(
                "/api/v1/memory-libraries/livingmemory_v8/Default",
                headers={
                    **proxy_headers,
                    "Authorization": "Bearer fixture-admin-key",
                },
            )
            access_client.cookies.update(webui_client.cookies)
            session_cookie = await access_client.get(
                "/api/v1/memory-libraries/livingmemory_v8/Default",
                headers=proxy_headers,
            )
            health = await access_client.get(
                "/api/v1/health",
                headers=proxy_headers,
            )

        assert login.status_code == 200
        assert webui_settings.status_code == 200
        assert webui_settings.headers["x-personalityrag-surface"] == "webui"
        assert typed.status_code == 200
        assert typed.headers["x-personalityrag-surface"] == "adapter-access"
        assert typed.headers["x-personalityrag-adapter-protocol"] == "1"
        assert typed.headers["cache-control"] == "no-store"
        assert typed.headers["x-content-type-options"] == "nosniff"
        assert global_key.status_code == 401
        assert session_cookie.status_code == 401
        assert health.status_code == 200
        assert health.headers["x-personalityrag-surface"] == "adapter-access"
    finally:
        await context.manager.close()


@pytest.mark.asyncio
async def test_adapter_listener_hides_every_management_surface(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    application = listener_app(create_app(context), ADAPTER_ACCESS_SURFACE)
    paths = (
        "/",
        "/favicon.ico",
        "/static/app.js",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/v1/auth/status",
        "/api/v1/settings",
        "/api/v1/jobs",
        "/api/v1/providers",
        "/api/v1/files",
        "/api/v1/logs",
        "/api/v1/database-types",
    )
    async with AsyncClient(
        transport=ASGITransport(app=application),
        base_url="https://memory.example.test",
        headers={"Authorization": "Bearer fixture-admin-key"},
    ) as client:
        responses = [await client.get(path) for path in paths]

    assert {response.status_code for response in responses} == {404}
    assert all(
        response.headers["x-personalityrag-surface"] == "adapter-access"
        for response in responses
    )
    assert all("x-request-id" in response.headers for response in responses)
