from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from fastapi.routing import APIRoute

from personalityrag import app as app_module


REPO_ROOT = Path(__file__).resolve().parents[1]


def _stable_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_app_state_is_isolated_from_live_tree() -> None:
    state_root = Path(os.environ["PERSONALITYRAG_STATE_ROOT"]).resolve()

    assert state_root == (REPO_ROOT / ".test-runtime").resolve()
    assert app_module.STATE_ROOT == state_root
    assert app_module.CONFIG_PATH.is_relative_to(state_root)
    assert app_module.manager.root == state_root
    assert app_module.STATIC_DIR == REPO_ROOT / "static"


def test_http_route_contract() -> None:
    registered_routes = []
    for item in app_module.app.routes:
        original_router = getattr(item, "original_router", None)
        if original_router is not None:
            registered_routes.extend(original_router.routes)
        else:
            registered_routes.append(item)
    routes = sorted(
        (
            method,
            route.path,
            route.name,
        )
        for route in registered_routes
        if isinstance(route, APIRoute)
        for method in sorted(route.methods or set())
        if method != "HEAD"
    )

    assert len(routes) == 105
    assert _stable_hash(routes) == (
        "25cfebc76b7fffc7770c430964f8cc3cf51131775796a6bff9729cf7458c4ec4"
    )
    assert ("POST", "/api/v1/providers/test", "provider_test_compat") in routes
    assert ("POST", "/api/v1/recall", "recall") in routes
    assert (
        "POST",
        "/api/v1/libraries/{library_id}/recall",
        "recall",
    ) in routes
    assert (
        "POST",
        "/api/v1/libraries/{library_id}/adapters/{adapter_id}/disconnect",
        "disconnect_adapter",
    ) in routes
    assert ("GET", "/api/v1/files", "list_files") in routes
    assert ("POST", "/api/v1/files/download", "download_files") in routes
    assert ("GET", "/api/v1/updates/status", "update_status") in routes
    assert ("GET", "/api/v1/updates/releases", "update_releases") in routes
    assert ("POST", "/api/v1/updates/switch", "switch_version") in routes


def test_openapi_contract() -> None:
    assert _stable_hash(app_module.app.openapi()) == (
        "2e7633b4c8ee8d8401e05e992885f935f22e7610f36e58d188583572cd857b0f"
    )


def test_library_adapter_ids_are_force_disconnect_buttons() -> None:
    source = (
        REPO_ROOT / "static" / "modules" / "libraries.js"
    ).read_text(encoding="utf-8")

    assert 'class="${["used-lib-jump", "disconnect-adapter"' in source
    assert "confirmForceDisconnectAdapter" in source
    assert "/adapters/${encodeURIComponent(adapterId)}/disconnect" in source


def test_webui_dom_id_contract() -> None:
    html = (REPO_ROOT / "static" / "index.html").read_text(encoding="utf-8")
    dom_ids = sorted(set(re.findall(r'\bid="([^"]+)"', html)))

    assert len(dom_ids) == 365
    assert _stable_hash(dom_ids) == (
        "fa6651bec28503fc6daf86f5e00a2e8680c8f4f866ecacb8333be9d8b9592ebe"
    )
    assert "page-files" in dom_ids
    assert "file-table-body" in dom_ids
    assert "file-auth-modal" in dom_ids
    assert "task-history-panel" in dom_ids
    assert "task-history-primary" in dom_ids
    assert "task-history-toggle" in dom_ids
    assert "tasks-finished-clear" in dom_ids
    assert "update-available-badge" in dom_ids
    assert "updates-modal" in dom_ids
    assert "provider-context-mode" in dom_ids
    assert "provider-context-source-label" in dom_ids


def test_settings_panels_keep_consistent_vertical_spacing() -> None:
    css = (REPO_ROOT / "static" / "styles.css").read_text(encoding="utf-8")

    assert ".runtime-residency-panel,.backup-migration-panel{margin-top:18px}" in css


def test_system_overview_grid_and_toggle_ownership() -> None:
    css = (REPO_ROOT / "static" / "styles.css").read_text(encoding="utf-8")
    system_js = (REPO_ROOT / "static" / "modules" / "system.js").read_text(
        encoding="utf-8"
    )
    libraries_js = (
        REPO_ROOT / "static" / "modules" / "libraries.js"
    ).read_text(encoding="utf-8")

    assert (
        ".system-grid{display:grid;grid-template-columns:"
        "minmax(0,1fr) minmax(0,1fr);gap:18px}"
    ) in css
    assert ".system-grid>.panel{min-width:0}" in css
    assert "overflow-wrap:anywhere;word-break:break-word" in css
    for toggle_id in ("system-provider-toggle", "system-index-toggle"):
        binding = f'$("{toggle_id}")?.addEventListener("click"'
        assert binding in system_js
        assert binding not in libraries_js


@pytest.mark.asyncio
async def test_http_status_and_static_cache_baseline() -> None:
    transport = ASGITransport(app=app_module.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        homepage = await client.get("/")
        health = await client.get("/api/v1/health")
        settings = await client.get("/api/v1/settings")
        missing = await client.get("/api/v1/not-a-route")
        static = await client.get(
            "/static/app.js",
            headers={"Accept-Encoding": "gzip"},
        )

    assert health.status_code == 200
    assert settings.status_code == 401
    assert missing.status_code == 404
    assert homepage.status_code == 200
    assert homepage.headers["cache-control"] == "no-store, max-age=0"
    assert homepage.headers["pragma"] == "no-cache"
    assert static.status_code == 200
    assert static.headers["cache-control"] == "public, max-age=0, must-revalidate"
    assert "etag" in static.headers
    assert static.headers["content-encoding"] == "gzip"


@pytest.mark.asyncio
async def test_health_and_settings_core_fields_remain_stable() -> None:
    health_fields = {
        "status",
        "product",
        "version",
        "livingmemory_database_version",
        "time",
    }
    settings_fields = {
        "host",
        "access_base_url",
        "configured_port",
        "configured_webui_url",
        "actual_port",
        "configured_access_port",
        "configured_api_access_url",
        "actual_access_port",
        "access_url",
        "webui_url",
        "api_access_url",
        "port_fallback_active",
        "access_port_fallback_active",
        "login_password_enabled",
        "login_mode",
        "api_key_fingerprint",
        "port_change_requires_restart",
        "access_port_change_requires_restart",
            "runtime_residency",
            "version",
        }

    assert set(app_module._settings_payload()) == settings_fields
    assert set(await app_module.health()) == health_fields


def test_public_compatibility_symbols_remain_available() -> None:
    from personalityrag.providers import (
        OpenAICompatibleEmbeddingProvider,
        VLLMEmbeddingProvider,
    )

    assert OpenAICompatibleEmbeddingProvider is VLLMEmbeddingProvider
