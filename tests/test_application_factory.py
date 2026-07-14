from __future__ import annotations

import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from personalityrag import app as app_module
from personalityrag import http_middleware
from personalityrag.application import create_app
from personalityrag.application_context import ApplicationContext
from personalityrag.auth import hash_password
from personalityrag.config import AppConfig
from personalityrag.routes import auth_settings


REPO_ROOT = Path(__file__).resolve().parents[1]


def _context(tmp_path: Path, name: str, config: AppConfig) -> ApplicationContext:
    return ApplicationContext.create(
        source_root=REPO_ROOT,
        state_root=tmp_path / name,
        config=config,
        configure_logs=False,
    )


@pytest.mark.asyncio
async def test_create_app_keeps_contexts_and_authentication_isolated(
    tmp_path: Path,
) -> None:
    first_context = _context(
        tmp_path,
        "first",
        AppConfig(api_key="first-api-key", session_secret="first-session"),
    )
    second_context = _context(
        tmp_path,
        "second",
        AppConfig(api_key="second-api-key", session_secret="second-session"),
    )
    first_app = create_app(first_context)
    second_app = create_app(second_context)

    async with AsyncClient(
        transport=ASGITransport(app=first_app),
        base_url="http://first:8765",
    ) as first_client, AsyncClient(
        transport=ASGITransport(app=second_app),
        base_url="http://second:8765",
    ) as second_client:
        first_ok = await first_client.get(
            "/api/v1/settings",
            headers={"Authorization": "Bearer first-api-key"},
        )
        first_rejects_second = await first_client.get(
            "/api/v1/settings",
            headers={"Authorization": "Bearer second-api-key"},
        )
        second_ok = await second_client.get(
            "/api/v1/settings",
            headers={"Authorization": "Bearer second-api-key"},
        )

    assert first_ok.status_code == 200
    assert first_rejects_second.status_code == 401
    assert second_ok.status_code == 200
    assert first_context.manager.root == tmp_path / "first"
    assert second_context.manager.root == tmp_path / "second"


@pytest.mark.asyncio
async def test_request_id_is_preserved_or_generated(tmp_path: Path) -> None:
    context = _context(tmp_path, "request-id", AppConfig(api_key="fixture-key"))
    application = create_app(context)
    async with AsyncClient(
        transport=ASGITransport(app=application),
        base_url="http://test:8765",
    ) as client:
        supplied = await client.get(
            "/api/v1/health",
            headers={"X-Request-ID": "fixture-request-id"},
        )
        generated = await client.get("/api/v1/health")

    assert supplied.headers["x-request-id"] == "fixture-request-id"
    assert len(generated.headers["x-request-id"]) == 32
    assert supplied.status_code == generated.status_code == 200


@pytest.mark.asyncio
async def test_update_status_is_authenticated_and_context_scoped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, "updates", AppConfig(api_key="fixture-key"))
    application = create_app(context)

    async def status(*, refresh: bool = False):
        return {"current_version": "0.1.0", "update_available": False, "refresh": refresh}

    monkeypatch.setattr(context.updates, "status", status)
    async with AsyncClient(
        transport=ASGITransport(app=application),
        base_url="http://test:8765",
    ) as client:
        rejected = await client.get("/api/v1/updates/status")
        accepted = await client.get(
            "/api/v1/updates/status?refresh=true",
            headers={"Authorization": "Bearer fixture-key"},
        )

    assert rejected.status_code == 401
    assert accepted.status_code == 200
    assert accepted.json() == {
        "current_version": "0.1.0",
        "update_available": False,
        "refresh": True,
    }


@pytest.mark.asyncio
async def test_log_poll_is_silent_and_heartbeat_does_not_raise_slow_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, "long-poll-logs", AppConfig(api_key="fixture-key"))
    debug_paths: list[str] = []
    warning_paths: list[str] = []
    clock = iter([0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0])

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        return None

    monkeypatch.setattr(http_middleware.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        http_middleware.logger,
        "debug",
        lambda message, *args: debug_paths.append(str(args[2])),
    )
    monkeypatch.setattr(
        http_middleware.logger,
        "warning",
        lambda message, *args: warning_paths.append(str(args[2])),
    )
    middleware = http_middleware.RequestContextMiddleware(app, context=context)

    for method, path in (
        ("GET", "/api/v1/logs"),
        ("GET", "/api/v1/jobs"),
        ("POST", "/api/v1/libraries/demo/adapters/heartbeat"),
        ("GET", "/api/v1/health"),
    ):
        await middleware(
            {
                "type": "http",
                "headers": [],
                "method": method,
                "path": path,
            },
            receive,
            send,
        )

    assert debug_paths == ["/api/v1/libraries/demo/adapters/heartbeat"]
    assert warning_paths == ["/api/v1/health"]


@pytest.mark.asyncio
async def test_pbkdf2_login_and_generation_run_through_worker_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    config = AppConfig(
        api_key="fixture-api-key",
        session_secret="fixture-session",
        webui_password_hash=hash_password("initial-password"),
    )
    context = _context(tmp_path, "password", config)
    application = create_app(context)

    async def inline_to_thread(function, /, *args, **kwargs):
        calls.append(function.__name__)
        return function(*args, **kwargs)

    monkeypatch.setattr(auth_settings.asyncio, "to_thread", inline_to_thread)
    async with AsyncClient(
        transport=ASGITransport(app=application),
        base_url="http://test:8765",
    ) as client:
        login = await client.post(
            "/api/v1/auth/login",
            json={"credential": "initial-password"},
        )
        update = await client.patch(
            "/api/v1/settings",
            headers={"Authorization": "Bearer fixture-api-key"},
            json={"new_password": "replacement-password"},
        )

    assert login.status_code == 200
    assert update.status_code == 200
    assert calls == ["verify_login_secret", "hash_password"]


@pytest.mark.asyncio
async def test_runtime_residency_settings_apply_immediately_and_persist(
    tmp_path: Path,
) -> None:
    context = _context(
        tmp_path,
        "runtime-residency",
        AppConfig(api_key="fixture-key"),
    )
    application = create_app(context)
    async with AsyncClient(
        transport=ASGITransport(app=application),
        base_url="http://test:8765",
    ) as client:
        response = await client.patch(
            "/api/v1/settings",
            headers={"Authorization": "Bearer fixture-key"},
            json={
                "runtime_idle_minutes": 12,
                "max_non_default_runtimes": 3,
            },
        )
        invalid = await client.patch(
            "/api/v1/settings",
            headers={"Authorization": "Bearer fixture-key"},
            json={"runtime_idle_minutes": 0},
        )

    assert response.status_code == 200
    assert response.json()["runtime_residency"] == {
        "idle_minutes": 12,
        "max_non_default_runtimes": 3,
    }
    assert invalid.status_code == 422
    persisted = json.loads(context.config_path.read_text(encoding="utf-8"))
    assert persisted["runtime_residency"] == {
        "idle_minutes": 12,
        "max_non_default_runtimes": 3,
    }


@pytest.mark.asyncio
async def test_adapter_status_reads_do_not_load_or_touch_library_runtime(
    tmp_path: Path,
) -> None:
    context = _context(
        tmp_path,
        "adapter-offline-status",
        AppConfig(api_key="fixture-key"),
    )
    await context.manager.initialize()
    try:
        await context.manager.create_library(
            {
                "id": "offline",
                "name": "offline",
                "provider_id": context.config.provider.id,
            }
        )
        await context.manager.unload_runtime("offline", reason="fixture")
        default_last_used = context.manager.runtime_residency_status()["runtimes"][
            "Default"
        ]["last_used_at"]
        application = create_app(context)
        headers = {
            "Authorization": "Bearer fixture-key",
            "X-PersonalityRAG-Adapter-ID": "Astrbot",
        }
        async with AsyncClient(
            transport=ASGITransport(app=application),
            base_url="http://test:8765",
        ) as client:
            detail = await client.get(
                "/api/v1/libraries/offline",
                headers=headers,
            )
            stats = await client.get(
                "/api/v1/libraries/offline/stats",
                headers=headers,
            )
            indexes = await client.get(
                "/api/v1/libraries/offline/indexes",
                headers=headers,
            )

        assert detail.status_code == stats.status_code == indexes.status_code == 200
        assert "offline" not in context.manager.runtimes
        assert context.manager.runtime_residency_status()["runtimes"]["Default"][
            "last_used_at"
        ] == default_last_used
        assert stats.json()["provider_status"]["cached"] is True
    finally:
        await context.manager.close()


def test_factory_preserves_openapi_paths_and_dual_server_uses_one_app() -> None:
    factory_app = create_app(app_module.DEFAULT_CONTEXT)
    assert factory_app.openapi()["paths"] == app_module.app.openapi()["paths"]

    run_source = (REPO_ROOT / "run.py").read_text(encoding="utf-8")
    assert '"personalityrag.app:app"' not in run_source
    assert run_source.count("app_module.app") >= 2
    assert run_source.count("access_log=False") == 2
