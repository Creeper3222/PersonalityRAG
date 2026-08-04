from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request

from personalityrag.application import create_app
from personalityrag.application_context import ApplicationContext
from personalityrag.auth import hash_password
from personalityrag.config import AppConfig, ProviderConfig
from personalityrag.database_types import DatabaseRef, LIVINGMEMORY_V8_TYPE
from personalityrag.database_types import database_type_registry
from personalityrag.library_types.text_media_v1.driver import TEXT_MEDIA_V1_TYPE
from personalityrag.listener_surface import (
    ADAPTER_ACCESS_SURFACE,
    WEBUI_SURFACE,
    listener_app,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _context(tmp_path: Path, name: str = "state") -> ApplicationContext:
    return ApplicationContext.create(
        source_root=REPO_ROOT,
        state_root=tmp_path / name,
        config=AppConfig(
            api_key="fixture-admin-key",
            session_secret="fixture-session-secret",
            library_psk_secret="fixture-database-secret",
            webui_password_hash=hash_password("fixture-password"),
            provider=ProviderConfig(
                id="fixture_embedding",
                display_name="Fixture Embedding",
                type="openai_embedding",
                api_base="http://127.0.0.1:9999/v1",
                api_key="provider-secret-value",
                model="fixture-model",
                dimensions=8,
            ),
        ),
        configure_logs=False,
    )


def _admin_headers() -> dict[str, str]:
    return {"Authorization": "Bearer fixture-admin-key"}


def _request(*, authorization: str = "Bearer fixture-admin-key") -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/debug/session",
            "headers": [(b"authorization", authorization.encode())],
            "query_string": b"",
            "server": ("127.0.0.1", 8765),
            "client": ("127.0.0.1", 43210),
            "scheme": "http",
        }
    )


@pytest.mark.asyncio
async def test_debug_session_is_local_webui_only_and_unlocks_with_password(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONALITYRAG_DEBUG_REVISION_API", "1")
    context = _context(tmp_path)
    await context.manager.initialize()
    try:
        application = create_app(context)
        webui = listener_app(application, WEBUI_SURFACE)
        adapter = listener_app(application, ADAPTER_ACCESS_SURFACE)
        async with (
            AsyncClient(
                transport=ASGITransport(app=webui),
                base_url="http://127.0.0.1:8765",
                headers=_admin_headers(),
            ) as client,
            AsyncClient(
                transport=ASGITransport(app=adapter),
                base_url="http://127.0.0.1:8766",
                headers=_admin_headers(),
            ) as adapter_client,
            AsyncClient(
                transport=ASGITransport(
                    app=webui,
                    client=("203.0.113.8", 4242),
                ),
                base_url="https://example.test",
                headers=_admin_headers(),
            ) as remote_client,
        ):
            locked = await client.get("/api/v1/debug/session")
            old_flag_cannot_unlock = await client.get(
                "/api/v1/debug/revisions/overview"
            )
            wrong_surface = await adapter_client.get("/api/v1/debug/session")
            remote = await remote_client.get("/api/v1/debug/session")
            unlocked = await client.post(
                "/api/v1/debug/session",
                json={"password": "fixture-password"},
            )
            overview = await client.get("/api/v1/debug/revisions/overview")
            refreshed = await client.get("/api/v1/debug/session")

        assert locked.status_code == 200
        assert locked.json()["unlocked"] is False
        assert old_flag_cannot_unlock.status_code == 404
        assert wrong_surface.status_code == 404
        assert remote.status_code == 404
        assert unlocked.status_code == 200
        assert unlocked.json()["unlocked"] is True
        cookie = unlocked.headers["set-cookie"].lower()
        assert "httponly" in cookie
        assert "samesite=strict" in cookie
        assert "max-age=900" in cookie
        assert overview.status_code == 200
        assert refreshed.json()["expires_at"] == unlocked.json()["expires_at"]
        assert refreshed.json()["remaining_seconds"] <= 900
    finally:
        await context.manager.close()


@pytest.mark.asyncio
async def test_debug_unlock_rate_limit_and_password_change_invalidate_session(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    await context.manager.initialize()
    try:
        application = listener_app(create_app(context), WEBUI_SURFACE)
        async with AsyncClient(
            transport=ASGITransport(app=application),
            base_url="http://127.0.0.1:8765",
            headers=_admin_headers(),
        ) as client:
            rejected = [
                await client.post(
                    "/api/v1/debug/session",
                    json={"password": "wrong-password"},
                )
                for _ in range(5)
            ]
            limited = await client.post(
                "/api/v1/debug/session",
                json={"password": "fixture-password"},
            )

        async with AsyncClient(
            transport=ASGITransport(
                app=application,
                client=("127.0.0.2", 2222),
            ),
            base_url="http://127.0.0.1:8765",
            headers=_admin_headers(),
        ) as fresh_client:
            unlocked = await fresh_client.post(
                "/api/v1/debug/session",
                json={"password": "fixture-password"},
            )
            changed = await fresh_client.patch(
                "/api/v1/settings",
                json={"new_password": "replacement-password"},
            )
            status = await fresh_client.get("/api/v1/debug/session")
            protected = await fresh_client.get(
                "/api/v1/debug/revisions/overview"
            )

        assert [response.status_code for response in rejected] == [401] * 5
        assert limited.status_code == 429
        assert int(limited.headers["retry-after"]) > 0
        assert unlocked.status_code == 200
        assert changed.status_code == 200
        assert status.status_code == 200
        assert status.json()["unlocked"] is False
        assert protected.status_code == 404
    finally:
        await context.manager.close()


@pytest.mark.asyncio
async def test_debug_session_cannot_be_reused_by_another_admin_credential_and_logout_revokes_it(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    await context.manager.initialize()
    try:
        application = listener_app(create_app(context), WEBUI_SURFACE)
        async with AsyncClient(
            transport=ASGITransport(app=application),
            base_url="http://127.0.0.1:8765",
            headers=_admin_headers(),
        ) as bearer_client:
            unlocked = await bearer_client.post(
                "/api/v1/debug/session",
                json={"password": "fixture-password"},
            )
            debug_cookie = bearer_client.cookies.get(
                "personalityrag_revision_debug"
            )
            assert unlocked.status_code == 200
            assert debug_cookie

            async with AsyncClient(
                transport=ASGITransport(app=application),
                base_url="http://127.0.0.1:8765",
            ) as password_client:
                login = await password_client.post(
                    "/api/v1/auth/login",
                    json={"credential": "fixture-password"},
                )
                password_client.cookies.set(
                    "personalityrag_revision_debug",
                    debug_cookie,
                )
                foreign_status = await password_client.get(
                    "/api/v1/debug/session"
                )

            logout = await bearer_client.post("/api/v1/auth/logout")
            bearer_client.cookies.set(
                "personalityrag_revision_debug",
                debug_cookie,
            )
            after_logout = await bearer_client.get("/api/v1/debug/session")

        assert login.status_code == 200
        assert foreign_status.status_code == 200
        assert foreign_status.json()["unlocked"] is False
        assert logout.status_code == 200
        assert after_logout.json()["unlocked"] is False
    finally:
        await context.manager.close()


@pytest.mark.asyncio
async def test_dangerous_revision_routes_require_password_and_risk_confirmation(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    await context.manager.initialize()
    try:
        provider_id = context.config.provider.id
        created = await context.manager.control.update_provider(
            provider_id,
            {"model": "functionally-different-model"},
        )
        application = listener_app(create_app(context), WEBUI_SURFACE)
        async with AsyncClient(
            transport=ASGITransport(app=application),
            base_url="http://127.0.0.1:8765",
            headers=_admin_headers(),
        ) as client:
            await client.post(
                "/api/v1/debug/session",
                json={"password": "fixture-password"},
            )
            reset_without_assertion = await client.post(
                f"/api/v1/debug/providers/{provider_id}/revisions/reset",
                json={"latest_revision": 1},
            )
            reset_without_risk = await client.post(
                f"/api/v1/debug/providers/{provider_id}/revisions/reset",
                json={
                    "latest_revision": 1,
                    "force_non_equivalent": True,
                    "password": "fixture-password",
                },
            )
            reset_wrong_password = await client.post(
                f"/api/v1/debug/providers/{provider_id}/revisions/reset",
                json={
                    "latest_revision": 1,
                    "force_non_equivalent": True,
                    "password": "wrong-password",
                    "risk_confirmed": True,
                },
            )
            reset = await client.post(
                f"/api/v1/debug/providers/{provider_id}/revisions/reset",
                json={
                    "latest_revision": 1,
                    "force_non_equivalent": True,
                    "password": "fixture-password",
                    "risk_confirmed": True,
                },
            )
            patch_without_risk = await client.patch(
                f"/api/v1/debug/providers/{provider_id}/revisions/{created.revision}",
                json={
                    "patch": {"display_name": "Unsafe"},
                    "password": "fixture-password",
                },
            )
            patched = await client.patch(
                f"/api/v1/debug/providers/{provider_id}/revisions/{created.revision}",
                json={
                    "patch": {"display_name": "Debug Patched"},
                    "password": "fixture-password",
                    "risk_confirmed": True,
                },
            )

        assert reset_without_assertion.status_code == 400
        assert reset_without_risk.status_code == 400
        assert reset_wrong_password.status_code == 401
        assert reset.status_code == 200
        assert reset.json()["latest_revision"] == 1
        assert patch_without_risk.status_code == 400
        assert patched.status_code == 200
        assert "provider-secret-value" not in json.dumps(patched.json())
    finally:
        await context.manager.close()


@pytest.mark.asyncio
async def test_debug_unlock_is_disabled_without_a_webui_password(
    tmp_path: Path,
) -> None:
    context = ApplicationContext.create(
        source_root=REPO_ROOT,
        state_root=tmp_path / "no-password",
        config=AppConfig(
            api_key="fixture-admin-key",
            session_secret="fixture-session-secret",
        ),
        configure_logs=False,
    )
    application = listener_app(create_app(context), WEBUI_SURFACE)
    async with AsyncClient(
        transport=ASGITransport(app=application),
        base_url="http://127.0.0.1:8765",
        headers=_admin_headers(),
    ) as client:
        status = await client.get("/api/v1/debug/session")
        unlock = await client.post(
            "/api/v1/debug/session",
            json={"password": "anything"},
        )

    assert status.status_code == 200
    assert status.json()["password_configured"] is False
    assert status.json()["unlocked"] is False
    assert unlock.status_code == 409


def test_debug_session_has_fixed_expiry_and_is_bound_to_admin_credential(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    request = _request()
    token, issued = context.revision_debug.issue(
        request,
        context.auth,
        now=1_000,
    )
    refreshed = context.revision_debug.status(
        token,
        request,
        context.auth,
        now=1_450,
    )
    other_credential = context.revision_debug.status(
        token,
        _request(authorization="Bearer invalid-credential"),
        context.auth,
        now=1_450,
    )
    expired = context.revision_debug.status(
        token,
        request,
        context.auth,
        now=1_901,
    )
    expired_at_boundary = context.revision_debug.status(
        token,
        request,
        context.auth,
        now=1_900,
    )

    assert issued["expires_at"] == 1_900
    assert refreshed["expires_at"] == issued["expires_at"]
    assert refreshed["remaining_seconds"] == 450
    assert other_credential["unlocked"] is False
    assert expired_at_boundary["unlocked"] is False
    assert expired["unlocked"] is False


@pytest.mark.asyncio
async def test_revision_overview_uses_typed_database_identity_and_redacts_secrets(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    await context.manager.initialize()
    try:
        provider_id = context.config.provider.id
        await context.manager.create_library(
            {
                "id": "shared_id",
                "database_type": LIVINGMEMORY_V8_TYPE,
                "name": "Shared Memory",
                "provider_id": provider_id,
            }
        )
        await context.manager.create_library(
            {
                "id": "shared_id",
                "database_type": TEXT_MEDIA_V1_TYPE,
                "name": "Shared Knowledge",
                "provider_id": provider_id,
            }
        )
        created_revision = await context.manager.control.update_provider(
            provider_id,
            {"model": "temporary-functional-change"},
        )
        await context.manager.debug_patch_provider_revision(
            provider_id,
            created_revision.revision,
            {"model": "fixture-model", "display_name": "Equivalent Name"},
        )
        overview = await context.manager.debug_revision_overview()

        shared = [
            item for item in overview["databases"]
            if item["database_id"] == "shared_id"
        ]
        provider = next(
            item for item in overview["providers"]
            if item["provider_id"] == provider_id
        )
        serialized = json.dumps(overview, ensure_ascii=False)

        assert {item["database_type"] for item in shared} == {
            LIVINGMEMORY_V8_TYPE,
            TEXT_MEDIA_V1_TYPE,
        }
        assert len(provider["revisions"]) == 2
        assert all(
            revision["functionally_equal_to_latest"]
            for revision in provider["revisions"]
        )
        assert "provider-secret-value" not in serialized
        assert all(
            "api_key" not in revision["config"]
            for revision in provider["revisions"]
        )
        assert all(
            revision["config"]["has_api_key"] is True
            for revision in provider["revisions"]
        )
    finally:
        await context.manager.close()


@pytest.mark.asyncio
async def test_equivalent_typed_binding_repairs_leave_payload_files_unchanged(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    await context.manager.initialize()
    try:
        provider_id = context.config.provider.id
        for database_type, name in (
            (LIVINGMEMORY_V8_TYPE, "Memory"),
            (TEXT_MEDIA_V1_TYPE, "Knowledge"),
        ):
            await context.manager.create_library(
                {
                    "id": "repair_shared",
                    "database_type": database_type,
                    "name": name,
                    "provider_id": provider_id,
                }
            )
        updated = await context.manager.control.update_provider(
            provider_id,
            {"model": "temporary-functional-change"},
        )
        await context.manager.debug_patch_provider_revision(
            provider_id,
            updated.revision,
            {"model": "fixture-model", "display_name": "Equivalent Revision"},
        )
        target_revision = int(updated.revision)
        text_directory = database_type_registry.data_dir(
            context.manager.data_dir,
            DatabaseRef(TEXT_MEDIA_V1_TYPE, "repair_shared"),
        )
        sentinel = text_directory / "payload-sentinel.bin"
        sentinel.write_bytes(b"vector-and-media-payload-must-not-change")
        before_hash = hashlib.sha256(sentinel.read_bytes()).hexdigest()

        memory_result = await context.manager.debug_repair_database_binding(
            database_type=LIVINGMEMORY_V8_TYPE,
            database_id="repair_shared",
            usage_kind="embedding",
            provider_id=provider_id,
            revision=target_revision,
        )
        knowledge_result = await context.manager.debug_repair_database_binding(
            database_type=TEXT_MEDIA_V1_TYPE,
            database_id="repair_shared",
            usage_kind="embedding",
            provider_id=provider_id,
            revision=target_revision,
        )
        after_hash = hashlib.sha256(sentinel.read_bytes()).hexdigest()

        assert memory_result["functionally_equivalent"] is True
        assert knowledge_result["functionally_equivalent"] is True
        assert before_hash == after_hash
        assert next(
            item for item in memory_result["after"]["bindings"]
            if item["usage_kind"] == "embedding"
        )["provider_revision"] == target_revision
        assert next(
            item for item in knowledge_result["after"]["bindings"]
            if item["usage_kind"] == "embedding"
        )["provider_revision"] == target_revision

        control_db = await context.manager.control.connect()
        try:
            await control_db.execute(
                """INSERT INTO jobs
                (id,library_id,database_type,database_id,kind,status,progress,
                 message,created_at,updated_at)
                VALUES(?,?,?,?,?,'running',0.5,'debug fixture',1,1)""",
                (
                    "revision-debug-active-job",
                    f"{TEXT_MEDIA_V1_TYPE}:repair_shared",
                    TEXT_MEDIA_V1_TYPE,
                    "repair_shared",
                    "text_media_index_rebuild",
                ),
            )
            await control_db.commit()
        finally:
            await control_db.close()
        with pytest.raises(ValueError, match="active tasks or adapters"):
            await context.manager.debug_repair_database_binding(
                database_type=TEXT_MEDIA_V1_TYPE,
                database_id="repair_shared",
                usage_kind="embedding",
                provider_id=provider_id,
                revision=1,
            )
        control_db = await context.manager.control.connect()
        try:
            await control_db.execute(
                "UPDATE jobs SET status='completed' WHERE id=?",
                ("revision-debug-active-job",),
            )
            await control_db.commit()
        finally:
            await control_db.close()

        await context.manager.control.register_adapter_connection(
            DatabaseRef(TEXT_MEDIA_V1_TYPE, "repair_shared"),
            adapter_id="fixture-adapter",
            instance_id="fixture-instance",
            adapter_type="fixture",
        )
        with pytest.raises(ValueError, match="active tasks or adapters"):
            await context.manager.debug_repair_database_binding(
                database_type=TEXT_MEDIA_V1_TYPE,
                database_id="repair_shared",
                usage_kind="embedding",
                provider_id=provider_id,
                revision=1,
            )
    finally:
        await context.manager.close()


@pytest.mark.asyncio
async def test_latest_reset_unloads_databases_that_follow_provider_latest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    await context.manager.initialize()
    try:
        await context.manager.control.create_provider(
            {
                "id": "fixture_rerank",
                "display_name": "Fixture Rerank",
                "type": "vllm_rerank",
                "enabled": True,
                "api_base": "http://127.0.0.1:9998",
                "api_suffix": "/v1/rerank",
                "model": "fixture-rerank-model",
                "dimensions": 0,
                "batch_size": 1,
                "concurrency": 1,
            }
        )
        updated = await context.manager.control.update_provider(
            "fixture_rerank",
            {"model": "temporary-rerank-model"},
        )
        await context.manager.debug_patch_provider_revision(
            "fixture_rerank",
            updated.revision,
            {
                "model": "fixture-rerank-model",
                "display_name": "Equivalent Rerank Name",
            },
        )
        await context.manager.create_library(
            {
                "id": "follows_latest",
                "database_type": LIVINGMEMORY_V8_TYPE,
                "name": "Follows Latest",
                "provider_id": context.config.provider.id,
                "rerank_provider_id": "fixture_rerank",
            }
        )

        unloaded: list[tuple[str, str]] = []
        original_unload = context.manager.unload_runtime

        async def recording_unload(ref: DatabaseRef, *, reason: str) -> None:
            unloaded.append((ref.key, reason))
            await original_unload(ref, reason=reason)

        monkeypatch.setattr(context.manager, "unload_runtime", recording_unload)
        result = await context.manager.debug_reset_provider_revisions(
            "fixture_rerank",
            latest_revision=1,
        )

        assert result["latest_revision"] == 1
        assert (
            f"{LIVINGMEMORY_V8_TYPE}:follows_latest",
            "revision_debug_latest_reset",
        ) in unloaded
    finally:
        await context.manager.close()
