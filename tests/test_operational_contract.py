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

    assert len(routes) == 149
    assert _stable_hash(routes) == (
        "c4bca7c216be5c76313e69fdc1a9573e9e6a9282b1afa139d5f5eada2af09d03"
    )
    assert (
        "GET",
        "/api/v1/jobs/{job_id}/details",
        "job_details",
    ) in routes
    assert ("POST", "/api/v1/providers/test", "provider_test_compat") in routes
    assert not any(path.startswith("/api/v1/libraries") for _, path, _ in routes)
    assert not any(path.startswith("/api/v1/databases/") for _, path, _ in routes)
    assert ("GET", "/api/v1/database-types", "database_types") in routes
    assert (
        "POST",
        "/api/v1/memory-libraries/livingmemory_v8/{memory_store_id}/recall",
        "recall",
    ) in routes
    assert (
        "GET",
        "/api/v1/memory-libraries/livingmemory_v8/{memory_store_id}/memories/{memory_id}/source",
        "memory_source",
    ) in routes
    assert (
        "POST",
        "/api/v1/memory-libraries/livingmemory_v8/{memory_store_id}/memories/{memory_id}/archive",
        "archive_memory",
    ) in routes
    assert (
        "POST",
        "/api/v1/memory-libraries/livingmemory_v8/{memory_store_id}/transfers/imports/{preview_id}/commit",
        "commit_memory_transfer",
    ) in routes
    assert (
        "POST",
        "/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/search",
        "search",
    ) in routes
    assert (
        "POST",
        "/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/ingest-batches",
        "ingest_batch",
    ) in routes
    assert (
        "POST",
        "/api/v1/knowledge-libraries/text_media_v1/imports/inspect",
        "inspect_import",
    ) in routes
    assert (
        "POST",
        "/api/v1/knowledge-libraries/text_media_v1/transfer-batches/exports",
        "create_batch_export",
    ) in routes
    assert (
        "POST",
        "/api/v1/knowledge-libraries/text_media_v1/transfer-batches/imports/{token}/commit",
        "commit_batch_import",
    ) in routes
    assert (
        "POST",
        "/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/copy",
        "copy_database",
    ) in routes
    assert (
        "POST",
        "/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/backup",
        "backup_database",
    ) in routes
    assert (
        "GET",
        "/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/media-calibrations",
        "list_media_calibrations",
    ) in routes
    assert (
        "PUT",
        "/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/document-media-relations/{document_id}/{asset_id}/semantic-calibration",
        "recalibrate_document_media",
    ) in routes
    assert (
        "GET",
        "/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/assets/{asset_id}/content",
        "asset_content",
    ) in routes
    assert (
        "DELETE",
        "/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/entries/{entry_id}",
        "delete_entry",
    ) in routes
    assert (
        "DELETE",
        "/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/assets/{asset_id}",
        "delete_image",
    ) in routes
    assert (
        "GET",
        "/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/documents/{document_id}",
        "document_detail",
    ) in routes
    assert (
        "POST",
        "/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/documents/batch-delete",
        "delete_documents_batch",
    ) in routes
    assert (
        "GET",
        "/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/chunks",
        "chunks",
    ) in routes
    assert (
        "GET",
        "/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/chunks/{chunk_id}",
        "chunk_detail",
    ) in routes
    assert (
        "GET",
        "/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/assets/{asset_id}",
        "asset_detail",
    ) in routes
    assert (
        "PUT",
        "/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/assets/{asset_id}/media-descriptions",
        "update_asset_media_descriptions",
    ) in routes
    assert (
        "PUT",
        "/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/visual-intent-policy",
        "update_visual_intent_policy",
    ) in routes
    assert (
        "POST",
        "/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/visual-intent-policy/export",
        "export_visual_intent_policy",
    ) in routes
    assert (
        "POST",
        "/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/visual-intent-policy/imports/inspect",
        "inspect_visual_intent_policy_import",
    ) in routes
    assert (
        "POST",
        "/api/v1/memory-libraries/livingmemory_v8/{memory_store_id}/adapters/{adapter_id}/disconnect",
        "disconnect_adapter",
    ) in routes
    assert (
        "POST",
        "/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/adapters/heartbeat",
        "knowledge_adapter_heartbeat",
    ) in routes
    assert (
        "POST",
        "/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/adapters/{adapter_id}/disconnect",
        "disconnect_knowledge_adapter",
    ) in routes
    assert ("GET", "/api/v1/files", "list_files") in routes
    assert ("POST", "/api/v1/files/download", "download_files") in routes
    assert ("GET", "/api/v1/updates/status", "update_status") in routes
    assert ("GET", "/api/v1/updates/releases", "update_releases") in routes
    assert ("POST", "/api/v1/updates/switch", "switch_version") in routes


def test_openapi_contract() -> None:
    openapi = app_module.app.openapi()
    assert _stable_hash(openapi) == (
        "2f7507c8332d83268a07c00b35c7aeccac491a9318a9cb19fde6ca596f468562"
    )
    assert (
        "memory_type"
        not in openapi["components"]["schemas"]["MemoryCreate"]["properties"]
    )
    assert (
        "memory_type"
        not in openapi["components"]["schemas"]["MemoryUpdate"]["properties"]
    )
    assert (
        "database_type"
        not in openapi["components"]["schemas"]["LivingMemoryV8Create"]["properties"]
    )
    search_threshold = openapi["components"]["schemas"]["SearchRequest"]["properties"][
        "media_relevance_pivot"
    ]
    assert {item.get("type") for item in search_threshold["anyOf"]} == {
        "number",
        "null",
    }
    legacy_search_threshold = openapi["components"]["schemas"]["SearchRequest"][
        "properties"
    ]["media_score_threshold"]
    assert legacy_search_threshold["deprecated"] is True
    rerank_switch = openapi["components"]["schemas"]["SearchRequest"]["properties"][
        "rerank"
    ]
    assert {item.get("type") for item in rerank_switch["anyOf"]} == {
        "boolean",
        "null",
    }
    retrieval = openapi["components"]["schemas"]["RetrievalSettingsRequest"][
        "properties"
    ]
    assert retrieval["rerank_candidate_limit"]["default"] == 10
    assert retrieval["rerank_fusion_weight"]["default"] == 0.30
    assert retrieval["text_lexical_boost"]["default"] == 0.6
    assert retrieval["rerank_rank_bonus_weight"]["default"] == 0.0
    assert retrieval["rerank_rank_reliability_exponent"]["default"] == 1.5
    retrieval_properties = openapi["components"]["schemas"]["RetrievalSettingsRequest"][
        "properties"
    ]
    assert retrieval_properties["media_relevance_pivot_fallback"]["default"] == 0.35
    assert retrieval_properties["media_pivot_positive_blend"]["default"] == 0.7
    assert retrieval_properties["media_pivot_negative_weight"]["default"] == 0.35
    assert (
        retrieval_properties["media_pivot_negative_attenuation_floor"]["default"]
        == 0.05
    )
    assert retrieval_properties["media_format_mismatch_factor"]["default"] == 0.1
    assert retrieval_properties["media_content_mismatch_factor"]["default"] == 0.1
    assert retrieval_properties["media_threshold_evidence_limit"]["default"] == 5
    assert retrieval_properties["unbound_media_candidate_limit"]["default"] == 10
    assert retrieval_properties["unbound_media_distinctive_boost"]["default"] == 0.35
    assert retrieval_properties["unbound_media_collection_boost"]["default"] == 0.55
    assert retrieval_properties["unbound_media_competition_floor"]["default"] == 0.35
    assert retrieval_properties["unbound_media_reliability_target"]["default"] == 0.25
    assert retrieval_properties["unbound_media_specificity_exponent"]["default"] == 1.0
    assert retrieval_properties["unbound_media_advantage_target"]["default"] == 0.04
    visual_policy = openapi["components"]["schemas"]["VisualIntentPolicyRequest"]
    assert set(visual_policy["required"]) == {
        "visual_object_terms",
        "lookup_action_terms",
        "generation_action_terms",
        "reference_connector_terms",
    }
    assert all(
        visual_policy["properties"][key]["maxItems"] == 256
        for key in visual_policy["required"]
    )
    assert (
        "/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/visual-intent-policy"
    ) in openapi["paths"]
    list_parameters = openapi["paths"][
        "/api/v1/memory-libraries/livingmemory_v8/{memory_store_id}/memories"
    ]["get"]["parameters"]
    assert "memory_type" not in {item["name"] for item in list_parameters}


def test_library_adapter_ids_are_force_disconnect_buttons() -> None:
    source = (REPO_ROOT / "static" / "modules" / "libraries.js").read_text(
        encoding="utf-8"
    )

    assert 'class="${["used-lib-jump", "disconnect-adapter"' in source
    assert "confirmForceDisconnectAdapter" in source
    assert "/adapters/${encodeURIComponent(adapterId)}/disconnect" in source


def test_webui_dom_id_contract() -> None:
    html = (REPO_ROOT / "static" / "index.html").read_text(encoding="utf-8")
    all_dom_ids = re.findall(r'\bid="([^"]+)"', html)
    dom_ids = set(all_dom_ids)

    assert len(all_dom_ids) == len(dom_ids)
    assert "database-type-nav" in dom_ids
    assert "page-database" in dom_ids
    assert "database-page-host" in dom_ids
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
    assert "text-media-create-modal" in dom_ids
    text_media_html = (
        REPO_ROOT / "static" / "database-types" / "text-media-v1.html"
    ).read_text(encoding="utf-8")
    living_html = (
        REPO_ROOT / "static" / "database-types" / "livingmemory-v8.html"
    ).read_text(encoding="utf-8")
    assert "text-media-workspace-modal" not in dom_ids
    assert 'id="text-media-import-preview"' in text_media_html
    assert '<select id="text-media-edit-provider" required>' in text_media_html
    assert 'id="text-media-edit-original-provider" type="hidden"' in text_media_html
    assert 'id="text-media-visual-policy-groups"' in text_media_html
    assert 'id="text-media-visual-policy-reset"' in text_media_html
    assert 'id="text-media-protected-blocker-list"' in text_media_html
    text_media_js = (REPO_ROOT / "static" / "modules" / "text-media-v1.js").read_text(
        encoding="utf-8"
    )
    assert '"text_media_index_rebuild"' in text_media_js
    assert 'reason: "library_edit_provider_switch"' in text_media_js
    assert 'id="page-graph"' in living_html
    assert 'id="page-memory"' in living_html
    assert 'id="page-recall"' in living_html
    assert 'id="page-library-overview"' in living_html


def test_settings_panels_keep_consistent_vertical_spacing() -> None:
    css = (REPO_ROOT / "static" / "styles.css").read_text(encoding="utf-8")

    assert ".runtime-residency-panel,.backup-migration-panel{margin-top:18px}" in css


def test_system_overview_grid_and_toggle_ownership() -> None:
    css = (REPO_ROOT / "static" / "styles.css").read_text(encoding="utf-8")
    system_js = (REPO_ROOT / "static" / "modules" / "system.js").read_text(
        encoding="utf-8"
    )
    libraries_js = (REPO_ROOT / "static" / "modules" / "libraries.js").read_text(
        encoding="utf-8"
    )

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
        "time",
    }
    settings_fields = {
        "host",
        "access_base_url",
        "public_adapter_url",
        "recommended_adapter_url",
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
        "deployment_mode",
        "managed_settings",
        "runtime_residency",
        "performance_profile",
        "effective_performance",
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
