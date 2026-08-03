from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = REPO_ROOT / "static"
MODULE_NAMES = {
    "api",
    "database-ui-registry",
    "files",
    "global-system",
    "libraries",
    "providers",
    "revision-debug",
    "settings",
    "tasks-logs",
    "ui-guard",
}


def test_revision_debug_webui_is_password_gated_typed_and_never_renders_raw_json() -> None:
    index = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    app = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    module = (STATIC_ROOT / "modules" / "revision-debug.js").read_text(
        encoding="utf-8"
    )
    styles = (STATIC_ROOT / "styles.css").read_text(encoding="utf-8")
    icon = (STATIC_ROOT / "icons" / "debug.svg").read_text(encoding="utf-8")

    assert 'id="revision-debug-settings-card"' in index
    assert 'id="revision-debug-nav" class="nav hidden"' in index
    assert 'id="page-revision-debug"' in index
    assert 'type="password" autocomplete="current-password"' in index
    assert '"revision-debug"' in app
    assert 'api("/debug/session"' in module
    assert 'api("/debug/revisions/overview"' in module
    assert "/debug/databases/${encode(databaseType)}/${encode(databaseId)}" in module
    assert 'revision.config?.has_api_key' in module
    assert "JSON.stringify(overview" not in module
    assert "<pre" not in module
    assert "max-height:min(58vh,680px)" in styles
    assert "var(--scrollbar-thumb)" in styles
    assert "<?xml" not in icon
    assert "<!DOCTYPE" not in icon
    assert "fill=" not in icon


def test_log_console_uses_compact_rows_and_a_larger_viewport() -> None:
    styles = (STATIC_ROOT / "styles.css").read_text(encoding="utf-8")

    assert ".log-console{height:clamp(520px,72vh,920px)" in styles
    assert "font-size:11px;line-height:1.38" in styles
    assert (
        ".log-entry{display:grid;grid-template-columns:50px minmax(0,1fr);"
        "align-items:start;gap:7px;border-bottom:0;padding:2px 0}"
    ) in styles
    assert (
        ".log-console{height:clamp(360px,68vh,720px);padding:8px;"
        "font-size:10px;line-height:1.35}"
    ) in styles
    assert (
        ".log-entry{grid-template-columns:42px minmax(0,1fr);"
        "gap:6px;padding:1px 0}"
    ) in styles


def test_global_system_overview_emphasizes_core_information() -> None:
    index = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    styles = (STATIC_ROOT / "styles.css").read_text(encoding="utf-8")
    module = (STATIC_ROOT / "modules" / "global-system.js").read_text(
        encoding="utf-8"
    )

    assert 'class="system-grid global-system-grid"' in index
    assert index.count("global-system-panel") == 3
    assert (
        ".global-system-grid{grid-template-columns:minmax(0,1.25fr) "
        "repeat(2,minmax(0,.9fr));align-items:stretch}"
    ) in styles
    assert ".global-system-service-hero strong{font-size:24px" in styles
    assert ".global-system-resource-grid{grid-template-columns:repeat(2,minmax(0,1fr))}" in styles
    assert ".global-system-runtime-grid{grid-template-columns:1fr}" in styles
    assert 'class="global-system-service-hero"' in module
    assert 'class="global-system-detail-list"' in module
    assert "global-system-resource-grid" in module
    assert "global-system-runtime-grid" in module


def test_webui_entrypoint_is_only_shared_assembly() -> None:
    source = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert len(source.splitlines()) < 1_380
    for name in MODULE_NAMES:
        assert f'from "./modules/{name}.js"' in source
    for name in ("graph", "memories", "recall", "system"):
        assert f'from "./modules/{name}.js"' not in source
        assert (
            f'from "../modules/{name}.js"'
            in (STATIC_ROOT / "database-types" / "livingmemory-v8.js").read_text(
                encoding="utf-8"
            )
        )
    assert 'from "./modules/text-media-v1.js"' not in source
    assert 'from "../modules/text-media-v1.js"' in (
        STATIC_ROOT / "database-types" / "text-media-v1.js"
    ).read_text(encoding="utf-8")
    assert "async function loadProviders" not in source
    assert "async function loadLibraries" not in source
    assert "async function loadMemories" not in source
    assert "async function loadSystem" not in source


def test_text_media_content_page_uses_management_name_and_icon() -> None:
    driver = (
        STATIC_ROOT / "database-types" / "text-media-v1.js"
    ).read_text(encoding="utf-8")
    template = (
        STATIC_ROOT / "database-types" / "text-media-v1.html"
    ).read_text(encoding="utf-8")
    zh_source = (STATIC_ROOT / "locales" / "zh.js").read_text(encoding="utf-8")
    icon = STATIC_ROOT / "icons" / "content-management.svg"

    assert 'labelKey: "knowledgeContentManagement"' in driver
    assert 'titleKey: "knowledgeContentManagement"' in driver
    assert 'navIcon("/static/icons/content-management.svg")' in driver
    assert 'data-i18n="knowledgeContentManagement">内容管理</button>' in template
    assert '"knowledgeContentManagement": "内容管理"' in zh_source
    assert "knowledgeContentSystem" not in driver + template + zh_source
    assert icon.is_file()
    assert "#707070" not in icon.read_text(encoding="utf-8")


def test_all_webui_javascript_has_valid_syntax() -> None:
    scripts = sorted(STATIC_ROOT.rglob("*.js"))
    assert scripts
    for script in scripts:
        subprocess.run(
            ["node", "--check", str(script)],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )


def test_database_ui_registry_is_explicit_and_rejects_unknown_types() -> None:
    module_url = (
        STATIC_ROOT / "modules" / "database-ui-registry.js"
    ).as_uri()
    script = f"""
      const {{ createDatabaseUiRegistry, databasePageRoute, databaseTypePageRoute, parsePageRoute }} =
        await import({module_url!r});
      const registry = createDatabaseUiRegistry();
      await registry.preload();
      if (!registry.has("livingmemory_v8") || !registry.has("text_media_v1")) {{
        throw new Error("known database UI drivers were not registered");
      }}
      if (registry.has("unknown_type") || await registry.load("unknown_type") !== null) {{
        throw new Error("unknown database type fell back to a known UI driver");
      }}
      const living = {{
        database_type: "livingmemory_v8",
        capabilities: ["backup", "graph", "memory_records", "recall"],
      }};
      const textMedia = {{
        database_type: "text_media_v1",
        capabilities: ["content_management", "image_assets", "search", "tmkb_export"],
      }};
      if (registry.resolvePage(living).id !== "overview") throw new Error("bad LivingMemory default page");
      if (registry.resolvePage(textMedia).id !== "content") throw new Error("bad text-media default page");
      const route = databasePageRoute("text_media_v1", "media");
      const parsed = parsePageRoute(route);
      if (parsed.databaseType !== "text_media_v1" || parsed.pageId !== "media") {{
        throw new Error("typed page route did not round-trip");
      }}
      const typeRoute = databaseTypePageRoute("text_media_v1", "settings");
      const parsedType = parsePageRoute(typeRoute);
      if (parsedType.scope !== "database_type" || parsedType.databaseType !== "text_media_v1") {{
        throw new Error("database-type page route did not round-trip");
      }}
      if (registry.resolveTypePage("text_media_v1", "settings")?.id !== "settings") {{
        throw new Error("text-media type settings page was not registered");
      }}
    """
    subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_livingmemory_sidebar_icons_preserve_the_pre_refactor_baseline() -> None:
    icons = (
        STATIC_ROOT / "database-types" / "livingmemory-v8-icons.js"
    ).read_text(encoding="utf-8")
    driver = (
        STATIC_ROOT / "database-types" / "livingmemory-v8.js"
    ).read_text(encoding="utf-8")
    app = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert 'viewBox=\\"0 0 1024 1024\\"' in icons
    assert "M128 469.33h341.33V128H128v341.33z" in icons
    assert "M490.212766 381.276596" in icons
    assert "M505.398652 376.952234" in icons
    assert 'recall: "<span>⌕</span>"' in icons
    assert "iconMarkup: LIVINGMEMORY_NAV_ICONS.overview" in driver
    assert "iconMarkup: LIVINGMEMORY_NAV_ICONS.graph" in driver
    assert "iconMarkup: LIVINGMEMORY_NAV_ICONS.memories" in driver
    assert "iconMarkup: LIVINGMEMORY_NAV_ICONS.recall" in driver
    assert "page.iconMarkup ||" in app


def test_api_client_coalesces_identical_inflight_gets() -> None:
    module_url = (STATIC_ROOT / "modules" / "api.js").as_uri()
    script = f"""
      globalThis.FormData ||= class FormData {{}};
      let calls = 0;
      let release;
      const gate = new Promise((resolve) => {{ release = resolve; }});
      globalThis.fetch = async () => {{
        calls += 1;
        await gate;
        return {{
          ok: true,
          status: 200,
          statusText: "OK",
          json: async () => ({{ value: calls }}),
        }};
      }};
      const {{ createApiClient }} = await import({module_url!r});
      const api = createApiClient();
      const first = api("/same");
      const second = api("/same");
      await Promise.resolve();
      if (calls !== 1) throw new Error(`expected one fetch, got ${{calls}}`);
      release();
      await Promise.all([first, second]);
      await api("/same");
      if (calls !== 2) throw new Error(`completed GET was not released: ${{calls}}`);
    """
    subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_webui_cancels_stale_pages_and_uses_adaptive_incremental_polling() -> None:
    app_source = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    api_source = (STATIC_ROOT / "modules" / "api.js").read_text(
        encoding="utf-8"
    )
    tasks_source = (STATIC_ROOT / "modules" / "tasks-logs.js").read_text(
        encoding="utf-8"
    )

    assert "const inFlightGets = new Map();" in api_source
    assert "function beginPageRequestGeneration()" in app_source
    assert "generation !== state.pageRequests.generation" in app_source
    assert 'error.name = "AbortError";' in app_source
    assert 'document.addEventListener("visibilitychange"' in tasks_source
    assert "hasActiveTasks ? 1200 : 5000" in tasks_source
    assert 'insertAdjacentHTML("beforeend"' in tasks_source
    assert "await onTaskFinished(job);" in tasks_source
    assert 'state.page === "graph"' not in tasks_source
    assert 'state.page === "memory"' not in tasks_source


def test_settings_module_has_one_settings_loader() -> None:
    source = (STATIC_ROOT / "modules" / "settings.js").read_text(
        encoding="utf-8"
    )

    assert source.count("async function loadSettings()") == 1
    assert "restartReturnPage" in source
    assert "RESTART_RETURN_PAGE" not in source


def test_webui_uses_database_type_instead_of_per_library_version() -> None:
    libraries_source = (STATIC_ROOT / "modules" / "libraries.js").read_text(
        encoding="utf-8"
    )
    system_source = (STATIC_ROOT / "modules" / "system.js").read_text(
        encoding="utf-8"
    )
    locale_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((STATIC_ROOT / "locales").glob("*.js"))
    )

    obsolete_names = (
        "livingmemory_database_version",
        "livingMemoryDbVersion",
        "dbVersionLabel",
        "dbVersionTarget",
        "dbVersionUnknown",
        "dbVersionMismatch",
    )
    combined = "\n".join((libraries_source, system_source, locale_sources))
    assert all(name not in combined for name in obsolete_names)
    assert "memoryLibraryType" in libraries_source


def test_visual_intent_csv_transfer_ui_uses_draft_and_requires_selection() -> None:
    template = (
        STATIC_ROOT / "database-types" / "text-media-v1.html"
    ).read_text(encoding="utf-8")
    module = (
        STATIC_ROOT / "modules" / "text-media-v1.js"
    ).read_text(encoding="utf-8")
    styles = (STATIC_ROOT / "styles.css").read_text(encoding="utf-8")

    assert 'id="text-media-visual-policy-import"' in template
    assert 'id="text-media-visual-policy-export"' in template
    assert 'accept=".csv,text/csv"' in template
    assert template.count("data-policy-export-category=") == 4
    assert 'id="text-media-visual-policy-import-modal"' in template
    assert 'id="text-media-visual-policy-export-modal"' in template
    assert "normalizedVisualIntentPolicyDraft()" in module
    assert 'excludeAcceptAllOption: true' in module
    assert 'accept: { "text/csv": [".csv"] }' in module
    assert 'importConfirm.disabled = policyModalCategories(' in module
    assert 'exportConfirm.disabled = policyModalCategories(' in module
    assert "visualIntentPolicyDraft[key] = [" in module
    assert ".text-media-policy-actions" in styles
    assert ".text-media-policy-category-picker" in styles


def test_text_media_v1_webui_uses_type_specific_protocol_and_image_precompression() -> None:
    app_source = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    libraries_source = (STATIC_ROOT / "modules" / "libraries.js").read_text(
        encoding="utf-8"
    )
    source = (STATIC_ROOT / "modules" / "text-media-v1.js").read_text(
        encoding="utf-8"
    )
    styles = (STATIC_ROOT / "styles.css").read_text(encoding="utf-8")
    html = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            STATIC_ROOT / "index.html",
            STATIC_ROOT / "database-types" / "text-media-v1.html",
        )
    )

    assert 'collection: "/knowledge-libraries/text_media_v1"' in app_source
    assert 'const TYPE_ID = "text_media_v1";' in source
    assert 'createImageBitmap(file)' in source
    assert 'canvas.toBlob(' in source
    assert '"image/webp"' in source
    assert '1536 / Math.max(bitmap.width, bitmap.height)' in source
    assert '/knowledge-libraries/${TYPE_ID}' in source
    assert "/memory-libraries/livingmemory_v8" not in source
    assert "databaseUiRegistry.get(button.dataset.type)" in libraries_source
    assert "driver.actions.openCreate()" in libraries_source
    assert "databasePageRoute(ref.databaseType, \"content\")" in libraries_source
    assert 'id="database-page-host"' in html
    assert 'classList.remove("modal", "text-media-workspace-modal")' in source
    assert 'data-i18n="tmkbAllPlaintextWarning"' in html
    assert 'id="text-media-settings-form"' not in html
    assert 'id="text-media-single-export-library"' in html
    assert 'id="text-media-batch-library-list"' in html
    assert 'id="text-media-batch-import-preview"' in html
    assert 'id="text-media-ingest-modal"' in html
    assert 'id="text-media-ingest-documents"' in html
    assert 'id="text-media-ingest-images"' in html
    assert 'data-i18n="chunkSize">最大分块大小' in html
    assert 'data-i18n="chunkOverlap">最大语义重叠' in html
    assert 'accept="image/png,image/jpeg,image/webp" multiple' in html
    assert 'id="text-media-search-mode-media-only"' in html
    assert 'id="text-media-search-mode-text-only"' in html
    assert 'retrieval_mode: searchRetrievalMode' in source
    assert 'unbound_media_candidate_limit' in source
    assert 'id="text-media-entry-form"' not in html
    assert 'id="text-media-image-form"' not in html
    assert 'id="text-media-relation-form"' not in html
    assert 'id="text-media-document-list"' in html
    assert 'id="text-media-chunk-list"' in html
    assert 'id="text-media-document-mode"' in html
    assert 'id="text-media-chunk-mode"' in html
    assert 'id="text-media-document-batch-delete"' in html
    assert 'id="text-media-detail-panel"' in html
    assert 'id="text-media-asset-grid"' in html
    assert 'id="text-media-ingest-semantic-enabled"' in html
    assert 'id="text-media-ingest-semantic-section"' in html
    assert 'accept=".txt,.md,.markdown,.pdf,.docx,' in html
    assert 'data-text-media-retrieval-mode="standard"' in html
    assert 'data-text-media-retrieval-mode="text_only"' in html
    assert 'data-text-media-retrieval-mode="media_only"' in html
    assert 'id="text-media-search-chunks-section"' in html
    assert 'id="text-media-edit-unbound-candidate-limit"' in html
    assert 'id="text-media-edit-unbound-competition-floor"' in html
    assert 'id="text-media-edit-unbound-reliability-target"' in html
    assert 'id="text-media-edit-unbound-specificity-exponent"' in html
    assert 'id="text-media-edit-unbound-advantage-target"' in html
    assert 'id="text-media-edit-uniform-strength" type="number" min="0" max="1" step="0.01"' in html
    assert 'id="text-media-edit-id" required pattern="[a-zA-Z0-9_\\-]+"' in html
    assert (
        'id="text-media-create-id" required pattern="[a-zA-Z0-9_\\-]+"'
        in (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    )
    for field_id in (
        "text-media-edit-pivot-negative-floor",
        "text-media-edit-format-mismatch-factor",
        "text-media-edit-content-mismatch-factor",
    ):
        assert (
            f'id="{field_id}" type="number" min="0.001" max="1" step="0.001"'
            in html
        )
    retrieval_section = html.split('id="text-media-retrieval-settings"', 1)[1].split(
        "</section>", 1
    )[0]
    assert 'data-retrieval-layout="compact"' in retrieval_section
    assert 'data-retrieval-layout="detailed"' in retrieval_section
    assert 'id="text-media-retrieval-reset"' in retrieval_section
    assert 'id="text-media-retrieval-primary"' in retrieval_section
    assert 'id="text-media-retrieval-toggle"' in retrieval_section
    assert 'class="switch text-media-intent-gate-switch"><input id="text-media-edit-intent-gate" type="checkbox" role="switch"><i></i></span>' in retrieval_section
    assert retrieval_section.count('<input id="text-media-edit-') == 36
    assert retrieval_section.count("<small data-i18n=") == 36
    assert retrieval_section.index("retrievalCoreTextGroup") < retrieval_section.index(
        "retrievalMediaGroup"
    ) < retrieval_section.index("retrievalRerankGroup")
    assert 'id="text-media-content-refresh"' not in html
    assert 'id="text-media-media-refresh"' not in html
    assert 'id="text-media-ingest-progress"' in html
    assert 'id="text-media-search-confidence-threshold"' in html
    assert 'id="text-media-search-score-threshold"' in html
    assert 'id="text-media-search-confidence-threshold" type="number" min="0" max="1" step="0.01"' in html
    assert 'id="text-media-search-score-threshold" type="number" min="0" max="1" step="0.01"' in html
    assert 'class="text-media-search-parameter-grid"' in html
    assert 'class="text-media-search-action-row"' in html
    assert 'class="text-media-search-meta-row"' in html
    assert 'class="switch"><input id="text-media-search-rerank" type="checkbox" role="switch"><i></i></span>' in html
    assert 'id="text-media-search-media"' in html
    assert 'id="text-media-search-decisions"' in html
    assert 'id="text-media-search-decisions-toggle"' in html
    assert 'aria-controls="text-media-search-decisions"' in html
    assert 'id="text-media-search-view-embedding"' in html
    assert 'id="text-media-search-view-rerank"' in html
    assert 'data-text-media-search-view="embedding"' in html
    assert 'data-text-media-search-view="rerank"' in html
    assert 'data-i18n="recallOnlyEmbedding"' in html
    assert 'data-i18n="recallWithRerank"' in html
    assert 'id="text-media-document-form"' not in html
    assert 'body.append("documents[]", file, file.name)' in source
    assert 'body.append("images[]", file, file.name)' in source
    assert '$("text-media-entry-form")' not in source
    assert '$("text-media-image-form")' not in source
    assert 'path(currentKnowledgeBase.id, "/ingest-batches")' in source
    assert "media_semantic_calibration_enabled" in source
    assert 'retrieval_mode: searchRetrievalMode' in source
    assert 'applySearchRetrievalMode("standard")' in source
    assert 'item.bindingMode === "none"' in source
    assert 'unbound_media_candidate_limit' in source
    assert 'unbound_media_competition_floor' in source
    assert 'unbound_media_reliability_target' in source
    assert '"hidden", !hasDocuments || !ingestImages.length' in source
    assert "uniform_media_strength" in source
    assert "media_description" in source
    assert "asset.original_name || asset.asset_id.slice" in source
    assert "/semantic-calibration" in source
    assert "semantic_strength" in source
    chunk_detail_source = source.split("async function openChunkDetail", 1)[1].split(
        "function relationTargetOptions", 1
    )[0]
    assert (
        'mediaPreviewButton(asset, "text-media-detail-thumbnail")'
        in chunk_detail_source
    )
    assert (
        '$("text-media-detail-body").querySelectorAll(".text-media-detail-thumbnail")'
        in chunk_detail_source
    )
    assert 't("associatedImages")' in chunk_detail_source
    assert "relationPolicyLabel(asset.output_policy)" in chunk_detail_source
    assert "collapsibleDetailContent(item.content)" in source
    assert "collapsibleDetailContent(item.text)" in source
    assert 'class="library-expand-toggle text-media-detail-content-toggle hidden"' in source
    assert "Math.max(280, Math.round(window.innerHeight * 0.5))" in source
    assert "primary.scrollHeight > collapsedHeight + 24" in source
    assert 't(expanded ? "collapseLibraryCard" : "expandLibraryCard")' in source
    assert ".text-media-detail-content-collapsed .text-media-detail-content-primary" in styles
    assert ".text-media-detail-content-expanded .text-media-detail-content-primary" in styles
    assert ".text-media-detail-content-toggle{bottom:4px}" in styles
    assert "MEDIA_DECISION_COLLAPSED_COUNT = 4" in source
    assert "Math.round(window.innerHeight / 3)" in source
    assert 'closest(".text-media-result-toggle")' in source
    assert 'function retrievalScoreTierClass(value)' in source
    assert 'if (score >= 0.7) return "score-high"' in source
    assert 'if (score >= 0.4) return "score-mid"' in source
    assert '<strong>ID ${Number(item.chunk_id)}</strong>' in source
    assert 'class="score ${retrievalScoreTierClass(item.score)}"' in source
    assert ".text-media-result header .score.score-high{color:#38a36a}" in styles
    assert ".text-media-result header .score.score-mid{color:#c48a2a}" in styles
    assert ".text-media-result header .score.score-low{color:#d04040}" in styles
    assert '[data-theme="dark"] .text-media-result header .score.score-high{background:var(--panel);border-color:#38a36a}' in styles
    assert '[data-theme="dark"] .text-media-result header .score.score-mid{background:var(--panel);border-color:#c48a2a}' in styles
    assert '[data-theme="dark"] .text-media-result header .score.score-low{background:var(--panel);border-color:#d04040}' in styles
    assert "mediaDecisionsExpanded = !mediaDecisionsExpanded" in source
    assert ".text-media-result.text-media-result-collapsed .text-media-result-primary" in styles
    assert ".text-media-result.text-media-result-expanded .text-media-result-primary" in styles
    assert ".text-media-search-decisions-panel.text-media-search-decisions-collapsed" in styles
    assert ".text-media-search-decisions-panel.text-media-search-decisions-expanded" in styles
    assert ".text-media-search-parameter-grid{display:flex;flex-wrap:wrap;align-items:center;gap:8px 18px}" in styles
    assert ".text-media-search-form .text-media-search-parameter{display:flex!important;align-items:center;gap:8px;flex:0 0 auto;width:auto" in styles
    assert ".text-media-search-form .text-media-search-parameter>input{flex:0 0 88px;width:88px;min-width:0;height:40px" in styles
    assert ".text-media-search-action-row{display:flex;align-items:center;align-self:flex-start;gap:8px;flex:0 0 auto;margin-left:auto;min-width:0}" in styles
    assert ".text-media-search-form .text-media-search-rerank-toggle{display:inline-flex!important;align-items:center;gap:6px;min-height:40px;margin:0;padding:0;border:0;background:transparent}" in styles
    assert ".text-media-search-meta-row{display:grid;gap:4px;margin-top:-4px}" in styles
    assert ".text-media-search-rerank-toggle .switch input{display:block;position:absolute" in styles
    assert "fillRetrievalSettings(DEFAULT_RETRIEVAL_SETTINGS)" in source
    assert 'title: t("confirmRetrievalDefaultsTitle")' in source
    assert 'message: t("confirmRetrievalDefaultsMessage")' in source
    assert "Math.round(window.innerHeight * 0.75)" in source
    assert 'button.setAttribute("aria-pressed", active ? "true" : "false")' in source
    assert '.text-media-retrieval-settings-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}' in styles
    assert '.text-media-retrieval-settings[data-layout="compact"] .text-media-retrieval-settings-grid label>small{display:none}' in styles
    assert '.text-media-retrieval-settings[data-layout="detailed"] .text-media-retrieval-settings-grid{grid-template-columns:minmax(0,1fr);gap:10px}' in styles
    assert '.text-media-retrieval-collapsed .text-media-retrieval-primary{max-height:var(--text-media-retrieval-collapsed-height,75vh)}' in styles
    assert '.text-media-retrieval-toggle .switch input{display:block;position:absolute;width:1px;height:1px;opacity:0}' in styles
    assert ".text-media-visual-intent-policy{margin-top:18px}" in styles
    assert "media_output_confidence_threshold" in source
    assert "media_relevance_pivot" in source
    assert 'number > 0 && number < 0.001 ? "<0.001"' in source
    assert "result.media_outputs" in source
    assert "result.media_decisions" in source
    assert "result.baseline_items || result.items || []" in source
    assert 'searchResultView === "rerank"' in source
    assert 'document.querySelectorAll("[data-text-media-search-view]")' in source
    assert "setSearchResultView(button.dataset.textMediaSearchView" in source
    assert "decision.confidence_threshold_met" in source
    assert 't(confidenceMet ? "thresholdMet" : "thresholdNotMet")' in source
    assert "decision.evidence_threshold_adjustment" in source
    assert "decision.qualifying_chunk_count" in source
    assert "decision.weakening_chunk_count" in source
    assert "media_match_ratio_threshold" not in source
    assert "/transfer-batches/exports" in source
    assert "/transfer-batches/imports/inspect" in source
    assert 'capabilities.has("copy")' in libraries_source
    assert 'capabilities.has("backup")' in libraries_source
    text_media_card = libraries_source.split('if (uiDriver.cardLayout === "knowledge")', 1)[1].split("const adapterButton", 1)[0]
    assert 'class="ghost copy-library"' in text_media_card
    assert 'class="ghost backup-library"' in text_media_card
    assert 't("modelDimension")' in text_media_card
    assert 't("generationLabel")' in text_media_card
    assert 't("indexStatus")' in text_media_card
    assert 't("connectedAdapters")' in text_media_card
    assert "knowledgeLibraryIndexState(library)" in text_media_card
    assert "connectedAdaptersMarkup(library, adapterConnections)" in text_media_card
    assert 't("status")' not in text_media_card
    assert 't("defaultPersona")' not in text_media_card
    assert 't("conversationBuffer")' not in text_media_card
    assert 'id="text-media-workspace-modal"' not in (
        STATIC_ROOT / "index.html"
    ).read_text(encoding="utf-8")


def test_text_media_batch_upload_uses_accessible_nested_drag_picker() -> None:
    source = (STATIC_ROOT / "modules" / "text-media-v1.js").read_text(
        encoding="utf-8"
    )
    styles = (STATIC_ROOT / "styles.css").read_text(encoding="utf-8")
    html = (
        STATIC_ROOT / "database-types" / "text-media-v1.html"
    ).read_text(encoding="utf-8")
    upload_icon = (STATIC_ROOT / "icons" / "upload.svg").read_text(encoding="utf-8")
    zh_source = (STATIC_ROOT / "locales" / "zh.js").read_text(encoding="utf-8")

    assert 'id="text-media-upload-picker-modal"' in html
    assert 'document.body.append(uploadPickerModal)' in source
    assert 'id="text-media-upload-drop-zone"' in html
    assert 'role="button" tabindex="0"' in html
    assert 'id="text-media-upload-browse"' in html
    assert html.count('data-ingest-picker="') == 2
    assert html.count('data-ingest-parameter="') == 10
    assert html.index('class="text-media-ingest-parameter-section"') < html.index(
        'data-i18n="selectedDocuments"'
    )
    picker_markup = html.split('id="text-media-upload-picker-modal"', 1)[1].split(
        'id="text-media-edit-modal"', 1
    )[0]
    assert 'class="text-media-upload-parameter-panel"' in picker_markup
    assert picker_markup.count('data-ingest-parameter="') == 5
    assert "function openIngestPicker(kind)" in source
    assert "function acceptIngestFiles(kind, files)" in source
    assert "function mergeIngestFiles(current, incoming)" in source
    assert "function syncIngestParameter(source)" in source
    assert "function syncIngestParameterMirrors()" in source
    assert '"dragenter", "dragover"' in source
    assert '"dragleave", "drop"' in source
    assert '["Enter", " "]' in source
    assert 'mask:url("/static/icons/upload.svg")' in styles
    assert ".text-media-upload-picker-overlay{z-index:72" in styles
    assert 'id="upload-icon"' in upload_icon
    assert "#8a8a8a" not in upload_icon
    assert '"chooseFromFileManager": "从文件管理器选择"' in zh_source
    assert '"dropBatchDocuments": "把文档拖到这里"' in zh_source
    assert '"dropBatchImages": "把图片拖到这里"' in zh_source
    assert '"batchParametersSyncedHint": "这里的调整会同步到本次批次上传。"' in zh_source


def test_recall_and_retrieval_refresh_reset_without_running_requests() -> None:
    app_source = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    recall_source = (STATIC_ROOT / "modules" / "recall.js").read_text(
        encoding="utf-8"
    )
    text_media_source = (
        STATIC_ROOT / "modules" / "text-media-v1.js"
    ).read_text(encoding="utf-8")
    text_media_html = (
        STATIC_ROOT / "database-types" / "text-media-v1.html"
    ).read_text(encoding="utf-8")

    recall_refresh = app_source.split(
        '$("recall-refresh")?.addEventListener("click", () => {', 1
    )[1].split("});", 1)[0]
    assert "resetRecallTest();" in recall_refresh
    assert '$("run-recall").click()' not in recall_refresh
    assert "function resetRecallTest()" in recall_source
    assert '$("recall-query").value = "";' not in recall_source
    assert '$("recall-persona").value = "";' in recall_source
    assert '$("recall-session").value = "";' in recall_source
    assert "recallGeneration += 1;" in recall_source
    assert '$("recall-results").innerHTML = "";' in recall_source

    search_refresh = text_media_source.split(
        'if (targetPage === "search") {', 1
    )[1].split('} else if (targetPage === "settings")', 1)[0]
    assert "resetSearchTest();" in search_refresh
    assert "runSearch(" not in search_refresh
    assert 'const query = $("text-media-search-query").value;' in text_media_source
    assert '$("text-media-search-form").reset();' in text_media_source
    assert '$("text-media-search-query").value = query;' in text_media_source
    assert "searchGeneration += 1;" in text_media_source
    assert 'id="text-media-search-top-k" type="number" min="1" max="50" value="10"' in text_media_html
    assert 'id="text-media-search-confidence-threshold" type="number" min="0" max="1" step="0.01" value="0.6"' in text_media_html
    assert 'id="text-media-search-score-threshold" type="number" min="0" max="1" step="0.01" value="0.35"' in text_media_html
    assert "media_relevance_pivot_fallback: 0.35" in text_media_source
    assert "media_pivot_positive_blend: 0.7" in text_media_source
    assert "media_pivot_negative_weight: 0.35" in text_media_source
    assert "media_format_mismatch_factor: 0.1" in text_media_source
    assert "media_content_mismatch_factor: 0.10" in text_media_source
    assert "media_threshold_evidence_limit: 5" in text_media_source
    assert 'id="text-media-search-max-outputs" type="number" min="0" max="20" value="5"' in text_media_html


def test_ui_error_logging_distinguishes_request_failures() -> None:
    api_source = (STATIC_ROOT / "modules" / "api.js").read_text(
        encoding="utf-8"
    )
    app_source = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert "suppressOperationalError = false" in api_source
    assert "if (!suppressOperationalError) onOperationalError(error);" in api_source
    assert "error.status >= 500 && !suppressOperationalError" in api_source
    assert 'if (cause?.name === "AbortError") throw cause;' in api_source
    assert 'new ApiError(cause?.message || "Network request failed", 0)' in api_source
    tasks_logs_source = (STATIC_ROOT / "modules" / "tasks-logs.js").read_text(
        encoding="utf-8"
    )
    assert tasks_logs_source.count("suppressOperationalError: true") == 2
    assert 'error ? "WARN" : "INFO"' in app_source


def test_finished_task_history_uses_collapsible_panel() -> None:
    app_source = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    styles = (STATIC_ROOT / "styles.css").read_text(encoding="utf-8")
    tasks_logs_source = (STATIC_ROOT / "modules" / "tasks-logs.js").read_text(
        encoding="utf-8"
    )

    assert 'finishedExpanded: false' in app_source
    assert 'id="task-history-panel"' in html
    assert 'id="task-history-toggle"' in html
    assert 'id="tasks-finished-clear"' in html
    assert 'class="library-expand-toggle task-history-toggle hidden"' in html
    assert ".task-history-panel.task-history-panel-collapsed" in styles
    assert ".task-tabs #tasks-finished-clear{margin-left:auto}" in styles
    assert "function applyTaskHistoryCollapseState()" in tasks_logs_source
    assert 'api("/jobs/finished/clear", { method: "POST" })' in tasks_logs_source
    assert 'renderFinishedTaskClearButton()' in tasks_logs_source
    assert 'state.tasks.finishedExpanded = !state.tasks.finishedExpanded;' in tasks_logs_source


def test_long_collapsibles_offer_expanded_only_top_collapse_controls() -> None:
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    livingmemory_html = (
        STATIC_ROOT / "database-types" / "livingmemory-v8.html"
    ).read_text(encoding="utf-8")
    text_media_html = (
        STATIC_ROOT / "database-types" / "text-media-v1.html"
    ).read_text(encoding="utf-8")
    styles = (STATIC_ROOT / "styles.css").read_text(encoding="utf-8")
    libraries_source = (STATIC_ROOT / "modules" / "libraries.js").read_text(
        encoding="utf-8"
    )
    system_source = (STATIC_ROOT / "modules" / "system.js").read_text(
        encoding="utf-8"
    )
    tasks_source = (STATIC_ROOT / "modules" / "tasks-logs.js").read_text(
        encoding="utf-8"
    )
    text_media_source = (
        STATIC_ROOT / "modules" / "text-media-v1.js"
    ).read_text(encoding="utf-8")

    assert ".collapse-top-toggle{display:flex" in styles
    assert ".collapse-top-toggle.hidden{display:none}" in styles
    assert 'id="task-history-collapse-top" class="collapse-top-toggle hidden"' in html
    assert 'id="system-provider-collapse-top" class="collapse-top-toggle hidden"' in livingmemory_html
    assert 'id="system-index-collapse-top" class="collapse-top-toggle hidden"' in livingmemory_html
    assert 'id="text-media-retrieval-collapse-top"' in text_media_html
    assert 'id="text-media-search-decisions-collapse-top"' in text_media_html
    assert "library-collapse-top" not in libraries_source
    assert 'topToggle.classList.toggle("hidden", !expanded);' in system_source
    assert 'state.systemProviderExpanded = false;' in system_source
    assert 'state.systemIndexExpanded = false;' in system_source
    assert 'topToggle.className = "collapse-top-toggle task-detail-collapse-top hidden";' in tasks_source
    assert 'item.topToggle.classList.toggle("hidden", !overflowing || !item.expanded);' in tasks_source
    assert 'state.tasks.finishedExpanded = false;' in tasks_source
    assert 'class="collapse-top-toggle text-media-detail-content-collapse-top hidden"' in text_media_source
    assert 'class="collapse-top-toggle text-media-result-collapse-top hidden"' in text_media_source
    assert 'topToggle.classList.toggle("hidden", !collapsible || !retrievalSettingsExpanded);' in text_media_source
    assert 'topToggle.classList.toggle("hidden", !mediaDecisionsExpanded);' in text_media_source
    assert 'retrievalSettingsExpanded = false;' in text_media_source
    assert 'mediaDecisionsExpanded = false;' in text_media_source


def test_finished_tasks_expose_responsive_detail_panel_and_state_comparison() -> None:
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    styles = (STATIC_ROOT / "styles.css").read_text(encoding="utf-8")
    tasks_logs_source = (STATIC_ROOT / "modules" / "tasks-logs.js").read_text(
        encoding="utf-8"
    )

    assert 'id="task-detail-overlay"' in html
    assert 'id="task-detail-panel"' in html
    assert 'id="task-detail-body"' in html
    assert 'aria-live="polite"' in html
    assert 'const detailClass = opensDetail ? " task-item-detail" : "";' in tasks_logs_source
    assert 'data-job-detail="${escapeHtml(String(job.id || ""))}" tabindex="0" role="button"' in tasks_logs_source
    assert '<button class="ghost" type="button" data-job-detail=' not in tasks_logs_source
    assert '$("task-list")?.addEventListener("keydown"' in tasks_logs_source
    assert 'event.key !== "Enter" && event.key !== " "' in tasks_logs_source
    assert 'api(`/jobs/${encodeURIComponent(jobId)}/details`' in tasks_logs_source
    assert "function renderTaskStateComparison(comparison)" in tasks_logs_source
    assert "function renderTaskData(value" in tasks_logs_source
    assert "function renderTaskSnapshot(snapshot, phase)" in tasks_logs_source
    assert "job.database_state_comparison" in tasks_logs_source
    assert 'data-label="${escapeHtml(t("taskStateAfter"))}"' in tasks_logs_source
    assert "taskJsonBlock" not in tasks_logs_source
    assert '<pre class="task-detail-json">' not in tasks_logs_source
    assert 'event.key === "Escape"' in tasks_logs_source
    assert ".task-detail-panel{position:fixed" in styles
    assert ".task-detail-snapshot{display:grid" in styles
    assert ".task-detail-data-list{display:grid" in styles
    assert ".task-detail-value-badge.positive" in styles
    assert ".task-item-detail{cursor:pointer" in styles
    assert ".task-item-detail:hover" in styles
    assert ".task-item-detail:focus-visible" in styles
    assert "function setupTaskDetailCollapsibles()" in tasks_logs_source
    assert "Math.floor(window.innerHeight * 0.5)" in tasks_logs_source
    assert 'class="task-detail-section task-detail-execution-section"' in tasks_logs_source
    assert '!section.classList.contains("task-detail-execution-section")' in tasks_logs_source
    assert 'body.querySelectorAll(".task-detail-snapshot")' in tasks_logs_source
    assert 'body.querySelectorAll(".task-detail-table-wrap")' in tasks_logs_source
    assert "setupTaskDetailCollapsibles();" in tasks_logs_source
    assert ".task-detail-collapsed{max-height:var(--task-detail-collapsed-height,50vh)" in styles
    assert ".task-detail-collapse-toggle{bottom:9px" in styles
    assert "@media(max-width:760px){.task-detail-panel{width:100vw" in styles
    assert ".task-detail-state-grid{grid-template-columns:minmax(0,1fr)}" in styles
    assert ".task-detail-diff-table{min-width:0}" in styles


def test_task_controller_receives_both_database_index_conflict_callbacks() -> None:
    app_source = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    libraries_source = (STATIC_ROOT / "modules" / "libraries.js").read_text(
        encoding="utf-8"
    )
    tasks_logs_source = (STATIC_ROOT / "modules" / "tasks-logs.js").read_text(
        encoding="utf-8"
    )

    controller_signature = tasks_logs_source.split("{", 1)[1].split("}", 1)[0]
    assert "markDatabaseIndexConflict" in controller_signature
    assert "clearDatabaseIndexConflict" in controller_signature
    assert (
        "markDatabaseIndexConflict: (...args) => markDatabaseIndexConflict(...args)"
        in app_source
    )
    assert (
        "clearDatabaseIndexConflict: (...args) => clearDatabaseIndexConflict(...args)"
        in app_source
    )
    library_controller_return = libraries_source.rsplit("return {", 1)[1].split(
        "};", 1
    )[0]
    assert "markDatabaseIndexConflict" in library_controller_return
    assert "clearDatabaseIndexConflict" in library_controller_return
    assert (
        "confirmSensitiveProviderEdit, markDatabaseIndexConflict, clearDatabaseIndexConflict"
        in app_source
    )


def test_library_index_health_uses_livingmemory_253_graph_vector_granularity() -> None:
    libraries_source = (STATIC_ROOT / "modules" / "libraries.js").read_text(
        encoding="utf-8"
    )

    assert 'String(manifest.graph_vector_granularity || "entry")' in libraries_source
    assert 'graphGranularity === "memory"' in libraries_source
    assert (
        "manifest.graph_source_memory_count ?? manifest.graph_vector_count ?? 0"
        in libraries_source
    )
    assert "stats.active_memories ?? stats.total_memories ?? 0" in libraries_source
    assert (
        "Number(indexes.graph_vectors || 0) === Number(stats.graph_entries || 0)"
        not in libraries_source
    )


def test_log_window_hides_debug_by_default() -> None:
    app_source = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")

    assert 'activeLevels: new Set(["INFO", "WARN", "ERROR"])' in app_source
    assert (
        '<button class="log-filter" type="button" data-log-level="DEBUG">Debug</button>'
        in html
    )
    assert (
        '<button class="log-filter active" type="button" data-log-level="DEBUG">'
        not in html
    )


def test_file_password_failure_does_not_invalidate_webui_login() -> None:
    api_source = (STATIC_ROOT / "modules" / "api.js").read_text(
        encoding="utf-8"
    )
    files_source = (STATIC_ROOT / "modules" / "files.js").read_text(
        encoding="utf-8"
    )

    assert "suppressUnauthorizedHandler = false" in api_source
    assert "if (!suppressUnauthorizedHandler)" in api_source
    assert "suppressUnauthorizedHandler: true" in files_source


def test_file_manager_module_preserves_rocketcat_file_type_icons() -> None:
    source = (STATIC_ROOT / "modules" / "files.js").read_text(
        encoding="utf-8"
    )

    assert 'suffix === ".txt"' in source
    assert '[".json", ".py", ".md"]' in source
    assert 'suffix === ".pdf"' in source
    assert '[".doc", ".docx"]' in source
    assert "file-icon--folder" in source
    assert "file-icon--image" in source
    for action in ("rename", "move", "copy", "download", "trash"):
        assert f"{action}:" in source


def test_file_manager_uses_personalityrag_light_and_dark_theme_tokens() -> None:
    styles = (STATIC_ROOT / "styles.css").read_text(encoding="utf-8")
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")

    assert 'data-page="files"' in html
    assert 'id="page-files"' in html
    assert 'id="file-table-body"' in html
    assert "--file-folder:var(--accent)" in styles
    assert "--file-code:var(--accent2)" in styles
    assert '[data-theme="dark"]{--file-pdf:' in styles
    assert ".file-panel" in styles
    assert ".file-image-viewer" in styles
    scrollbar_theme = styles.split(
        ":where(html,.file-breadcrumb,.file-table-shell,.file-preview-content,.file-move-tree",
        1,
    )[1].split("html{--scrollbar-surface:var(--bg)}", 1)[0]
    assert "scrollbar-color:var(--scrollbar-thumb) var(--scrollbar-surface,var(--panel-soft))" in styles
    assert "::-webkit-scrollbar-thumb{background:var(--scrollbar-thumb)" in styles
    for scroll_container in (
        ".file-breadcrumb",
        ".file-table-shell",
        ".file-preview-content",
        ".file-move-tree",
        ".updates-release-list",
        ".text-media-tabs",
        ".text-media-workspace-body",
        ".text-media-batch-library-list",
        ".text-media-ingest-body",
        ".text-media-upload-picker-body",
        ".text-media-policy-term-list",
        ".task-detail-collapse-content-scroll",
    ):
        assert scroll_container in scrollbar_theme
    assert "html{--scrollbar-surface:var(--bg)}" in styles
    assert ".sidebar-scroll::-webkit-scrollbar{width:0;height:0;display:none}" in styles
    assert ".provider-kind-tabs::-webkit-scrollbar{display:none}" in styles


def test_light_theme_uses_astrbot_blue_and_preserves_dark_accent() -> None:
    styles = (STATIC_ROOT / "styles.css").read_text(encoding="utf-8")
    graph_source = (STATIC_ROOT / "modules" / "graph.js").read_text(
        encoding="utf-8"
    )
    renderer_source = (
        STATIC_ROOT / "modules" / "livingmemory-253-graph-2d.js"
    ).read_text(encoding="utf-8")

    assert "--accent:#3c96ca;--accent2:#2f86bd" in styles
    assert "--accent-rgb:60,150,202;--accent2-rgb:47,134,189" in styles
    assert "--accent-soft:#e8f3fa" in styles
    assert "--task-progress-fill-start:#90caf9" in styles
    assert "--task-progress-fill-mid:#8cc4e1" in styles
    assert "--accent:#ff5b96;--accent2:#a594ff" in styles
    assert "--accent-rgb:239,77,134;--accent2-rgb:113,98,212" in styles
    assert "rgba(239,77,134" not in styles
    assert "rgba(113,98,212" not in styles
    assert ".key-library-fab svg{width:17px;height:17px;fill:currentColor}" in styles
    assert "border:2px solid rgba(var(--accent-rgb),.32)!important" in styles
    assert "box-shadow:0 4px 16px rgba(var(--accent-rgb),.18)" in styles
    assert ".key-library-fab svg{width:17px;height:17px;fill:#f6a23a}" not in styles
    assert 'summary: "var(--graph-summary)"' in graph_source
    assert 'themeColor("--graph-summary", TYPE_COLORS.summary)' in renderer_source


def test_livingmemory_graph_uses_the_253_renderer_and_large_graph_contract() -> None:
    index = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    graph_html = (
        STATIC_ROOT / "database-types" / "livingmemory-v8.html"
    ).read_text(encoding="utf-8")
    controller = (STATIC_ROOT / "modules" / "graph.js").read_text(
        encoding="utf-8"
    )
    renderer = (
        STATIC_ROOT / "modules" / "livingmemory-253-graph-2d.js"
    ).read_text(encoding="utf-8")

    assert index.index("livingmemory-253-graph-2d.js") < index.index("/static/app.js")
    assert "LivingMemory 2.5.3 renderer" in renderer
    for contract in (
        "FORCE_ITERATIONS: 400",
        "FORCE_REPULSION: 1680",
        "FORCE_LINK_DISTANCE: 108",
        "LARGE_NODE_THRESHOLD: 1200",
        "LARGE_EDGE_THRESHOLD: 4500",
        "MASSIVE_NODE_THRESHOLD: 3500",
        "MASSIVE_EDGE_THRESHOLD: 12000",
        "AMBIENT_NODE_LIMIT: 700",
        "AMBIENT_EDGE_LIMIT: 1800",
        "Renderer.prototype.prepareGraph",
        "Renderer.prototype._rebuildNodeHitGrid",
        "Graph2D.prototype.getDiagnostics",
        'var hoverColor = themeColor("--accent", dark ? "#ff5b96" : "#3c96ca")',
        "var hoverEdges = this._nodeEdges[hoverNodeId] || []",
        "isHoverConnected: isHoverConnected",
        "hoverConnectedEdges:",
        "Interaction.prototype._clearHover",
    ):
        assert contract in renderer
    assert 'topic: "#5d35c7", person: "#59c2ff", fact: "#d5a20a"' in renderer
    assert "limit_memories: 24" in controller
    assert "limit_entries: 80" in controller
    assert "limit_nodes: 80" in controller
    assert "limit_edges: 120" in controller
    assert '$("graph-focus")?.addEventListener("click", focusMemory)' in controller
    assert 'id="graph-canvas-state"' in graph_html
    assert 'id="graph-focus"' in graph_html


def test_graph_overview_omits_an_empty_session_filter() -> None:
    source = (STATIC_ROOT / "modules" / "graph.js").read_text(
        encoding="utf-8"
    )

    assert 'new URLSearchParams({ full_graph: "true" })' in source
    assert (
        'if (overviewSessionId) overviewParams.set("session_id", overviewSessionId)'
        in source
    )
    assert '"/graph/overview?" + overviewParams.toString()' in source
    assert "session_id=\" + encodeURIComponent" not in source


def test_acceptance_runner_is_not_bound_to_live_library_paths() -> None:
    source = (REPO_ROOT / "tools" / "acceptance_webui.py").read_text(
        encoding="utf-8"
    )

    assert '"beileite"' not in source
    assert "--base-url" in source
    assert "--state-root" in source
    assert "--api-key" in source
    assert "Network.setExtraHTTPHeaders" in source
    assert "--report-dir" in source
    assert "tempfile.gettempdir()" in source
    assert 'result["provider_editor"]["type_cards"] >= 3' in source


def test_first_library_form_uses_neutral_defaults_and_explicit_provider_choice() -> None:
    index = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    libraries = (STATIC_ROOT / "modules" / "libraries.js").read_text(
        encoding="utf-8"
    )
    providers = (STATIC_ROOT / "modules" / "providers.js").read_text(
        encoding="utf-8"
    )

    assert 'id="library-name" required placeholder="Default"' in index
    assert 'firstLivingMemoryLibrary ? "Default" : ""' in libraries
    assert "requireExplicit: !library" in libraries
    assert "options.optional || options.requireExplicit" in providers
    assert "例如 贝雷特" not in index


def test_memory_persona_editor_is_separate_from_content_editor() -> None:
    source = (STATIC_ROOT / "modules" / "memories.js").read_text(
        encoding="utf-8"
    )

    persona_editor = source[
        source.index("async function saveMemoryPersonaEdit") :
        source.index("function renderMemoryEditView")
    ]
    content_editor = source[
        source.index("async function saveMemoryDetailEdit") :
        source.index(
            '$("memory-detail-close")?.addEventListener',
            source.index("async function saveMemoryDetailEdit"),
        )
    ]
    assert '`/memories/${detail.id}/persona`' in persona_editor
    assert "JSON.stringify({ persona_id: personaId })" in persona_editor
    assert "persona_id" not in content_editor
    assert "if (content !== detail.text) payload.content = content;" in content_editor


def test_livingmemory_v8_webui_has_no_user_selectable_memory_type() -> None:
    source = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    memories = (STATIC_ROOT / "modules" / "memories.js").read_text(
        encoding="utf-8"
    )
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")

    for obsolete in (
        "form-type",
        "memory-edit-type",
        "displayMemoryType",
        "memoryTypeTag",
        "type_asc",
    ):
        assert obsolete not in source
        assert obsolete not in memories
        assert obsolete not in html
    assert 'id="memory-type"' not in html
    assert '"memory-type"' not in source
    assert '"memory-type"' not in memories
    assert '"memory_type"' not in source
    assert 'memory_type:' not in source
    assert '"memory_type"' not in memories
    assert 'memory_type:' not in memories
    assert 'name="memory_type"' not in html


def test_settings_module_exposes_version_switch_workflow() -> None:
    source = (STATIC_ROOT / "modules" / "settings.js").read_text(encoding="utf-8")
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")

    assert 'api("/updates/switch"' in source
    assert 'api(`/updates/status' in source
    assert 'api(`/updates/releases' in source
    assert "data-update-tag" in source
    assert 'id="update-available-badge"' in html
    assert 'id="updates-modal-busy"' in html


def test_webui_async_action_guard_is_wired_to_high_risk_submits() -> None:
    app_source = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    guard_source = (STATIC_ROOT / "modules" / "ui-guard.js").read_text(
        encoding="utf-8"
    )
    libraries_source = (STATIC_ROOT / "modules" / "libraries.js").read_text(
        encoding="utf-8"
    )

    assert "createAsyncActionGuard" in guard_source
    assert "active.has(guardKey)" in guard_source
    assert 'setAttribute("aria-busy", "true")' in guard_source
    assert 'from "./modules/ui-guard.js"' in app_source
    assert "const asyncGuard = createAsyncActionGuard();" in app_source
    assert "asyncGuard" in app_source
    assert "database:${databaseRefKey(databaseType, databaseId)}:livingmemory-import" in libraries_source
    assert 'form: event.currentTarget' in libraries_source
    assert 'busyText: t("taskSubmitting")' in libraries_source


def test_webui_uses_typed_database_categories_and_compound_identity() -> None:
    app_source = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    libraries_source = (STATIC_ROOT / "modules" / "libraries.js").read_text(
        encoding="utf-8"
    )
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")

    assert 'api("/databases?stats_mode=summary")' in libraries_source
    assert 'api("/database-types")' in libraries_source
    assert "/libraries" not in libraries_source
    assert "databaseRefKey" in app_source
    assert "selectedDatabaseRefByCategory" in app_source
    assert 'databaseCategory: "memory"' in app_source
    assert "loadStoredDatabaseRef" not in app_source
    assert 'localStorage.getItem("prag_database_category")' not in app_source
    assert 'localStorage.getItem("prag_database_ref")' not in app_source
    assert 'localStorage.getItem("prag_library_id")' not in app_source
    assert 'data-database-type="${escapeHtml(databaseType)}"' in libraries_source
    assert 'id="database-type-modal"' in html
    assert 'data-database-category="memory"' in html
    assert 'data-database-category="knowledge"' in html
    assert "/static/icons/livingmemory-v8.svg" in (
        STATIC_ROOT / "styles.css"
    ).read_text(encoding="utf-8")


def test_webui_runtime_state_and_module_contract_use_database_identifiers() -> None:
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            STATIC_ROOT / "app.js",
            STATIC_ROOT / "modules" / "libraries.js",
            STATIC_ROOT / "modules" / "providers.js",
            STATIC_ROOT / "modules" / "recall.js",
            STATIC_ROOT / "modules" / "tasks-logs.js",
            STATIC_ROOT / "modules" / "text-media-v1.js",
        )
    )

    assert "databases: []" in sources
    assert "selectedDatabaseId" in sources
    assert "expandedDatabaseRefs" in sources
    assert "selectedDatabaseApi" in sources
    assert "selectedDatabase" in sources
    assert "selectDatabase" in sources
    assert "databaseId" in sources
    for legacy_symbol in (
        "selectedLibraryId",
        "expandedLibraryIds",
        "libraryApi",
        "selectedLibrary",
        "selectLibrary",
    ):
        assert legacy_symbol not in sources
    assert "let currentLibrary =" not in sources
    assert "options.libraryId" not in sources
    assert "libraryId =" not in sources
    assert "libraryId:" not in sources


def test_desktop_shell_and_library_cards_use_fluid_container_layout() -> None:
    styles = (STATIC_ROOT / "styles.css").read_text(encoding="utf-8")

    assert "--desktop-sidebar-width:clamp(230px,11vw,300px)" in styles
    assert "--desktop-page-gutter:clamp(24px,1.75vw,56px)" in styles
    assert "grid-template-columns:var(--desktop-sidebar-width) minmax(0,1fr)" in styles
    assert "#page-libraries{width:100%;max-width:var(--library-page-max);margin-inline:auto}" in styles
    assert "#library-cards{container-name:library-grid;container-type:inline-size" in styles
    assert "repeat(auto-fit,minmax(min(100%,var(--library-card-min)),1fr))" in styles
    assert "#library-cards:has(>.library-card:nth-child(3):last-child)>.library-card" in styles
    assert "max-width:620px;justify-self:start" in styles
    assert "#library-cards>.library-card:last-child:nth-child(3n + 1){grid-column:2}" not in styles
    assert "#library-cards>.library-card:last-child:nth-child(2n + 1){grid-column:1/-1;justify-self:center}" not in styles
    assert ".selected-library-badge{display:inline-flex" in styles
    assert "white-space:nowrap;word-break:keep-all" in styles
    assert ".library-card,.library-card.library-card-collapsed{min-height:0}" in styles


def test_adapter_protected_mutations_refresh_both_library_categories_without_polling() -> None:
    libraries_source = (STATIC_ROOT / "modules" / "libraries.js").read_text(
        encoding="utf-8"
    )
    text_media_source = (
        STATIC_ROOT / "modules" / "text-media-v1.js"
    ).read_text(encoding="utf-8")

    assert "async function refreshCatalogAfterMutationConflict(error)" in libraries_source
    assert 'Number(error?.status || 0) !== 409' in libraries_source
    assert (
        'loadDatabases(state.page === "libraries", { force: true })'
        in libraries_source
    )
    assert libraries_source.count(
        "await refreshCatalogAfterMutationConflict(error);"
    ) == 2
    assert "refreshCatalogOnConflict: true" in text_media_source
    assert "refreshCatalogOnConflict && Number(error?.status || 0) === 409" in (
        text_media_source
    )
    assert (
        'loadDatabases(state.page === "libraries", { force: true })'
        in text_media_source
    )
    for source in (libraries_source, text_media_source):
        assert "setInterval" not in source


def test_provider_used_library_jump_syncs_database_category_tab() -> None:
    providers_source = (STATIC_ROOT / "modules" / "providers.js").read_text(
        encoding="utf-8"
    )

    assert 'localStorage.setItem("prag_database_category"' not in providers_source
    assert 'document.querySelectorAll(".database-category-tab").forEach' in providers_source
    assert 'item.dataset.databaseCategory === databaseCategory' in providers_source
    assert 'item.classList.toggle("active", active)' in providers_source
    assert 'item.setAttribute("aria-selected", active ? "true" : "false")' in providers_source
    assert "selectDatabase(databaseId, {" in providers_source
    assert "button.dataset.databaseId" in providers_source
    assert "button.dataset.libraryId" not in providers_source
    assert 'databaseType: button.dataset.databaseType || "livingmemory_v8"' in providers_source


def test_database_category_switch_restores_in_process_selection() -> None:
    libraries_source = (STATIC_ROOT / "modules" / "libraries.js").read_text(
        encoding="utf-8"
    )
    app_source = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert "function ensureCategorySelections()" in libraries_source
    assert "function selectionForCategory(category)" in libraries_source
    assert "function selectedDatabaseNameForCategory(category)" in libraries_source
    assert "function setDatabaseCategory(category" in libraries_source
    assert "database-category-tab-current" in libraries_source
    assert '`· ${currentName}`' in libraries_source
    assert 'localStorage.setItem("prag_database_category"' not in libraries_source
    assert "ensureCategorySelections()[library.database_category]" in libraries_source
    assert "applySelection(selectionForCategory(state.databaseCategory))" in libraries_source
    assert 'setDatabaseCategory(button.dataset.databaseCategory || "memory")' in libraries_source
    assert 'window.addEventListener("prag-database-selection-changed", updateDatabaseCategoryTabs)' in libraries_source
    assert 'window.dispatchEvent(new CustomEvent("prag-database-selection-changed"))' in app_source


def test_startup_uses_default_memory_context_and_cached_catalog() -> None:
    app_source = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    libraries_source = (STATIC_ROOT / "modules" / "libraries.js").read_text(
        encoding="utf-8"
    )

    assert 'const RESTART_RETURN_PAGE = "libraries";' in app_source
    assert "status.version ? `v${status.version}` : \"v-\"" in app_source
    assert 'url.searchParams.has("restart") ? "libraries"' in app_source
    assert 'requestedRoute?.scope === "global" && requestedRoute.pageId === "libraries"' in app_source
    assert "const CATALOG_CACHE_MS = 750;" in libraries_source
    assert "let catalogLoadPromise = null;" in libraries_source
    assert "async function fetchCatalog(options = {})" in libraries_source
    assert "return catalogLoadPromise;" in libraries_source
    assert "await fetchCatalog(options);" in libraries_source
    assert "await fetchCatalog();" in libraries_source


def test_database_cards_identify_embedding_and_rerank_providers() -> None:
    libraries_source = (STATIC_ROOT / "modules" / "libraries.js").read_text(
        encoding="utf-8"
    )
    zh_source = (STATIC_ROOT / "locales" / "zh.js").read_text(encoding="utf-8")
    en_source = (STATIC_ROOT / "locales" / "en.js").read_text(encoding="utf-8")
    ru_source = (STATIC_ROOT / "locales" / "ru.js").read_text(encoding="utf-8")

    assert '"providerLabel": "嵌入模型提供商"' in zh_source
    assert '"providerLabel": "Embedding Model Provider"' in en_source
    assert '"providerLabel": "Провайдер модели эмбеддингов"' in ru_source
    assert libraries_source.count('t("providerLabel")') == 2
    assert libraries_source.count('t("rerankProvider")') == 2
    assert libraries_source.count(
        'library.rerank_provider?.display_name || library.rerank_provider_id || t("none")'
    ) == 2
