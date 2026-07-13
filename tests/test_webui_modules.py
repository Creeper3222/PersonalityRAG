from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = REPO_ROOT / "static"
MODULE_NAMES = {
    "api",
    "files",
    "graph",
    "libraries",
    "memories",
    "providers",
    "recall",
    "settings",
    "system",
    "tasks-logs",
}


def test_webui_entrypoint_is_only_shared_assembly() -> None:
    source = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert len(source.splitlines()) < 1_200
    for name in MODULE_NAMES:
        assert f'from "./modules/{name}.js"' in source
    assert "async function loadProviders" not in source
    assert "async function loadLibraries" not in source
    assert "async function loadMemories" not in source
    assert "async function loadSystem" not in source


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


def test_settings_module_has_one_settings_loader() -> None:
    source = (STATIC_ROOT / "modules" / "settings.js").read_text(
        encoding="utf-8"
    )

    assert source.count("async function loadSettings()") == 1
    assert "restartReturnPage" in source
    assert "RESTART_RETURN_PAGE" not in source


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


def test_task_controller_receives_both_library_index_conflict_callbacks() -> None:
    app_source = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    libraries_source = (STATIC_ROOT / "modules" / "libraries.js").read_text(
        encoding="utf-8"
    )
    tasks_logs_source = (STATIC_ROOT / "modules" / "tasks-logs.js").read_text(
        encoding="utf-8"
    )

    controller_signature = tasks_logs_source.split("{", 1)[1].split("}", 1)[0]
    assert "markLibraryIndexConflict" in controller_signature
    assert "clearLibraryIndexConflict" in controller_signature
    assert (
        "markLibraryIndexConflict: (...args) => markLibraryIndexConflict(...args)"
        in app_source
    )
    assert (
        "clearLibraryIndexConflict: (...args) => clearLibraryIndexConflict(...args)"
        in app_source
    )
    library_controller_return = libraries_source.rsplit("return {", 1)[1].split(
        "};", 1
    )[0]
    assert "markLibraryIndexConflict" in library_controller_return
    assert "clearLibraryIndexConflict" in library_controller_return
    assert (
        "confirmSensitiveProviderEdit, markLibraryIndexConflict, clearLibraryIndexConflict"
        in app_source
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


def test_acceptance_runner_is_not_bound_to_live_library_paths() -> None:
    source = (REPO_ROOT / "tools" / "acceptance_webui.py").read_text(
        encoding="utf-8"
    )

    assert '"beileite"' not in source
    assert "--base-url" in source
    assert "--state-root" in source
    assert "--report-dir" in source
    assert "tempfile.gettempdir()" in source


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


def test_memory_source_types_have_localized_display_labels() -> None:
    source = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")

    assert 'GROUP_CHAT: t("typeGroupChat")' in source
    assert 'PRIVATE_CHAT: t("typePrivateChat")' in source
    assert 'MANUAL: t("typeManual")' in source
    for value in ("GENERAL", "GROUP_CHAT", "PRIVATE_CHAT", "MANUAL"):
        assert f'<option value="{value}"' in html


def test_settings_module_exposes_version_switch_workflow() -> None:
    source = (STATIC_ROOT / "modules" / "settings.js").read_text(encoding="utf-8")
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")

    assert 'api("/updates/switch"' in source
    assert 'api(`/updates/status' in source
    assert 'api(`/updates/releases' in source
    assert "data-update-tag" in source
    assert 'id="update-available-badge"' in html
    assert 'id="updates-modal-busy"' in html
