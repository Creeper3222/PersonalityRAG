"""Compatibility entry point for the PersonalityRAG ASGI application."""

from __future__ import annotations

from .application import create_app
from .application_context import (
    ApplicationContext,
    SOURCE_ROOT,
    set_default_context,
)
from .http_shared import (
    RESTART_PROBE_SCAN_LIMIT,
    _restart_probe_urls,
    jobs,
    require_auth,
    runtime,
    set_process_shutdown_callback,
)
from .routes.auth_settings import _settings_payload, health
from .routes.memories import _normalize_memory_update_payload
from .routes.recall_graph import recall


DEFAULT_CONTEXT = ApplicationContext.create()
set_default_context(DEFAULT_CONTEXT)

ROOT = SOURCE_ROOT
STATE_ROOT = DEFAULT_CONTEXT.state_root
CONFIG_PATH = DEFAULT_CONTEXT.config_path
STATIC_DIR = DEFAULT_CONTEXT.static_dir
ASSETS_DIR = DEFAULT_CONTEXT.assets_dir
config = DEFAULT_CONTEXT.config
auth = DEFAULT_CONTEXT.auth
manager = DEFAULT_CONTEXT.manager
app = create_app(DEFAULT_CONTEXT)


__all__ = [
    "ASSETS_DIR",
    "CONFIG_PATH",
    "DEFAULT_CONTEXT",
    "ROOT",
    "RESTART_PROBE_SCAN_LIMIT",
    "STATE_ROOT",
    "STATIC_DIR",
    "_normalize_memory_update_payload",
    "_restart_probe_urls",
    "_settings_payload",
    "app",
    "auth",
    "config",
    "create_app",
    "health",
    "jobs",
    "manager",
    "recall",
    "require_auth",
    "runtime",
    "set_process_shutdown_callback",
]
