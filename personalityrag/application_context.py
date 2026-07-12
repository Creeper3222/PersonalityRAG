from __future__ import annotations

import os
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .auth import AuthManager
from .config import AppConfig, load_config
from .file_manager import FileManager
from .libraries import LibraryManager
from .logger import configure_logging


SOURCE_ROOT = Path(__file__).resolve().parents[1]
_current_context: ContextVar[ApplicationContext | None] = ContextVar(
    "personalityrag_application_context",
    default=None,
)
_default_context: ApplicationContext | None = None
_runtime_lease_scope: ContextVar[RuntimeLeaseScope | None] = ContextVar(
    "personalityrag_runtime_lease_scope",
    default=None,
)


@dataclass(slots=True)
class RuntimeLeaseScope:
    manager: LibraryManager
    leases: dict[str, bool]

    async def acquire(self, library_id: str, *, touch: bool = True):
        if library_id not in self.leases:
            runtime = await self.manager.acquire_runtime(library_id, touch=touch)
            self.leases[library_id] = touch
            return runtime
        return self.manager.runtimes[library_id]

    async def release_all(self) -> None:
        leases = list(self.leases.items())
        self.leases.clear()
        for library_id, touch in reversed(leases):
            await self.manager.release_runtime(library_id, touch=touch)


@dataclass(slots=True)
class ApplicationContext:
    source_root: Path
    state_root: Path
    config_path: Path
    static_dir: Path
    assets_dir: Path
    config: AppConfig
    auth: AuthManager
    manager: LibraryManager
    file_manager: FileManager
    process_shutdown_callback: Callable[[], None] | None = None
    restart_in_progress: bool = False

    @classmethod
    def create(
        cls,
        *,
        source_root: Path | None = None,
        state_root: Path | None = None,
        config: AppConfig | None = None,
        configure_logs: bool = True,
    ) -> ApplicationContext:
        source = (source_root or SOURCE_ROOT).expanduser().resolve()
        state = (
            state_root
            or Path(os.environ.get("PERSONALITYRAG_STATE_ROOT", str(source)))
        ).expanduser().resolve()
        config_path = state / "config" / "config.json"
        app_config = config or load_config(config_path)
        if configure_logs:
            configure_logging(
                state / "data" / "logs" / "personalityrag.log",
                level_name=app_config.logging.level,
                file_max_bytes=app_config.logging.file_max_bytes,
                file_backup_count=app_config.logging.file_backup_count,
                web_max_entries=app_config.logging.web_max_entries,
                web_max_bytes=app_config.logging.web_max_bytes,
                web_max_entry_bytes=app_config.logging.web_max_entry_bytes,
            )
        auth = AuthManager(
            app_config.api_key,
            app_config.session_secret,
            password_hash=app_config.webui_password_hash,
        )
        file_manager = FileManager(source, state, auth)
        return cls(
            source_root=source,
            state_root=state,
            config_path=config_path,
            static_dir=source / "static",
            assets_dir=source / "assets",
            config=app_config,
            auth=auth,
            manager=LibraryManager(state, app_config),
            file_manager=file_manager,
        )


def set_default_context(context: ApplicationContext) -> None:
    global _default_context
    _default_context = context


def current_context() -> ApplicationContext:
    context = _current_context.get() or _default_context
    if context is None:
        raise RuntimeError("application context is not configured")
    return context


def activate_context(context: ApplicationContext) -> Token:
    return _current_context.set(context)


def reset_context(token: Token) -> None:
    _current_context.reset(token)


def activate_runtime_lease_scope(manager: LibraryManager) -> Token:
    return _runtime_lease_scope.set(RuntimeLeaseScope(manager=manager, leases={}))


def current_runtime_lease_scope() -> RuntimeLeaseScope | None:
    return _runtime_lease_scope.get()


async def release_runtime_lease_scope(token: Token) -> None:
    scope = _runtime_lease_scope.get()
    try:
        if scope is not None:
            await scope.release_all()
    finally:
        _runtime_lease_scope.reset(token)


class ContextAttributeProxy:
    __slots__ = ("_attribute",)

    def __init__(self, attribute: str):
        object.__setattr__(self, "_attribute", attribute)

    def _target(self) -> Any:
        return getattr(current_context(), self._attribute)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._target(), name, value)

    def __repr__(self) -> str:
        return repr(self._target())


config = ContextAttributeProxy("config")
auth = ContextAttributeProxy("auth")
manager = ContextAttributeProxy("manager")
