from __future__ import annotations

from .auth_settings import router as auth_settings_router
from .diagnostics import router as diagnostics_router
from .files import router as files_router
from .libraries import router as libraries_router
from .logs import router as logs_router
from .memories import router as memories_router
from .providers import router as providers_router
from .recall_graph import router as recall_graph_router
from .tasks_migration import router as tasks_migration_router


ROUTERS = (
    auth_settings_router,
    files_router,
    logs_router,
    libraries_router,
    providers_router,
    memories_router,
    recall_graph_router,
    tasks_migration_router,
    diagnostics_router,
)
