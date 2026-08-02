from __future__ import annotations

from .auth_settings import router as auth_settings_router
from .files import router as files_router
from .libraries import catalog_router
from .logs import router as logs_router
from .providers import router as providers_router
from .revision_debug import router as revision_debug_router
from .tasks_migration import router as tasks_router
from .updates import router as updates_router


ROUTERS = (
    auth_settings_router,
    files_router,
    logs_router,
    catalog_router,
    providers_router,
    revision_debug_router,
    tasks_router,
    updates_router,
)
