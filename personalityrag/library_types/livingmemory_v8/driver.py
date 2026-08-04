from __future__ import annotations

import base64
import hashlib
import hmac
from typing import Iterable

from fastapi import APIRouter

from ...database_types import (
    DATABASE_CATEGORY_MEMORY,
    LIVINGMEMORY_V8_TYPE,
    DatabaseDriverContext,
    DatabaseTypeDescriptor,
)


class LivingMemoryV8Driver:
    descriptor = DatabaseTypeDescriptor(
        id=LIVINGMEMORY_V8_TYPE,
        category=DATABASE_CATEGORY_MEMORY,
        display_name="LivingMemory v8",
        description="Bot 根据消息记录总结、写入并召回的人格记忆库。",
        icon="/static/icons/livingmemory-v8.svg",
        capabilities=(
            "adapter_access",
            "backup",
            "conversation_buffer",
            "copy",
            "graph",
            "index_rebuild",
            "livingmemory_import",
            "memory_records",
            "recall",
            "rerank",
        ),
        key_prefix="psk-",
        key_derivation_id="livingmemory-v8-legacy-hmac-sha256-id-v1",
    )

    def resource_key(self, database_id: str) -> str:
        # Keep the legacy key so v0.1.0 control rows remain readable after rollback.
        return database_id

    def create_manager(self, context: DatabaseDriverContext):
        from .manager import LivingMemoryV8Manager

        return LivingMemoryV8Manager(context)

    def api_routers(self) -> Iterable[APIRouter]:
        from ...routes.diagnostics import router as diagnostics_router
        from ...routes.libraries import livingmemory_router
        from ...routes.memories import router as memories_router
        from ...routes.recall_graph import router as recall_graph_router
        from ...routes.tasks_migration import livingmemory_router as tasks_router

        return (
            livingmemory_router,
            memories_router,
            recall_graph_router,
            diagnostics_router,
            tasks_router,
        )

    def derive_access_key(self, root_secret: str, database_id: str) -> str:
        digest = hmac.new(
            root_secret.encode("utf-8"),
            database_id.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return "psk-" + base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")

    def verify_access_key(
        self, root_secret: str, database_id: str, candidate: str | None
    ) -> bool:
        if not candidate or not candidate.startswith(self.descriptor.key_prefix):
            return False
        return hmac.compare_digest(
            candidate,
            self.derive_access_key(root_secret, database_id),
        )


driver = LivingMemoryV8Driver()
