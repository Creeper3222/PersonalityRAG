from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from .database_types import LIVINGMEMORY_V8_TYPE


TaskLane = Literal["short", "long"]
EmbeddingContextPolicy = Literal[
    "none",
    "probe_each_long_task",
    "trust_valid_config",
]


@dataclass(frozen=True, slots=True)
class TaskDefinition:
    """Execution contract owned by one database type and one task kind."""

    database_type: str
    kind: str
    lane: TaskLane
    resumable: bool = False
    adapter_blocking: bool = False
    runtime_pause: bool = True
    read_only: bool = False
    embedding_context_policy: EmbeddingContextPolicy = "none"
    database_state_comparison: bool = False

    def public(self) -> dict[str, object]:
        return {
            "lane": self.lane,
            "resumable": self.resumable,
            "adapter_blocking": self.adapter_blocking,
            "runtime_pause": self.runtime_pause,
            "read_only": self.read_only,
            "embedding_context_policy": self.embedding_context_policy,
            "database_state_comparison": self.database_state_comparison,
        }


class TaskTypeRegistry:
    def __init__(self) -> None:
        self._definitions: dict[tuple[str, str], TaskDefinition] = {}

    def register(self, definitions: Iterable[TaskDefinition]) -> None:
        for definition in definitions:
            key = (definition.database_type, definition.kind)
            current = self._definitions.get(key)
            if current is not None and current != definition:
                raise ValueError(
                    "task definition is already registered: "
                    f"{definition.database_type}/{definition.kind}"
                )
            self._definitions[key] = definition

    def get(self, database_type: str, kind: str) -> TaskDefinition:
        key = (str(database_type or LIVINGMEMORY_V8_TYPE), str(kind or ""))
        return self._definitions.get(
            key,
            TaskDefinition(
                database_type=key[0],
                kind=key[1],
                lane="short",
            ),
        )

    def list(self, database_type: str | None = None) -> tuple[TaskDefinition, ...]:
        return tuple(
            sorted(
                (
                    definition
                    for definition in self._definitions.values()
                    if database_type is None
                    or definition.database_type == database_type
                ),
                key=lambda item: (item.database_type, item.kind),
            )
        )


task_type_registry = TaskTypeRegistry()


ACTIVE_JOB_STATUSES = (
    "queued",
    "running",
    "pausing",
    "paused",
    "interrupted",
    "stopping",
)

TERMINAL_JOB_STATUSES = (
    "completed",
    "failed",
    "stopped",
    "cancelled",
)


# Backward-compatible exports for extensions that still import the pre-registry
# constants. Core scheduling and busy checks use task_type_registry instead.
ADAPTER_BUSY_JOB_KINDS = (
    "index_rebuild",
    "graph_rebuild",
    "livingmemory_import",
    "livingmemory_migration",
    "text_media_index_rebuild",
)

RESUMABLE_JOB_KINDS = (
    "index_rebuild",
    "graph_rebuild",
    "livingmemory_import",
    "livingmemory_migration",
    "text_media_index_rebuild",
    "text_media_document_ingest",
    "text_media_ingest_batch",
    "text_media_entry_create",
    "text_media_entry_update",
    "text_media_document_delete",
    "text_media_entry_delete",
    "text_media_media_calibration",
    "text_media_media_descriptions_update",
    "tmkb_import",
    "tmkbs_import",
)

GLOBAL_LONG_JOB_KINDS = (
    *ADAPTER_BUSY_JOB_KINDS,
    "library_copy",
    "library_backup",
    "tmkb_export",
    "tmkb_import",
    "tmkbs_export",
    "tmkbs_import",
    "text_media_document_ingest",
    "text_media_ingest_batch",
    "text_media_entry_create",
    "text_media_entry_update",
    "text_media_document_delete",
    "text_media_entry_delete",
    "text_media_media_calibration",
    "text_media_media_descriptions_update",
)
