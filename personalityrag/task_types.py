from __future__ import annotations


ADAPTER_BUSY_JOB_KINDS = (
    "index_rebuild",
    "graph_rebuild",
    "livingmemory_import",
    "livingmemory_migration",
)

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

RESUMABLE_JOB_KINDS = (
    "index_rebuild",
    "graph_rebuild",
    "livingmemory_import",
    "livingmemory_migration",
)
