from __future__ import annotations

from ...database_types import LIVINGMEMORY_V8_TYPE
from ...task_types import TaskDefinition, task_type_registry


TASK_DEFINITIONS = (
    TaskDefinition(
        LIVINGMEMORY_V8_TYPE,
        "index_rebuild",
        "long",
        resumable=True,
        adapter_blocking=False,
        embedding_context_policy="probe_each_long_task",
        database_state_comparison=True,
    ),
    TaskDefinition(
        LIVINGMEMORY_V8_TYPE,
        "graph_rebuild",
        "long",
        resumable=True,
        adapter_blocking=False,
        embedding_context_policy="probe_each_long_task",
        database_state_comparison=True,
    ),
    TaskDefinition(
        LIVINGMEMORY_V8_TYPE,
        "livingmemory_import",
        "long",
        resumable=True,
        adapter_blocking=True,
        embedding_context_policy="probe_each_long_task",
        database_state_comparison=True,
    ),
    TaskDefinition(
        LIVINGMEMORY_V8_TYPE,
        "livingmemory_migration",
        "long",
        resumable=True,
        adapter_blocking=True,
        embedding_context_policy="probe_each_long_task",
        database_state_comparison=True,
    ),
    TaskDefinition(
        LIVINGMEMORY_V8_TYPE,
        "memory_transfer_import",
        "long",
        resumable=True,
        adapter_blocking=True,
        embedding_context_policy="trust_valid_config",
        database_state_comparison=True,
    ),
    TaskDefinition(
        LIVINGMEMORY_V8_TYPE,
        "memory_create",
        "short",
        embedding_context_policy="trust_valid_config",
    ),
    TaskDefinition(
        LIVINGMEMORY_V8_TYPE,
        "memory_update",
        "short",
        embedding_context_policy="trust_valid_config",
    ),
    TaskDefinition(LIVINGMEMORY_V8_TYPE, "memory_delete", "short"),
    TaskDefinition(
        LIVINGMEMORY_V8_TYPE,
        "library_copy",
        "long",
        read_only=True,
        runtime_pause=False,
    ),
    TaskDefinition(
        LIVINGMEMORY_V8_TYPE,
        "library_backup",
        "long",
        read_only=True,
        runtime_pause=False,
    ),
)


task_type_registry.register(TASK_DEFINITIONS)
