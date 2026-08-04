from __future__ import annotations

from ...database_types import TEXT_MEDIA_V1_TYPE
from ...task_types import (
    EmbeddingContextPolicy,
    TaskDefinition,
    task_type_registry,
)


def _long(
    kind: str,
    *,
    resumable: bool = True,
    adapter_blocking: bool = False,
    read_only: bool = False,
    runtime_pause: bool = True,
    embedding_context_policy: EmbeddingContextPolicy = "none",
    database_state_comparison: bool = False,
) -> TaskDefinition:
    return TaskDefinition(
        TEXT_MEDIA_V1_TYPE,
        kind,
        "long",
        resumable=resumable,
        adapter_blocking=adapter_blocking,
        read_only=read_only,
        runtime_pause=runtime_pause,
        embedding_context_policy=embedding_context_policy,
        database_state_comparison=database_state_comparison,
    )


TASK_DEFINITIONS = (
    _long(
        "text_media_index_rebuild",
        adapter_blocking=True,
        embedding_context_policy="probe_each_long_task",
        database_state_comparison=True,
    ),
    _long(
        "text_media_document_ingest",
        embedding_context_policy="probe_each_long_task",
        database_state_comparison=True,
    ),
    _long(
        "text_media_ingest_batch",
        embedding_context_policy="probe_each_long_task",
        database_state_comparison=True,
    ),
    _long(
        "text_media_entry_create",
        embedding_context_policy="probe_each_long_task",
        database_state_comparison=True,
    ),
    _long(
        "text_media_entry_update",
        embedding_context_policy="probe_each_long_task",
        database_state_comparison=True,
    ),
    _long("text_media_document_delete", database_state_comparison=True),
    _long("text_media_entry_delete", database_state_comparison=True),
    _long(
        "text_media_media_calibration",
        embedding_context_policy="probe_each_long_task",
        database_state_comparison=True,
    ),
    _long(
        "text_media_media_descriptions_update",
        embedding_context_policy="probe_each_long_task",
        database_state_comparison=True,
    ),
    TaskDefinition(TEXT_MEDIA_V1_TYPE, "text_media_image_upload", "short"),
    TaskDefinition(TEXT_MEDIA_V1_TYPE, "text_media_image_delete", "short"),
    TaskDefinition(TEXT_MEDIA_V1_TYPE, "text_media_relation_update", "short"),
    TaskDefinition(TEXT_MEDIA_V1_TYPE, "text_media_relation_delete", "short"),
    TaskDefinition(
        TEXT_MEDIA_V1_TYPE,
        "text_media_visual_intent_policy_update",
        "short",
        embedding_context_policy="none",
    ),
    _long("tmkb_import", database_state_comparison=True),
    _long("tmkbs_import"),
    _long("library_copy", resumable=False, read_only=True, runtime_pause=False),
    _long("library_backup", resumable=False, read_only=True, runtime_pause=False),
    _long("tmkb_export", resumable=False, read_only=True, runtime_pause=False),
    _long("tmkbs_export", resumable=False, read_only=True, runtime_pause=False),
)


task_type_registry.register(TASK_DEFINITIONS)
