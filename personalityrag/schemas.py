from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from .config import normalize_access_base_url


class LoginRequest(BaseModel):
    api_key: str | None = None
    credential: str | None = None


class SettingsUpdate(BaseModel):
    access_base_url: str | None = Field(default=None, max_length=255)
    port: int | None = Field(default=None, ge=1, le=65535)
    access_port: int | None = Field(default=None, ge=1, le=65535)
    new_password: str | None = None
    clear_password: bool = False

    @field_validator("access_base_url")
    @classmethod
    def clean_access_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            return normalize_access_base_url(value)
        except ValueError as exc:
            raise ValueError(
                "服务接入端点 URL 需为不带端口、路径、查询参数或尾斜杠的 http(s) 基址"
            ) from exc


class UiLogRequest(BaseModel):
    level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARN|ERROR)$")
    message: str = Field(min_length=1, max_length=2000)
    context: dict[str, Any] = Field(default_factory=dict)


class AtomInput(BaseModel):
    atom_type: str = "unknown"
    content: str
    entities: list[str] = Field(default_factory=list)
    importance: float = Field(default=0.5, ge=0, le=1)
    confidence: float = Field(default=0.7, ge=0, le=1)
    event_time: float | None = None
    session_id: str | None = None
    persona_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryCreate(BaseModel):
    content: str
    canonical_summary: str | None = None
    persona_summary: str | None = None
    persona_id: str | None = None
    session_id: str | None = None
    importance: float = Field(default=0.5, ge=0, le=1)
    status: str = "active"
    memory_type: str = "GENERAL"
    topics: list[str] = Field(default_factory=list)
    participants: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    atoms: list[AtomInput] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryUpdate(BaseModel):
    content: str | None = None
    importance: float | None = Field(default=None, ge=0, le=10)
    value_scale: str | None = Field(
        default=None,
        pattern="^(auto|display|0-10|ten|stored|normalized|0-1)$",
    )
    status: str | None = None
    memory_type: str | None = None
    session_id: str | None = None
    persona_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecallRequest(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=50)
    rerank_k: int | None = Field(default=None, ge=1, le=50)
    session_id: str | None = None
    persona_id: str | None = None
    rerank: bool | None = None
    include_baseline: bool = False


class ConversationMessageCreate(BaseModel):
    session_id: str = Field(min_length=1)
    role: str = Field(pattern="^(user|assistant|system)$")
    content: str
    sender_id: str | None = None
    sender_name: str | None = None
    group_id: str | None = None
    platform: str = "astrbot"
    timestamp: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    dedup_key: str | None = None


class ConversationMetadataUpdate(BaseModel):
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationTrimRequest(BaseModel):
    delete_count: int = Field(default=0, ge=0)


class BatchDelete(BaseModel):
    memory_ids: list[int]


class BatchUpdate(BaseModel):
    memory_ids: list[int]
    updates: MemoryUpdate


class GraphQuery(BaseModel):
    query: str = ""
    memory_id: int | None = None
    session_id: str | None = None
    persona_id: str | None = None
    limit_memories: int = Field(default=10, ge=1, le=24)


class RebuildRequest(BaseModel):
    reason: str = "manual"
    provider_id: str | None = None


class MigrationRequest(BaseModel):
    source_path: str
    mode: str = Field(pattern="^(rehearsal|formal)$")


class ProviderCreate(BaseModel):
    id: str
    display_name: str
    type: str
    enabled: bool = False
    api_base: str
    api_key: str = ""
    model: str
    dimensions: int = Field(default=0, ge=0)
    max_context_tokens: int = Field(default=0, ge=0)
    max_context_tokens_source: str = ""
    timeout_seconds: int = Field(default=30, gt=0)
    proxy: str = ""
    batch_size: int = Field(default=64, gt=0)
    concurrency: int = Field(default=2, gt=0)
    max_retries: int = Field(default=5, gt=0)
    api_suffix: str = ""
    return_documents: bool = False
    instruct: str = ""
    model_endpoint: str = ""
    truncate: str = ""
    launch_model_if_not_running: bool = False


class ProviderUpdate(BaseModel):
    id: str | None = None
    display_name: str | None = None
    enabled: bool | None = None
    api_base: str | None = None
    api_key: str | None = None
    clear_api_key: bool = False
    model: str | None = None
    dimensions: int | None = Field(default=None, ge=0)
    max_context_tokens: int | None = Field(default=None, ge=0)
    max_context_tokens_source: str | None = None
    timeout_seconds: int | None = Field(default=None, gt=0)
    proxy: str | None = None
    batch_size: int | None = Field(default=None, gt=0)
    concurrency: int | None = Field(default=None, gt=0)
    max_retries: int | None = Field(default=None, gt=0)
    api_suffix: str | None = None
    return_documents: bool | None = None
    instruct: str | None = None
    model_endpoint: str | None = None
    truncate: str | None = None
    launch_model_if_not_running: bool | None = None


class ProviderCopy(BaseModel):
    new_id: str | None = None


class DebugProviderRevisionPatch(BaseModel):
    patch: dict[str, Any] = Field(default_factory=dict)


class DebugProviderRevisionReset(BaseModel):
    latest_revision: int | None = Field(default=None, ge=1)
    bind_libraries_to_latest: bool = False
    library_revisions: dict[str, int] = Field(default_factory=dict)
    delete_revisions_after_latest: bool = False


class LibraryCreate(BaseModel):
    id: str
    name: str
    description: str = ""
    default_persona_id: str = ""
    provider_id: str
    rerank_provider_id: str | None = None
    recall_settings: dict[str, Any] | None = None
    maintenance_settings: dict[str, Any] | None = None


class LibraryUpdate(BaseModel):
    id: str | None = None
    name: str | None = None
    description: str | None = None
    default_persona_id: str | None = None
    provider_id: str | None = None
    rerank_provider_id: str | None = None
    recall_settings: dict[str, Any] | None = None
    maintenance_settings: dict[str, Any] | None = None


class LibraryPskRequest(BaseModel):
    password: str = Field(min_length=1, max_length=512)
