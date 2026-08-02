from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from .config import normalize_access_base_url, normalize_public_adapter_url
from .identifiers import validate_identifier


def _validate_required_id(value: str, *, field: str) -> str:
    return validate_identifier(value, field=field)


def _validate_optional_id(value: str | None, *, field: str) -> str | None:
    if value is None or value == "":
        return value
    return validate_identifier(value, field=field)


def _validate_present_id(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    return validate_identifier(value, field=field)


class LoginRequest(BaseModel):
    api_key: str | None = None
    credential: str | None = None


class SettingsUpdate(BaseModel):
    access_base_url: str | None = Field(default=None, max_length=255)
    public_adapter_url: str | None = Field(default=None, max_length=512)
    port: int | None = Field(default=None, ge=1, le=65535)
    access_port: int | None = Field(default=None, ge=1, le=65535)
    new_password: str | None = None
    clear_password: bool = False
    runtime_idle_minutes: int | None = Field(default=None, ge=1)
    max_non_default_runtimes: int | None = Field(default=None, ge=1)

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

    @field_validator("public_adapter_url")
    @classmethod
    def clean_public_adapter_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            return normalize_public_adapter_url(value)
        except ValueError as exc:
            raise ValueError(
                "公网适配器 URL 必须是不带路径、凭据、查询参数或片段的 HTTPS 地址"
            ) from exc


class BackupMigrationExportRequest(BaseModel):
    password: str = Field(default="", max_length=512)
    include_libraries: bool = True
    include_providers: bool = True


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


class ParticipantIdentity(BaseModel):
    identity_key: str | None = Field(default=None, max_length=512)
    sender_id: str = Field(min_length=1, max_length=256)
    platform: str = Field(default="unknown", min_length=1, max_length=128)
    display_name: str = Field(min_length=1, max_length=256)
    aliases: list[str] = Field(default_factory=list, max_length=32)
    is_bot: bool = False

    @field_validator("sender_id", "platform", "display_name")
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("participant identity fields cannot be empty")
        return normalized

    @field_validator("aliases")
    @classmethod
    def normalize_aliases(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for raw in values:
            value = str(raw or "").strip()
            key = value.casefold()
            if not value or len(value) > 256 or key in seen:
                continue
            seen.add(key)
            result.append(value)
        return result

    @model_validator(mode="after")
    def normalize_identity_key(self):
        self.platform = self.platform.lower()
        expected = f"{self.platform}:{self.sender_id}"
        supplied = str(self.identity_key or "").strip()
        if supplied and supplied.casefold() != expected.casefold():
            raise ValueError("participant identity_key does not match platform/sender_id")
        self.identity_key = expected
        if self.display_name.casefold() not in {
            alias.casefold() for alias in self.aliases
        }:
            self.aliases.append(self.display_name)
        return self


class MemoryCreate(BaseModel):
    content: str
    canonical_summary: str | None = None
    persona_summary: str | None = None
    persona_id: str | None = None
    session_id: str | None = None
    importance: float = Field(default=0.5, ge=0, le=1)
    status: str = "active"
    topics: list[str] = Field(default_factory=list)
    participants: list[str] = Field(default_factory=list)
    participant_identities: list[ParticipantIdentity] = Field(
        default_factory=list,
        max_length=64,
    )
    key_facts: list[str] = Field(default_factory=list)
    atoms: list[AtomInput] = Field(default_factory=list)
    source_messages: list[dict[str, Any]] = Field(default_factory=list)
    source_time_strategy: str = Field(
        default="preserve", pattern="^(preserve|derive|none)$"
    )
    source_time_tags: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryUpdate(BaseModel):
    content: str | None = None
    importance: float | None = Field(default=None, ge=0, le=10)
    value_scale: str | None = Field(
        default=None,
        pattern="^(auto|display|0-10|ten|stored|normalized|0-1)$",
    )
    status: str | None = None
    session_id: str | None = None
    persona_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryPersonaUpdate(BaseModel):
    persona_id: str = ""


class MemorySourceUpdate(BaseModel):
    source_messages: list[dict[str, Any]] = Field(min_length=1)


class MemoryResummaryCommit(BaseModel):
    canonical_summary: str = Field(min_length=1)
    persona_summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    expected_content_sha256: str | None = Field(
        default=None, pattern="^[0-9a-fA-F]{64}$"
    )


class MemoryTransferSummary(BaseModel):
    preview_item_id: str = Field(min_length=1, max_length=128)
    canonical_summary: str = Field(min_length=1)
    persona_summary: str | None = None
    importance: float | None = Field(default=None, ge=0, le=1)
    topics: list[str] = Field(default_factory=list)
    participants: list[str] = Field(default_factory=list)
    participant_identities: list[ParticipantIdentity] = Field(
        default_factory=list,
        max_length=64,
    )
    key_facts: list[str] = Field(default_factory=list)


class MemoryTransferCommit(BaseModel):
    duplicate_mode: str = Field(default="skip", pattern="^(skip|allow)$")
    summaries: list[MemoryTransferSummary] = Field(default_factory=list)


class RecallRequest(BaseModel):
    query: str
    k: int = Field(default=5, ge=1)
    rerank_k: int | None = Field(default=None, ge=1)
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


class AdapterHeartbeatRequest(BaseModel):
    wait_seconds: float = Field(default=0, ge=0, le=55)
    manual_reconnect: bool = False


class AdapterDisconnectRequest(BaseModel):
    instance_id: str

    @field_validator("instance_id")
    @classmethod
    def validate_instance_id(cls, value: str) -> str:
        return _validate_required_id(value, field="适配器实例ID")


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
    limit_entries: int = Field(default=40, ge=12, le=80)
    limit_nodes: int = Field(default=56, ge=12, le=80)
    limit_edges: int = Field(default=96, ge=12, le=120)


class RebuildRequest(BaseModel):
    reason: str = "manual"
    provider_id: str | None = None

    @field_validator("provider_id")
    @classmethod
    def validate_provider_id(cls, value: str | None) -> str | None:
        return _validate_optional_id(value, field="Provider ID")


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
    context_length_mode: str = Field(default="auto", pattern="^(auto|manual)$")
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
    index_rebuild_settings: dict[str, Any] | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _validate_required_id(value, field="Provider ID")


class ProviderUpdate(BaseModel):
    id: str | None = None
    display_name: str | None = None
    enabled: bool | None = None
    api_base: str | None = None
    api_key: str | None = None
    clear_api_key: bool = False
    model: str | None = None
    dimensions: int | None = Field(default=None, ge=0)
    context_length_mode: str | None = Field(default=None, pattern="^(auto|manual)$")
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
    index_rebuild_settings: dict[str, Any] | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str | None) -> str | None:
        return _validate_present_id(value, field="Provider ID")


class ProviderCopy(BaseModel):
    new_id: str | None = None

    @field_validator("new_id")
    @classmethod
    def validate_new_id(cls, value: str | None) -> str | None:
        return _validate_optional_id(value, field="Provider ID")


class DebugProviderRevisionPatch(BaseModel):
    patch: dict[str, Any] = Field(default_factory=dict)
    password: str = Field(default="", max_length=512)
    risk_confirmed: bool = False


class DebugProviderRevisionReset(BaseModel):
    latest_revision: int | None = Field(default=None, ge=1)
    bind_libraries_to_latest: bool = False
    library_revisions: dict[str, int] = Field(default_factory=dict)
    delete_revisions_after_latest: bool = False
    force_non_equivalent: bool = False
    password: str = Field(default="", max_length=512)
    risk_confirmed: bool = False


class DebugSessionUnlock(BaseModel):
    password: str = Field(min_length=1, max_length=512)


class DebugDatabaseBindingPatch(BaseModel):
    provider_id: str
    revision: int = Field(ge=1)
    assert_functional_compatibility: bool = False
    password: str = Field(default="", max_length=512)
    risk_confirmed: bool = False

    @field_validator("provider_id")
    @classmethod
    def validate_provider_id(cls, value: str) -> str:
        return _validate_required_id(value, field="Provider ID")


class LibraryCreate(BaseModel):
    id: str
    database_type: str = "livingmemory_v8"
    name: str
    description: str = ""
    default_persona_id: str = ""
    provider_id: str
    rerank_provider_id: str | None = None
    conversation_settings: dict[str, Any] | None = None
    recall_settings: dict[str, Any] | None = None
    maintenance_settings: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _validate_required_id(value, field="记忆库 ID")

    @field_validator("database_type")
    @classmethod
    def validate_database_type(cls, value: str) -> str:
        return _validate_required_id(value, field="数据库类型")

    @field_validator("provider_id")
    @classmethod
    def validate_provider_id(cls, value: str) -> str:
        return _validate_required_id(value, field="Provider ID")

    @field_validator("rerank_provider_id")
    @classmethod
    def validate_rerank_provider_id(cls, value: str | None) -> str | None:
        return _validate_optional_id(value, field="Provider ID")


class LibraryUpdate(BaseModel):
    id: str | None = None
    name: str | None = None
    description: str | None = None
    default_persona_id: str | None = None
    provider_id: str | None = None
    rerank_provider_id: str | None = None
    conversation_settings: dict[str, Any] | None = None
    recall_settings: dict[str, Any] | None = None
    maintenance_settings: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str | None) -> str | None:
        return _validate_present_id(value, field="记忆库 ID")

    @field_validator("provider_id")
    @classmethod
    def validate_provider_id(cls, value: str | None) -> str | None:
        return _validate_present_id(value, field="Provider ID")

    @field_validator("rerank_provider_id")
    @classmethod
    def validate_rerank_provider_id(cls, value: str | None) -> str | None:
        return _validate_optional_id(value, field="Provider ID")


class LibraryPskRequest(BaseModel):
    password: str = Field(min_length=1, max_length=512)
