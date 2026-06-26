from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    api_key: str | None = None
    credential: str | None = None


class SettingsUpdate(BaseModel):
    port: int | None = Field(default=None, ge=1, le=65535)
    new_password: str | None = None
    clear_password: bool = False


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
    importance: float | None = Field(default=None, ge=0, le=1)
    status: str | None = None
    memory_type: str | None = None
    session_id: str | None = None
    persona_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecallRequest(BaseModel):
    query: str
    k: int = Field(default=10, ge=1, le=50)
    session_id: str | None = None
    persona_id: str | None = None


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
    timeout_seconds: int = Field(default=30, gt=0)
    proxy: str = ""
    batch_size: int = Field(default=64, gt=0)
    concurrency: int = Field(default=2, gt=0)
    max_retries: int = Field(default=5, gt=0)


class ProviderUpdate(BaseModel):
    display_name: str | None = None
    enabled: bool | None = None
    api_base: str | None = None
    api_key: str | None = None
    clear_api_key: bool = False
    model: str | None = None
    dimensions: int | None = Field(default=None, ge=0)
    timeout_seconds: int | None = Field(default=None, gt=0)
    proxy: str | None = None
    batch_size: int | None = Field(default=None, gt=0)
    concurrency: int | None = Field(default=None, gt=0)
    max_retries: int | None = Field(default=None, gt=0)


class ProviderCopy(BaseModel):
    new_id: str | None = None


class LibraryCreate(BaseModel):
    id: str
    name: str
    description: str = ""
    default_persona_id: str = ""
    provider_id: str
    recall_settings: dict[str, Any] | None = None
    maintenance_settings: dict[str, Any] | None = None


class LibraryUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    default_persona_id: str | None = None
    recall_settings: dict[str, Any] | None = None
    maintenance_settings: dict[str, Any] | None = None
