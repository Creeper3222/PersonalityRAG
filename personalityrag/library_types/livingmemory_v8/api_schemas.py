from __future__ import annotations

from typing import Any

from pydantic import BaseModel, field_validator

from ...identifiers import validate_identifier


class LivingMemoryV8Create(BaseModel):
    id: str
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
        return validate_identifier(value, field="记忆库 ID")

    @field_validator("provider_id")
    @classmethod
    def validate_provider_id(cls, value: str) -> str:
        return validate_identifier(value, field="Provider ID")

    @field_validator("rerank_provider_id")
    @classmethod
    def validate_rerank_provider_id(cls, value: str | None) -> str | None:
        if value in (None, ""):
            return value
        return validate_identifier(value, field="Provider ID")
