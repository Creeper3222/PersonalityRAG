from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import shutil
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, field_validator, model_validator

from ...application_context import auth, config, current_context, manager
from ...auth import COOKIE_NAME, verify_password
from ...control import ADAPTER_CONNECTION_TTL_SECONDS
from ...database_types import (
    TEXT_MEDIA_V1_TYPE,
    DatabaseRef,
    database_identity_fields,
    database_type_registry,
)
from ...http_shared import (
    ADAPTER_ID_HEADER,
    ADAPTER_INSTANCE_HEADER,
    _adapter_forced_offline_detail,
    _adapter_header,
    _current_adapter_connection_url,
    jobs,
    require_admin_auth,
    require_auth,
    require_existing_database_ref,
    runtime,
)
from ...identifiers import validate_identifier
from ...io_utils import run_blocking, save_upload_file
from ...providers import provider_kind
from .document_parsers import SUPPORTED_DOCUMENT_SUFFIXES
from .batch_package import (
    MAX_BATCH_LIBRARIES,
    MAX_BATCH_PACKAGE_BYTES,
    export_tmkbs,
    inspect_tmkbs,
)
from .images import MAX_SOURCE_BYTES
from .package import MAX_PACKAGE_BYTES, export_tmkb, extract_tmkb, inspect_tmkb
from .request_logging import (
    log_search_completed,
    log_search_failed,
    log_search_rejected,
    log_search_request_received,
)
from .text import normalize_visual_intent_policy
from .visual_intent_policy import (
    MAX_VISUAL_INTENT_POLICY_CSV_BYTES,
    VISUAL_INTENT_POLICY_KEYS,
    parse_visual_intent_policy_csv,
    serialize_visual_intent_policy_csv,
)


router = APIRouter(prefix="/api/v1/knowledge-libraries/text_media_v1")
_import_tokens: dict[str, dict[str, Any]] = {}
_export_tokens: dict[str, dict[str, Any]] = {}
_batch_import_tokens: dict[str, dict[str, Any]] = {}
_batch_export_tokens: dict[str, dict[str, Any]] = {}
TOKEN_TTL_SECONDS = 30 * 60
MEDIA_SCOPE_VERSION = 1


def _task_input_root(knowledge_base_id: str, token: str | None = None) -> Path:
    return (
        current_context().state_root
        / "data"
        / "task_inputs"
        / TEXT_MEDIA_V1_TYPE
        / knowledge_base_id
        / (token or uuid.uuid4().hex)
    )


async def _start_text_media_resumable(
    kind: str,
    *,
    knowledge_base_id: str,
    operation: dict[str, Any],
) -> str:
    return await jobs().start_resumable(
        kind,
        operation,
        database_id=knowledge_base_id,
        database_type=TEXT_MEDIA_V1_TYPE,
        dedupe_active=False,
        lease_runtime=False,
    )


async def _run_text_media_short(
    kind: str,
    *,
    knowledge_base_id: str,
    operation,
) -> dict[str, Any]:
    job_id = await jobs().start(
        kind,
        operation,
        database_id=knowledge_base_id,
        database_type=TEXT_MEDIA_V1_TYPE,
        dedupe_active=False,
        lease_runtime=False,
    )
    completed = await jobs().wait(job_id)
    if completed.get("status") != "completed":
        raise RuntimeError(
            str(completed.get("error") or completed.get("message") or "task failed")
        )
    result = completed.get("result")
    return {
        **(dict(result) if isinstance(result, dict) else {"result": result}),
        "job_id": job_id,
    }


class CreateRequest(BaseModel):
    id: str
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=4000)
    provider_id: str
    rerank_provider_id: str | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return validate_identifier(value, field="知识库 ID")

    @field_validator("provider_id")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        return validate_identifier(value, field="Provider ID")

    @field_validator("rerank_provider_id")
    @classmethod
    def validate_rerank_provider(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        return validate_identifier(value, field="Rerank Provider ID")


class RetrievalSettingsRequest(BaseModel):
    rrf_k: int = Field(default=60, ge=1, le=1000)
    rerank_candidate_limit: int = Field(default=10, ge=10, le=200)
    rerank_fusion_weight: float = Field(default=0.30, ge=0, le=1)
    rerank_rank_bonus_weight: float = Field(default=0.0, ge=0, le=1)
    rerank_rank_reliability_exponent: float = Field(
        default=1.5, ge=0.5, le=4
    )
    text_lexical_boost: float = Field(default=0.60, ge=0, le=1)
    media_candidate_limit: int = Field(default=30, ge=10, le=500)
    unbound_media_candidate_limit: int = Field(default=10, ge=10, le=500)
    media_relevance_pivot_fallback: float = Field(default=0.35, ge=0, le=1)
    media_pivot_positive_blend: float = Field(default=0.70, ge=0, le=1)
    media_pivot_negative_weight: float = Field(default=0.35, ge=0, le=1)
    media_pivot_negative_attenuation_floor: float = Field(
        default=0.05, ge=0.001, le=1
    )
    media_format_mismatch_factor: float = Field(
        default=0.10, ge=0.001, le=1
    )
    media_content_mismatch_factor: float = Field(
        default=0.10, ge=0.001, le=1
    )
    media_score_threshold_fallback: float | None = Field(
        default=None,
        ge=0,
        le=1,
        deprecated=True,
        description="Deprecated alias for media_relevance_pivot_fallback.",
    )
    media_threshold_evidence_limit: int = Field(default=5, ge=1, le=100)
    media_threshold_rank_decay_exponent: float = Field(default=1.5, ge=0, le=4)
    media_threshold_negative_reliability_exponent: float = Field(
        default=1.5, ge=1, le=4
    )
    media_threshold_reinforcement_weight: float | None = Field(
        default=None,
        ge=0,
        le=1,
        deprecated=True,
        description="Deprecated alias for media_pivot_positive_blend.",
    )
    media_threshold_weakening_weight: float | None = Field(
        default=None,
        ge=0,
        le=1,
        deprecated=True,
        description="Deprecated alias for media_pivot_negative_weight.",
    )
    visual_intent_gate_enabled: bool = True
    media_semantic_floor: float = Field(default=0.35, ge=0, le=0.99)
    media_semantic_weight: float = Field(default=0.80, ge=0, le=1)
    media_lexical_boost: float = Field(default=0.30, ge=0, le=1)
    media_lexical_coverage_exponent: float = Field(default=1.20, ge=0.1, le=4)
    media_lexical_common_floor: float = Field(default=0.00, ge=0, le=1)
    media_lexical_oov_penalty: float = Field(default=0.30, ge=0, le=2)
    media_distinctive_rarity_exponent: float = Field(
        default=1.50, ge=0.25, le=4
    )
    unbound_media_distinctive_boost: float = Field(default=0.35, ge=0, le=1)
    unbound_media_collection_boost: float = Field(default=0.55, ge=0, le=1)
    unbound_media_competition_floor: float = Field(default=0.35, ge=0, le=1)
    unbound_media_reliability_target: float = Field(default=0.25, ge=0.01, le=1)
    unbound_media_specificity_exponent: float = Field(
        default=1.0, ge=0.25, le=4
    )
    unbound_media_advantage_target: float = Field(default=0.04, ge=0.01, le=1)
    media_bound_distinctive_boost: float = Field(default=0.10, ge=0, le=1)
    media_bound_distinctive_rescue_min: float = Field(
        default=0.80, ge=0, le=1
    )
    media_rank_decay_exponent: float = Field(default=1.0, ge=0, le=4)
    media_corroboration_weight: float = Field(default=0.50, ge=0, le=1)
    media_corroboration_limit: int = Field(default=5, ge=1, le=50)


class UpdateRequest(BaseModel):
    id: str | None = None
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=4000)
    uniform_media_strength: float | None = Field(default=None, ge=0, le=1)
    retrieval_settings: RetrievalSettingsRequest | None = None
    rerank_provider_id: str | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str | None) -> str | None:
        return validate_identifier(value, field="知识库 ID") if value is not None else None


    @field_validator("rerank_provider_id")
    @classmethod
    def validate_rerank_provider(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        return validate_identifier(value, field="Rerank Provider ID")


class VisualIntentPolicyRequest(BaseModel):
    visual_object_terms: list[str] = Field(max_length=256)
    lookup_action_terms: list[str] = Field(max_length=256)
    generation_action_terms: list[str] = Field(
        max_length=256
    )
    reference_connector_terms: list[str] = Field(
        max_length=256
    )

    @model_validator(mode="after")
    def validate_policy(self) -> "VisualIntentPolicyRequest":
        normalized = normalize_visual_intent_policy(
            self.model_dump(),
            require_complete=True,
        )
        for key, terms in normalized.items():
            setattr(self, key, terms)
        return self


class VisualIntentPolicyExportRequest(BaseModel):
    categories: list[
        Literal[
            "visual_object_terms",
            "lookup_action_terms",
            "generation_action_terms",
            "reference_connector_terms",
        ]
    ] = Field(min_length=1, max_length=4)
    policy: VisualIntentPolicyRequest

    @field_validator("categories")
    @classmethod
    def validate_categories(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("visual intent policy categories cannot be duplicated")
        return value


class RebuildRequest(BaseModel):
    provider_id: str | None = None
    reason: str = Field(default="manual", max_length=120)

    @field_validator("provider_id")
    @classmethod
    def validate_rebuild_provider(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        return validate_identifier(value, field="Provider ID")


class AccessKeyRequest(BaseModel):
    password: str = Field(min_length=1, max_length=512)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=20000)
    retrieval_mode: Literal["standard", "text_only", "media_only"] = "standard"
    media_response_mode: Literal["full", "descriptions_only"] = Field(
        default="full",
        description=(
            "Full returns the established media diagnostics and asset URLs. "
            "Descriptions-only returns compact qualified media candidates "
            "without creating asset URLs."
        ),
    )
    top_k: int = Field(default=10, ge=1, le=50)
    media_output_confidence_threshold: float = Field(
        default=0.6,
        ge=0,
        le=1,
        description="Final automatic media output-confidence threshold.",
    )
    media_relevance_pivot: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            "Universal media relevance pivot. It calibrates both bound chunk "
            "evidence and unbound asset relevance; missing or null uses the "
            "single-library fallback."
        ),
    )
    media_score_threshold: float | None = Field(
        default=None,
        ge=0,
        le=1,
        deprecated=True,
        description=(
            "Deprecated alias for media_relevance_pivot. Supplying different "
            "numeric values for both fields is rejected."
        ),
    )
    max_media_outputs: int = Field(default=5, ge=0, le=20)
    rerank: bool | None = Field(
        default=None,
        description=(
            "False disables all query-time rerank paths; missing or null uses "
            "the bound-provider default."
        ),
    )

    @model_validator(mode="after")
    def validate_media_relevance_pivot_alias(self) -> "SearchRequest":
        canonical = self.__dict__.get("media_relevance_pivot")
        legacy = self.__dict__.get("media_score_threshold")
        if (
            canonical is not None
            and legacy is not None
            and abs(float(canonical) - float(legacy)) > 1e-12
        ):
            raise ValueError(
                "media_relevance_pivot and deprecated "
                "media_score_threshold must match when both are provided"
            )
        return self


class IngestImageMapping(BaseModel):
    document_indexes: list[int] = Field(default_factory=list, max_length=10)
    media_description: str = Field(default="", max_length=2000)
    media_descriptions: list[str] = Field(
        default_factory=list, max_length=20
    )

    @field_validator("media_descriptions")
    @classmethod
    def validate_media_descriptions(cls, values: list[str]) -> list[str]:
        from .text import media_description_list

        return (
            media_description_list(values, allow_empty_input=True)
            if values
            else []
        )

    @model_validator(mode="after")
    def validate_legacy_description(self):
        from .text import normalize_media_description

        if (
            self.media_descriptions
            and self.media_description.strip()
            and normalize_media_description(self.media_description)
            != normalize_media_description(self.media_descriptions[0])
        ):
            raise ValueError(
                "media_description must equal the first media_descriptions item"
            )
        return self


class IngestBatchManifest(BaseModel):
    chunk_target: int = Field(default=1200, ge=200, le=4000)
    chunk_overlap: int = Field(default=150, ge=0, le=2000)
    embedding_batch_size: int = Field(default=32, ge=1, le=128)
    concurrency: int = Field(default=3, ge=1, le=8)
    max_retries: int = Field(default=3, ge=1, le=8)
    media_semantic_calibration_enabled: bool = False
    images: list[IngestImageMapping] = Field(default_factory=list, max_length=10)


class SemanticCalibrationRequest(BaseModel):
    enabled: bool = True
    media_description: str = Field(default="", max_length=2000)


class MediaDescriptionsUpdateRequest(BaseModel):
    media_descriptions: list[str] = Field(min_length=1, max_length=20)

    @field_validator("media_descriptions")
    @classmethod
    def validate_media_descriptions(cls, values: list[str]) -> list[str]:
        from .text import media_description_list

        return media_description_list(values)


class EntryRequest(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    body: str = Field(min_length=1, max_length=5_000_000)


class BatchDocumentDeleteRequest(BaseModel):
    document_ids: list[str] = Field(min_length=1, max_length=200)

    @field_validator("document_ids")
    @classmethod
    def validate_document_ids(cls, values: list[str]) -> list[str]:
        normalized = [str(value).strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("document ID cannot be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("duplicate document ID")
        return normalized


class RelationRequest(BaseModel):
    role: str = Field(default="illustration", max_length=64)
    relation_weight: float = Field(default=1.0, ge=0, le=1)
    caption: str = Field(default="", max_length=2000)
    alt_text: str = Field(default="", max_length=2000)
    output_policy: Literal["auto", "with_result", "metadata_only", "disabled"] = "auto"
    sort_order: int = 0


class ImportCommitRequest(BaseModel):
    target_id: str
    name: str | None = Field(default=None, max_length=200)

    @field_validator("target_id")
    @classmethod
    def validate_target_id(cls, value: str) -> str:
        return validate_identifier(value, field="知识库 ID")


class BatchExportRequest(BaseModel):
    database_ids: list[str] = Field(min_length=1, max_length=MAX_BATCH_LIBRARIES)

    @field_validator("database_ids")
    @classmethod
    def validate_database_ids(cls, values: list[str]) -> list[str]:
        normalized = [validate_identifier(value, field="knowledge library ID") for value in values]
        if len(set(normalized)) != len(normalized):
            raise ValueError("duplicate knowledge library selection")
        return normalized


class BatchImportItem(BaseModel):
    source_id: str
    target_id: str
    name: str | None = Field(default=None, max_length=200)

    @field_validator("source_id", "target_id")
    @classmethod
    def validate_database_id(cls, value: str) -> str:
        return validate_identifier(value, field="knowledge library ID")


class BatchImportCommitRequest(BaseModel):
    libraries: list[BatchImportItem] = Field(min_length=1, max_length=MAX_BATCH_LIBRARIES)


class SignedUrlRequest(BaseModel):
    variant: Literal["content", "thumbnail"] = "content"
    expires_in: int = Field(default=300, ge=30, le=600)


async def _ref(knowledge_base_id: str, capability: str | None = None) -> DatabaseRef:
    return await require_existing_database_ref(
        TEXT_MEDIA_V1_TYPE, knowledge_base_id, capability=capability
    )


def _text_manager():
    return manager._managers[TEXT_MEDIA_V1_TYPE]


def _connection_epoch(value: Any) -> str:
    try:
        return format(float(value), ".17g")
    except (TypeError, ValueError):
        return ""


def _encode_media_scope(connection: dict[str, Any]) -> str:
    payload = {
        "v": MEDIA_SCOPE_VERSION,
        "adapter_id": str(connection.get("adapter_id") or ""),
        "instance_id": str(connection.get("instance_id") or ""),
        "connected_at": _connection_epoch(connection.get("connected_at")),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("=")


def _decode_media_scope(scope: str) -> dict[str, str]:
    if not scope or len(scope) > 2048:
        raise ValueError("invalid media scope")
    try:
        decoded = base64.urlsafe_b64decode(scope + "=" * (-len(scope) % 4))
        payload = json.loads(decoded)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid media scope") from exc
    if not isinstance(payload, dict) or payload.get("v") != MEDIA_SCOPE_VERSION:
        raise ValueError("invalid media scope")
    result = {
        "adapter_id": str(payload.get("adapter_id") or ""),
        "instance_id": str(payload.get("instance_id") or ""),
        "connected_at": str(payload.get("connected_at") or ""),
    }
    if not all(result.values()):
        raise ValueError("invalid media scope")
    return result


def _media_signature(
    knowledge_base_id: str,
    asset_id: str,
    variant: str,
    expires: int,
    *,
    scope: str = "",
) -> str:
    payload = (
        f"{TEXT_MEDIA_V1_TYPE}\0{knowledge_base_id}\0{asset_id}\0{variant}\0{expires}"
    ).encode("utf-8")
    if scope:
        payload += b"\0" + scope.encode("ascii")
    return hmac.new(
        config.library_psk_secret.encode("utf-8"), payload, hashlib.sha256
    ).hexdigest()


def _adapter_connection_changed_detail(
    knowledge_base_id: str,
    adapter_id: str,
) -> dict[str, Any]:
    ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, knowledge_base_id)
    return {
        "code": "adapter_connection_changed",
        "message": "Adapter connection is no longer active",
        **database_identity_fields(ref, include_deprecated=True),
        "adapter_id": adapter_id,
    }


async def _active_media_adapter_connection(
    knowledge_base_id: str,
    *,
    adapter_id: str,
    instance_id: str,
    connected_at: str | None = None,
) -> dict[str, Any]:
    ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, knowledge_base_id)
    connection = await manager.control.adapter_connection(ref, adapter_id)
    if connection and connection.get("state") == "forced_offline":
        raise HTTPException(
            status_code=409,
            detail=_adapter_forced_offline_detail(connection),
        )
    active = bool(
        connection
        and connection.get("state") == "active"
        and str(connection.get("instance_id") or "") == instance_id
        and float(connection.get("last_seen") or 0)
        >= time.time() - ADAPTER_CONNECTION_TTL_SECONDS
    )
    if connected_at is not None:
        active = bool(
            active
            and connection
            and _connection_epoch(connection.get("connected_at")) == connected_at
        )
    if not active:
        raise HTTPException(
            status_code=409,
            detail=_adapter_connection_changed_detail(knowledge_base_id, adapter_id),
        )
    return connection


def _database_access_key_authenticated(request: Request, knowledge_base_id: str) -> bool:
    authorization = str(request.headers.get("authorization") or "")
    if not authorization.lower().startswith("bearer "):
        return False
    candidate = authorization[7:].strip()
    driver = database_type_registry.require(TEXT_MEDIA_V1_TYPE)
    return driver.verify_access_key(
        config.library_psk_secret,
        knowledge_base_id,
        candidate,
    )


async def _require_media_access(
    request: Request,
    *,
    knowledge_base_id: str,
    asset_id: str,
    variant: str,
) -> bool:
    now = int(time.time())
    try:
        expires = int(request.query_params.get("expires") or 0)
    except ValueError:
        expires = 0
    signature = str(request.query_params.get("signature") or "")
    scope = str(request.query_params.get("scope") or "")
    if (
        expires >= now
        and expires <= now + 600
        and signature
        and hmac.compare_digest(
            signature,
            _media_signature(
                knowledge_base_id,
                asset_id,
                variant,
                expires,
                scope=scope,
            ),
        )
    ):
        if scope:
            try:
                connection_scope = _decode_media_scope(scope)
            except ValueError as exc:
                raise HTTPException(
                    status_code=401,
                    detail="invalid or expired media URL",
                ) from exc
            await _active_media_adapter_connection(
                knowledge_base_id,
                adapter_id=connection_scope["adapter_id"],
                instance_id=connection_scope["instance_id"],
                connected_at=connection_scope["connected_at"],
            )
            return True
        return False
    await require_auth(
        request,
        authorization=request.headers.get("authorization"),
        session=request.cookies.get(COOKIE_NAME),
    )
    return False


def _expire_tokens(registry: dict[str, dict[str, Any]]) -> None:
    cutoff = time.time() - TOKEN_TTL_SECONDS
    for token, item in list(registry.items()):
        if float(item.get("created_at") or 0) >= cutoff:
            continue
        path = Path(str(item.get("path") or ""))
        path.unlink(missing_ok=True)
        registry.pop(token, None)


@router.post("", dependencies=[Depends(require_admin_auth)])
async def create_database(payload: CreateRequest):
    try:
        return await manager.create_library(
            {**payload.model_dump(), "database_type": TEXT_MEDIA_V1_TYPE}
        )
    except ValueError as exc:
        raise HTTPException(409 if "已存在" in str(exc) else 400, str(exc)) from exc


@router.get("/{knowledge_base_id}", dependencies=[Depends(require_auth)])
async def database_detail(knowledge_base_id: str):
    ref = await _ref(knowledge_base_id)
    return await manager.library_detail(ref)


@router.patch("/{knowledge_base_id}", dependencies=[Depends(require_admin_auth)])
async def update_database(knowledge_base_id: str, payload: UpdateRequest):
    ref = await _ref(knowledge_base_id)
    try:
        return await manager.update_library(
            ref, payload.model_dump(exclude_unset=True)
        )
    except ValueError as exc:
        detail = str(exc)
        status = 409 if any(word in detail for word in ("已存在", "连接", "任务")) else 400
        raise HTTPException(status, detail) from exc


@router.put(
    "/{knowledge_base_id}/visual-intent-policy",
    dependencies=[Depends(require_admin_auth)],
)
async def update_visual_intent_policy(
    knowledge_base_id: str,
    payload: VisualIntentPolicyRequest,
):
    await _ref(knowledge_base_id)

    async def operation(_progress):
        return await _text_manager().update_visual_intent_policy(
            knowledge_base_id,
            payload.model_dump(),
        )

    try:
        return await _run_text_media_short(
            "text_media_visual_intent_policy_update",
            knowledge_base_id=knowledge_base_id,
            operation=operation,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post(
    "/{knowledge_base_id}/visual-intent-policy/export",
    dependencies=[Depends(require_admin_auth)],
)
async def export_visual_intent_policy(
    knowledge_base_id: str,
    payload: VisualIntentPolicyExportRequest,
):
    await _ref(knowledge_base_id)
    categories = tuple(
        key for key in VISUAL_INTENT_POLICY_KEYS if key in payload.categories
    )
    try:
        content = serialize_visual_intent_policy_csv(
            payload.policy.model_dump(),
            categories=categories,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    filename = f"{knowledge_base_id}-visual-intent-policy.csv"
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post(
    "/{knowledge_base_id}/visual-intent-policy/imports/inspect",
    dependencies=[Depends(require_admin_auth)],
)
async def inspect_visual_intent_policy_import(
    knowledge_base_id: str,
    file: UploadFile = File(...),
):
    await _ref(knowledge_base_id)
    filename = Path(file.filename or "").name
    if Path(filename).suffix.casefold() != ".csv":
        raise HTTPException(400, "visual intent policy import only accepts .csv")
    try:
        content = await file.read(MAX_VISUAL_INTENT_POLICY_CSV_BYTES + 1)
        policy, categories = parse_visual_intent_policy_csv(content)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        await file.close()
    return {
        "filename": filename,
        "categories": [
            {"key": key, "count": len(policy[key])} for key in categories
        ],
        "policy": policy,
    }


@router.delete("/{knowledge_base_id}", dependencies=[Depends(require_admin_auth)])
async def delete_database(knowledge_base_id: str):
    ref = await _ref(knowledge_base_id)
    try:
        return await manager.delete_library(ref)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/{knowledge_base_id}/copy", dependencies=[Depends(require_admin_auth)])
async def copy_database(knowledge_base_id: str):
    await _ref(knowledge_base_id, "copy")

    async def operation(progress):
        return await _text_manager().copy_library(knowledge_base_id, progress)

    try:
        job_id = await jobs().start(
            "library_copy",
            operation,
            database_id=knowledge_base_id,
            database_type=TEXT_MEDIA_V1_TYPE,
            lease_runtime=False,
        )
        return {"job_id": job_id}
    except KeyError as exc:
        raise HTTPException(404, "knowledge library not found") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post(
    "/{knowledge_base_id}/backup",
    dependencies=[Depends(require_admin_auth)],
    status_code=202,
)
async def backup_database(knowledge_base_id: str):
    await _ref(knowledge_base_id, "backup")
    try:
        job_id = await jobs().start(
            "library_backup",
            lambda _progress: _text_manager().backup_library(knowledge_base_id),
            database_id=knowledge_base_id,
            database_type=TEXT_MEDIA_V1_TYPE,
            lease_runtime=False,
        )
        return {"job_id": job_id}
    except KeyError as exc:
        raise HTTPException(404, "knowledge library not found") from exc


@router.post("/{knowledge_base_id}/access-key", dependencies=[Depends(require_admin_auth)])
async def access_key(knowledge_base_id: str, payload: AccessKeyRequest):
    ref = await _ref(knowledge_base_id)
    if not auth.password_enabled:
        raise HTTPException(400, "请先设置 WebUI 登录密码")
    if not await run_blocking(verify_password, payload.password, auth.password_hash):
        raise HTTPException(401, "invalid password")
    driver = database_type_registry.require(TEXT_MEDIA_V1_TYPE)
    return {
        **database_identity_fields(ref),
        "database_category": "knowledge",
        "key_prefix": "pkb-",
        "access_key": driver.derive_access_key(config.library_psk_secret, ref.id),
        "adapter_url": _current_adapter_connection_url(),
    }


@router.get("/{knowledge_base_id}/documents", dependencies=[Depends(require_auth)])
async def documents(
    knowledge_base_id: str,
    query: str = Query(default="", max_length=500),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
    sort: Literal["created_desc", "created_asc", "title_asc", "title_desc"] = "created_desc",
):
    service = await runtime(await _ref(knowledge_base_id, "content_management"))
    return await service.storage.list_document_summaries(
        query=query,
        offset=offset,
        limit=limit,
        sort=sort,
    )


@router.get(
    "/{knowledge_base_id}/documents/{document_id}",
    dependencies=[Depends(require_auth)],
)
async def document_detail(knowledge_base_id: str, document_id: str):
    service = await runtime(await _ref(knowledge_base_id, "content_management"))
    item = await service.storage.get_document_detail(document_id)
    if item is None:
        raise HTTPException(404, "document not found")
    return item


@router.post(
    "/{knowledge_base_id}/documents/batch-delete",
    dependencies=[Depends(require_admin_auth)],
    status_code=202,
)
async def delete_documents_batch(
    knowledge_base_id: str,
    payload: BatchDocumentDeleteRequest,
):
    await _ref(knowledge_base_id, "content_management")
    job_id = await _start_text_media_resumable(
        "text_media_document_delete",
        knowledge_base_id=knowledge_base_id,
        operation={"document_ids": payload.document_ids},
    )
    return {"job_id": job_id, "document_ids": payload.document_ids}


@router.post(
    "/{knowledge_base_id}/documents",
    dependencies=[Depends(require_admin_auth)],
    status_code=202,
)
async def upload_document(
    knowledge_base_id: str,
    file: UploadFile = File(...),
    title: str = Form(default=""),
):
    await _ref(knowledge_base_id, "content_management")
    filename = Path(file.filename or "document.txt").name
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_DOCUMENT_SUFFIXES:
        await file.close()
        raise HTTPException(400, f"unsupported document format: {filename}")
    input_root = _task_input_root(knowledge_base_id)
    target = input_root / f"document{suffix}"
    try:
        await save_upload_file(file, target, max_bytes=20 * 1024 * 1024)
        job_id = await _start_text_media_resumable(
            "text_media_document_ingest",
            knowledge_base_id=knowledge_base_id,
            operation={
                "input_root": str(input_root),
                "document_path": str(target),
                "filename": filename,
                "title": title,
            },
        )
    except Exception as exc:
        await run_blocking(shutil.rmtree, input_root, True)
        raise HTTPException(400, str(exc)) from exc
    return {"job_id": job_id, "filename": filename}


@router.post(
    "/{knowledge_base_id}/ingest-batches",
    dependencies=[Depends(require_admin_auth)],
    status_code=202,
)
async def ingest_batch(
    knowledge_base_id: str,
    manifest: str = Form(...),
    documents: list[UploadFile] | None = File(None, alias="documents[]"),
    images: list[UploadFile] | None = File(None, alias="images[]"),
):
    await _ref(knowledge_base_id, "content_management")
    document_files = list(documents or [])
    image_files = list(images or [])
    if not document_files and not image_files:
        raise HTTPException(400, "batch requires at least one document or image")
    if len(document_files) > 10 or len(image_files) > 10:
        raise HTTPException(400, "batch accepts at most 10 documents and 10 images")
    try:
        parsed = IngestBatchManifest.model_validate(json.loads(manifest))
    except Exception as exc:
        for upload in [*document_files, *image_files]:
            await upload.close()
        raise HTTPException(400, f"invalid ingest batch manifest: {exc}") from exc
    if parsed.chunk_overlap > parsed.chunk_target // 2:
        for upload in [*document_files, *image_files]:
            await upload.close()
        raise HTTPException(400, "chunk overlap cannot exceed half of chunk target")
    if len(parsed.images) != len(image_files):
        for upload in [*document_files, *image_files]:
            await upload.close()
        raise HTTPException(400, "image mapping count does not match uploaded images")
    for mapping in parsed.images:
        indexes = set(mapping.document_indexes)
        if indexes and (min(indexes) < 0 or max(indexes) >= len(document_files)):
            for upload in [*document_files, *image_files]:
                await upload.close()
            raise HTTPException(400, "image document mapping is out of range")

    driver = database_type_registry.require(TEXT_MEDIA_V1_TYPE)
    active = await jobs().active_long_job(driver.resource_key(knowledge_base_id))
    if active:
        for upload in [*document_files, *image_files]:
            await upload.close()
        raise HTTPException(409, "knowledge library already has an active task")

    batch_id = uuid.uuid4().hex
    staging = _task_input_root(knowledge_base_id, batch_id)
    staged_documents: list[dict[str, Any]] = []
    staged_images: list[dict[str, Any]] = []
    try:
        for index, upload in enumerate(document_files):
            filename = Path(upload.filename or f"document-{index}.txt").name
            suffix = Path(filename).suffix.lower()
            if suffix not in SUPPORTED_DOCUMENT_SUFFIXES:
                raise ValueError(f"unsupported document format: {filename}")
            target = staging / "documents" / f"{index:02d}{suffix}"
            await save_upload_file(upload, target, max_bytes=20 * 1024 * 1024)
            staged_documents.append(
                {"path": str(target), "filename": filename, "title": Path(filename).stem}
            )
        for index, (upload, mapping) in enumerate(
            zip(image_files, parsed.images, strict=True)
        ):
            filename = Path(upload.filename or f"image-{index}").name
            target = staging / "images" / f"{index:02d}.upload"
            await save_upload_file(upload, target, max_bytes=MAX_SOURCE_BYTES)
            staged_images.append(
                {
                    "path": str(target),
                    "filename": filename,
                    "document_indexes": mapping.document_indexes,
                    "media_description": mapping.media_description,
                    "media_descriptions": mapping.media_descriptions,
                }
            )
    except Exception as exc:
        await run_blocking(shutil.rmtree, staging, True)
        for upload in [*document_files, *image_files]:
            await upload.close()
        raise HTTPException(400, str(exc)) from exc

    try:
        job_id = await _start_text_media_resumable(
            "text_media_ingest_batch",
            knowledge_base_id=knowledge_base_id,
            operation={
                "input_root": str(staging),
                "batch_id": batch_id,
                "documents": staged_documents,
                "images": staged_images,
                "chunk_target": parsed.chunk_target,
                "chunk_overlap": parsed.chunk_overlap,
                "embedding_batch_size": parsed.embedding_batch_size,
                "concurrency": parsed.concurrency,
                "max_retries": parsed.max_retries,
                "media_semantic_calibration_enabled": (
                    parsed.media_semantic_calibration_enabled
                ),
            },
        )
    except Exception:
        await run_blocking(shutil.rmtree, staging, True)
        raise
    return {"job_id": job_id, "batch_id": batch_id}


@router.delete(
    "/{knowledge_base_id}/documents/{document_id}",
    dependencies=[Depends(require_admin_auth)],
    status_code=202,
)
async def delete_document(knowledge_base_id: str, document_id: str):
    await _ref(knowledge_base_id, "content_management")
    job_id = await _start_text_media_resumable(
        "text_media_document_delete",
        knowledge_base_id=knowledge_base_id,
        operation={"document_ids": [document_id]},
    )
    return {"job_id": job_id, "document_ids": [document_id]}


@router.get("/{knowledge_base_id}/chunks", dependencies=[Depends(require_auth)])
async def chunks(
    knowledge_base_id: str,
    query: str = Query(default="", max_length=500),
    document_id: str = Query(default="", max_length=128),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
    sort: Literal["ordinal_asc", "ordinal_desc", "id_desc"] = "ordinal_asc",
):
    service = await runtime(await _ref(knowledge_base_id, "content_management"))
    return await service.storage.list_chunk_summaries(
        query=query,
        document_id=document_id,
        offset=offset,
        limit=limit,
        sort=sort,
    )


@router.get(
    "/{knowledge_base_id}/chunks/{chunk_id}",
    dependencies=[Depends(require_auth)],
)
async def chunk_detail(knowledge_base_id: str, chunk_id: int):
    service = await runtime(await _ref(knowledge_base_id, "content_management"))
    item = await service.storage.get_chunk_detail(chunk_id)
    if item is None:
        raise HTTPException(404, "chunk not found")
    for asset in item["associated_media"]:
        asset_id = asset["asset_id"]
        asset["content_url"] = (
            f"/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}"
            f"/assets/{asset_id}/content"
        )
        asset["thumbnail_url"] = (
            f"/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}"
            f"/assets/{asset_id}/thumbnail"
        )
    return item


@router.get("/{knowledge_base_id}/entries", dependencies=[Depends(require_auth)])
async def entries(knowledge_base_id: str):
    service = await runtime(await _ref(knowledge_base_id, "content_management"))
    return {"items": await service.storage.list_entries()}


@router.post(
    "/{knowledge_base_id}/entries",
    dependencies=[Depends(require_admin_auth)],
    status_code=202,
)
async def create_entry(knowledge_base_id: str, payload: EntryRequest):
    await _ref(knowledge_base_id, "content_management")
    input_root = _task_input_root(knowledge_base_id)
    body_path = input_root / "body.txt"
    try:
        await run_blocking(body_path.parent.mkdir, parents=True, exist_ok=True)
        await run_blocking(body_path.write_text, payload.body, encoding="utf-8")
        job_id = await _start_text_media_resumable(
            "text_media_entry_create",
            knowledge_base_id=knowledge_base_id,
            operation={
                "input_root": str(input_root),
                "body_path": str(body_path),
                "title": payload.title,
            },
        )
    except Exception as exc:
        await run_blocking(shutil.rmtree, input_root, True)
        raise HTTPException(400, str(exc)) from exc
    return {"job_id": job_id}


@router.patch(
    "/{knowledge_base_id}/entries/{entry_id}",
    dependencies=[Depends(require_admin_auth)],
    status_code=202,
)
async def update_entry(knowledge_base_id: str, entry_id: str, payload: EntryRequest):
    await _ref(knowledge_base_id, "content_management")
    input_root = _task_input_root(knowledge_base_id)
    body_path = input_root / "body.txt"
    try:
        await run_blocking(body_path.parent.mkdir, parents=True, exist_ok=True)
        await run_blocking(body_path.write_text, payload.body, encoding="utf-8")
        job_id = await _start_text_media_resumable(
            "text_media_entry_update",
            knowledge_base_id=knowledge_base_id,
            operation={
                "input_root": str(input_root),
                "body_path": str(body_path),
                "entry_id": entry_id,
                "title": payload.title,
            },
        )
    except Exception as exc:
        await run_blocking(shutil.rmtree, input_root, True)
        raise HTTPException(400, str(exc)) from exc
    return {"job_id": job_id, "entry_id": entry_id}


@router.delete(
    "/{knowledge_base_id}/entries/{entry_id}",
    dependencies=[Depends(require_admin_auth)],
    status_code=202,
)
async def delete_entry(knowledge_base_id: str, entry_id: str):
    await _ref(knowledge_base_id, "content_management")
    job_id = await _start_text_media_resumable(
        "text_media_entry_delete",
        knowledge_base_id=knowledge_base_id,
        operation={"entry_id": entry_id},
    )
    return {"job_id": job_id, "entry_id": entry_id}


@router.get("/{knowledge_base_id}/assets", dependencies=[Depends(require_auth)])
async def assets(knowledge_base_id: str):
    service = await runtime(await _ref(knowledge_base_id, "image_assets"))
    items = await service.storage.list_assets(kind="image")
    for item in items:
        item["content_url"] = f"/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/assets/{item['id']}/content"
        item["thumbnail_url"] = f"/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/assets/{item['id']}/thumbnail"
    return {"items": items}


@router.get(
    "/{knowledge_base_id}/assets/{asset_id}",
    dependencies=[Depends(require_auth)],
)
async def asset_detail(knowledge_base_id: str, asset_id: str):
    service = await runtime(await _ref(knowledge_base_id, "image_assets"))
    item = await service.storage.get_asset_detail(asset_id)
    if item is None or item.get("kind") != "image":
        raise HTTPException(404, "asset not found")
    item["content_url"] = f"/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/assets/{asset_id}/content"
    item["thumbnail_url"] = f"/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/assets/{asset_id}/thumbnail"
    return item


@router.post("/{knowledge_base_id}/assets/images", dependencies=[Depends(require_admin_auth)])
async def upload_image(
    knowledge_base_id: str,
    file: UploadFile = File(...),
    media_description: str = Form(""),
    media_descriptions: str = Form(""),
):
    ref = await _ref(knowledge_base_id, "image_assets")
    data = await file.read(25 * 1024 * 1024 + 1)
    filename = Path(file.filename or "image").name
    await file.close()
    try:
        parsed_descriptions = (
            json.loads(media_descriptions) if media_descriptions.strip() else []
        )
        if not isinstance(parsed_descriptions, list):
            raise ValueError("media_descriptions must be a JSON array")
        if media_description.strip():
            if parsed_descriptions:
                from .text import normalize_media_description

            if (
                parsed_descriptions
                and normalize_media_description(media_description)
                != normalize_media_description(parsed_descriptions[0])
            ):
                raise ValueError(
                    "media_description must match the first media_descriptions item"
                )
            if not parsed_descriptions:
                parsed_descriptions = [media_description]
        payload = MediaDescriptionsUpdateRequest(
            media_descriptions=parsed_descriptions
        ).media_descriptions if parsed_descriptions else None
    except Exception as exc:
        raise HTTPException(422, str(exc)) from exc

    async def operation(_progress):
        return await (await runtime(ref)).upload_image(
            filename=filename,
            data=data,
            media_descriptions=payload,
        )

    try:
        return await _run_text_media_short(
            "text_media_image_upload",
            knowledge_base_id=knowledge_base_id,
            operation=operation,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.put(
    "/{knowledge_base_id}/assets/{asset_id}/media-descriptions",
    dependencies=[Depends(require_admin_auth)],
    status_code=202,
)
async def update_asset_media_descriptions(
    knowledge_base_id: str,
    asset_id: str,
    payload: MediaDescriptionsUpdateRequest,
):
    await _ref(knowledge_base_id, "image_assets")
    job_id = await _start_text_media_resumable(
        "text_media_media_descriptions_update",
        knowledge_base_id=knowledge_base_id,
        operation={
            "asset_id": asset_id,
            "media_descriptions": payload.media_descriptions,
        },
    )
    return {"job_id": job_id}


@router.delete(
    "/{knowledge_base_id}/assets/{asset_id}",
    dependencies=[Depends(require_admin_auth)],
)
async def delete_image(knowledge_base_id: str, asset_id: str):
    ref = await _ref(knowledge_base_id, "image_assets")
    async def operation(_progress):
        return await (await runtime(ref)).delete_image(asset_id)

    try:
        return await _run_text_media_short(
            "text_media_image_delete",
            knowledge_base_id=knowledge_base_id,
            operation=operation,
        )
    except (KeyError, RuntimeError) as exc:
        raise HTTPException(404, "asset not found") from exc


@router.put(
    "/{knowledge_base_id}/relations/{scope}/{target_id}/assets/{asset_id}",
    dependencies=[Depends(require_admin_auth)],
)
async def link_asset(
    knowledge_base_id: str,
    scope: Literal["document", "entry", "chunk"],
    target_id: str,
    asset_id: str,
    payload: RelationRequest,
):
    ref = await _ref(knowledge_base_id, "image_assets")
    normalized_target: str | int = int(target_id) if scope == "chunk" else target_id
    async def operation(_progress):
        service = await runtime(ref)
        return await service.storage.link_asset(
            scope=scope,
            target_id=normalized_target,
            asset_id=asset_id,
            payload=payload.model_dump(),
        )

    try:
        return await _run_text_media_short(
            "text_media_relation_update",
            knowledge_base_id=knowledge_base_id,
            operation=operation,
        )
    except (KeyError, RuntimeError) as exc:
        raise HTTPException(404, "asset or relation target not found") from exc


@router.delete(
    "/{knowledge_base_id}/relations/{scope}/{target_id}/assets/{asset_id}",
    dependencies=[Depends(require_admin_auth)],
)
async def unlink_asset(
    knowledge_base_id: str,
    scope: Literal["document", "entry", "chunk"],
    target_id: str,
    asset_id: str,
):
    ref = await _ref(knowledge_base_id, "image_assets")
    normalized_target: str | int = int(target_id) if scope == "chunk" else target_id
    async def operation(_progress):
        service = await runtime(ref)
        deleted = await service.storage.unlink_asset(
            scope=scope, target_id=normalized_target, asset_id=asset_id
        )
        if not deleted:
            raise KeyError((scope, normalized_target, asset_id))
        return {
            "scope": scope,
            "target_id": normalized_target,
            "asset_id": asset_id,
            "deleted": True,
        }

    try:
        return await _run_text_media_short(
            "text_media_relation_delete",
            knowledge_base_id=knowledge_base_id,
            operation=operation,
        )
    except (KeyError, RuntimeError) as exc:
        raise HTTPException(404, "relation not found") from exc


@router.get(
    "/{knowledge_base_id}/media-calibrations",
    dependencies=[Depends(require_admin_auth)],
)
async def list_media_calibrations(knowledge_base_id: str):
    service = await runtime(await _ref(knowledge_base_id, "image_assets"))
    return {"items": await service.storage.list_document_media_calibrations()}


@router.put(
    "/{knowledge_base_id}/document-media-relations/{document_id}/{asset_id}/semantic-calibration",
    dependencies=[Depends(require_admin_auth)],
    status_code=202,
)
async def recalibrate_document_media(
    knowledge_base_id: str,
    document_id: str,
    asset_id: str,
    payload: SemanticCalibrationRequest,
):
    await _ref(knowledge_base_id, "image_assets")
    job_id = await _start_text_media_resumable(
        "text_media_media_calibration",
        knowledge_base_id=knowledge_base_id,
        operation={
            "document_id": document_id,
            "asset_id": asset_id,
            "enabled": payload.enabled,
            "media_description": payload.media_description,
        },
    )
    return {"job_id": job_id}


@router.post("/{knowledge_base_id}/search", dependencies=[Depends(require_auth)])
async def search(
    knowledge_base_id: str,
    payload: SearchRequest,
    request: Request,
):
    request_id = log_search_request_received(
        request,
        knowledge_base_id=knowledge_base_id,
        payload=payload,
    )
    started = time.perf_counter()
    try:
        service = await runtime(await _ref(knowledge_base_id, "search"))
        result = await service.search(
            payload.query,
            top_k=payload.top_k,
            media_output_confidence_threshold=(
                payload.media_output_confidence_threshold
            ),
            media_relevance_pivot=payload.media_relevance_pivot,
            media_score_threshold=payload.media_score_threshold,
            max_media_outputs=payload.max_media_outputs,
            rerank=payload.rerank,
            retrieval_mode=payload.retrieval_mode,
        )
    except ValueError as exc:
        status_code = 409 if "绑定" in str(exc) else 400
        log_search_rejected(
            request_id=request_id,
            knowledge_base_id=knowledge_base_id,
            status_code=status_code,
            error=exc,
        )
        raise HTTPException(status_code, str(exc)) from exc
    except Exception as exc:
        log_search_failed(
            request_id=request_id,
            knowledge_base_id=knowledge_base_id,
            error=exc,
        )
        raise
    log_search_completed(
        request_id=request_id,
        knowledge_base_id=knowledge_base_id,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        result=result,
    )
    if payload.media_response_mode == "descriptions_only":
        result["media_candidates"] = [
            {
                "rank": rank,
                "asset_id": str(asset.get("asset_id") or ""),
                "original_name": str(asset.get("original_name") or ""),
                "media_description": str(
                    asset.get("media_description") or ""
                ),
                "matched_media_description": str(
                    asset.get("matched_media_description") or ""
                ),
                "output_policy": str(asset.get("output_policy") or "auto"),
                "output_confidence": asset.get("output_confidence"),
                "media_relevance": asset.get(
                    "media_relevance",
                    asset.get("association_score"),
                ),
                "reason": str(asset.get("reason") or ""),
                "fetchable": bool(
                    asset.get("asset_id")
                    and asset.get("output_policy") != "metadata_only"
                ),
            }
            for rank, asset in enumerate(
                result.get("media_outputs") or [],
                start=1,
            )
            if isinstance(asset, dict)
        ]
        result["media_outputs"] = []
        result["media_decisions"] = []
        return result
    for asset in result["media_outputs"]:
        asset_id = asset["asset_id"]
        if asset.get("output_policy") != "metadata_only":
            asset["content_url"] = f"/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/assets/{asset_id}/content"
            asset["thumbnail_url"] = f"/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/assets/{asset_id}/thumbnail"
    for asset in result["media_decisions"]:
        asset_id = asset["asset_id"]
        asset["thumbnail_url"] = f"/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/assets/{asset_id}/thumbnail"
        if asset.get("output_policy") != "metadata_only":
            asset["content_url"] = f"/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/assets/{asset_id}/content"
    return result


@router.post(
    "/{knowledge_base_id}/assets/{asset_id}/signed-url",
    dependencies=[Depends(require_auth)],
)
async def signed_asset_url(
    request: Request,
    knowledge_base_id: str,
    asset_id: str,
    payload: SignedUrlRequest,
):
    service = await runtime(await _ref(knowledge_base_id, "image_assets"))
    asset = await service.storage.get_asset(asset_id)
    if not asset or asset["kind"] != "image":
        raise HTTPException(404, "asset not found")
    scope = ""
    if _database_access_key_authenticated(request, knowledge_base_id):
        adapter_id = _adapter_header(request, ADAPTER_ID_HEADER)
        instance_id = _adapter_header(request, ADAPTER_INSTANCE_HEADER)
        if not adapter_id or not instance_id:
            raise HTTPException(
                status_code=400,
                detail="adapter id and instance id headers are required",
            )
        connection = await _active_media_adapter_connection(
            knowledge_base_id,
            adapter_id=adapter_id,
            instance_id=instance_id,
        )
        scope = _encode_media_scope(connection)
        expires = int(time.time()) + 300
    else:
        expires = int(time.time()) + payload.expires_in
    signature = _media_signature(
        knowledge_base_id,
        asset_id,
        payload.variant,
        expires,
        scope=scope,
    )
    query = f"expires={expires}&signature={signature}"
    if scope:
        query = f"expires={expires}&scope={scope}&signature={signature}"
    return {
        "url": (
            f"/api/v1/knowledge-libraries/text_media_v1/{knowledge_base_id}/assets/"
            f"{asset_id}/{payload.variant}?{query}"
        ),
        "expires": expires,
    }


@router.post(
    "/{knowledge_base_id}/indexes/rebuild",
    dependencies=[Depends(require_admin_auth)],
    status_code=202,
)
async def rebuild_index(
    knowledge_base_id: str,
    payload: RebuildRequest | None = None,
):
    ref = await _ref(knowledge_base_id, "index_rebuild")
    detail = await manager.library_detail(ref)
    request = payload or RebuildRequest()
    provider_id = request.provider_id or str(detail.get("provider_id") or "")
    target = await _text_manager().control.get_provider(provider_id)
    if (
        target is None
        or not target.config.enabled
        or provider_kind(target.config.type) != "embedding"
    ):
        raise HTTPException(400, "target Embedding Provider is unavailable")
    job_id = await jobs().start_resumable(
        "text_media_index_rebuild",
        {
            "provider_id": provider_id,
            "provider_revision": target.revision,
            "provider_fingerprint": target.config_sha256,
            "reason": request.reason,
        },
        database_id=knowledge_base_id,
        database_type=TEXT_MEDIA_V1_TYPE,
        dedupe_active=False,
        lease_runtime=False,
    )
    return {
        "job_id": job_id,
        "database_type": TEXT_MEDIA_V1_TYPE,
        "database_id": knowledge_base_id,
        "provider_id": provider_id,
    }


@router.get("/{knowledge_base_id}/assets/{asset_id}/content")
async def asset_content(request: Request, knowledge_base_id: str, asset_id: str):
    adapter_scoped = await _require_media_access(
        request,
        knowledge_base_id=knowledge_base_id,
        asset_id=asset_id,
        variant="content",
    )
    service = await runtime(await _ref(knowledge_base_id, "image_assets"))
    asset = await service.storage.get_asset(asset_id)
    if not asset or asset["kind"] != "image":
        raise HTTPException(404, "asset not found")
    path = service.root.joinpath(*PurePosixPath(asset["storage_key"]).parts)
    return FileResponse(
        path,
        media_type="image/webp",
        headers={
            "Cache-Control": (
                "private, no-store"
                if adapter_scoped
                else "private, max-age=3600"
            ),
            "ETag": f'"{asset["sha256"]}"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/{knowledge_base_id}/assets/{asset_id}/thumbnail")
async def asset_thumbnail(request: Request, knowledge_base_id: str, asset_id: str):
    adapter_scoped = await _require_media_access(
        request,
        knowledge_base_id=knowledge_base_id,
        asset_id=asset_id,
        variant="thumbnail",
    )
    service = await runtime(await _ref(knowledge_base_id, "image_assets"))
    asset = await service.storage.get_asset(asset_id)
    if not asset or asset["kind"] != "image":
        raise HTTPException(404, "asset not found")
    path = service.root / "derived" / "previews" / f"{asset['sha256']}.webp"
    if not path.exists():
        raise HTTPException(404, "thumbnail not found")
    return FileResponse(
        path,
        media_type="image/webp",
        headers={
            "Cache-Control": (
                "private, no-store"
                if adapter_scoped
                else "private, max-age=3600"
            ),
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post("/{knowledge_base_id}/exports", dependencies=[Depends(require_admin_auth)])
async def create_export(knowledge_base_id: str):
    service = await runtime(await _ref(knowledge_base_id, "tmkb_export"))
    _expire_tokens(_export_tokens)
    token = secrets.token_urlsafe(24)
    path = current_context().state_root / "data" / "tmkb_exports" / f"{knowledge_base_id}-{int(time.time())}.tmkb"

    async def operation(progress):
        await progress(0.1, "正在创建一致性知识库快照")
        result = await export_tmkb(service=service, target=path)
        _export_tokens[token] = {"path": str(path), "created_at": time.time(), "database_id": knowledge_base_id}
        await progress(1.0, "知识库封包已生成")
        return {**result, "download_token": token}

    job_id = await jobs().start(
        "tmkb_export",
        operation,
        database_id=knowledge_base_id,
        database_type=TEXT_MEDIA_V1_TYPE,
        lease_runtime=False,
    )
    return {"job_id": job_id}


@router.get("/{knowledge_base_id}/exports/{token}", dependencies=[Depends(require_admin_auth)])
async def download_export(knowledge_base_id: str, token: str, background_tasks: BackgroundTasks):
    _expire_tokens(_export_tokens)
    item = _export_tokens.pop(token, None)
    if not item or item["database_id"] != knowledge_base_id:
        raise HTTPException(404, "export not found or expired")
    path = Path(item["path"])
    background_tasks.add_task(path.unlink, missing_ok=True)
    return FileResponse(path, filename=path.name, media_type="application/vnd.personalityrag.tmkb", background=background_tasks)


@router.post("/imports/inspect", dependencies=[Depends(require_admin_auth)])
async def inspect_import(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    if Path(filename).suffix.lower() != ".tmkb":
        await file.close()
        raise HTTPException(400, "请选择 .tmkb 知识库封包")
    _expire_tokens(_import_tokens)
    token = secrets.token_urlsafe(24)
    upload_dir = current_context().state_root / "data" / "import_uploads" / TEXT_MEDIA_V1_TYPE
    path = upload_dir / f"{token}.tmkb"
    try:
        await save_upload_file(file, path, max_bytes=MAX_PACKAGE_BYTES)
        manifest = await run_blocking(inspect_tmkb, path)
        with tempfile.TemporaryDirectory(prefix="tmkb-inspect-") as raw:
            await run_blocking(extract_tmkb, path, Path(raw))
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(400, str(exc)) from exc
    _import_tokens[token] = {"path": str(path), "created_at": time.time(), "manifest": manifest}
    conflict = bool(await manager.control.database_identity(DatabaseRef(TEXT_MEDIA_V1_TYPE, str(manifest["database_id"]))))
    return {"import_token": token, "manifest": manifest, "id_conflict": conflict, "unencrypted": True}


@router.post("/imports/{token}/commit", dependencies=[Depends(require_admin_auth)])
async def commit_import(token: str, payload: ImportCommitRequest):
    _expire_tokens(_import_tokens)
    item = _import_tokens.pop(token, None)
    if not item:
        raise HTTPException(404, "import not found or expired")
    path = Path(item["path"])

    try:
        job_id = await jobs().start_resumable(
            "tmkb_import",
            {
                "package_path": str(path),
                "target_id": payload.target_id,
                "name": payload.name,
            },
            database_id=payload.target_id,
            database_type=TEXT_MEDIA_V1_TYPE,
            dedupe_active=False,
            lease_runtime=False,
        )
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return {"job_id": job_id}


@router.post(
    "/transfer-batches/exports", dependencies=[Depends(require_admin_auth)]
)
async def create_batch_export(payload: BatchExportRequest):
    for knowledge_base_id in payload.database_ids:
        await _ref(knowledge_base_id, "tmkb_export")
    _expire_tokens(_batch_export_tokens)
    token = secrets.token_urlsafe(24)
    extension = ".tmkb" if len(payload.database_ids) == 1 else ".tmkbs"
    filename = f"text-media-{int(time.time())}{extension}"
    path = current_context().state_root / "data" / "tmkb_exports" / filename

    async def operation(progress):
        if len(payload.database_ids) == 1:
            knowledge_base_id = payload.database_ids[0]
            await progress(0.1, "creating a consistent knowledge library snapshot")
            service = await _text_manager().get_runtime(knowledge_base_id)
            result = await export_tmkb(service=service, target=path)
        else:
            await progress(0.05, "creating consistent knowledge library snapshots")
            result = await export_tmkbs(
                manager=_text_manager(),
                database_ids=payload.database_ids,
                target=path,
                progress=progress,
            )
        _batch_export_tokens[token] = {
            "path": str(path),
            "created_at": time.time(),
            "database_ids": payload.database_ids,
        }
        await progress(1.0, "knowledge library transfer package is ready")
        return {**result, "download_token": token}

    job_id = await jobs().start(
        "tmkbs_export",
        operation,
        database_id=None,
        database_type=TEXT_MEDIA_V1_TYPE,
        dedupe_active=False,
        lease_runtime=False,
    )
    return {"job_id": job_id}


@router.get(
    "/transfer-batches/exports/{token}", dependencies=[Depends(require_admin_auth)]
)
async def download_batch_export(token: str, background_tasks: BackgroundTasks):
    _expire_tokens(_batch_export_tokens)
    item = _batch_export_tokens.pop(token, None)
    if not item:
        raise HTTPException(404, "export not found or expired")
    path = Path(item["path"])
    if not path.is_file():
        raise HTTPException(404, "export file not found")
    background_tasks.add_task(path.unlink, missing_ok=True)
    media_type = (
        "application/vnd.personalityrag.tmkbs"
        if path.suffix.lower() == ".tmkbs"
        else "application/vnd.personalityrag.tmkb"
    )
    return FileResponse(
        path,
        filename=path.name,
        media_type=media_type,
        background=background_tasks,
    )


@router.post(
    "/transfer-batches/imports/inspect", dependencies=[Depends(require_admin_auth)]
)
async def inspect_batch_import(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    if Path(filename).suffix.lower() != ".tmkbs":
        await file.close()
        raise HTTPException(400, "select a .tmkbs knowledge library batch package")
    _expire_tokens(_batch_import_tokens)
    token = secrets.token_urlsafe(24)
    upload_dir = (
        current_context().state_root
        / "data"
        / "import_uploads"
        / TEXT_MEDIA_V1_TYPE
    )
    path = upload_dir / f"{token}.tmkbs"
    try:
        await save_upload_file(file, path, max_bytes=MAX_BATCH_PACKAGE_BYTES)
        manifest = await run_blocking(inspect_tmkbs, path)
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(400, str(exc)) from exc
    previews = []
    for item in manifest["libraries"]:
        child = dict(item.get("manifest") or {})
        source_id = str(item["database_id"])
        previews.append(
            {
                "source_id": source_id,
                "name": str(child.get("name") or item.get("name") or source_id),
                "description": str(child.get("description") or ""),
                "counts": dict(child.get("counts") or {}),
                "provider": dict(child.get("provider") or {}),
                "size_bytes": int(item["size"]),
                "id_conflict": bool(
                    await manager.control.database_identity(
                        DatabaseRef(TEXT_MEDIA_V1_TYPE, source_id)
                    )
                ),
            }
        )
    _batch_import_tokens[token] = {
        "path": str(path),
        "created_at": time.time(),
        "source_ids": [item["source_id"] for item in previews],
    }
    return {
        "import_token": token,
        "libraries": previews,
        "unencrypted": True,
    }


@router.post(
    "/transfer-batches/imports/{token}/commit",
    dependencies=[Depends(require_admin_auth)],
)
async def commit_batch_import(token: str, payload: BatchImportCommitRequest):
    _expire_tokens(_batch_import_tokens)
    item = _batch_import_tokens.pop(token, None)
    if not item:
        raise HTTPException(404, "import not found or expired")
    requested = [entry.model_dump() for entry in payload.libraries]
    if not set(entry["source_id"] for entry in requested).issubset(set(item["source_ids"])):
        _batch_import_tokens[token] = item
        raise HTTPException(400, "batch import selection is not present in the inspected package")
    path = Path(item["path"])

    try:
        job_id = await jobs().start_resumable(
            "tmkbs_import",
            {"package_path": str(path), "items": requested},
            database_id=None,
            database_type=TEXT_MEDIA_V1_TYPE,
            dedupe_active=False,
            lease_runtime=False,
        )
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return {"job_id": job_id}


# Static transfer-batch paths must precede the legacy database-id export paths,
# otherwise Starlette would interpret "transfer-batches" as a database ID.
_batch_routes = [
    route for route in router.routes if "/transfer-batches/" in route.path
]
router.routes[:] = _batch_routes + [
    route for route in router.routes if route not in _batch_routes
]
