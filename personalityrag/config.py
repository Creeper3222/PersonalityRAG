from __future__ import annotations

import hashlib
import json
import os
import secrets
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .resource_limits import (
    DEFAULT_PERFORMANCE_PROFILE,
    normalize_performance_profile,
)


DEFAULT_ACCESS_BASE_URL = "http://127.0.0.1"
DEFAULT_PUBLIC_ADAPTER_URL = ""
LOOPBACK_BIND_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
DEPLOYMENT_ENV = "PERSONALITYRAG_DEPLOYMENT"
DOCKER_DEPLOYMENT = "docker"
DOCKER_HOST = "0.0.0.0"
DOCKER_WEBUI_PORT = 8765
DOCKER_ACCESS_PORT = 8766
DOCKER_MANAGED_CONFIG_FIELDS = ("host", "port", "access_port")


def deployment_mode() -> str:
    return str(os.environ.get(DEPLOYMENT_ENV, "desktop") or "desktop").strip().lower()


def is_docker_deployment() -> bool:
    return deployment_mode() == DOCKER_DEPLOYMENT


def normalize_bind_host(value: str | None) -> str:
    host = str(value or "127.0.0.1").strip().lower()
    if is_docker_deployment() and host == DOCKER_HOST:
        return host
    if host not in LOOPBACK_BIND_HOSTS:
        raise ValueError(
            "host must be a loopback address; expose adapter access through "
            "an HTTPS reverse proxy"
        )
    return host


def normalize_access_base_url(value: str | None) -> str:
    raw = (value or DEFAULT_ACCESS_BASE_URL).strip().rstrip("/")
    if not raw:
        return DEFAULT_ACCESS_BASE_URL
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("access_base_url must be an http(s) URL host")
    if parsed.username or parsed.password:
        raise ValueError("access_base_url must not include credentials")
    try:
        explicit_port = parsed.port is not None
    except ValueError as exc:
        raise ValueError("access_base_url has an invalid port") from exc
    if explicit_port:
        raise ValueError("access_base_url must not include a port")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("access_base_url must not include a path, query, or fragment")
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"{parsed.scheme}://{host}"


def build_access_url(base_url: str, port: int) -> str:
    return f"{normalize_access_base_url(base_url)}:{port}/"


def normalize_public_adapter_url(value: str | None) -> str:
    raw = (value or DEFAULT_PUBLIC_ADAPTER_URL).strip().rstrip("/")
    if not raw:
        return DEFAULT_PUBLIC_ADAPTER_URL
    parsed = urlsplit(raw)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("public_adapter_url must be an https URL origin")
    if parsed.username or parsed.password:
        raise ValueError("public_adapter_url must not include credentials")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("public_adapter_url has an invalid port") from exc
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError(
            "public_adapter_url must not include a path, query, or fragment"
        )
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port_suffix = f":{port}" if port is not None else ""
    return f"https://{host}{port_suffix}"


def build_adapter_connection_url(
    config: "AppConfig",
    *,
    access_port: int | None = None,
) -> str:
    if config.public_adapter_url:
        return normalize_public_adapter_url(config.public_adapter_url)
    return build_access_url(
        config.access_base_url,
        access_port if access_port is not None else config.access_port,
    ).rstrip("/")


@dataclass(slots=True)
class IndexRebuildSettings:
    batch_size: int = 50
    embedding_batch_size: int = 8
    tasks_limit: int = 1
    max_retries: int = 5
    retry_base_delay: float = 30.0
    batch_delay: float = 5.0
    request_delay: float = 5.0
    max_failure_ratio: float = 0.02


@dataclass(slots=True)
class ProviderConfig:
    id: str = "vllm_embedding"
    display_name: str = "本机 bge-m3"
    type: str = "vllm_embedding"
    enabled: bool = True
    api_base: str = "http://127.0.0.1:8001/v1"
    api_key: str = "vllm"
    model: str = "BAAI/bge-m3"
    context_length_mode: str = "auto"
    dimensions: int = 1024
    max_context_tokens: int = 0
    max_context_tokens_source: str = ""
    timeout_seconds: int = 30
    proxy: str = ""
    batch_size: int = 64
    concurrency: int = 2
    max_retries: int = 5
    api_suffix: str = ""
    return_documents: bool = False
    instruct: str = ""
    model_endpoint: str = ""
    truncate: str = ""
    input_type: str = ""
    launch_model_if_not_running: bool = False
    index_rebuild_settings: IndexRebuildSettings = field(
        default_factory=IndexRebuildSettings
    )


@dataclass(slots=True)
class RecallConfig:
    rrf_k: int = 60
    decay_rate: float = 0.0
    access_decay_window_days: float = 30.0
    access_decay_max_count: int = 10
    access_count_decay_multiplier: float = 0.5
    score_alpha: float = 0.5
    score_beta: float = 0.25
    score_gamma: float = 0.25
    importance_weight: float = 1.0
    min_importance_for_retrieval: float = 0.0
    min_similarity_for_retrieval: float = 0.0
    recent_memory_count: int = 2
    recent_memory_max_age_hours: int = 72
    memory_type_filter: str = "all"
    mmr_lambda: float = 0.7
    graph_memory_enabled: bool = True
    document_route_weight: float = 0.65
    graph_route_weight: float = 0.35
    cross_route_bonus: float = 0.08
    graph_expansion_limit: int = 24
    graph_expansion_hops: int = 1
    graph_second_hop_weight: float = 0.4
    dynamic_route_weighting: bool = True
    graph_max_topics: int = 6
    graph_max_participants: int = 8
    graph_max_facts: int = 8
    use_persona_filtering: bool = True
    use_session_filtering: bool = False
    search_cache_enabled: bool = True
    search_cache_ttl_seconds: float = 45.0
    search_cache_max_size: int = 256


@dataclass(slots=True)
class MaintenanceConfig:
    atom_enabled: bool = True
    atom_maintenance_interval_hours: float = 24.0
    atom_forget_delay_days: float = 7.0
    atom_purge_delay_days: float = 30.0
    auto_cleanup_enabled: bool = False
    auto_archived_enabled: bool = False
    cleanup_days_threshold: int = 7
    cleanup_importance_threshold: float = 0.3
    protected_importance_threshold: float = 1.0
    backup_enabled: bool = True
    backup_keep_days: int = 7


@dataclass(slots=True)
class ConversationConfig:
    max_sessions: int = 100
    session_ttl: int = 3600
    context_window_size: int = 300
    max_messages_per_session: int = 1000
    cleanup_batch_size: int = 50


@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"
    file_max_bytes: int = 10 * 1024 * 1024
    file_backup_count: int = 3
    web_max_entries: int = 2000
    web_max_bytes: int = 4 * 1024 * 1024
    web_max_entry_bytes: int = 32 * 1024


@dataclass(slots=True)
class RuntimeResidencyConfig:
    idle_minutes: int = 30
    max_non_default_runtimes: int = 4


@dataclass(slots=True)
class AppConfig:
    version: int = 1
    host: str = "127.0.0.1"
    access_base_url: str = DEFAULT_ACCESS_BASE_URL
    public_adapter_url: str = DEFAULT_PUBLIC_ADAPTER_URL
    port: int = 8765
    access_port: int = 8766
    performance_profile: str = DEFAULT_PERFORMANCE_PROFILE
    api_key: str = field(default_factory=lambda: f"prag_{secrets.token_urlsafe(32)}")
    session_secret: str = field(default_factory=lambda: secrets.token_urlsafe(48))
    library_psk_secret: str = field(default_factory=lambda: secrets.token_urlsafe(48))
    webui_password_hash: str = ""
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    # Compatibility bridge for installs created before providers moved into the
    # control database.  Direct AppConfig users keep the historical seed
    # behaviour, while a genuinely new config file disables it explicitly.
    bootstrap_provider_enabled: bool = field(default=True, repr=False)
    recall: RecallConfig = field(default_factory=RecallConfig)
    maintenance: MaintenanceConfig = field(default_factory=MaintenanceConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    runtime_residency: RuntimeResidencyConfig = field(
        default_factory=RuntimeResidencyConfig
    )

    @property
    def api_key_fingerprint(self) -> str:
        return hashlib.sha256(self.api_key.encode("utf-8")).hexdigest()[:12]


def apply_deployment_constraints(config: AppConfig) -> list[str]:
    """Keep container networking Compose-managed without seeding a Provider."""

    if not is_docker_deployment():
        return []
    changed: list[str] = []
    for field_name, value in (
        ("host", DOCKER_HOST),
        ("port", DOCKER_WEBUI_PORT),
        ("access_port", DOCKER_ACCESS_PORT),
    ):
        if getattr(config, field_name) != value:
            setattr(config, field_name, value)
            changed.append(field_name)
    return changed


def _merge_dataclass(cls, raw: dict[str, Any] | None):
    raw = raw or {}
    allowed = {item.name for item in cls.__dataclass_fields__.values()}
    payload = {key: value for key, value in raw.items() if key in allowed}
    if cls is ProviderConfig:
        if payload.get("type") == "openai_compatible":
            payload["type"] = "vllm_embedding"
        if "context_length_mode" not in payload:
            source = str(payload.get("max_context_tokens_source") or "")
            tokens = int(payload.get("max_context_tokens") or 0)
            if source.startswith("auto:"):
                payload["context_length_mode"] = "auto"
            elif tokens >= 128:
                payload["context_length_mode"] = "manual"
                payload.setdefault("max_context_tokens_source", "manual")
            else:
                payload["context_length_mode"] = "auto"
                payload["max_context_tokens"] = 0
                payload["max_context_tokens_source"] = ""
    if cls is ProviderConfig and isinstance(
        payload.get("index_rebuild_settings"), dict
    ):
        payload["index_rebuild_settings"] = _merge_dataclass(
            IndexRebuildSettings,
            payload["index_rebuild_settings"],
        )
    return cls(**payload)


def _normalize_runtime_residency(config: AppConfig) -> None:
    config.performance_profile = normalize_performance_profile(
        config.performance_profile
    )
    config.runtime_residency.idle_minutes = max(
        1,
        int(config.runtime_residency.idle_minutes),
    )
    config.runtime_residency.max_non_default_runtimes = max(
        1,
        int(config.runtime_residency.max_non_default_runtimes),
    )


def load_config(path: Path) -> AppConfig:
    if not path.exists():
        config = AppConfig(bootstrap_provider_enabled=False)
        _normalize_runtime_residency(config)
        apply_deployment_constraints(config)
        save_config(path, config)
        return config
    raw = json.loads(path.read_text(encoding="utf-8"))
    top = {
        key: value
        for key, value in raw.items()
        if key
        in {
            "version",
            "host",
            "access_base_url",
            "public_adapter_url",
            "port",
            "access_port",
            "performance_profile",
            "api_key",
            "session_secret",
            "library_psk_secret",
            "webui_password_hash",
        }
    }
    provider_raw = dict(raw.get("provider") or {})
    top["bootstrap_provider_enabled"] = bool(provider_raw)
    if provider_raw.get("type") == "openai_compatible":
        # v0.1.0 初版的 openai_compatible 实际实现的是 vLLM 语义：
        # 不发送 dimensions，并自动对齐 served-model-name。
        provider_raw["type"] = "vllm_embedding"
    top["provider"] = _merge_dataclass(ProviderConfig, provider_raw)
    top["recall"] = _merge_dataclass(RecallConfig, raw.get("recall"))
    top["maintenance"] = _merge_dataclass(
        MaintenanceConfig, raw.get("maintenance")
    )
    top["conversation"] = _merge_dataclass(
        ConversationConfig, raw.get("conversation")
    )
    top["logging"] = _merge_dataclass(LoggingConfig, raw.get("logging"))
    top["runtime_residency"] = _merge_dataclass(
        RuntimeResidencyConfig, raw.get("runtime_residency")
    )
    config = AppConfig(**top)
    _normalize_runtime_residency(config)
    if not config.api_key:
        config.api_key = f"prag_{secrets.token_urlsafe(32)}"
    if not config.session_secret:
        config.session_secret = secrets.token_urlsafe(48)
    if not config.library_psk_secret:
        config.library_psk_secret = secrets.token_urlsafe(48)
    config.host = normalize_bind_host(config.host)
    config.access_base_url = normalize_access_base_url(config.access_base_url)
    config.public_adapter_url = normalize_public_adapter_url(
        config.public_adapter_url
    )
    apply_deployment_constraints(config)
    save_config(path, config)
    return config


def app_config_from_dict(
    raw: dict[str, Any],
    *,
    provider: ProviderConfig | dict[str, Any] | None = None,
) -> AppConfig:
    payload = dict(raw or {})
    if provider is not None:
        payload["provider"] = (
            asdict(provider) if isinstance(provider, ProviderConfig) else dict(provider)
        )
    top = {
        key: value
        for key, value in payload.items()
        if key
        in {
            "version",
            "host",
            "access_base_url",
            "public_adapter_url",
            "port",
            "access_port",
            "performance_profile",
            "api_key",
            "session_secret",
            "library_psk_secret",
            "webui_password_hash",
        }
    }
    provider_raw = dict(payload.get("provider") or {})
    top["bootstrap_provider_enabled"] = bool(provider_raw)
    if provider_raw.get("type") == "openai_compatible":
        provider_raw["type"] = "vllm_embedding"
    top["provider"] = _merge_dataclass(ProviderConfig, provider_raw)
    top["recall"] = _merge_dataclass(RecallConfig, payload.get("recall"))
    top["maintenance"] = _merge_dataclass(
        MaintenanceConfig, payload.get("maintenance")
    )
    top["conversation"] = _merge_dataclass(
        ConversationConfig, payload.get("conversation")
    )
    top["logging"] = _merge_dataclass(LoggingConfig, payload.get("logging"))
    top["runtime_residency"] = _merge_dataclass(
        RuntimeResidencyConfig, payload.get("runtime_residency")
    )
    config = AppConfig(**top)
    _normalize_runtime_residency(config)
    if not config.api_key:
        config.api_key = f"prag_{secrets.token_urlsafe(32)}"
    if not config.session_secret:
        config.session_secret = secrets.token_urlsafe(48)
    if not config.library_psk_secret:
        config.library_psk_secret = secrets.token_urlsafe(48)
    config.host = normalize_bind_host(config.host)
    config.access_base_url = normalize_access_base_url(config.access_base_url)
    config.public_adapter_url = normalize_public_adapter_url(
        config.public_adapter_url
    )
    apply_deployment_constraints(config)
    return config


def save_config(path: Path, config: AppConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    payload = asdict(config)
    payload.pop("bootstrap_provider_enabled", None)
    if not config.bootstrap_provider_enabled:
        payload.pop("provider", None)
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp.replace(path)
