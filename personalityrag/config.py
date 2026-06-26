from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ProviderConfig:
    id: str = "vllm_embedding"
    display_name: str = "本机 bge-m3"
    type: str = "vllm_embedding"
    enabled: bool = True
    api_base: str = "http://127.0.0.1:8001/v1"
    api_key: str = "vllm"
    model: str = "BAAI/bge-m3"
    dimensions: int = 1024
    timeout_seconds: int = 30
    proxy: str = ""
    batch_size: int = 64
    concurrency: int = 2
    max_retries: int = 5


@dataclass(slots=True)
class RecallConfig:
    top_k: int = 10
    rrf_k: int = 60
    decay_rate: float = 0.0
    score_alpha: float = 0.5
    score_beta: float = 0.25
    score_gamma: float = 0.25
    mmr_lambda: float = 0.7
    document_route_weight: float = 0.65
    graph_route_weight: float = 0.35
    cross_route_bonus: float = 0.08
    graph_expansion_limit: int = 24
    graph_expansion_hops: int = 1
    graph_second_hop_weight: float = 0.4
    dynamic_route_weighting: bool = True
    use_persona_filtering: bool = True
    use_session_filtering: bool = False
    search_cache_ttl_seconds: float = 45.0
    search_cache_max_size: int = 256


@dataclass(slots=True)
class MaintenanceConfig:
    atom_maintenance_interval_hours: float = 24.0
    atom_forget_delay_days: float = 7.0
    atom_purge_delay_days: float = 30.0
    auto_cleanup_enabled: bool = True
    cleanup_days_threshold: int = 30
    cleanup_importance_threshold: float = 0.3
    backup_enabled: bool = True
    backup_keep_days: int = 7


@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"
    file_max_bytes: int = 10 * 1024 * 1024
    file_backup_count: int = 3
    web_max_entries: int = 2000
    web_max_bytes: int = 4 * 1024 * 1024
    web_max_entry_bytes: int = 32 * 1024


@dataclass(slots=True)
class AppConfig:
    version: int = 1
    host: str = "127.0.0.1"
    port: int = 8765
    api_key: str = field(default_factory=lambda: f"prag_{secrets.token_urlsafe(32)}")
    session_secret: str = field(default_factory=lambda: secrets.token_urlsafe(48))
    webui_password_hash: str = ""
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    recall: RecallConfig = field(default_factory=RecallConfig)
    maintenance: MaintenanceConfig = field(default_factory=MaintenanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @property
    def api_key_fingerprint(self) -> str:
        return hashlib.sha256(self.api_key.encode("utf-8")).hexdigest()[:12]


def _merge_dataclass(cls, raw: dict[str, Any] | None):
    raw = raw or {}
    allowed = {item.name for item in cls.__dataclass_fields__.values()}
    return cls(**{key: value for key, value in raw.items() if key in allowed})


def load_config(path: Path) -> AppConfig:
    if not path.exists():
        config = AppConfig()
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
            "port",
            "api_key",
            "session_secret",
            "webui_password_hash",
        }
    }
    provider_raw = dict(raw.get("provider") or {})
    if provider_raw.get("type") == "openai_compatible":
        # v0.1.0 初版的 openai_compatible 实际实现的是 vLLM 语义：
        # 不发送 dimensions，并自动对齐 served-model-name。
        provider_raw["type"] = "vllm_embedding"
    top["provider"] = _merge_dataclass(ProviderConfig, provider_raw)
    top["recall"] = _merge_dataclass(RecallConfig, raw.get("recall"))
    top["maintenance"] = _merge_dataclass(
        MaintenanceConfig, raw.get("maintenance")
    )
    top["logging"] = _merge_dataclass(LoggingConfig, raw.get("logging"))
    config = AppConfig(**top)
    if not config.api_key:
        config.api_key = f"prag_{secrets.token_urlsafe(32)}"
    if not config.session_secret:
        config.session_secret = secrets.token_urlsafe(48)
    save_config(path, config)
    return config


def save_config(path: Path, config: AppConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp.replace(path)
