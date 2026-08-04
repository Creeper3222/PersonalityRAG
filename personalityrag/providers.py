from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, replace
from typing import Any
from urllib.parse import urlparse

import httpx

from .config import IndexRebuildSettings, ProviderConfig
from .context_lengths import (
    MIN_VALID_CONTEXT_TOKENS,
    lookup_static_context_length,
)
from .identifiers import validate_identifier
from .logger import logger, safe_summary
from .http_pool import PooledAsyncClient, acquire_http_client


PROVIDER_TEMPLATES: dict[str, dict[str, Any]] = {
    "openai_embedding": {
        "type": "openai_embedding",
        "display_name": "OpenAI Embedding",
        "api_base": "https://api.openai.com/v1",
        "api_key": "",
        "model": "text-embedding-3-small",
        "dimensions": 1536,
        "context_length_mode": "auto",
        "max_context_tokens": 0,
        "max_context_tokens_source": "",
        "timeout_seconds": 30,
        "proxy": "",
        "batch_size": 64,
        "concurrency": 2,
        "max_retries": 5,
    },
    "gemini_embedding": {
        "type": "gemini_embedding",
        "display_name": "Gemini Embedding",
        "api_base": "https://generativelanguage.googleapis.com/v1beta",
        "api_key": "",
        "model": "gemini-embedding-exp-03-07",
        "dimensions": 768,
        "context_length_mode": "auto",
        "max_context_tokens": 0,
        "max_context_tokens_source": "",
        "timeout_seconds": 20,
        "proxy": "",
        "batch_size": 64,
        "concurrency": 2,
        "max_retries": 5,
    },
    "nvidia_embedding": {
        "type": "nvidia_embedding",
        "display_name": "NVIDIA Embedding",
        "api_base": "https://integrate.api.nvidia.com/v1",
        "api_key": "",
        "model": "nvidia/llama-nemotron-embed-1b-v2",
        "dimensions": 1024,
        "context_length_mode": "auto",
        "max_context_tokens": 0,
        "max_context_tokens_source": "",
        "timeout_seconds": 20,
        "proxy": "",
        "batch_size": 64,
        "concurrency": 2,
        "max_retries": 5,
        "input_type": "passage",
    },
    "ollama_embedding": {
        "type": "ollama_embedding",
        "display_name": "Ollama Embedding",
        "api_base": "http://127.0.0.1:11434",
        "api_key": "",
        "model": "nomic-embed-text",
        "dimensions": 768,
        "context_length_mode": "auto",
        "max_context_tokens": 0,
        "max_context_tokens_source": "",
        "timeout_seconds": 60,
        "proxy": "",
        "batch_size": 32,
        "concurrency": 2,
        "max_retries": 3,
    },
    "vllm_embedding": {
        "type": "vllm_embedding",
        "display_name": "vLLM Embedding",
        "api_base": "http://127.0.0.1:8001/v1",
        "api_key": "",
        "model": "BAAI/bge-m3",
        "dimensions": 1024,
        "context_length_mode": "auto",
        "max_context_tokens": 0,
        "max_context_tokens_source": "",
        "timeout_seconds": 30,
        "proxy": "",
        "batch_size": 64,
        "concurrency": 2,
        "max_retries": 5,
    },
    "vllm_rerank": {
        "type": "vllm_rerank",
        "display_name": "vLLM Rerank",
        "api_base": "http://127.0.0.1:8002",
        "api_key": "",
        "model": "BAAI/bge-reranker-v2-m3",
        "dimensions": 0,
        "max_context_tokens": 0,
        "max_context_tokens_source": "",
        "timeout_seconds": 30,
        "proxy": "",
        "batch_size": 1,
        "concurrency": 1,
        "max_retries": 3,
        "api_suffix": "/v1/rerank",
    },
    "xinference_rerank": {
        "type": "xinference_rerank",
        "display_name": "Xinference Rerank",
        "api_base": "http://127.0.0.1:9997",
        "api_key": "",
        "model": "BAAI/bge-reranker-base",
        "dimensions": 0,
        "max_context_tokens": 0,
        "max_context_tokens_source": "",
        "timeout_seconds": 30,
        "proxy": "",
        "batch_size": 1,
        "concurrency": 1,
        "max_retries": 3,
        "api_suffix": "/v1/rerank",
        "launch_model_if_not_running": False,
    },
    "bailian_rerank": {
        "type": "bailian_rerank",
        "display_name": "阿里云百炼重排序",
        "api_base": "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
        "api_key": "",
        "model": "qwen3-rerank",
        "dimensions": 0,
        "max_context_tokens": 0,
        "max_context_tokens_source": "",
        "timeout_seconds": 30,
        "proxy": "",
        "batch_size": 1,
        "concurrency": 1,
        "max_retries": 3,
        "return_documents": False,
        "instruct": "",
    },
    "nvidia_rerank": {
        "type": "nvidia_rerank",
        "display_name": "NVIDIA Rerank",
        "api_base": "https://ai.api.nvidia.com/v1/retrieval",
        "api_key": "",
        "model": "nv-rerank-qa-mistral-4b:1",
        "dimensions": 0,
        "max_context_tokens": 0,
        "max_context_tokens_source": "",
        "timeout_seconds": 30,
        "proxy": "",
        "batch_size": 1,
        "concurrency": 1,
        "max_retries": 3,
        "model_endpoint": "/reranking",
        "truncate": "",
    },
}

EMBEDDING_PROVIDER_TYPES = {
    "openai_embedding",
    "gemini_embedding",
    "nvidia_embedding",
    "ollama_embedding",
    "vllm_embedding",
    "openai_compatible",
}
RERANK_PROVIDER_TYPES = {
    "vllm_rerank",
    "xinference_rerank",
    "bailian_rerank",
    "nvidia_rerank",
}
OPENAI_EMBEDDING_CONTEXT_LENGTHS = {
    "text-embedding-3-small": 8192,
    "text-embedding-3-large": 8192,
    "text-embedding-ada-002": 8192,
}


def provider_kind(provider_type: str) -> str:
    if provider_type in RERANK_PROVIDER_TYPES:
        return "rerank"
    return "embedding"


def provider_config_hash(config: ProviderConfig) -> str:
    payload = asdict(config)
    # Index rebuild throttling is an operational setting, not an embedding
    # semantics setting. Keep it out of the revision hash used for index drift.
    payload.pop("index_rebuild_settings", None)
    payload.pop("batch_size", None)
    payload.pop("concurrency", None)
    payload.pop("max_retries", None)
    payload.pop("display_name", None)
    payload.pop("context_length_mode", None)
    payload.pop("max_context_tokens_source", None)
    payload = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def masked_config(config: ProviderConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["api_key"] = "********" if config.api_key else ""
    payload["has_api_key"] = bool(config.api_key)
    payload["provider_kind"] = provider_kind(config.type)
    return payload


def validate_provider_config(config: ProviderConfig) -> None:
    validate_identifier(config.id, field="Provider ID")
    if config.type not in PROVIDER_TEMPLATES:
        raise ValueError(f"不支持的 Provider 类型: {config.type}")
    if not config.api_base:
        raise ValueError("API Base URL 不能为空")
    if not config.model:
        raise ValueError("嵌入模型不能为空")
    if config.dimensions < 0:
        raise ValueError("嵌入维度不能小于 0")
    if config.max_context_tokens < 0:
        raise ValueError("max_context_tokens must not be negative")
    if config.context_length_mode not in {"auto", "manual"}:
        raise ValueError("context_length_mode must be auto or manual")
    if (
        provider_kind(config.type) == "embedding"
        and config.context_length_mode == "manual"
        and config.max_context_tokens < MIN_VALID_CONTEXT_TOKENS
    ):
        raise ValueError(
            f"manual max_context_tokens must be >= {MIN_VALID_CONTEXT_TOKENS}"
        )
    if config.timeout_seconds <= 0:
        raise ValueError("超时时间必须大于 0")
    if config.batch_size <= 0 or config.concurrency <= 0:
        raise ValueError("批量大小和并发数必须大于 0")
    if config.max_retries <= 0:
        raise ValueError("最大重试次数必须大于 0")
    rebuild = config.index_rebuild_settings
    if (
        rebuild.batch_size <= 0
        or rebuild.embedding_batch_size <= 0
        or rebuild.tasks_limit <= 0
        or rebuild.max_retries <= 0
    ):
        raise ValueError("index rebuild batch, concurrency, and retries must be positive")
    if (
        rebuild.retry_base_delay < 0
        or rebuild.batch_delay < 0
        or rebuild.request_delay < 0
        or rebuild.max_failure_ratio < 0
    ):
        raise ValueError("index rebuild delays and failure ratio must not be negative")


def config_from_dict(
    payload: dict[str, Any],
    *,
    base: ProviderConfig | None = None,
    keep_secret: bool = False,
) -> ProviderConfig:
    source = asdict(base) if base else {}
    context_mode_supplied = "context_length_mode" in payload
    allowed = set(ProviderConfig.__dataclass_fields__)
    had_nested_rebuild_settings = isinstance(
        source.get("index_rebuild_settings"), dict
    )
    for key, value in payload.items():
        if key in allowed:
            source[key] = value
    if keep_secret and base and payload.get("api_key", None) in {None, "", "********"}:
        source["api_key"] = base.api_key
    source.pop("has_api_key", None)
    raw_rebuild_settings = source.get("index_rebuild_settings")
    if isinstance(raw_rebuild_settings, IndexRebuildSettings):
        pass
    elif isinstance(raw_rebuild_settings, dict):
        defaults = asdict(IndexRebuildSettings())
        defaults.update(
            {
                key: value
                for key, value in raw_rebuild_settings.items()
                if key in defaults
            }
        )
        source["index_rebuild_settings"] = IndexRebuildSettings(**defaults)
    else:
        defaults = asdict(IndexRebuildSettings())
        if not had_nested_rebuild_settings:
            if "batch_size" in source:
                defaults["batch_size"] = source["batch_size"]
            if "concurrency" in source:
                defaults["tasks_limit"] = source["concurrency"]
            if "max_retries" in source:
                defaults["max_retries"] = source["max_retries"]
        source["index_rebuild_settings"] = IndexRebuildSettings(**defaults)
    if not context_mode_supplied and base is None:
        max_context_tokens = int(source.get("max_context_tokens") or 0)
        max_context_source = str(source.get("max_context_tokens_source") or "")
        if max_context_source.startswith("auto:"):
            source["context_length_mode"] = "auto"
        elif max_context_tokens >= MIN_VALID_CONTEXT_TOKENS:
            source["context_length_mode"] = "manual"
            source.setdefault("max_context_tokens_source", "manual")
        else:
            source["context_length_mode"] = "auto"
            source["max_context_tokens"] = 0
            source["max_context_tokens_source"] = ""
    if str(source.get("context_length_mode") or "auto") == "manual":
        source["max_context_tokens_source"] = str(
            source.get("max_context_tokens_source") or "manual"
        )
    elif int(source.get("max_context_tokens") or 0) <= 0:
        source["max_context_tokens_source"] = ""
    config = ProviderConfig(**source)
    validate_provider_config(config)
    return config


class EmbeddingProvider(ABC):
    config: ProviderConfig

    @abstractmethod
    async def get_embedding(self, text: str) -> list[float]: ...

    @abstractmethod
    async def get_embeddings(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    async def get_dimension(self) -> int: ...

    @abstractmethod
    async def list_models(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def detect_context_length(self) -> dict[str, Any]: ...

    @abstractmethod
    async def test_connection(self) -> dict[str, Any]: ...

    @abstractmethod
    async def close(self) -> None: ...


class HTTPEmbeddingProvider(EmbeddingProvider):
    CONTEXT_LENGTH_KEYS = (
        "max_model_len",
        "max_context_length",
        "context_length",
        "max_sequence_length",
        "max_seq_len",
        "max_position_embeddings",
        "n_ctx",
        "num_ctx",
        "inputTokenLimit",
    )

    def __init__(self, config: ProviderConfig):
        validate_provider_config(config)
        self.config = config
        self._dimension: int | None = config.dimensions or None
        self._resolved_model: str | None = None

    @staticmethod
    def _is_local_or_private(url: str) -> bool:
        host = (urlparse(url).hostname or "").lower()
        if host in {"localhost", "127.0.0.1", "::1", "host.docker.internal"}:
            return True
        try:
            value = ipaddress.ip_address(host)
        except ValueError:
            return False
        return value.is_private or value.is_loopback

    def _http_client(self, *, base_url: str) -> PooledAsyncClient:
        trust_env = not self._is_local_or_private(base_url)
        if self.config.proxy:
            trust_env = False
        return acquire_http_client(
            base_url=base_url,
            timeout=self.config.timeout_seconds,
            proxy=self.config.proxy or None,
            trust_env=trust_env,
            headers=(
                {"Authorization": f"Bearer {self.config.api_key}"}
                if self.config.api_key
                else None
            ),
        )

    def _model_matches(self, item: dict[str, Any]) -> bool:
        configured = self.config.model.strip().casefold()
        basename = self.config.model.strip().rsplit("/", 1)[-1].casefold()
        candidates = {
            str(item.get("id") or "").casefold(),
            str(item.get("root") or "").casefold(),
            str(item.get("name") or "").casefold(),
            str(item.get("model") or "").casefold(),
        }
        candidates.update(
            value.rsplit("/", 1)[-1] for value in tuple(candidates) if value
        )
        return configured in candidates or basename in candidates

    @classmethod
    def _extract_context_length(
        cls, payload: Any, prefix: str = ""
    ) -> tuple[int, str] | None:
        if isinstance(payload, dict):
            for key, value in payload.items():
                path = f"{prefix}.{key}" if prefix else key
                normalized_key = key.rsplit(".", 1)[-1]
                if normalized_key in cls.CONTEXT_LENGTH_KEYS:
                    try:
                        number = int(value)
                    except (TypeError, ValueError):
                        number = 0
                    if number > 0:
                        return number, path
                nested = cls._extract_context_length(value, path)
                if nested:
                    return nested
        elif isinstance(payload, list):
            for index, value in enumerate(payload):
                nested = cls._extract_context_length(value, f"{prefix}[{index}]")
                if nested:
                    return nested
        return None

    async def _detect_context_length_from_models(self) -> dict[str, Any]:
        try:
            for item in await self.list_models():
                if not isinstance(item, dict) or not self._model_matches(item):
                    continue
                detected = self._extract_context_length(item)
                if detected:
                    value, source = detected
                    return {
                        "max_context_tokens": value,
                        "max_context_tokens_source": f"auto:{self.config.type}:models.{source}",
                    }
        except Exception as exc:
            logger.info(
                "Provider context length detection skipped: provider=%s type=%s err=%s",
                self.config.id,
                self.config.type,
                safe_summary(exc, max_chars=160),
            )
        return {"max_context_tokens": 0, "max_context_tokens_source": ""}

    def _detect_static_context_length(self) -> dict[str, Any]:
        candidates = [self.config.model]
        if self._resolved_model and self._resolved_model != self.config.model:
            candidates.insert(0, self._resolved_model)
        for model_name in candidates:
            detected = lookup_static_context_length(model_name)
            if detected.get("max_context_tokens"):
                return detected
        return {"max_context_tokens": 0, "max_context_tokens_source": ""}

    async def detect_context_length(self) -> dict[str, Any]:
        return self._detect_static_context_length()

    async def _retry(self, operation):
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                return await operation()
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= self.config.max_retries:
                    break
                await asyncio.sleep(min(30.0, 2**attempt))
        raise RuntimeError(f"嵌入请求失败: {last_error}") from last_error

    def _validate_vectors(
        self, vectors: list[list[float]], expected: int
    ) -> list[list[float]]:
        if len(vectors) != expected:
            raise RuntimeError(
                f"向量数量不匹配: expected={expected}, actual={len(vectors)}"
            )
        dimension = 0
        for vector in vectors:
            if not vector:
                raise RuntimeError("Provider 返回了空向量")
            if not all(math.isfinite(float(value)) for value in vector):
                raise RuntimeError("Provider 返回了非有限向量值")
            norm = math.sqrt(sum(float(value) ** 2 for value in vector))
            if norm <= 0:
                raise RuntimeError("Provider 返回了零向量")
            if not dimension:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise RuntimeError("同一批次返回了不同维度的向量")
        if self._dimension and dimension != self._dimension:
            raise RuntimeError(
                f"向量维度不匹配: configured={self._dimension}, actual={dimension}"
            )
        self._dimension = dimension
        return vectors

    async def get_embedding(self, text: str) -> list[float]:
        return (await self.get_embeddings([text]))[0]

    async def get_dimension(self) -> int:
        if self._dimension:
            return self._dimension
        self._dimension = len(await self.get_embedding("维度检测"))
        return self._dimension

    async def test_connection(self) -> dict[str, Any]:
        try:
            started = asyncio.get_running_loop().time()
            try:
                models = await self.list_models()
            except Exception:
                models = []
            vector = await self.get_embedding("PersonalityRAG Provider 测试")
            elapsed = (asyncio.get_running_loop().time() - started) * 1000
            norm = math.sqrt(sum(float(value) ** 2 for value in vector))
            logger.info(
                "Embedding Provider 连接成功：provider=%s type=%s model=%s resolved=%s dimension=%s elapsed_ms=%.2f",
                self.config.id,
                self.config.type,
                self.config.model,
                self._resolved_model or self.config.model,
                len(vector),
                elapsed,
            )
            return {
                "available": True,
                "provider_id": self.config.id,
                "provider_type": self.config.type,
                "configured_model": self.config.model,
                "resolved_model": self._resolved_model or self.config.model,
                "dimension": len(vector),
                "vector_norm": round(norm, 6),
                "models": models,
                "elapsed_ms": round(elapsed, 2),
            }
        except Exception as exc:
            logger.warning(
                "Embedding Provider 连接失败：provider=%s type=%s model=%s err=%s",
                self.config.id,
                self.config.type,
                self.config.model,
                safe_summary(exc, max_chars=240),
            )
            return {
                "available": False,
                "provider_id": self.config.id,
                "provider_type": self.config.type,
                "configured_model": self.config.model,
                "resolved_model": self._resolved_model or "",
                "dimension": self._dimension or 0,
                "error": str(exc),
            }


class OpenAIEmbeddingProvider(HTTPEmbeddingProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        api_base = config.api_base.rstrip("/").removesuffix("/embeddings")
        if not api_base.endswith(("/v1", "/v4")):
            api_base += "/v1"
        self.api_base = api_base
        self._client = self._http_client(base_url=api_base)

    async def list_models(self) -> list[dict[str, Any]]:
        response = await self._client.get("/models")
        response.raise_for_status()
        return list(response.json().get("data") or [])

    async def detect_context_length(self) -> dict[str, Any]:
        return await super().detect_context_length()

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        async def request():
            payload: dict[str, Any] = {
                "model": self.config.model,
                "input": texts,
            }
            if self.config.dimensions > 0:
                payload["dimensions"] = self.config.dimensions
            response = await self._client.post("/embeddings", json=payload)
            response.raise_for_status()
            data = sorted(
                response.json().get("data") or [],
                key=lambda item: int(item.get("index", 0)),
            )
            return [list(map(float, item["embedding"])) for item in data]

        vectors = await self._retry(request)
        self._resolved_model = self.config.model
        return self._validate_vectors(vectors, len(texts))

    async def close(self) -> None:
        await self._client.aclose()


class GeminiEmbeddingProvider(HTTPEmbeddingProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        api_base = config.api_base.rstrip("/")
        if not api_base.endswith(("/v1", "/v1beta")):
            api_base += "/v1beta"
        self.api_base = api_base
        self.model = config.model.strip().removeprefix("models/")
        self._client = acquire_http_client(
            base_url=api_base,
            timeout=config.timeout_seconds,
            proxy=config.proxy or None,
            trust_env=(
                not self._is_local_or_private(api_base)
                if not config.proxy
                else False
            ),
            headers=(
                {"x-goog-api-key": config.api_key}
                if config.api_key
                else None
            ),
        )
        self._resolved_model = self.model

    async def list_models(self) -> list[dict[str, Any]]:
        response = await self._client.get("/models", params={"pageSize": 1000})
        response.raise_for_status()
        return list(response.json().get("models") or [])

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        async def request():
            model_path = f"models/{self.model}"
            requests = []
            for text in texts:
                item: dict[str, Any] = {
                    "model": model_path,
                    "content": {"parts": [{"text": text}]},
                }
                if self.config.dimensions > 0:
                    item["outputDimensionality"] = self.config.dimensions
                requests.append(item)
            response = await self._client.post(
                f"/models/{self.model}:batchEmbedContents",
                json={"requests": requests},
            )
            response.raise_for_status()
            return [
                list(map(float, item.get("values") or []))
                for item in response.json().get("embeddings") or []
            ]

        vectors = await self._retry(request)
        return self._validate_vectors(vectors, len(texts))

    async def close(self) -> None:
        await self._client.aclose()


class NvidiaEmbeddingProvider(HTTPEmbeddingProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.api_base = config.api_base.rstrip("/").removesuffix("/embeddings")
        self._client = self._http_client(base_url=self.api_base)
        self._resolved_model = config.model

    async def list_models(self) -> list[dict[str, Any]]:
        response = await self._client.get("/models")
        response.raise_for_status()
        return list(response.json().get("data") or [])

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        async def request():
            response = await self._client.post(
                "/embeddings",
                json={
                    "input": texts,
                    "model": self.config.model,
                    "input_type": self.config.input_type or "passage",
                    "encoding_format": "float",
                },
            )
            response.raise_for_status()
            data = sorted(
                response.json().get("data") or [],
                key=lambda item: int(item.get("index", 0)),
            )
            return [list(map(float, item.get("embedding") or [])) for item in data]

        vectors = await self._retry(request)
        return self._validate_vectors(vectors, len(texts))

    async def close(self) -> None:
        await self._client.aclose()


class VLLMEmbeddingProvider(HTTPEmbeddingProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        api_base = config.api_base.rstrip("/").removesuffix("/embeddings")
        if not api_base.endswith(("/v1", "/v4")):
            api_base += "/v1"
        self.api_base = api_base
        self._client = self._http_client(base_url=api_base)

    async def list_models(self) -> list[dict[str, Any]]:
        response = await self._client.get("/models")
        response.raise_for_status()
        return list(response.json().get("data") or [])

    async def detect_context_length(self) -> dict[str, Any]:
        detected = await self._detect_context_length_from_models()
        if detected.get("max_context_tokens"):
            return detected
        try:
            model = await self._resolve_model()
            root_base = self.api_base.rsplit("/", 1)[0]
            response = await self._client.post(
                f"{root_base}/tokenize",
                json={"model": model, "prompt": "PersonalityRAG context probe"},
            )
            response.raise_for_status()
            payload = response.json()
            value = int(payload.get("max_model_len") or 0)
            if value > 0:
                return {
                    "max_context_tokens": value,
                    "max_context_tokens_source": f"auto:{self.config.type}:/tokenize.max_model_len",
                }
        except Exception as exc:
            logger.info(
                "vLLM context length detection via /tokenize skipped: provider=%s err=%s",
                self.config.id,
                safe_summary(exc, max_chars=160),
            )
        return self._detect_static_context_length()

    async def _resolve_model(self) -> str:
        if self._resolved_model:
            return self._resolved_model
        configured = self.config.model.strip()
        try:
            models = await self.list_models()
        except Exception:
            models = []
        configured_lower = configured.casefold()
        basename = configured.rsplit("/", 1)[-1].casefold()
        for item in models:
            model_id = str(item.get("id") or "")
            root = str(item.get("root") or "")
            if (
                model_id.casefold() == configured_lower
                or root.casefold() == configured_lower
                or model_id.casefold() == basename
            ):
                self._resolved_model = model_id
                return model_id
        self._resolved_model = configured.rsplit("/", 1)[-1] or configured
        return self._resolved_model

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = await self._resolve_model()

        async def request():
            # vLLM 的原生维度由 served model 决定，禁止发送 dimensions。
            response = await self._client.post(
                "/embeddings", json={"model": model, "input": texts}
            )
            response.raise_for_status()
            data = sorted(
                response.json().get("data") or [],
                key=lambda item: int(item.get("index", 0)),
            )
            return [list(map(float, item["embedding"])) for item in data]

        vectors = await self._retry(request)
        return self._validate_vectors(vectors, len(texts))

    async def close(self) -> None:
        await self._client.aclose()


class OllamaEmbeddingProvider(HTTPEmbeddingProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.api_base = config.api_base.rstrip("/").removesuffix("/api/embed")
        self._client = self._http_client(base_url=self.api_base)
        self._resolved_model = config.model

    async def list_models(self) -> list[dict[str, Any]]:
        response = await self._client.get("/api/tags")
        response.raise_for_status()
        return [
            {
                "id": item.get("name") or item.get("model"),
                "name": item.get("name"),
                "model": item.get("model"),
                "size": item.get("size"),
            }
            for item in response.json().get("models") or []
        ]

    async def detect_context_length(self) -> dict[str, Any]:
        try:
            response = await self._client.post(
                "/api/show", json={"model": self.config.model}
            )
            response.raise_for_status()
            payload = response.json()
            detected = self._extract_context_length(payload)
            if detected:
                value, source = detected
                return {
                    "max_context_tokens": value,
                    "max_context_tokens_source": f"auto:{self.config.type}:/api/show.{source}",
                }
        except Exception as exc:
            logger.info(
                "Ollama context length detection skipped: provider=%s err=%s",
                self.config.id,
                safe_summary(exc, max_chars=160),
            )
        return self._detect_static_context_length()

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        async def request():
            payload: dict[str, Any] = {
                "model": self.config.model,
                "input": texts,
            }
            if self.config.dimensions > 0:
                payload["dimensions"] = self.config.dimensions
            response = await self._client.post("/api/embed", json=payload)
            response.raise_for_status()
            return [
                list(map(float, item))
                for item in response.json().get("embeddings") or []
            ]

        vectors = await self._retry(request)
        return self._validate_vectors(vectors, len(texts))

    async def close(self) -> None:
        await self._client.aclose()


@dataclass(slots=True)
class RerankResult:
    index: int
    relevance_score: float


class RerankProvider(ABC):
    config: ProviderConfig

    @abstractmethod
    async def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[RerankResult]: ...

    @abstractmethod
    async def test_connection(self) -> dict[str, Any]: ...

    @abstractmethod
    async def close(self) -> None: ...


class HTTPRerankProvider(RerankProvider):
    def __init__(self, config: ProviderConfig):
        validate_provider_config(config)
        if provider_kind(config.type) != "rerank":
            raise ValueError(f"Provider 类型不是 Rerank: {config.type}")
        self.config = config
        self._resolved_model: str | None = None

    def _http_client(self, *, base_url: str) -> PooledAsyncClient:
        trust_env = not HTTPEmbeddingProvider._is_local_or_private(base_url)
        if self.config.proxy:
            trust_env = False
        return acquire_http_client(
            base_url=base_url,
            timeout=self.config.timeout_seconds,
            proxy=self.config.proxy or None,
            trust_env=trust_env,
            headers=(
                {"Authorization": f"Bearer {self.config.api_key}"}
                if self.config.api_key
                else None
            ),
        )

    async def _retry(self, operation):
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                return await operation()
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= self.config.max_retries:
                    break
                await asyncio.sleep(min(30.0, 2**attempt))
        raise RuntimeError(f"Rerank 请求失败: {last_error}") from last_error

    @staticmethod
    def _normalize_suffix(value: str | None, default: str = "/v1/rerank") -> str:
        suffix = default if value is None or value == "" else str(value)
        if suffix and not suffix.startswith("/"):
            suffix = "/" + suffix
        return suffix

    @staticmethod
    def _parse_standard_results(data: dict[str, Any]) -> list[RerankResult]:
        results = data.get("results") or []
        parsed: list[RerankResult] = []
        for fallback_index, item in enumerate(results):
            try:
                parsed.append(
                    RerankResult(
                        index=int(item.get("index", fallback_index)),
                        relevance_score=float(
                            item.get("relevance_score", item.get("score", 0.0))
                        ),
                    )
                )
            except (TypeError, ValueError):
                continue
        return parsed

    async def list_models(self) -> list[dict[str, Any]]:
        return []

    async def test_connection(self) -> dict[str, Any]:
        try:
            started = asyncio.get_running_loop().time()
            try:
                models = await self.list_models()
            except Exception:
                models = []
            results = await self.rerank(
                "PersonalityRAG Rerank Provider 测试",
                ["PersonalityRAG 支持记忆召回重排。", "完全无关的文本。"],
                2,
            )
            elapsed = (asyncio.get_running_loop().time() - started) * 1000
            if not results:
                raise RuntimeError("Rerank Provider 返回了空结果")
            logger.info(
                "Rerank Provider 连接成功：provider=%s type=%s model=%s resolved=%s elapsed_ms=%.2f",
                self.config.id,
                self.config.type,
                self.config.model,
                self._resolved_model or self.config.model,
                elapsed,
            )
            return {
                "available": True,
                "provider_id": self.config.id,
                "provider_type": self.config.type,
                "provider_kind": "rerank",
                "configured_model": self.config.model,
                "resolved_model": self._resolved_model or self.config.model,
                "models": models,
                "top_score": round(results[0].relevance_score, 6),
                "elapsed_ms": round(elapsed, 2),
            }
        except Exception as exc:
            logger.warning(
                "Rerank Provider 连接失败：provider=%s type=%s model=%s err=%s",
                self.config.id,
                self.config.type,
                self.config.model,
                safe_summary(exc, max_chars=240),
            )
            return {
                "available": False,
                "provider_id": self.config.id,
                "provider_type": self.config.type,
                "provider_kind": "rerank",
                "configured_model": self.config.model,
                "resolved_model": self._resolved_model or "",
                "error": str(exc),
            }


class VLLMRerankProvider(HTTPRerankProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.api_base = config.api_base.rstrip("/")
        self.api_suffix = self._normalize_suffix(config.api_suffix)
        version = self.api_base.rsplit("/", 1)[-1]
        if version in {"v1", "v4"} and self.api_suffix.startswith(f"/{version}/"):
            self.api_suffix = self.api_suffix.removeprefix(f"/{version}")
        self._client = self._http_client(base_url=self.api_base)

    async def list_models(self) -> list[dict[str, Any]]:
        versioned_base = self.api_base.rsplit("/", 1)[-1] in {"v1", "v4"}
        paths = ["/models", "/v1/models"] if versioned_base else ["/v1/models", "/models"]
        last_response: httpx.Response | None = None
        for path in paths:
            response = await self._client.get(path)
            last_response = response
            if response.status_code != 404:
                response.raise_for_status()
                return list(response.json().get("data") or [])
        if last_response is not None:
            last_response.raise_for_status()
        return []

    async def _resolve_model(self) -> str:
        if self._resolved_model:
            return self._resolved_model
        configured = self.config.model.strip()
        try:
            models = await self.list_models()
        except Exception:
            models = []
        configured_lower = configured.casefold()
        basename = configured.rsplit("/", 1)[-1].casefold()
        for item in models:
            model_id = str(item.get("id") or "")
            root = str(item.get("root") or "")
            if (
                model_id.casefold() == configured_lower
                or root.casefold() == configured_lower
                or model_id.casefold() == basename
            ):
                self._resolved_model = model_id
                return model_id
        self._resolved_model = configured
        return self._resolved_model

    async def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[RerankResult]:
        if not query.strip() or not documents:
            return []
        model = await self._resolve_model()

        async def request():
            payload: dict[str, Any] = {
                "query": query,
                "documents": documents,
                "model": model,
            }
            if top_n is not None:
                payload["top_n"] = top_n
            response = await self._client.post(self.api_suffix, json=payload)
            response.raise_for_status()
            return self._parse_standard_results(response.json())

        return await self._retry(request)

    async def close(self) -> None:
        await self._client.aclose()


class XinferenceRerankProvider(VLLMRerankProvider):
    pass


class BailianRerankProvider(HTTPRerankProvider):
    QWEN3_RERANK_MODEL = "qwen3-rerank"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.api_base = config.api_base.rstrip("/")
        self._client = self._http_client(base_url=self.api_base)
        self._resolved_model = config.model

    def _build_payload(
        self, query: str, documents: list[str], top_n: int | None
    ) -> dict[str, Any]:
        normalized_top_n = top_n if top_n is not None and top_n > 0 else None
        if self.config.model.strip().lower() == self.QWEN3_RERANK_MODEL:
            payload: dict[str, Any] = {
                "model": self.config.model,
                "query": query,
                "documents": documents,
            }
            if normalized_top_n is not None:
                payload["top_n"] = normalized_top_n
            if self.config.instruct:
                payload["instruct"] = self.config.instruct
            return payload
        payload_input = {"query": query, "documents": documents}
        params: dict[str, Any] = {}
        if normalized_top_n is not None:
            params["top_n"] = normalized_top_n
        if self.config.return_documents:
            params["return_documents"] = True
        payload = {"model": self.config.model, "input": payload_input}
        if params:
            payload["parameters"] = params
        return payload

    def _parse_results(self, data: dict[str, Any]) -> list[RerankResult]:
        if "compatible-api" in self.api_base:
            if data.get("code"):
                raise RuntimeError(
                    f"百炼 Rerank API 错误: {data.get('code')} {data.get('message', '')}"
                )
            return self._parse_standard_results(data)
        code = str(data.get("code", "200"))
        if code != "200":
            raise RuntimeError(
                f"百炼 Rerank API 错误: {code} {data.get('message', '')}"
            )
        results = (data.get("output") or {}).get("results") or []
        return self._parse_standard_results({"results": results})

    async def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[RerankResult]:
        if not query.strip() or not documents:
            return []
        documents = documents[:500]

        async def request():
            response = await self._client.post(
                "", json=self._build_payload(query, documents, top_n)
            )
            response.raise_for_status()
            return self._parse_results(response.json())

        return await self._retry(request)

    async def close(self) -> None:
        await self._client.aclose()


class NvidiaRerankProvider(HTTPRerankProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.api_base = config.api_base.rstrip("/")
        self.model_endpoint = self._normalize_suffix(
            config.model_endpoint or "/reranking", "/reranking"
        )
        self._client = self._http_client(base_url=self.api_base)
        self._resolved_model = config.model

    def _endpoint(self) -> str:
        model_path = "nvidia"
        if "/" in self.config.model:
            model_path = self.config.model.strip("/").replace(".", "_")
        return f"/{model_path}{self.model_endpoint}"

    def _parse_results(
        self, data: dict[str, Any], top_n: int | None = None
    ) -> list[RerankResult]:
        parsed: list[RerankResult] = []
        for fallback_index, item in enumerate(data.get("rankings") or []):
            try:
                parsed.append(
                    RerankResult(
                        index=int(item.get("index", fallback_index)),
                        relevance_score=float(
                            item.get("relevance_score", item.get("logit", 0.0))
                        ),
                    )
                )
            except (TypeError, ValueError):
                continue
        parsed.sort(key=lambda item: item.relevance_score, reverse=True)
        return parsed[:top_n] if top_n is not None and top_n > 0 else parsed

    async def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[RerankResult]:
        if not query.strip() or not documents:
            return []

        async def request():
            payload: dict[str, Any] = {
                "model": self.config.model,
                "query": {"text": query},
                "passages": [{"text": doc} for doc in documents],
            }
            if self.config.truncate:
                payload["truncate"] = self.config.truncate
            response = await self._client.post(self._endpoint(), json=payload)
            response.raise_for_status()
            return self._parse_results(response.json(), top_n)

        return await self._retry(request)

    async def close(self) -> None:
        await self._client.aclose()

def build_rerank_provider(config: ProviderConfig) -> RerankProvider:
    if config.type == "vllm_rerank":
        return VLLMRerankProvider(config)
    if config.type == "xinference_rerank":
        return XinferenceRerankProvider(config)
    if config.type == "bailian_rerank":
        return BailianRerankProvider(config)
    if config.type == "nvidia_rerank":
        return NvidiaRerankProvider(config)
    raise ValueError(f"不支持的 Rerank Provider 类型: {config.type}")


def build_provider(config: ProviderConfig) -> EmbeddingProvider:
    if provider_kind(config.type) != "embedding":
        raise ValueError(f"Provider 类型不是 Embedding: {config.type}")
    if config.type == "openai_embedding":
        return OpenAIEmbeddingProvider(config)
    if config.type == "gemini_embedding":
        return GeminiEmbeddingProvider(config)
    if config.type == "nvidia_embedding":
        return NvidiaEmbeddingProvider(config)
    if config.type == "ollama_embedding":
        return OllamaEmbeddingProvider(config)
    if config.type in {"vllm_embedding", "openai_compatible"}:
        if config.type == "openai_compatible":
            config = replace(config, type="vllm_embedding")
        return VLLMEmbeddingProvider(config)
    raise ValueError(f"不支持的 Provider 类型: {config.type}")


# 初版公开类名兼容。它原本就是 vLLM 风格的 OpenAI-compatible 实现。
OpenAICompatibleEmbeddingProvider = VLLMEmbeddingProvider
