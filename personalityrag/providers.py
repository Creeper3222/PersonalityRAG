from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import math
from abc import ABC, abstractmethod
from dataclasses import asdict, replace
from typing import Any
from urllib.parse import urlparse

import httpx

from .config import ProviderConfig
from .logger import logger, safe_summary


PROVIDER_TEMPLATES: dict[str, dict[str, Any]] = {
    "openai_embedding": {
        "type": "openai_embedding",
        "display_name": "OpenAI Embedding",
        "api_base": "https://api.openai.com/v1",
        "api_key": "",
        "model": "text-embedding-3-small",
        "dimensions": 1536,
        "timeout_seconds": 30,
        "proxy": "",
        "batch_size": 64,
        "concurrency": 2,
        "max_retries": 5,
    },
    "ollama_embedding": {
        "type": "ollama_embedding",
        "display_name": "Ollama Embedding",
        "api_base": "http://127.0.0.1:11434",
        "api_key": "",
        "model": "nomic-embed-text",
        "dimensions": 768,
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
        "timeout_seconds": 30,
        "proxy": "",
        "batch_size": 64,
        "concurrency": 2,
        "max_retries": 5,
    },
}


def provider_config_hash(config: ProviderConfig) -> str:
    payload = json.dumps(
        asdict(config), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def masked_config(config: ProviderConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["api_key"] = "********" if config.api_key else ""
    payload["has_api_key"] = bool(config.api_key)
    return payload


def validate_provider_config(config: ProviderConfig) -> None:
    if not config.id or not config.id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Provider ID 只能包含字母、数字、下划线和连字符")
    if config.type not in PROVIDER_TEMPLATES:
        raise ValueError(f"不支持的 Provider 类型: {config.type}")
    if not config.api_base:
        raise ValueError("API Base URL 不能为空")
    if not config.model:
        raise ValueError("嵌入模型不能为空")
    if config.dimensions < 0:
        raise ValueError("嵌入维度不能小于 0")
    if config.timeout_seconds <= 0:
        raise ValueError("超时时间必须大于 0")
    if config.batch_size <= 0 or config.concurrency <= 0:
        raise ValueError("批量大小和并发数必须大于 0")
    if config.max_retries <= 0:
        raise ValueError("最大重试次数必须大于 0")


def config_from_dict(
    payload: dict[str, Any],
    *,
    base: ProviderConfig | None = None,
    keep_secret: bool = False,
) -> ProviderConfig:
    source = asdict(base) if base else {}
    allowed = set(ProviderConfig.__dataclass_fields__)
    for key, value in payload.items():
        if key in allowed:
            source[key] = value
    if keep_secret and base and payload.get("api_key", None) in {None, "", "********"}:
        source["api_key"] = base.api_key
    source.pop("has_api_key", None)
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
    async def test_connection(self) -> dict[str, Any]: ...

    @abstractmethod
    async def close(self) -> None: ...


class HTTPEmbeddingProvider(EmbeddingProvider):
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

    def _http_client(self, *, base_url: str) -> httpx.AsyncClient:
        kwargs: dict[str, Any] = {
            "base_url": base_url,
            "timeout": self.config.timeout_seconds,
            "trust_env": not self._is_local_or_private(base_url),
        }
        if self.config.proxy:
            kwargs["proxy"] = self.config.proxy
            kwargs["trust_env"] = False
        if self.config.api_key:
            kwargs["headers"] = {
                "Authorization": f"Bearer {self.config.api_key}"
            }
        return httpx.AsyncClient(**kwargs)

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


def build_provider(config: ProviderConfig) -> EmbeddingProvider:
    if config.type == "openai_embedding":
        return OpenAIEmbeddingProvider(config)
    if config.type == "ollama_embedding":
        return OllamaEmbeddingProvider(config)
    if config.type in {"vllm_embedding", "openai_compatible"}:
        if config.type == "openai_compatible":
            config = replace(config, type="vllm_embedding")
        return VLLMEmbeddingProvider(config)
    raise ValueError(f"不支持的 Provider 类型: {config.type}")


# 初版公开类名兼容。它原本就是 vLLM 风格的 OpenAI-compatible 实现。
OpenAICompatibleEmbeddingProvider = VLLMEmbeddingProvider
