from __future__ import annotations

import base64
import hashlib
import hmac
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable

from fastapi import APIRouter

from .identifiers import validate_identifier


DATABASE_CATEGORY_MEMORY = "memory"
DATABASE_CATEGORY_KNOWLEDGE = "knowledge"
DATABASE_CATEGORIES = frozenset(
    {DATABASE_CATEGORY_MEMORY, DATABASE_CATEGORY_KNOWLEDGE}
)
DATABASE_CATEGORY_DIRECTORIES = {
    DATABASE_CATEGORY_MEMORY: "memory_stores",
    DATABASE_CATEGORY_KNOWLEDGE: "knowledge_bases",
}
LIVINGMEMORY_V8_TYPE = "livingmemory_v8"
TEXT_MEDIA_V1_TYPE = "text_media_v1"


@dataclass(frozen=True, slots=True, order=True)
class DatabaseRef:
    database_type: str
    id: str

    def __post_init__(self) -> None:
        validate_identifier(self.database_type, field="数据库类型")
        validate_identifier(self.id, field="数据库 ID")

    @property
    def key(self) -> str:
        return f"{self.database_type}:{self.id}"

    def public(self) -> dict[str, str]:
        return {
            "id": self.id,
            **database_identity_fields(self),
        }


def database_identity_fields(
    database: DatabaseRef | tuple[str, str],
    *,
    include_deprecated: bool = False,
) -> dict[str, str]:
    """Return the canonical generic and category-specific identity fields.

    ``library_*`` is intentionally emitted only at explicit compatibility
    boundaries. SQL columns and frozen package manifests are not affected.
    """

    ref = (
        database
        if isinstance(database, DatabaseRef)
        else DatabaseRef(str(database[0]), str(database[1]))
    )
    result = {
        "database_id": ref.id,
        "database_type": ref.database_type,
    }
    if ref.database_type == LIVINGMEMORY_V8_TYPE:
        result.update(
            {
                "memory_store_id": ref.id,
                "memory_store_type": ref.database_type,
            }
        )
    elif ref.database_type == TEXT_MEDIA_V1_TYPE:
        result.update(
            {
                "knowledge_base_id": ref.id,
                "knowledge_base_type": ref.database_type,
            }
        )
    if include_deprecated:
        result["library_id"] = ref.id
    return result


@dataclass(slots=True)
class DatabaseDriverContext:
    root: Path
    data_root: Path
    system_path: Path
    config: Any
    control: Any
    jobs: Any = None


@dataclass(frozen=True, slots=True)
class DatabaseTypeDescriptor:
    id: str
    category: str
    display_name: str
    description: str
    icon: str
    capabilities: tuple[str, ...]
    key_prefix: str
    key_derivation_id: str

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "display_name": self.display_name,
            "description": self.description,
            "icon": self.icon,
            "capabilities": list(self.capabilities),
            "key_prefix": self.key_prefix,
        }


@runtime_checkable
class DatabaseTypeDriver(Protocol):
    descriptor: DatabaseTypeDescriptor

    def resource_key(self, database_id: str) -> str: ...

    def create_manager(self, context: DatabaseDriverContext) -> Any: ...

    def derive_access_key(self, root_secret: str, database_id: str) -> str: ...

    def verify_access_key(
        self, root_secret: str, database_id: str, candidate: str | None
    ) -> bool: ...

    def api_routers(self) -> Iterable[APIRouter]: ...


class DatabaseTypeRegistry:
    def __init__(self) -> None:
        self._drivers: dict[str, DatabaseTypeDriver] = {}
        self._derivations: dict[str, str] = {}

    def register(self, driver: DatabaseTypeDriver) -> None:
        descriptor = driver.descriptor
        validate_identifier(descriptor.id, field="数据库类型")
        if not callable(getattr(driver, "create_manager", None)):
            raise ValueError(f"数据库类型 {descriptor.id} 未提供运行管理器工厂")
        if descriptor.category not in DATABASE_CATEGORIES:
            raise ValueError(f"不支持的数据库大类: {descriptor.category}")
        expected_prefix = (
            "psk-"
            if descriptor.category == DATABASE_CATEGORY_MEMORY
            else "pkb-"
        )
        if descriptor.key_prefix != expected_prefix:
            raise ValueError(
                f"数据库类型 {descriptor.id} 的密钥前缀必须是 {expected_prefix}"
            )
        if descriptor.id in self._drivers:
            raise ValueError(f"数据库类型已注册: {descriptor.id}")
        owner = self._derivations.get(descriptor.key_derivation_id)
        if owner:
            raise ValueError(
                f"数据库密钥派生标识已由 {owner} 使用: "
                f"{descriptor.key_derivation_id}"
            )
        probe = driver.derive_access_key("database-type-registry-probe", "probe")
        if not probe.startswith(expected_prefix):
            raise ValueError(f"数据库类型 {descriptor.id} 返回了错误的密钥前缀")
        self._drivers[descriptor.id] = driver
        self._derivations[descriptor.key_derivation_id] = descriptor.id

    def require(self, database_type: str) -> DatabaseTypeDriver:
        driver = self._drivers.get(str(database_type or ""))
        if driver is None:
            raise KeyError(database_type)
        return driver

    def list(self, category: str | None = None) -> list[DatabaseTypeDescriptor]:
        if category is not None and category not in DATABASE_CATEGORIES:
            raise ValueError(f"不支持的数据库大类: {category}")
        return [
            driver.descriptor
            for driver in self._drivers.values()
            if category is None or driver.descriptor.category == category
        ]

    def type_root(self, data_root: Path, database_type: str) -> Path:
        descriptor = self.require(database_type).descriptor
        category_directory = DATABASE_CATEGORY_DIRECTORIES[descriptor.category]
        return Path(data_root) / "databases" / category_directory / descriptor.id

    def data_dir(
        self,
        data_root: Path,
        database: DatabaseRef | tuple[str, str],
    ) -> Path:
        ref = (
            database
            if isinstance(database, DatabaseRef)
            else DatabaseRef(str(database[0]), str(database[1]))
        )
        return self.type_root(data_root, ref.database_type) / ref.id

    def trash_type_root(self, data_root: Path, database_type: str) -> Path:
        descriptor = self.require(database_type).descriptor
        category_directory = DATABASE_CATEGORY_DIRECTORIES[descriptor.category]
        return (
            Path(data_root)
            / "trash"
            / "databases"
            / category_directory
            / descriptor.id
        )

    def api_routers(self) -> list[APIRouter]:
        routers: list[APIRouter] = []
        for driver in self._drivers.values():
            factory = getattr(driver, "api_routers", None)
            if callable(factory):
                routers.extend(factory())
        return routers


def derive_domain_separated_access_key(
    *,
    root_secret: str,
    database_type: str,
    database_id: str,
    prefix: str,
) -> str:
    payload = f"personalityrag\0{database_type}\0{database_id}".encode("utf-8")
    digest = hmac.new(root_secret.encode("utf-8"), payload, hashlib.sha256).digest()
    return prefix + base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


database_type_registry = DatabaseTypeRegistry()
