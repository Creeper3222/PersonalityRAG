from __future__ import annotations

import hashlib
import json
import re
import shutil
import stat
import tempfile
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

import pyzipper

from .config import AppConfig, app_config_from_dict, save_config
from .context_lengths import (
    MANUAL_CONTEXT_FALLBACK_TOKENS,
    MIN_VALID_CONTEXT_TOKENS,
    static_context_table_metadata,
)
from .database_types import (
    LIVINGMEMORY_V8_TYPE,
    TEXT_MEDIA_V1_TYPE,
    DatabaseRef,
    database_type_registry,
)
from .identifiers import validate_identifier
from .io_utils import run_blocking
from . import library_types as _registered_database_types  # noqa: F401
from .libraries import DatabaseManager
from .logger import logger
from .library_types.livingmemory_v8.migration import (
    sha256_file,
    sqlite_backup,
    validate_conversations_db_file,
    validate_livingmemory_db_file,
)
from .library_types.livingmemory_v8.storage import Storage
from .library_types.text_media_v1.package import (
    export_tmkb,
    extract_tmkb,
    install_tmkb,
    inspect_tmkb,
)


PACKAGE_FORMAT = "personalityrag.prag"
PACKAGE_VERSION = 2
SUPPORTED_PACKAGE_VERSIONS = frozenset({1, 2})
MAX_PACKAGE_BYTES = 2 * 1024 * 1024 * 1024
MAX_EXTRACTED_BYTES = 8 * 1024 * 1024 * 1024
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_CONFIG_JSON_BYTES = 64 * 1024 * 1024
ARCHIVE_CHUNK_BYTES = 1024 * 1024
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class PragPackageError(RuntimeError):
    pass


def _normalize_provider_context_config(config: dict[str, Any]) -> dict[str, int]:
    summary = {
        "auto_pending": 0,
        "manual_fallback": 0,
        "legacy_manual": 0,
    }
    provider_type = str(config.get("type") or "")
    if "embedding" not in provider_type:
        config["context_length_mode"] = "auto"
        config["max_context_tokens"] = 0
        config["max_context_tokens_source"] = ""
        return summary
    try:
        tokens = int(config.get("max_context_tokens") or 0)
    except (TypeError, ValueError):
        tokens = 0
    source = str(config.get("max_context_tokens_source") or "")
    mode = str(config.get("context_length_mode") or "").strip().lower()
    if mode not in {"auto", "manual"}:
        if source.startswith("auto:"):
            mode = "auto"
        elif tokens >= MIN_VALID_CONTEXT_TOKENS:
            mode = "manual"
            summary["legacy_manual"] += 1
        else:
            mode = "auto"
            tokens = 0
            source = ""
    if mode == "manual":
        if tokens < MIN_VALID_CONTEXT_TOKENS:
            tokens = MANUAL_CONTEXT_FALLBACK_TOKENS
            source = "manual:fallback-undetected"
            summary["manual_fallback"] += 1
        elif not source:
            source = "manual:user"
    elif tokens < MIN_VALID_CONTEXT_TOKENS:
        tokens = 0
        source = ""
        summary["auto_pending"] += 1
    config["context_length_mode"] = mode
    config["max_context_tokens"] = tokens
    config["max_context_tokens_source"] = source
    return summary


def _normalize_provider_snapshot_context(snapshot: dict[str, Any]) -> dict[str, int]:
    summary = {
        "auto_pending": 0,
        "manual_fallback": 0,
        "legacy_manual": 0,
    }
    for revision in snapshot.get("provider_revisions") or []:
        config = revision.get("config")
        if not isinstance(config, dict):
            continue
        item_summary = _normalize_provider_context_config(config)
        for key, value in item_summary.items():
            summary[key] = summary.get(key, 0) + int(value or 0)
    return summary


@asynccontextmanager
async def _temporary_directory(prefix: str):
    path = Path(await run_blocking(tempfile.mkdtemp, prefix=prefix))
    try:
        yield path
    finally:
        await run_blocking(shutil.rmtree, path, ignore_errors=True)


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode(
        "utf-8"
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_member(name: str) -> str:
    value = str(name or "").replace("\\", "/")
    path = PurePosixPath(value)
    parts = path.parts
    if not value or path.is_absolute() or value.endswith("/"):
        raise PragPackageError(f"invalid package member path: {name}")
    if any(part in {"", ".", ".."} or ":" in part for part in parts):
        raise PragPackageError(f"invalid package member path: {name}")
    normalized = "/".join(parts)
    if not normalized or "\x00" in normalized:
        raise PragPackageError(f"invalid package member path: {name}")
    return normalized


def _global_config_payload(config: AppConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload.pop("provider", None)
    payload.pop("bootstrap_provider_enabled", None)
    return payload


def _package_password(password: str) -> bytes:
    value = str(password or "")
    if not value.strip():
        raise PragPackageError("请输入配置验证密码")
    return value.encode("utf-8")


async def _library_empty(library_dir: Path, system_path: Path) -> bool:
    if not (library_dir / "livingmemory.db").exists():
        return True
    storage = Storage(library_dir, system_path=system_path)
    stats = await storage.statistics()
    values = (
        stats.get("total_memories"),
        stats.get("graph_nodes"),
        stats.get("graph_edges"),
        stats.get("graph_entries"),
        stats.get("atom_count"),
        (stats.get("conversation_counts") or {}).get("sessions"),
        (stats.get("conversation_counts") or {}).get("messages"),
    )
    return all(int(value or 0) == 0 for value in values)


async def _add_sqlite_file(
    files: dict[str, dict[str, Any]],
    archive_name: str,
    source: Path,
    work_dir: Path,
) -> None:
    target = work_dir / archive_name.replace("/", "__")
    await run_blocking(sqlite_backup, source, target)
    digest = await run_blocking(sha256_file, target)
    files[archive_name] = {
        "path": target,
        "sha256": digest,
        "size": target.stat().st_size,
    }


async def _add_path_file(
    files: dict[str, dict[str, Any]], archive_name: str, source: Path
) -> None:
    digest = await run_blocking(sha256_file, source)
    files[archive_name] = {
        "path": source,
        "sha256": digest,
        "size": source.stat().st_size,
    }


def _add_json_file(
    files: dict[str, dict[str, Any]], archive_name: str, payload: dict[str, Any]
) -> None:
    data = _json_bytes(payload)
    files[archive_name] = {
        "bytes": data,
        "sha256": _sha256_bytes(data),
        "size": len(data),
    }


def _write_package(target: Path, password: str, files: dict[str, dict[str, Any]]) -> None:
    manifest_files = {
        name: {"sha256": item["sha256"], "size": item["size"]}
        for name, item in sorted(files.items())
        if name != "manifest.json"
    }
    manifest = files["manifest.json"]["manifest"]
    manifest["files"] = manifest_files
    manifest_bytes = _json_bytes(manifest)
    extracted_size = len(manifest_bytes) + sum(
        int(item["size"]) for item in manifest_files.values()
    )
    if extracted_size > MAX_EXTRACTED_BYTES:
        raise PragPackageError("配置包解压后总大小超过安全上限")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with pyzipper.AESZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            encryption=pyzipper.WZ_AES,
        ) as archive:
            archive.setpassword(_package_password(password))
            archive.setencryption(pyzipper.WZ_AES, nbits=256)
            for name, item in sorted(files.items()):
                if name == "manifest.json":
                    continue
                _safe_member(name)
                if "bytes" in item:
                    archive.writestr(name, item["bytes"])
                else:
                    archive.write(
                        item["path"],
                        name,
                        compress_type=(
                            zipfile.ZIP_STORED
                            if name.endswith(".tmkb")
                            else zipfile.ZIP_DEFLATED
                        ),
                    )
            archive.writestr("manifest.json", manifest_bytes)
        if temporary.stat().st_size > MAX_PACKAGE_BYTES:
            raise PragPackageError("配置包过大")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


async def export_prag_package(
    *,
    root: Path,
    config: AppConfig,
    manager: DatabaseManager,
    target: Path,
    password: str,
    include_libraries: bool,
    include_providers: bool,
) -> dict[str, Any]:
    _package_password(password)
    export_id = time.strftime("%Y%m%d-%H%M%S")
    files: dict[str, dict[str, Any]] = {}
    async with _temporary_directory("prag-export-") as work_dir:
        _add_json_file(files, "config/global.json", _global_config_payload(config))
        provider_snapshot: dict[str, Any] | None = None
        if include_providers:
            provider_snapshot = await manager.control.export_provider_snapshot()
            provider_snapshot["context_length_table"] = static_context_table_metadata()
            _add_json_file(files, "providers/providers.json", provider_snapshot)

        library_entries = []
        if include_libraries:
            library_snapshot = await manager.control.export_library_snapshot()
            database_records = list(library_snapshot["libraries"])
            for detail in await manager._managers[TEXT_MEDIA_V1_TYPE].list_libraries(
                stats_mode="summary"
            ):
                database_records.append(
                    {
                        "id": detail["id"],
                        "database_type": TEXT_MEDIA_V1_TYPE,
                        "name": detail["name"],
                        "description": detail.get("description") or "",
                        "provider_id": detail.get("provider_id") or "",
                        "provider_revision": detail.get("provider_revision") or 0,
                        "status": detail.get("status") or "ready",
                        "is_default": False,
                        "created_at": detail.get("created_at"),
                        "updated_at": detail.get("updated_at"),
                    }
                )
            for index, library in enumerate(database_records, start=1):
                library_id = str(library["id"])
                database_type = str(
                    library.get("database_type") or LIVINGMEMORY_V8_TYPE
                )
                driver = database_type_registry.require(database_type)
                base = f"libraries/{index:04d}"
                library_dir = database_type_registry.data_dir(
                    manager.data_dir,
                    DatabaseRef(database_type, library_id),
                )
                empty = (
                    await _library_empty(library_dir, manager.system_path)
                    if database_type == LIVINGMEMORY_V8_TYPE
                    else bool((await manager.library_detail(
                        DatabaseRef(database_type, library_id)
                    ))["stats"]["documents"] == 0)
                )
                entry = {
                    "id": library_id,
                    "database_type": database_type,
                    "path": base,
                    "empty": empty,
                    "files": {},
                }
                _add_json_file(
                    files,
                    f"{base}/library.json",
                    {"library": library, "empty": empty},
                )
                if database_type == TEXT_MEDIA_V1_TYPE:
                    package_name = f"databases/{TEXT_MEDIA_V1_TYPE}/{library_id}.tmkb"
                    package_path = work_dir / f"{TEXT_MEDIA_V1_TYPE}-{library_id}.tmkb"
                    service = await manager.get_runtime(
                        DatabaseRef(TEXT_MEDIA_V1_TYPE, library_id)
                    )
                    await export_tmkb(service=service, target=package_path)
                    await _add_path_file(files, package_name, package_path)
                    entry["files"]["native_package"] = package_name
                elif not empty:
                    livingmemory_db = library_dir / "livingmemory.db"
                    conversations_db = library_dir / "conversations.db"
                    if livingmemory_db.exists():
                        archive_name = f"{base}/livingmemory.db"
                        await _add_sqlite_file(
                            files, archive_name, livingmemory_db, work_dir
                        )
                        entry["files"]["livingmemory_db"] = archive_name
                    if conversations_db.exists():
                        archive_name = f"{base}/conversations.db"
                        await _add_sqlite_file(
                            files, archive_name, conversations_db, work_dir
                        )
                        entry["files"]["conversations_db"] = archive_name
                library_entries.append(entry)
            _add_json_file(
                files,
                "libraries/libraries.json",
                {
                    "libraries": [item["id"] for item in library_entries],
                    "databases": [
                        {
                            "database_type": item["database_type"],
                            "id": item["id"],
                        }
                        for item in library_entries
                    ],
                },
            )

        default_library_id = ""
        default_database: dict[str, str] | None = None
        if include_libraries:
            for entry in library_snapshot["libraries"]:
                if entry.get("is_default"):
                    default_library_id = str(entry.get("id") or "")
                    default_database = {
                        "database_type": str(
                            entry.get("database_type") or LIVINGMEMORY_V8_TYPE
                        ),
                        "id": default_library_id,
                    }
                    break
        files["manifest.json"] = {
            "manifest": {
                "format": PACKAGE_FORMAT,
                "version": PACKAGE_VERSION,
                "created_at": time.time(),
                "export_id": export_id,
                "scope": {
                    "include_libraries": bool(include_libraries),
                    "include_providers": bool(include_providers),
                },
                "default_library_id": default_library_id,
                "default_database": default_database,
                "libraries": library_entries,
                "providers": {
                    "count": len((provider_snapshot or {}).get("providers") or [])
                },
            }
        }
        await run_blocking(_write_package, target, password, files)
    logger.warning(
        "备份迁移配置包已导出：path=%s include_libraries=%s include_providers=%s",
        target,
        include_libraries,
        include_providers,
    )
    return {
        "path": str(target),
        "filename": target.name,
        "include_libraries": include_libraries,
        "include_providers": include_providers,
        "size_bytes": target.stat().st_size,
    }


def _zip_member_is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (int(info.external_attr) >> 16) & 0o170000
    return mode == stat.S_IFLNK


def _read_manifest_member(
    archive: pyzipper.AESZipFile,
    info: zipfile.ZipInfo,
) -> bytes:
    if int(info.file_size) > MAX_MANIFEST_BYTES:
        raise PragPackageError("配置包 manifest.json 过大")
    chunks: list[bytes] = []
    total = 0
    with archive.open(info, "r") as source:
        while chunk := source.read(ARCHIVE_CHUNK_BYTES):
            total += len(chunk)
            if total > MAX_MANIFEST_BYTES:
                raise PragPackageError("配置包 manifest.json 实际大小超过安全上限")
            chunks.append(chunk)
    if total != int(info.file_size):
        raise PragPackageError("配置包 manifest.json 尺寸不一致")
    return b"".join(chunks)


def _validated_archive_members(
    archive: pyzipper.AESZipFile,
) -> tuple[dict[str, zipfile.ZipInfo], zipfile.ZipInfo]:
    members: dict[str, zipfile.ZipInfo] = {}
    manifest_info: zipfile.ZipInfo | None = None
    total_declared_size = 0
    for info in archive.infolist():
        if info.is_dir():
            raise PragPackageError(f"配置包不允许目录成员：{info.filename}")
        name = _safe_member(info.filename)
        if name in members or (manifest_info is not None and name == "manifest.json"):
            raise PragPackageError(f"配置包包含重复成员：{name}")
        if _zip_member_is_symlink(info):
            raise PragPackageError(f"配置包不允许符号链接成员：{name}")
        if not int(info.flag_bits) & 0x1:
            raise PragPackageError(f"配置包成员未加密：{name}")
        size = int(info.file_size)
        if size < 0:
            raise PragPackageError(f"配置包成员尺寸不合法：{name}")
        total_declared_size += size
        if total_declared_size > MAX_EXTRACTED_BYTES:
            raise PragPackageError("配置包解压后总大小超过安全上限")
        if name == "manifest.json":
            manifest_info = info
        else:
            members[name] = info
    if manifest_info is None:
        raise PragPackageError("配置包缺少 manifest.json")
    return members, manifest_info


def _validate_package_structure(
    manifest: dict[str, Any],
    declared: dict[str, dict[str, Any]],
) -> None:
    scope = manifest.get("scope")
    if not isinstance(scope, dict):
        raise PragPackageError("配置包 manifest.json 缺少导出范围")
    for key in ("include_libraries", "include_providers"):
        if not isinstance(scope.get(key), bool):
            raise PragPackageError(f"配置包导出范围字段不合法：{key}")

    required = {"config/global.json"}
    if scope["include_providers"]:
        required.add("providers/providers.json")

    raw_libraries = manifest.get("libraries") or []
    if not isinstance(raw_libraries, list):
        raise PragPackageError("配置包记忆库清单不合法")
    if scope["include_libraries"]:
        required.add("libraries/libraries.json")
    elif raw_libraries:
        raise PragPackageError("配置包范围未包含记忆库但清单非空")

    database_refs: set[tuple[str, str]] = set()
    library_paths: set[str] = set()
    for entry in raw_libraries:
        if not isinstance(entry, dict):
            raise PragPackageError("配置包记忆库条目不合法")
        library_id = str(entry.get("id") or "")
        database_type = str(
            entry.get("database_type") or LIVINGMEMORY_V8_TYPE
        )
        try:
            validate_identifier(library_id, field="记忆库 ID")
            validate_identifier(database_type, field="数据库类型")
            database_type_registry.require(database_type)
        except KeyError as exc:
            raise PragPackageError(f"配置包包含未注册的数据库类型：{database_type}") from exc
        except ValueError as exc:
            raise PragPackageError(str(exc)) from exc
        base = _safe_member(entry.get("path") or "")
        database_ref = (database_type, library_id)
        if not library_id or database_ref in database_refs or base in library_paths:
            raise PragPackageError("配置包记忆库 ID 或路径重复")
        database_refs.add(database_ref)
        library_paths.add(base)
        required.add(f"{base}/library.json")
        raw_files = entry.get("files") or {}
        if not isinstance(raw_files, dict):
            raise PragPackageError(f"配置包记忆库文件清单不合法：{library_id}")
        allowed_files = (
            {"native_package"}
            if database_type == TEXT_MEDIA_V1_TYPE
            else {"livingmemory_db", "conversations_db"}
        )
        unknown = set(raw_files) - allowed_files
        if unknown:
            raise PragPackageError(f"配置包记忆库文件类型不合法：{library_id}")
        for file_name, raw_path in raw_files.items():
            member = _safe_member(raw_path)
            if file_name == "native_package":
                expected = f"databases/{TEXT_MEDIA_V1_TYPE}/{library_id}.tmkb"
                if member != expected:
                    raise PragPackageError(
                        f"配置包知识库子封包路径不合法：{library_id}"
                    )
                required.add(member)
                continue
            if not member.startswith(f"{base}/"):
                raise PragPackageError(
                    f"配置包记忆库文件路径与条目不匹配：{library_id}"
                )
            expected_leaf = (
                "livingmemory.db"
                if file_name == "livingmemory_db"
                else "conversations.db"
            )
            if PurePosixPath(member).name != expected_leaf:
                raise PragPackageError(
                    f"配置包记忆库文件名不合法：{library_id}"
                )
            required.add(member)
        if database_type == TEXT_MEDIA_V1_TYPE and "native_package" not in raw_files:
            raise PragPackageError(f"知识库缺少 .tmkb 子封包：{library_id}")
        if (
            database_type == LIVINGMEMORY_V8_TYPE
            and not bool(entry.get("empty"))
            and "livingmemory_db" not in raw_files
        ):
            raise PragPackageError(f"非空记忆库缺少 livingmemory.db：{library_id}")

    raw_default_database = manifest.get("default_database")
    if isinstance(raw_default_database, dict):
        default_database = (
            str(raw_default_database.get("database_type") or LIVINGMEMORY_V8_TYPE),
            str(raw_default_database.get("id") or ""),
        )
    else:
        default_database = (
            LIVINGMEMORY_V8_TYPE,
            str(manifest.get("default_library_id") or ""),
        )
    if scope["include_libraries"] and default_database not in database_refs:
        raise PragPackageError("配置包默认记忆库不在记忆库清单中")

    missing_required = sorted(required - set(declared))
    if missing_required:
        raise PragPackageError(f"配置包缺少必要文件：{missing_required[0]}")
    for name, expected in declared.items():
        if name.endswith(".json") and int(expected["size"]) > MAX_CONFIG_JSON_BYTES:
            raise PragPackageError(f"配置包 JSON 文件过大：{name}")


def _extract_verified_package(
    package_path: Path,
    password: str,
    target: Path,
) -> dict[str, Any]:
    if package_path.stat().st_size > MAX_PACKAGE_BYTES:
        raise PragPackageError("配置包过大")
    try:
        with pyzipper.AESZipFile(package_path, "r") as archive:
            archive.setpassword(_package_password(password))
            members, manifest_info = _validated_archive_members(archive)
            try:
                manifest_bytes = _read_manifest_member(archive, manifest_info)
                manifest = json.loads(manifest_bytes.decode("utf-8"))
            except RuntimeError as exc:
                raise PragPackageError("配置验证密码错误或配置包已损坏") from exc
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PragPackageError("配置包 manifest.json 已损坏") from exc
            if not isinstance(manifest, dict):
                raise PragPackageError("配置包 manifest.json 结构不合法")
            if manifest.get("format") != PACKAGE_FORMAT:
                raise PragPackageError("不是有效的 PersonalityRAG 配置包")
            if int(manifest.get("version") or 0) not in SUPPORTED_PACKAGE_VERSIONS:
                raise PragPackageError("不支持的 PersonalityRAG 配置包版本")
            raw_file_manifest = manifest.get("files")
            if not isinstance(raw_file_manifest, dict):
                raise PragPackageError("配置包 manifest.json 缺少文件清单")

            declared: dict[str, dict[str, Any]] = {}
            declared_total = len(manifest_bytes)
            for raw_name, raw_expected in raw_file_manifest.items():
                name = _safe_member(raw_name)
                if name == "manifest.json" or name in declared:
                    raise PragPackageError(f"配置包清单包含重复成员：{name}")
                if not isinstance(raw_expected, dict):
                    raise PragPackageError(f"配置包文件清单损坏：{name}")
                digest = str(raw_expected.get("sha256") or "").lower()
                size_value = raw_expected.get("size")
                if isinstance(size_value, bool):
                    raise PragPackageError(f"配置包文件尺寸不合法：{name}")
                try:
                    size = int(size_value)
                except (TypeError, ValueError) as exc:
                    raise PragPackageError(
                        f"配置包文件尺寸不合法：{name}"
                    ) from exc
                if size < 0 or not SHA256_PATTERN.fullmatch(digest):
                    raise PragPackageError(f"配置包文件清单损坏：{name}")
                declared_total += size
                if declared_total > MAX_EXTRACTED_BYTES:
                    raise PragPackageError("配置包解压后总大小超过安全上限")
                declared[name] = {"sha256": digest, "size": size}

            _validate_package_structure(manifest, declared)

            actual_names = set(members)
            declared_names = set(declared)
            missing = sorted(declared_names - actual_names)
            undeclared = sorted(actual_names - declared_names)
            if missing:
                raise PragPackageError(f"配置包缺少文件：{missing[0]}")
            if undeclared:
                raise PragPackageError(f"配置包包含未声明文件：{undeclared[0]}")

            extracted_total = len(manifest_bytes)
            target_root = target.resolve()
            for name in sorted(declared):
                expected = declared[name]
                info = members[name]
                if int(info.file_size) != expected["size"]:
                    raise PragPackageError(f"配置包文件声明尺寸不一致：{name}")
                destination = target.joinpath(*PurePosixPath(name).parts)
                resolved_destination = destination.resolve()
                if target_root not in resolved_destination.parents:
                    raise PragPackageError(f"invalid package member path: {name}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                actual_size = 0
                try:
                    with archive.open(info, "r") as source, destination.open(
                        "xb"
                    ) as output:
                        while chunk := source.read(ARCHIVE_CHUNK_BYTES):
                            actual_size += len(chunk)
                            extracted_total += len(chunk)
                            if actual_size > expected["size"]:
                                raise PragPackageError(
                                    f"配置包文件实际尺寸超过声明：{name}"
                                )
                            if extracted_total > MAX_EXTRACTED_BYTES:
                                raise PragPackageError(
                                    "配置包解压后总大小超过安全上限"
                                )
                            digest.update(chunk)
                            output.write(chunk)
                except RuntimeError as exc:
                    raise PragPackageError(
                        "配置验证密码错误或配置包已损坏"
                    ) from exc
                if actual_size != expected["size"]:
                    raise PragPackageError(f"配置包文件实际尺寸不一致：{name}")
                if digest.hexdigest() != expected["sha256"]:
                    raise PragPackageError(f"配置包文件校验失败：{name}")
    except zipfile.BadZipFile as exc:
        raise PragPackageError("配置包不是有效的 zip 文件") from exc
    return manifest


def _load_json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > MAX_CONFIG_JSON_BYTES:
        raise PragPackageError(f"JSON 配置文件过大：{path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PragPackageError(f"JSON 配置文件损坏：{path.name}") from exc
    if not isinstance(payload, dict):
        raise PragPackageError(f"JSON 配置文件结构不合法：{path.name}")
    return payload


def _copytree_replace(source: Path, target: Path, data_dir: Path) -> None:
    resolved_target = target.resolve()
    resolved_data_dir = data_dir.resolve()
    if (
        resolved_target.parent != resolved_data_dir
        or resolved_target.name != "databases"
    ):
        raise PragPackageError("refusing to restore an unsafe database directory")
    if target.exists():
        shutil.rmtree(target)
    if source.exists():
        shutil.copytree(source, target)
    else:
        target.mkdir(parents=True, exist_ok=True)


def _reset_database_type_directory(target: Path, data_dir: Path) -> None:
    resolved_target = target.resolve()
    resolved_data_dir = data_dir.resolve()
    if (
        resolved_target.parent.parent.parent != resolved_data_dir
        or resolved_target.parent.parent.name != "databases"
    ):
        raise PragPackageError("refusing to replace an unsafe database type directory")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)


async def _backup_current_state(
    *,
    config_path: Path,
    manager: DatabaseManager,
    rollback_dir: Path,
) -> None:
    await run_blocking(
        rollback_dir.mkdir,
        parents=True,
        exist_ok=False,
    )
    if config_path.exists():
        await run_blocking(
            shutil.copy2,
            config_path,
            rollback_dir / "config.json",
        )
    if manager.system_path.exists():
        await run_blocking(
            sqlite_backup, manager.system_path, rollback_dir / "personalityrag_system.db"
        )
    databases_root = manager.data_dir / "databases"
    if databases_root.exists():
        await run_blocking(
            shutil.copytree, databases_root, rollback_dir / "databases"
        )


async def _restore_rollback(
    *,
    config_path: Path,
    manager: DatabaseManager,
    rollback_dir: Path,
) -> None:
    backup_config = rollback_dir / "config.json"
    if backup_config.exists():
        await run_blocking(
            config_path.parent.mkdir,
            parents=True,
            exist_ok=True,
        )
        await run_blocking(shutil.copy2, backup_config, config_path)
    backup_system = rollback_dir / "personalityrag_system.db"
    if backup_system.exists():
        for suffix in ("-wal", "-shm"):
            await run_blocking(
                Path(str(manager.system_path) + suffix).unlink,
                missing_ok=True,
            )
        await run_blocking(sqlite_backup, backup_system, manager.system_path)
    await run_blocking(
        _copytree_replace,
        rollback_dir / "databases",
        manager.data_dir / "databases",
        manager.data_dir,
    )


async def _close_runtimes(manager: DatabaseManager) -> None:
    for library_id in list(manager.runtimes):
        await manager.unload_runtime(library_id, reason="backup_migration")


async def _prepare_library_files(
    *,
    manager: DatabaseManager,
    extract_dir: Path,
    manifest: dict[str, Any],
    available_provider_ids: set[str],
) -> dict[str, Any]:
    libraries = []
    missing_providers: set[str] = set()
    for entry in manifest.get("libraries") or []:
        database_type = str(
            entry.get("database_type") or LIVINGMEMORY_V8_TYPE
        )
        try:
            driver = database_type_registry.require(database_type)
        except KeyError as exc:
            raise PragPackageError(f"配置包包含未注册的数据库类型：{database_type}") from exc
        base = extract_dir / _safe_member(entry.get("path") or "")
        library_payload = await run_blocking(
            _load_json,
            base / "library.json",
        )
        library = dict(library_payload.get("library") or {})
        if str(library.get("id") or "") != str(entry.get("id") or ""):
            raise PragPackageError("配置包记忆库 ID 与条目不一致")
        payload_type = str(library.get("database_type") or database_type)
        if payload_type != database_type:
            raise PragPackageError("配置包数据库类型与条目不一致")
        library["database_type"] = database_type
        provider_id = str(library.get("provider_id") or "")
        if provider_id and provider_id not in available_provider_ids:
            missing_providers.add(provider_id)
        rerank_provider_id = str(library.get("rerank_provider_id") or "")
        if rerank_provider_id and rerank_provider_id not in available_provider_ids:
            missing_providers.add(rerank_provider_id)
        files = dict(entry.get("files") or {})
        if database_type == TEXT_MEDIA_V1_TYPE:
            native_package = extract_dir / _safe_member(files["native_package"])
            await run_blocking(inspect_tmkb, native_package)
            async with _temporary_directory("prag-tmkb-validate-") as validate_dir:
                await run_blocking(extract_tmkb, native_package, validate_dir)
            libraries.append(
                {
                    "library": library,
                    "driver": driver,
                    "empty": bool(entry.get("empty")),
                    "native_package": native_package,
                }
            )
            continue
        livingmemory_db = (
            extract_dir / _safe_member(files["livingmemory_db"])
            if files.get("livingmemory_db")
            else None
        )
        conversations_db = (
            extract_dir / _safe_member(files["conversations_db"])
            if files.get("conversations_db")
            else None
        )
        if livingmemory_db is not None:
            await run_blocking(
                validate_livingmemory_db_file,
                livingmemory_db,
            )
        if conversations_db is not None:
            await run_blocking(
                validate_conversations_db_file,
                conversations_db,
            )
        libraries.append(
            {
                "library": library,
                "driver": driver,
                "empty": bool(entry.get("empty")),
                "livingmemory_db": livingmemory_db,
                "conversations_db": conversations_db,
            }
        )
    return {
        "libraries": libraries,
        "missing_providers": sorted(missing_providers),
    }


async def _install_library_files(
    *, manager: DatabaseManager, prepared: dict[str, Any]
) -> None:
    memory_root = database_type_registry.type_root(
        manager.data_dir, LIVINGMEMORY_V8_TYPE
    )
    text_root = database_type_registry.type_root(manager.data_dir, TEXT_MEDIA_V1_TYPE)
    await run_blocking(
        _reset_database_type_directory,
        memory_root,
        manager.data_dir,
    )
    await run_blocking(
        _reset_database_type_directory,
        text_root,
        manager.data_dir,
    )
    await manager.control.delete_database_identities_by_type(TEXT_MEDIA_V1_TYPE)
    for item in prepared.get("libraries") or []:
        library_id = str(item["library"]["id"])
        if item["library"]["database_type"] == TEXT_MEDIA_V1_TYPE:
            await install_tmkb(
                manager=manager._managers[TEXT_MEDIA_V1_TYPE],
                package_path=item["native_package"],
                target_id=library_id,
                name_override=str(item["library"].get("name") or library_id),
            )
            continue
        library_dir = database_type_registry.data_dir(
            manager.data_dir,
            DatabaseRef(item["library"]["database_type"], library_id),
        )
        await run_blocking(
            library_dir.mkdir,
            parents=True,
            exist_ok=True,
        )
        livingmemory_db = item.get("livingmemory_db")
        conversations_db = item.get("conversations_db")
        if livingmemory_db is not None:
            await run_blocking(
                sqlite_backup, livingmemory_db, library_dir / "livingmemory.db"
            )
        if conversations_db is not None:
            await run_blocking(
                sqlite_backup, conversations_db, library_dir / "conversations.db"
            )
        if livingmemory_db is None and conversations_db is None:
            storage = Storage(library_dir, system_path=manager.system_path)
            await storage.initialize()
        index_dir = library_dir / "indexes"
        if index_dir.exists():
            await run_blocking(shutil.rmtree, index_dir)


async def import_prag_package(
    *,
    root: Path,
    config_path: Path,
    config: AppConfig,
    manager: DatabaseManager,
    package_path: Path,
    password: str,
) -> tuple[AppConfig, dict[str, Any]]:
    _package_password(password)
    if await manager.control.has_any_running_jobs():
        raise PragPackageError("存在运行中任务，暂时不能导入配置包")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    rollback_dir = manager.data_dir / "backup_migration_rollbacks" / timestamp
    async with _temporary_directory("prag-import-") as extract_dir:
        manifest = await run_blocking(
            _extract_verified_package, package_path, password, extract_dir
        )
        scope = dict(manifest.get("scope") or {})
        global_config = await run_blocking(
            _load_json,
            extract_dir / "config" / "global.json",
        )
        next_config = app_config_from_dict(
            global_config,
            provider=(
                config.provider
                if config.bootstrap_provider_enabled
                else None
            ),
        )
        provider_snapshot = None
        if scope.get("include_providers"):
            provider_snapshot = await run_blocking(
                _load_json,
                extract_dir / "providers" / "providers.json",
            )
            context_length_migration = _normalize_provider_snapshot_context(
                provider_snapshot
            )
        else:
            context_length_migration = {
                "auto_pending": 0,
                "manual_fallback": 0,
                "legacy_manual": 0,
            }
        prepared_libraries = None
        if scope.get("include_libraries"):
            current_providers = await manager.control.export_provider_snapshot()
            available_provider_ids = {
                str(item.get("id") or "")
                for item in (provider_snapshot or current_providers).get("providers", [])
            }
            prepared_libraries = await _prepare_library_files(
                manager=manager,
                extract_dir=extract_dir,
                manifest=manifest,
                available_provider_ids=available_provider_ids,
            )
        await _close_runtimes(manager)
        try:
            await _backup_current_state(
                config_path=config_path,
                manager=manager,
                rollback_dir=rollback_dir,
            )
        except Exception:
            await manager.refresh_default_library(load=True)
            raise
        try:
            save_config(config_path, next_config)
            manager.config = next_config
            if provider_snapshot is not None:
                await manager.control.restore_provider_snapshot(provider_snapshot)
            if prepared_libraries is not None:
                await manager.control.restore_library_snapshot(
                    {
                        "libraries": [
                            item["library"]
                            for item in prepared_libraries["libraries"]
                            if item["library"]["database_type"]
                            == LIVINGMEMORY_V8_TYPE
                        ]
                    }
                )
                await _install_library_files(
                    manager=manager, prepared=prepared_libraries
                )
            await manager.refresh_default_library(load=False)
            if rollback_dir.exists():
                await run_blocking(
                    shutil.rmtree,
                    rollback_dir,
                    ignore_errors=True,
                )
        except Exception:
            logger.exception("备份迁移配置包导入失败，正在回滚")
            await _close_runtimes(manager)
            await manager.control.pool.close()
            await _restore_rollback(
                config_path=config_path,
                manager=manager,
                rollback_dir=rollback_dir,
            )
            manager.config = config
            await manager.refresh_default_library(load=True)
            raise
    result = {
        "scope": scope,
        "default_library_id": manifest.get("default_library_id") or "",
        "libraries_imported": len((prepared_libraries or {}).get("libraries") or []),
        "providers_imported": len((provider_snapshot or {}).get("providers") or []),
        "context_length_migration": context_length_migration,
        "missing_provider_ids": (prepared_libraries or {}).get(
            "missing_providers", []
        ),
        "indexes_pending": bool(scope.get("include_libraries")),
        "restart_required": True,
    }
    logger.warning(
        "备份迁移配置包导入完成：include_libraries=%s include_providers=%s libraries=%s providers=%s",
        bool(scope.get("include_libraries")),
        bool(scope.get("include_providers")),
        result["libraries_imported"],
        result["providers_imported"],
    )
    return next_config, result
