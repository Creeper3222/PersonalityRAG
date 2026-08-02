from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import tempfile
import time
import zipfile
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from ...database_types import TEXT_MEDIA_V1_TYPE, DatabaseRef, database_type_registry
from ...io_utils import run_blocking
from ..livingmemory_v8.migration import sqlite_backup
from .images import build_preview
from .indexes import TextMediaIndex
from .retrieval import normalize_retrieval_config
from .storage import DATABASE_FILENAME, TextMediaStorage
from .visual_intent_policy import (
    VISUAL_INTENT_POLICY_FILENAME,
    VISUAL_INTENT_POLICY_KEYS,
    migrate_legacy_visual_intent_policy,
    parse_visual_intent_policy_csv,
    read_visual_intent_policy_override,
)


TMKB_FORMAT = "personalityrag.text_media_knowledge.tmkb"
TMKB_VERSION = 1
MAX_PACKAGE_BYTES = 2 * 1024 * 1024 * 1024
MAX_EXTRACTED_BYTES = 8 * 1024 * 1024 * 1024
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class TmkbPackageError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreparedTmkbInstall:
    ref: DatabaseRef
    root: Path
    manifest: dict[str, Any]


def _safe_name(value: str) -> str:
    candidate = str(value or "").replace("\\", "/")
    path = PurePosixPath(candidate)
    if (
        not candidate
        or candidate.startswith("/")
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or ":" in path.parts[0]
    ):
        raise TmkbPackageError(f"不安全的封包路径：{candidate}")
    return path.as_posix()


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    return ((int(info.external_attr) >> 16) & 0o170000) == stat.S_IFLNK


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _database_summary(path: Path) -> dict[str, Any]:
    with closing(sqlite3.connect(path)) as db:
        db.row_factory = sqlite3.Row
        integrity = [row[0] for row in db.execute("PRAGMA integrity_check")]
        foreign_keys = list(db.execute("PRAGMA foreign_key_check"))
        if integrity != ["ok"] or foreign_keys:
            raise TmkbPackageError("知识库数据库完整性校验失败")
        meta = db.execute("SELECT * FROM library_meta WHERE singleton=1").fetchone()
        if meta is None:
            raise TmkbPackageError("知识库数据库缺少元数据")
        meta_dict = dict(meta)
        try:
            normalize_retrieval_config(meta_dict.get("retrieval_config_json"))
        except ValueError as exc:
            raise TmkbPackageError("知识库检索融合配置无效") from exc
        counts = {
            table: int(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("documents", "entries", "chunks", "assets")
        }
        existing_tables = {
            str(row[0])
            for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        for key, table in (
            ("ingest_batches", "ingest_batches"),
            ("batch_documents", "ingest_batch_documents"),
            ("batch_assets", "ingest_batch_assets"),
            ("document_asset_relations", "document_assets"),
            ("entry_asset_relations", "entry_assets"),
            ("chunk_asset_relations", "chunk_assets"),
            ("chunk_media_strengths", "chunk_media_strengths"),
            ("asset_media_metadata", "asset_media_metadata"),
            ("asset_media_tokens", "asset_media_tokens"),
        ):
            if table in existing_tables:
                counts[key] = int(
                    db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                )
        active = db.execute(
            "SELECT generation_id FROM active_generation WHERE singleton=1"
        ).fetchone()
        vector_digest = hashlib.sha256()
        vector_count = 0
        dimensions = 0
        if active:
            rows = db.execute(
                "SELECT chunk_id,vector,vector_sha256 FROM chunk_embeddings WHERE generation_id=? ORDER BY chunk_id",
                (active[0],),
            )
            for row in rows:
                data = bytes(row[1])
                if hashlib.sha256(data).hexdigest() != row[2] or len(data) % 4:
                    raise TmkbPackageError("知识库向量校验失败")
                values = np.frombuffer(data, dtype="<f4")
                if not values.size or not np.isfinite(values).all():
                    raise TmkbPackageError("知识库向量包含无效数值")
                dimensions = dimensions or int(values.size)
                if dimensions != int(values.size):
                    raise TmkbPackageError("知识库向量维度不一致")
                vector_digest.update(str(int(row[0])).encode("ascii"))
                vector_digest.update(data)
                vector_count += 1
        media_vector_digest = hashlib.sha256()
        media_vector_count = 0
        media_vector_dimensions = 0
        document_asset_columns = {
            str(row[1]) for row in db.execute("PRAGMA table_info(document_assets)")
        }
        if "media_description_vector" in document_asset_columns:
            rows = db.execute(
                """SELECT document_id,asset_id,media_description_vector
                FROM document_assets WHERE media_description_vector IS NOT NULL
                ORDER BY document_id,asset_id"""
            )
            for row in rows:
                data = bytes(row[2])
                if not data or len(data) % 4:
                    raise TmkbPackageError("知识库媒体描述向量校验失败")
                values = np.frombuffer(data, dtype="<f4")
                if not values.size or not np.isfinite(values).all():
                    raise TmkbPackageError("知识库媒体描述向量包含无效数值")
                media_vector_dimensions = media_vector_dimensions or int(values.size)
                if media_vector_dimensions != int(values.size):
                    raise TmkbPackageError("知识库媒体描述向量维度不一致")
                if dimensions and dimensions != int(values.size):
                    raise TmkbPackageError("媒体描述与分块向量维度不一致")
                media_vector_digest.update(str(row[0]).encode("utf-8"))
                media_vector_digest.update(str(row[1]).encode("utf-8"))
                media_vector_digest.update(data)
                media_vector_count += 1
        if "asset_media_metadata" in existing_tables:
            rows = db.execute(
                """SELECT id,asset_id,sort_order,media_description_vector,vector_sha256
                FROM asset_media_metadata
                WHERE media_description_vector IS NOT NULL
                ORDER BY asset_id,sort_order,id"""
            )
            for row in rows:
                data = bytes(row[3])
                if (
                    not data
                    or len(data) % 4
                    or hashlib.sha256(data).hexdigest() != str(row[4] or "")
                ):
                    raise TmkbPackageError("知识库资产级媒体向量校验失败")
                values = np.frombuffer(data, dtype="<f4")
                if not values.size or not np.isfinite(values).all():
                    raise TmkbPackageError("知识库资产级媒体向量包含无效数值")
                media_vector_dimensions = media_vector_dimensions or int(values.size)
                if media_vector_dimensions != int(values.size):
                    raise TmkbPackageError("知识库媒体描述向量维度不一致")
                if dimensions and dimensions != int(values.size):
                    raise TmkbPackageError("媒体描述与分块向量维度不一致")
                media_vector_digest.update(b"asset_metadata\0")
                media_vector_digest.update(str(row[1]).encode("utf-8"))
                media_vector_digest.update(b"\0")
                media_vector_digest.update(str(int(row[2])).encode("ascii"))
                media_vector_digest.update(b"\0")
                media_vector_digest.update(str(int(row[0])).encode("ascii"))
                media_vector_digest.update(data)
                media_vector_count += 1
        assets = [dict(row) for row in db.execute("SELECT * FROM assets WHERE state='active' ORDER BY kind,id")]
        return {
            "meta": meta_dict,
            "counts": counts,
            "assets": assets,
            "active_generation": str(active[0]) if active else None,
            "vector_count": vector_count,
            "vector_dimensions": dimensions,
            "vector_sha256": vector_digest.hexdigest(),
            "media_vector_count": media_vector_count,
            "media_vector_dimensions": media_vector_dimensions,
            "media_vector_sha256": media_vector_digest.hexdigest(),
        }


def _write_tmkb(source_root: Path, database_snapshot: Path, target: Path) -> dict[str, Any]:
    summary = _database_summary(database_snapshot)
    members: dict[str, Path] = {f"database/{DATABASE_FILENAME}": database_snapshot}
    visual_policy = read_visual_intent_policy_override(source_root)
    if visual_policy is not None:
        members[f"settings/{VISUAL_INTENT_POLICY_FILENAME}"] = (
            source_root / VISUAL_INTENT_POLICY_FILENAME
        )
    for asset in summary["assets"]:
        storage_key = _safe_name(str(asset["storage_key"]))
        if not storage_key.startswith(("assets/documents/", "assets/images/")):
            raise TmkbPackageError("知识库资源路径不在允许目录")
        source = source_root / Path(*PurePosixPath(storage_key).parts)
        if not source.is_file():
            raise TmkbPackageError(f"知识库资源缺失：{storage_key}")
        digest, size = _hash_file(source)
        if digest != asset["sha256"] or size != int(asset["size_bytes"]):
            raise TmkbPackageError(f"知识库资源校验失败：{storage_key}")
        members[storage_key] = source

    file_manifest: dict[str, dict[str, Any]] = {}
    for name, path in sorted(members.items()):
        digest, size = _hash_file(path)
        file_manifest[name] = {"sha256": digest, "size": size}
    meta = summary["meta"]
    manifest = {
        "format": TMKB_FORMAT,
        "version": TMKB_VERSION,
        "created_at": time.time(),
        "database_type": TEXT_MEDIA_V1_TYPE,
        "database_id": meta["database_id"],
        "name": meta["name"],
        "description": meta["description"],
        "provider": {
            "id": meta["provider_id"],
            "revision": meta["provider_revision"],
            "fingerprint": meta["provider_fingerprint"],
        },
        "rerank_provider": {
            "id": meta.get("rerank_provider_id") or "",
            "revision": int(meta.get("rerank_provider_revision") or 0),
            "fingerprint": meta.get("rerank_provider_fingerprint") or "",
        },
        "image_profile": {
            "format": "webp",
            "max_long_edge": 1536,
            "initial_quality": 80,
            "max_bytes": 1536 * 1024,
        },
        "counts": summary["counts"],
        "vectors": {
            "generation": summary["active_generation"],
            "count": summary["vector_count"],
            "dimensions": summary["vector_dimensions"],
            "sha256": summary["vector_sha256"],
        },
        "media_vectors": {
            "count": summary["media_vector_count"],
            "dimensions": summary["media_vector_dimensions"],
            "sha256": summary["media_vector_sha256"],
        },
        "files": file_manifest,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            archive.writestr(
                "manifest.json",
                json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8"),
            )
            for name, path in sorted(members.items()):
                archive.write(path, name)
        if temporary.stat().st_size > MAX_PACKAGE_BYTES:
            raise TmkbPackageError(".tmkb 封包超过 2 GiB")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


async def export_tmkb(*, service: Any, target: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="tmkb-export-") as raw:
        snapshot = Path(raw) / DATABASE_FILENAME
        await run_blocking(sqlite_backup, service.storage.path, snapshot)
        manifest = await run_blocking(_write_tmkb, service.root, snapshot, target)
    await run_blocking(inspect_tmkb, target)
    return {
        "path": str(target),
        "filename": target.name,
        "size_bytes": target.stat().st_size,
        "sha256": (await run_blocking(_hash_file, target))[0],
        "manifest": manifest,
    }


def _read_manifest(archive: zipfile.ZipFile) -> tuple[dict[str, Any], dict[str, zipfile.ZipInfo]]:
    infos: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        name = _safe_name(info.filename)
        if info.is_dir() or _is_symlink(info) or name in infos:
            raise TmkbPackageError(".tmkb 包含目录、符号链接或重复成员")
        infos[name] = info
    info = infos.get("manifest.json")
    if info is None or info.file_size > MAX_MANIFEST_BYTES:
        raise TmkbPackageError(".tmkb 缺少有效 manifest.json")
    try:
        manifest = json.loads(archive.read(info).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RuntimeError) as exc:
        raise TmkbPackageError(".tmkb manifest.json 已损坏") from exc
    if not isinstance(manifest, dict):
        raise TmkbPackageError(".tmkb manifest.json 结构无效")
    return manifest, infos


def _validate_manifest(manifest: dict[str, Any], infos: dict[str, zipfile.ZipInfo]) -> dict[str, dict[str, Any]]:
    if manifest.get("format") != TMKB_FORMAT or int(manifest.get("version") or 0) != TMKB_VERSION:
        raise TmkbPackageError("不是受支持的文本媒体知识库封包")
    if manifest.get("database_type") != TEXT_MEDIA_V1_TYPE:
        raise TmkbPackageError(".tmkb 数据库类型不匹配")
    files = manifest.get("files")
    if not isinstance(files, dict) or f"database/{DATABASE_FILENAME}" not in files:
        raise TmkbPackageError(".tmkb 文件清单不完整")
    declared: dict[str, dict[str, Any]] = {}
    total = 0
    for raw_name, value in files.items():
        name = _safe_name(raw_name)
        if name == "manifest.json" or not isinstance(value, dict):
            raise TmkbPackageError(".tmkb 文件清单无效")
        digest = str(value.get("sha256") or "").lower()
        try:
            size = int(value.get("size"))
        except (TypeError, ValueError) as exc:
            raise TmkbPackageError(".tmkb 文件尺寸无效") from exc
        if size < 0 or not SHA256_RE.fullmatch(digest):
            raise TmkbPackageError(".tmkb 文件校验信息无效")
        total += size
        if total > MAX_EXTRACTED_BYTES:
            raise TmkbPackageError(".tmkb 解压后超过安全上限")
        declared[name] = {"sha256": digest, "size": size}
    actual = set(infos) - {"manifest.json"}
    if actual != set(declared):
        raise TmkbPackageError(".tmkb 实际成员与清单不一致")
    return declared


def inspect_tmkb(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > MAX_PACKAGE_BYTES:
        raise TmkbPackageError(".tmkb 文件不存在或超过安全上限")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            manifest, infos = _read_manifest(archive)
            _validate_manifest(manifest, infos)
            return manifest
    except zipfile.BadZipFile as exc:
        raise TmkbPackageError("不是有效的 .tmkb ZIP 封包") from exc


def extract_tmkb(path: Path, target: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path, "r") as archive:
        manifest, infos = _read_manifest(archive)
        declared = _validate_manifest(manifest, infos)
        root = target.resolve()
        for name, expected in sorted(declared.items()):
            info = infos[name]
            if int(info.file_size) != expected["size"]:
                raise TmkbPackageError(f".tmkb 成员尺寸不一致：{name}")
            destination = target.joinpath(*PurePosixPath(name).parts)
            resolved = destination.resolve()
            if root not in resolved.parents:
                raise TmkbPackageError(".tmkb 成员路径越界")
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            size = 0
            with archive.open(info, "r") as source, destination.open("xb") as output:
                while chunk := source.read(CHUNK_BYTES):
                    size += len(chunk)
                    if size > expected["size"]:
                        raise TmkbPackageError(f".tmkb 成员超过声明尺寸：{name}")
                    digest.update(chunk)
                    output.write(chunk)
            if size != expected["size"] or digest.hexdigest() != expected["sha256"]:
                raise TmkbPackageError(f".tmkb 成员校验失败：{name}")
    database = target / "database" / DATABASE_FILENAME
    summary = _database_summary(database)
    meta = summary["meta"]
    if meta["database_id"] != manifest.get("database_id") or meta["name"] != manifest.get("name"):
        raise TmkbPackageError(".tmkb manifest 与数据库元数据不一致")
    if summary["counts"] != manifest.get("counts"):
        raise TmkbPackageError(".tmkb 数据计数不一致")
    vectors = manifest.get("vectors") or {}
    if (
        summary["vector_count"] != int(vectors.get("count") or 0)
        or summary["vector_dimensions"] != int(vectors.get("dimensions") or 0)
        or summary["vector_sha256"] != str(vectors.get("sha256") or "")
    ):
        raise TmkbPackageError(".tmkb 向量清单不一致")
    media_vectors = manifest.get("media_vectors") or {}
    if (
        summary["media_vector_count"] != int(media_vectors.get("count") or 0)
        or summary["media_vector_dimensions"]
        != int(media_vectors.get("dimensions") or 0)
        or summary["media_vector_sha256"]
        != str(media_vectors.get("sha256") or hashlib.sha256().hexdigest())
    ):
        raise TmkbPackageError(".tmkb 媒体描述向量清单不一致")
    for asset in summary["assets"]:
        member = target.joinpath(*PurePosixPath(_safe_name(asset["storage_key"])).parts)
        digest, size = _hash_file(member)
        if digest != asset["sha256"] or size != int(asset["size_bytes"]):
            raise TmkbPackageError(".tmkb 资源闭包校验失败")
    visual_policy_member = (
        target / "settings" / VISUAL_INTENT_POLICY_FILENAME
    )
    if visual_policy_member.exists():
        try:
            _policy, categories = parse_visual_intent_policy_csv(
                visual_policy_member.read_bytes(),
                require_complete=True,
            )
            if categories != VISUAL_INTENT_POLICY_KEYS:
                raise ValueError("visual intent policy CSV columns are out of order")
        except (OSError, ValueError) as exc:
            raise TmkbPackageError(".tmkb 视觉意图词表 CSV 无效") from exc
    return manifest


async def prepare_tmkb_install(
    *,
    manager: Any,
    package_path: Path,
    target_id: str,
    name_override: str | None = None,
    staging: Path,
) -> PreparedTmkbInstall:
    ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, target_id)
    if await manager.control.database_identity(ref):
        raise TmkbPackageError(f"知识库 ID {target_id} 已存在")
    extracted = staging / "extracted"
    manifest = await run_blocking(extract_tmkb, package_path, extracted)
    library_root = staging / "library"
    (library_root / "assets").mkdir(parents=True, exist_ok=True)
    await run_blocking(
        shutil.copy2,
        extracted / "database" / DATABASE_FILENAME,
        library_root / DATABASE_FILENAME,
    )
    if (extracted / "assets").exists():
        await run_blocking(
            shutil.copytree,
            extracted / "assets",
            library_root / "assets",
            dirs_exist_ok=True,
        )
    visual_policy = (
        extracted / "settings" / VISUAL_INTENT_POLICY_FILENAME
    )
    if visual_policy.exists():
        await run_blocking(
            shutil.copy2,
            visual_policy,
            library_root / VISUAL_INTENT_POLICY_FILENAME,
        )
    storage = TextMediaStorage(library_root)
    try:
        await storage.initialize()
        await migrate_legacy_visual_intent_policy(storage, library_root)
        provider_info = dict(manifest.get("provider") or {})
        revision = await manager.control.get_provider(
            str(provider_info.get("id") or ""), int(provider_info.get("revision") or 0)
        ) if provider_info.get("id") else None
        matches = bool(revision and revision.config_sha256 == provider_info.get("fingerprint") and revision.config.enabled)
        await storage.update_metadata(
            {
                "database_id": target_id,
                "name": (name_override or str(manifest.get("name") or target_id)).strip(),
                "status": "ready" if matches else "provider_binding_required",
            }
        )
        indexes = TextMediaIndex(library_root)
        await indexes.rebuild(storage)
        for asset in await storage.list_assets(kind="image"):
            source = library_root.joinpath(*PurePosixPath(asset["storage_key"]).parts)
            preview = library_root / "derived" / "previews" / f"{asset['sha256']}.webp"
            await run_blocking(build_preview, source, preview)
        await storage.validate()
    finally:
        await storage.close()
    return PreparedTmkbInstall(ref=ref, root=library_root, manifest=manifest)


async def install_tmkb(
    *,
    manager: Any,
    package_path: Path,
    target_id: str,
    name_override: str | None = None,
) -> dict[str, Any]:
    ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, target_id)
    final = database_type_registry.data_dir(
        manager.data_dir,
        DatabaseRef(TEXT_MEDIA_V1_TYPE, target_id),
    )
    parent = final.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target_id}.import-", dir=parent))
    installed = False
    try:
        prepared = await prepare_tmkb_install(
            manager=manager,
            package_path=package_path,
            target_id=target_id,
            name_override=name_override,
            staging=staging,
        )
        await manager.control.register_database_identity(ref, category="knowledge")
        try:
            await run_blocking(os.replace, prepared.root, final)
        except Exception:
            await manager.control.delete_database_identity(ref)
            raise
        installed = True
        return await manager.library_detail(target_id)
    finally:
        if not installed and final.exists() and not await manager.control.database_identity(ref):
            await run_blocking(shutil.rmtree, final, True)
        await run_blocking(shutil.rmtree, staging, True)
