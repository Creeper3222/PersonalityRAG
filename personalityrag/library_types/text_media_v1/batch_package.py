from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Awaitable, Callable

from ...database_types import (
    DATABASE_CATEGORY_KNOWLEDGE,
    TEXT_MEDIA_V1_TYPE,
    DatabaseRef,
    database_type_registry,
)
from ...io_utils import run_blocking
from .package import (
    CHUNK_BYTES,
    MAX_MANIFEST_BYTES,
    SHA256_RE,
    TmkbPackageError,
    export_tmkb,
    inspect_tmkb,
    prepare_tmkb_install,
)


TMKBS_FORMAT = "personalityrag.text_media_knowledge.tmkbs"
TMKBS_VERSION = 1
MAX_BATCH_PACKAGE_BYTES = 8 * 1024 * 1024 * 1024
MAX_BATCH_LIBRARIES = 100


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
        raise TmkbPackageError(f"unsafe .tmkbs member path: {candidate}")
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


def _read_archive(path: Path) -> tuple[dict[str, Any], dict[str, zipfile.ZipInfo]]:
    if not path.is_file() or path.stat().st_size > MAX_BATCH_PACKAGE_BYTES:
        raise TmkbPackageError(".tmkbs file is missing or exceeds the safety limit")
    try:
        archive = zipfile.ZipFile(path, "r")
    except zipfile.BadZipFile as exc:
        raise TmkbPackageError("invalid .tmkbs ZIP package") from exc
    try:
        infos: dict[str, zipfile.ZipInfo] = {}
        for info in archive.infolist():
            name = _safe_name(info.filename)
            if info.is_dir() or _is_symlink(info) or name in infos:
                raise TmkbPackageError(
                    ".tmkbs contains a directory, symbolic link, or duplicate member"
                )
            infos[name] = info
        manifest_info = infos.get("manifest.json")
        if manifest_info is None or manifest_info.file_size > MAX_MANIFEST_BYTES:
            raise TmkbPackageError(".tmkbs is missing a valid manifest.json")
        try:
            manifest = json.loads(archive.read(manifest_info).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, RuntimeError) as exc:
            raise TmkbPackageError(".tmkbs manifest.json is damaged") from exc
        if not isinstance(manifest, dict):
            raise TmkbPackageError(".tmkbs manifest.json has an invalid structure")
        return manifest, infos
    finally:
        archive.close()


def _validate_manifest(
    manifest: dict[str, Any], infos: dict[str, zipfile.ZipInfo]
) -> list[dict[str, Any]]:
    if (
        manifest.get("format") != TMKBS_FORMAT
        or int(manifest.get("version") or 0) != TMKBS_VERSION
    ):
        raise TmkbPackageError("unsupported .tmkbs package")
    raw_libraries = manifest.get("libraries")
    if not isinstance(raw_libraries, list) or not 2 <= len(raw_libraries) <= MAX_BATCH_LIBRARIES:
        raise TmkbPackageError(".tmkbs must contain between 2 and 100 libraries")
    declared: list[dict[str, Any]] = []
    names: set[str] = set()
    source_ids: set[str] = set()
    total = 0
    for raw in raw_libraries:
        if not isinstance(raw, dict):
            raise TmkbPackageError(".tmkbs library manifest entry is invalid")
        source_id = str(raw.get("database_id") or "")
        name = _safe_name(str(raw.get("member") or ""))
        digest = str(raw.get("sha256") or "").lower()
        try:
            size = int(raw.get("size"))
        except (TypeError, ValueError) as exc:
            raise TmkbPackageError(".tmkbs member size is invalid") from exc
        if (
            not source_id
            or name != f"libraries/{source_id}.tmkb"
            or name in names
            or source_id in source_ids
            or size < 0
            or not SHA256_RE.fullmatch(digest)
        ):
            raise TmkbPackageError(".tmkbs library manifest entry is invalid")
        info = infos.get(name)
        if info is None or int(info.file_size) != size:
            raise TmkbPackageError(f".tmkbs member size mismatch: {name}")
        total += size
        if total > MAX_BATCH_PACKAGE_BYTES:
            raise TmkbPackageError(".tmkbs extracted members exceed the safety limit")
        item = dict(raw)
        item.update({"database_id": source_id, "member": name, "size": size, "sha256": digest})
        declared.append(item)
        names.add(name)
        source_ids.add(source_id)
    if set(infos) != {"manifest.json", *names}:
        raise TmkbPackageError(".tmkbs contains undeclared members")
    return declared


def write_tmkbs(packages: list[Path], target: Path) -> dict[str, Any]:
    if not 2 <= len(packages) <= MAX_BATCH_LIBRARIES:
        raise TmkbPackageError("batch export requires between 2 and 100 libraries")
    libraries: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    for package in packages:
        child = inspect_tmkb(package)
        source_id = str(child.get("database_id") or "")
        if not source_id or source_id in source_ids:
            raise TmkbPackageError("batch export contains duplicate library IDs")
        digest, size = _hash_file(package)
        libraries.append(
            {
                "database_id": source_id,
                "name": str(child.get("name") or source_id),
                "member": f"libraries/{source_id}.tmkb",
                "size": size,
                "sha256": digest,
            }
        )
        source_ids.add(source_id)
    manifest = {
        "format": TMKBS_FORMAT,
        "version": TMKBS_VERSION,
        "created_at": time.time(),
        "database_type": TEXT_MEDIA_V1_TYPE,
        "libraries": libraries,
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
            for item, package in zip(libraries, packages, strict=True):
                archive.write(
                    package,
                    item["member"],
                    compress_type=zipfile.ZIP_STORED,
                )
        if temporary.stat().st_size > MAX_BATCH_PACKAGE_BYTES:
            raise TmkbPackageError(".tmkbs package exceeds 8 GiB")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def extract_tmkbs(path: Path, target: Path) -> dict[str, Any]:
    manifest, infos = _read_archive(path)
    declared = _validate_manifest(manifest, infos)
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    with zipfile.ZipFile(path, "r") as archive:
        for item in declared:
            destination = target.joinpath(*PurePosixPath(item["member"]).parts)
            resolved = destination.resolve()
            if root not in resolved.parents:
                raise TmkbPackageError(".tmkbs member path escaped the target directory")
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            size = 0
            with archive.open(infos[item["member"]], "r") as source, destination.open("xb") as output:
                while chunk := source.read(CHUNK_BYTES):
                    size += len(chunk)
                    if size > item["size"]:
                        raise TmkbPackageError(f".tmkbs member exceeds declared size: {item['member']}")
                    digest.update(chunk)
                    output.write(chunk)
            if size != item["size"] or digest.hexdigest() != item["sha256"]:
                raise TmkbPackageError(f".tmkbs member hash mismatch: {item['member']}")
            child = inspect_tmkb(destination)
            if str(child.get("database_id") or "") != item["database_id"]:
                raise TmkbPackageError(".tmkbs child manifest does not match the outer manifest")
            item["manifest"] = child
    return {**manifest, "libraries": declared}


def inspect_tmkbs(path: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="tmkbs-inspect-") as raw:
        return extract_tmkbs(path, Path(raw))


async def export_tmkbs(
    *,
    manager: Any,
    database_ids: list[str],
    target: Path,
    progress: Callable[[float, str], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    unique = list(dict.fromkeys(database_ids))
    if len(unique) != len(database_ids) or not 2 <= len(unique) <= MAX_BATCH_LIBRARIES:
        raise TmkbPackageError("batch export requires 2 to 100 unique libraries")
    with tempfile.TemporaryDirectory(prefix="tmkbs-export-") as raw:
        root = Path(raw)
        packages: list[Path] = []
        for index, database_id in enumerate(unique):
            service = await manager.get_runtime(DatabaseRef(TEXT_MEDIA_V1_TYPE, database_id))
            package = root / f"{database_id}.tmkb"
            await export_tmkb(service=service, target=package)
            packages.append(package)
            if progress:
                await progress(0.1 + 0.7 * ((index + 1) / len(unique)), f"exported {index + 1}/{len(unique)}")
        manifest = await run_blocking(write_tmkbs, packages, target)
    verified = await run_blocking(inspect_tmkbs, target)
    digest, size = await run_blocking(_hash_file, target)
    return {
        "path": str(target),
        "filename": target.name,
        "size_bytes": size,
        "sha256": digest,
        "manifest": manifest,
        "libraries": verified["libraries"],
    }


async def install_tmkbs_atomic(
    *,
    manager: Any,
    package_path: Path,
    items: list[dict[str, Any]],
    progress: Callable[[float, str], Awaitable[None]] | None = None,
) -> list[dict[str, Any]]:
    if not items:
        raise TmkbPackageError("select at least one library to import")
    target_ids = [str(item.get("target_id") or "") for item in items]
    source_ids = [str(item.get("source_id") or "") for item in items]
    if len(set(target_ids)) != len(target_ids):
        raise TmkbPackageError("duplicate target library IDs in batch")
    if len(set(source_ids)) != len(source_ids):
        raise TmkbPackageError("duplicate source libraries in batch")
    driver = database_type_registry.require(TEXT_MEDIA_V1_TYPE)
    parent = database_type_registry.type_root(manager.data_dir, TEXT_MEDIA_V1_TYPE)
    parent.mkdir(parents=True, exist_ok=True)
    batch_root = Path(tempfile.mkdtemp(prefix=".tmkbs-import-", dir=parent))
    moved: list[tuple[DatabaseRef, Path]] = []
    registered = False
    try:
        extracted = batch_root / "package"
        manifest = await run_blocking(extract_tmkbs, package_path, extracted)
        available = {item["database_id"]: item for item in manifest["libraries"]}
        prepared = []
        for index, item in enumerate(items):
            source_id = str(item.get("source_id") or "")
            target_id = str(item.get("target_id") or "")
            child = available.get(source_id)
            if child is None:
                raise TmkbPackageError(f"source library is not present in package: {source_id}")
            ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, target_id)
            final = database_type_registry.data_dir(
                manager.data_dir,
                DatabaseRef(TEXT_MEDIA_V1_TYPE, target_id),
            )
            if await manager.control.database_identity(ref) or final.exists():
                raise TmkbPackageError(f"target library ID already exists: {target_id}")
            staging = batch_root / "staging" / target_id
            staging.mkdir(parents=True, exist_ok=False)
            prepared_item = await prepare_tmkb_install(
                manager=manager,
                package_path=extracted.joinpath(*PurePosixPath(child["member"]).parts),
                target_id=target_id,
                name_override=str(item.get("name") or "").strip() or None,
                staging=staging,
            )
            prepared.append(prepared_item)
            if progress:
                await progress(0.1 + 0.65 * ((index + 1) / len(items)), f"prepared {index + 1}/{len(items)}")
        refs = [item.ref for item in prepared]
        for prepared_item in prepared:
            final = database_type_registry.data_dir(
                manager.data_dir,
                prepared_item.ref,
            )
            await run_blocking(os.replace, prepared_item.root, final)
            moved.append((prepared_item.ref, final))
        await manager.control.register_database_identities(
            refs, category=DATABASE_CATEGORY_KNOWLEDGE
        )
        registered = True
        if progress:
            await progress(0.95, "registered imported libraries")
        return [await manager.library_detail(ref.id) for ref in refs]
    except Exception:
        if registered:
            for ref, _ in moved:
                await manager.control.delete_database_identity(ref)
        for ref, final in reversed(moved):
            await manager.unload_runtime(ref)
            if final.exists():
                await run_blocking(shutil.rmtree, final, True)
        raise
    finally:
        await run_blocking(shutil.rmtree, batch_root, True)
