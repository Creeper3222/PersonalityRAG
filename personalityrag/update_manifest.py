from __future__ import annotations

import hashlib
import json
import platform
import re
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .compat import LIVINGMEMORY_DATABASE_VERSION
from .version import (
    DOCKER_REPOSITORY,
    PLATFORM_NAME,
    PRODUCT_NAME,
    RELEASE_ROOT_NAME,
    TAG_NAME,
    VERSION,
)


MANIFEST_NAME = "update-manifest.json"
MANIFEST_FORMAT_VERSION = 1
CONFIG_SCHEMA_VERSION = 1
CONTROL_SCHEMA_VERSION = 1
MAX_RELEASE_BYTES = 512 * 1024 * 1024
MAX_RELEASE_FILES = 10_000
RELEASE_DIRECTORIES = ("assets", "docker", "personalityrag", "static")
RELEASE_FILES = (
    ".dockerignore",
    ".env.example",
    "Dockerfile",
    "docker-compose.local.yml",
    "docker-compose.yml",
    "requirements.txt",
    "requirements-runtime.lock",
    "run.py",
)
PROTECTED_TOP_LEVEL = frozenset(
    {".git", ".venv", "config", "data", "docs", "tests", "runtime"}
)
TAG_PATTERN = re.compile(
    r"^v(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<pre>[0-9A-Za-z.-]+))?$"
)
DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class UpdatePackageError(ValueError):
    pass


def parse_tag(tag: str) -> tuple[int, int, int, tuple[tuple[int, object], ...]]:
    match = TAG_PATTERN.fullmatch(str(tag or ""))
    if not match:
        raise UpdatePackageError("invalid semantic version tag")
    pre = match.group("pre")
    if pre is None:
        pre_key: tuple[tuple[int, object], ...] = ((1, ""),)
    else:
        parts: list[tuple[int, object]] = []
        for part in pre.split("."):
            parts.append((0, int(part)) if part.isdigit() else (1, part.lower()))
        pre_key = ((0, ""), *parts)
    return int(match.group("major")), int(match.group("minor")), int(match.group("patch")), pre_key


def compare_tags(left: str, right: str) -> int:
    return (parse_tag(left) > parse_tag(right)) - (parse_tag(left) < parse_tag(right))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _release_files(root: Path) -> Iterable[Path]:
    for directory in RELEASE_DIRECTORIES:
        base = root / directory
        if not base.is_dir():
            raise UpdatePackageError(f"missing release directory: {directory}")
        yield from (path for path in base.rglob("*") if path.is_file())
    for relative in RELEASE_FILES:
        path = root / relative
        if not path.is_file():
            raise UpdatePackageError(f"missing release file: {relative}")
        yield path


def _validate_digest(value: object, label: str) -> str:
    digest = str(value or "").lower()
    if not DIGEST_PATTERN.fullmatch(digest):
        raise UpdatePackageError(f"invalid Docker digest: {label}")
    return digest


def build_manifest(
    root: Path,
    *,
    version: str = VERSION,
    tag_name: str = TAG_NAME,
    source_commit: str,
    index_digest: str,
    amd64_digest: str,
    arm64_digest: str,
) -> dict[str, Any]:
    if tag_name != f"v{version}":
        raise UpdatePackageError("version and tag do not match")
    parse_tag(tag_name)
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise UpdatePackageError("invalid Linux-Docker source commit")
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(_release_files(root), key=lambda item: item.as_posix().lower())
    ]
    return {
        "manifest_format": MANIFEST_FORMAT_VERSION,
        "product": PRODUCT_NAME,
        "platform": PLATFORM_NAME,
        "version": version,
        "tag_name": tag_name,
        "root_directory": RELEASE_ROOT_NAME,
        "source": {"branch": "Linux-Docker", "commit": source_commit},
        "release_directories": list(RELEASE_DIRECTORIES),
        "release_files": list(RELEASE_FILES),
        "python": {"implementation": "cpython", "minimum": "3.12", "architecture_bits": 64},
        "persistent_compatibility": {
            "config_schema": {"minimum": CONFIG_SCHEMA_VERSION, "maximum": CONFIG_SCHEMA_VERSION},
            "control_schema": {"minimum": CONTROL_SCHEMA_VERSION, "maximum": CONTROL_SCHEMA_VERSION},
            "livingmemory_database": {
                "minimum": LIVINGMEMORY_DATABASE_VERSION,
                "maximum": LIVINGMEMORY_DATABASE_VERSION,
            },
        },
        "docker": {
            "repository": DOCKER_REPOSITORY,
            "tag": tag_name,
            "index_digest": _validate_digest(index_digest, "index"),
            "platforms": {
                "linux/amd64": _validate_digest(amd64_digest, "linux/amd64"),
                "linux/arm64": _validate_digest(arm64_digest, "linux/arm64"),
            },
        },
        "files": files,
    }


def write_manifest(root: Path, **kwargs: Any) -> Path:
    target = root / MANIFEST_NAME
    target.write_text(
        json.dumps(build_manifest(root, **kwargs), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def _validate_persistent_compatibility(contract: dict[str, Any]) -> None:
    current = {
        "config_schema": CONFIG_SCHEMA_VERSION,
        "control_schema": CONTROL_SCHEMA_VERSION,
        "livingmemory_database": LIVINGMEMORY_DATABASE_VERSION,
    }
    for key, value in current.items():
        bounds = contract.get(key) or {}
        try:
            minimum, maximum = int(bounds["minimum"]), int(bounds["maximum"])
        except (KeyError, TypeError, ValueError) as exc:
            raise UpdatePackageError(f"missing persistent compatibility range: {key}") from exc
        if not minimum <= value <= maximum:
            raise UpdatePackageError(f"target release is incompatible with {key}={value}")


def _validated_relative_path(value: object) -> str:
    text = str(value or "").replace("\\", "/")
    pure = PurePosixPath(text)
    if not text or pure.is_absolute() or ".." in pure.parts or ":" in text:
        raise UpdatePackageError("release contains an unsafe path")
    if pure.parts[0].lower() in PROTECTED_TOP_LEVEL:
        raise UpdatePackageError("release attempts to include a protected path")
    if pure.parts[0] not in RELEASE_DIRECTORIES and text not in RELEASE_FILES:
        raise UpdatePackageError(f"release contains an unknown path: {text}")
    return pure.as_posix()


def validate_manifest(
    manifest: dict[str, Any], *, expected_tag: str, candidate_root: Path | None = None
) -> dict[str, Any]:
    expected_version = expected_tag.removeprefix("v")
    if manifest.get("manifest_format") != MANIFEST_FORMAT_VERSION:
        raise UpdatePackageError("unsupported update manifest format")
    if manifest.get("product") != PRODUCT_NAME or manifest.get("platform") != PLATFORM_NAME:
        raise UpdatePackageError("release product or platform does not match")
    if manifest.get("tag_name") != expected_tag or manifest.get("version") != expected_version:
        raise UpdatePackageError("release tag and internal version do not match")
    if manifest.get("root_directory") != RELEASE_ROOT_NAME:
        raise UpdatePackageError("release root directory is invalid")
    if tuple(manifest.get("release_directories") or ()) != RELEASE_DIRECTORIES:
        raise UpdatePackageError("release directory contract does not match")
    if tuple(manifest.get("release_files") or ()) != RELEASE_FILES:
        raise UpdatePackageError("release file contract does not match")
    parse_tag(expected_tag)
    python_contract = manifest.get("python") or {}
    if python_contract.get("implementation") != "cpython" or int(python_contract.get("architecture_bits") or 0) != 64:
        raise UpdatePackageError("release Python implementation is incompatible")
    _validate_persistent_compatibility(manifest.get("persistent_compatibility") or {})
    source = manifest.get("source") or {}
    if source.get("branch") != "Linux-Docker" or not re.fullmatch(r"[0-9a-f]{40}", str(source.get("commit") or "")):
        raise UpdatePackageError("release source contract is invalid")
    docker = manifest.get("docker") or {}
    if docker.get("repository") != DOCKER_REPOSITORY or docker.get("tag") != expected_tag:
        raise UpdatePackageError("release Docker repository or tag does not match")
    _validate_digest(docker.get("index_digest"), "index")
    platforms = docker.get("platforms") or {}
    for key in ("linux/amd64", "linux/arm64"):
        _validate_digest(platforms.get(key), key)

    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise UpdatePackageError("release manifest contains no files")
    declared: set[str] = set()
    for entry in entries:
        relative = _validated_relative_path(entry.get("path"))
        if relative in declared:
            raise UpdatePackageError("release manifest contains duplicate paths")
        declared.add(relative)
        if candidate_root is not None:
            path = candidate_root / relative
            if not path.is_file() or path.stat().st_size != int(entry.get("size") or -1):
                raise UpdatePackageError(f"release file is missing or has the wrong size: {relative}")
            if sha256_file(path) != str(entry.get("sha256") or ""):
                raise UpdatePackageError(f"release file hash mismatch: {relative}")
    if candidate_root is not None:
        actual = {
            path.relative_to(candidate_root).as_posix()
            for path in candidate_root.rglob("*")
            if path.is_file() and path.name != MANIFEST_NAME
        }
        if actual != declared:
            raise UpdatePackageError("release file list does not match the manifest")
        version_source = (candidate_root / "personalityrag" / "version.py").read_text(encoding="utf-8")
        match = re.search(r'^VERSION\s*=\s*["\']([^"\']+)["\']', version_source, re.MULTILINE)
        if not match or match.group(1) != expected_version:
            raise UpdatePackageError("release internal runtime version does not match the tag")
    return manifest


def inspect_and_extract_zip(
    zip_path: Path, destination: Path, *, expected_tag: str
) -> tuple[Path, dict[str, Any]]:
    if zip_path.stat().st_size > MAX_RELEASE_BYTES:
        raise UpdatePackageError("release asset exceeds the size limit")
    with zipfile.ZipFile(zip_path) as archive:
        infos = archive.infolist()
        if not infos or len(infos) > MAX_RELEASE_FILES:
            raise UpdatePackageError("release archive file count is invalid")
        roots: set[str] = set()
        total = 0
        normalized: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
        for info in infos:
            name = info.filename.replace("\\", "/")
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or ":" in name or not pure.parts:
                raise UpdatePackageError("release archive contains an unsafe path")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise UpdatePackageError("release archive contains a symbolic link")
            roots.add(pure.parts[0])
            total += int(info.file_size)
            if total > MAX_RELEASE_BYTES:
                raise UpdatePackageError("release archive expands beyond the size limit")
            normalized.append((info, pure))
        if roots != {RELEASE_ROOT_NAME}:
            raise UpdatePackageError("release archive must contain one PersonalityRAG-linux root")
        for info, pure in normalized:
            parts = pure.parts[1:]
            if not parts or info.is_dir():
                continue
            relative = PurePosixPath(*parts).as_posix()
            if relative != MANIFEST_NAME:
                _validated_relative_path(relative)
            target = destination.joinpath(*pure.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
    root = destination / RELEASE_ROOT_NAME
    manifest_path = root / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UpdatePackageError("release manifest is missing or invalid") from exc
    return root, validate_manifest(manifest, expected_tag=expected_tag, candidate_root=root)


def current_architecture() -> str:
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "linux/amd64"
    if machine in {"aarch64", "arm64"}:
        return "linux/arm64"
    raise UpdatePackageError(f"unsupported Docker architecture: {machine}")


def current_release_contract() -> dict[str, Any]:
    return {
        "product": PRODUCT_NAME,
        "platform": PLATFORM_NAME,
        "version": VERSION,
        "tag_name": TAG_NAME,
        "architecture": current_architecture(),
        "python": f"{sys.implementation.name} {sys.version_info.major}.{sys.version_info.minor}",
        "persistent_compatibility": {
            "config_schema": CONFIG_SCHEMA_VERSION,
            "control_schema": CONTROL_SCHEMA_VERSION,
            "livingmemory_database": LIVINGMEMORY_DATABASE_VERSION,
        },
    }
