from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .storage_layout import DATABASE_LAYOUT_VERSION
from .version import PLATFORM_NAME, PRODUCT_NAME, TAG_NAME, VERSION


MANIFEST_NAME = "update-manifest.json"
MANIFEST_FORMAT_VERSION = 1
CONFIG_SCHEMA_VERSION = 1
CONTROL_SCHEMA_VERSION = 1
# v0.1.0 requires this target-manifest key before it will install an update.
# New runtimes do not read it; keep it only until v0.1.0 is no longer an
# accepted update origin.
V010_LIVINGMEMORY_MANIFEST_BRIDGE = 8
MAX_WINDOWS_RELEASE_BYTES = 512 * 1024 * 1024
MAX_RELEASE_FILES = 10_000
MANAGED_DIRECTORIES = ("personalityrag", "static")
MANAGED_FILES = (
    "launcher.bat",
    "run.py",
    "requirements.txt",
    "requirements-runtime.lock",
    "tools/runtime_bootstrap.py",
    "tools/update_helper.py",
)
SOURCE_ONLY_DIRECTORIES = frozenset({"assets", "config", "docs", "tests"})
SOURCE_ONLY_TOP_LEVEL_FILES = frozenset(
    {
        ".editorconfig",
        ".gitignore",
        "DELIVERY.md",
        "IMPLEMENTATION_AUDIT.md",
        "LICENSE",
        "README.md",
        "THIRD_PARTY_NOTICES.md",
        "pyproject.toml",
        "requirements-dev.txt",
    }
)
SOURCE_ONLY_TOOL_FILES = frozenset(
    {
        "tools/acceptance_livingmemory_253_webui.py",
        "tools/acceptance_text_media_fusion_webui.py",
        "tools/acceptance_text_media_modes_webui.py",
        "tools/acceptance_text_media_webui.py",
        "tools/acceptance_webui.py",
        "tools/audit_migration.py",
        "tools/benchmark_operations.py",
        "tools/benchmark_text_media_modes.py",
        "tools/benchmark_text_media_relevance_pivot.py",
        "tools/benchmark_text_media_v1.py",
        "tools/benchmark_text_media_v1_rerank.py",
        "tools/benchmark_text_media_v1_semantic_rerank.py",
        "tools/benchmark_text_media_v1_text_relevance.py",
        "tools/build_windows_release.py",
        "tools/compare_indexes.py",
        "tools/export_embedding_context_lengths.py",
        "tools/migrate_livingmemory.py",
        "tools/rebuild_text_media_benchmark.py",
        "tools/soak_runtime.py",
    }
)
PROTECTED_TOP_LEVEL = frozenset(
    {".git", ".venv", "config", "data", "docs", "tests", "assets"}
)
TAG_PATTERN = re.compile(
    r"^v(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<pre>[0-9A-Za-z.-]+))?$"
)


class UpdatePackageError(ValueError):
    pass


def audit_release_source_contract(root: Path) -> tuple[str, ...]:
    """Reject tracked source files that have no explicit release classification.

    The v0.1.0 updater requires the manifest-v1 managed path contract to remain
    byte-for-byte stable.  New runtime code therefore belongs below an existing
    managed directory (``personalityrag`` or ``static``), while every tracked
    source-only file must be classified here deliberately.  This prevents a
    future release builder from silently omitting a newly added runtime entry.
    """

    root = root.resolve()
    tracked_result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=False,
        capture_output=True,
    )
    if tracked_result.returncode != 0:
        raise UpdatePackageError("release source must be a readable Git repository")

    dirty_result = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if dirty_result.returncode != 0:
        raise UpdatePackageError("release source Git status is unavailable")
    if dirty_result.stdout.strip():
        raise UpdatePackageError("release source contains uncommitted tracked changes")

    tracked = tuple(
        item.decode("utf-8", "strict").replace("\\", "/")
        for item in tracked_result.stdout.split(b"\0")
        if item
    )
    unclassified: list[str] = []
    for relative in tracked:
        pure = PurePosixPath(relative)
        if not pure.parts or pure.is_absolute() or ".." in pure.parts:
            unclassified.append(relative)
            continue
        if pure.parts[0] in MANAGED_DIRECTORIES or relative in MANAGED_FILES:
            continue
        if pure.parts[0] in SOURCE_ONLY_DIRECTORIES:
            continue
        if relative in SOURCE_ONLY_TOP_LEVEL_FILES or relative in SOURCE_ONLY_TOOL_FILES:
            continue
        unclassified.append(relative)

    if unclassified:
        sample = ", ".join(sorted(unclassified)[:5])
        raise UpdatePackageError(
            "tracked files are not classified by the frozen update contract: " + sample
        )
    return tracked


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
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
        pre_key,
    )


def compare_tags(left: str, right: str) -> int:
    left_key = parse_tag(left)
    right_key = parse_tag(right)
    return (left_key > right_key) - (left_key < right_key)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_files(root: Path) -> Iterable[Path]:
    for directory in MANAGED_DIRECTORIES:
        base = root / directory
        if not base.is_dir():
            raise UpdatePackageError(f"missing managed directory: {directory}")
        yield from (path for path in base.rglob("*") if path.is_file())
    for relative in MANAGED_FILES:
        path = root / Path(relative)
        if not path.is_file():
            raise UpdatePackageError(f"missing managed file: {relative}")
        yield path


def build_manifest(
    root: Path,
    *,
    version: str = VERSION,
    tag_name: str = TAG_NAME,
) -> dict[str, Any]:
    if tag_name != f"v{version}":
        raise UpdatePackageError("version and tag do not match")
    parse_tag(tag_name)
    files: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(_relative_files(root), key=lambda item: item.as_posix().lower()):
        relative = path.relative_to(root).as_posix()
        if relative in seen:
            continue
        seen.add(relative)
        files.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "manifest_format": MANIFEST_FORMAT_VERSION,
        "product": PRODUCT_NAME,
        "platform": PLATFORM_NAME,
        "version": version,
        "tag_name": tag_name,
        "root_directory": PRODUCT_NAME,
        "managed_directories": list(MANAGED_DIRECTORIES),
        "managed_files": list(MANAGED_FILES),
        "python": {
            "implementation": "cpython",
            "minimum": "3.10",
            "architecture_bits": 64,
        },
        "persistent_compatibility": {
            "config_schema": {"minimum": CONFIG_SCHEMA_VERSION, "maximum": CONFIG_SCHEMA_VERSION},
            "control_schema": {"minimum": CONTROL_SCHEMA_VERSION, "maximum": CONTROL_SCHEMA_VERSION},
            "livingmemory_database": {
                "minimum": V010_LIVINGMEMORY_MANIFEST_BRIDGE,
                "maximum": V010_LIVINGMEMORY_MANIFEST_BRIDGE,
            },
            "database_layout": {
                "minimum": DATABASE_LAYOUT_VERSION,
                "maximum": DATABASE_LAYOUT_VERSION,
            },
        },
        "files": files,
    }


def write_manifest(root: Path, *, version: str = VERSION, tag_name: str = TAG_NAME) -> Path:
    target = root / MANIFEST_NAME
    target.write_text(
        json.dumps(build_manifest(root, version=version, tag_name=tag_name), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def validate_manifest(
    manifest: dict[str, Any],
    *,
    expected_tag: str,
    candidate_root: Path | None = None,
) -> dict[str, Any]:
    expected_version = expected_tag.removeprefix("v")
    if manifest.get("manifest_format") != MANIFEST_FORMAT_VERSION:
        raise UpdatePackageError("unsupported update manifest format")
    if manifest.get("product") != PRODUCT_NAME or manifest.get("platform") != PLATFORM_NAME:
        raise UpdatePackageError("release product or platform does not match")
    if manifest.get("tag_name") != expected_tag or manifest.get("version") != expected_version:
        raise UpdatePackageError("release tag and internal version do not match")
    if manifest.get("root_directory") != PRODUCT_NAME:
        raise UpdatePackageError("release root directory is invalid")
    if tuple(manifest.get("managed_directories") or ()) != MANAGED_DIRECTORIES:
        raise UpdatePackageError("managed directory contract does not match")
    if tuple(manifest.get("managed_files") or ()) != MANAGED_FILES:
        raise UpdatePackageError("managed file contract does not match")
    parse_tag(expected_tag)
    python_contract = manifest.get("python") or {}
    if python_contract.get("implementation") != "cpython" or int(python_contract.get("architecture_bits") or 0) != 64:
        raise UpdatePackageError("release Python implementation is incompatible")
    minimum = tuple(int(part) for part in str(python_contract.get("minimum") or "").split("."))
    if sys.implementation.name != "cpython" or sys.maxsize <= 2**32 or sys.version_info[:2] < minimum:
        raise UpdatePackageError("current Python cannot run the target release")
    _validate_persistent_compatibility(manifest.get("persistent_compatibility") or {})

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
            path = candidate_root / Path(relative)
            if not path.is_file():
                raise UpdatePackageError(f"release file is missing: {relative}")
            if path.stat().st_size != int(entry.get("size") or -1):
                raise UpdatePackageError(f"release file size mismatch: {relative}")
            if sha256_file(path) != str(entry.get("sha256") or ""):
                raise UpdatePackageError(f"release file hash mismatch: {relative}")
    if candidate_root is not None:
        actual = {
            path.relative_to(candidate_root).as_posix()
            for path in candidate_root.rglob("*")
            if path.is_file() and path.name != MANIFEST_NAME
        }
        if actual != declared:
            extras = sorted(actual - declared)
            missing = sorted(declared - actual)
            raise UpdatePackageError(f"release file list mismatch: extra={extras[:3]} missing={missing[:3]}")
        version_source = (candidate_root / "personalityrag" / "version.py").read_text(encoding="utf-8")
        version_match = re.search(r'^VERSION\s*=\s*["\']([^"\']+)["\']', version_source, re.MULTILINE)
        if not version_match or version_match.group(1) != expected_version:
            raise UpdatePackageError("release internal runtime version does not match the tag")
    return manifest


def _validate_persistent_compatibility(contract: dict[str, Any]) -> None:
    current = {
        "config_schema": CONFIG_SCHEMA_VERSION,
        "control_schema": CONTROL_SCHEMA_VERSION,
        "database_layout": DATABASE_LAYOUT_VERSION,
    }
    for key, value in current.items():
        bounds = contract.get(key) or {}
        try:
            minimum = int(bounds["minimum"])
            maximum = int(bounds["maximum"])
        except (KeyError, TypeError, ValueError) as exc:
            raise UpdatePackageError(f"missing persistent compatibility range: {key}") from exc
        if not minimum <= value <= maximum:
            if key == "database_layout":
                raise UpdatePackageError(
                    f"target release does not support database layout {value}; "
                    "install a layout-aware release instead"
                )
            raise UpdatePackageError(f"target release is incompatible with {key}={value}")


def _validated_relative_path(value: object) -> str:
    text = str(value or "").replace("\\", "/")
    pure = PurePosixPath(text)
    if not text or pure.is_absolute() or ".." in pure.parts or ":" in text:
        raise UpdatePackageError("release contains an unsafe path")
    if pure.parts[0].lower() in PROTECTED_TOP_LEVEL:
        raise UpdatePackageError("release attempts to manage a protected path")
    allowed = pure.parts[0] in MANAGED_DIRECTORIES or text in MANAGED_FILES
    if not allowed:
        raise UpdatePackageError(f"release contains an unknown managed path: {text}")
    return pure.as_posix()


def inspect_and_extract_zip(zip_path: Path, destination: Path, *, expected_tag: str) -> tuple[Path, dict[str, Any]]:
    if zip_path.stat().st_size > MAX_WINDOWS_RELEASE_BYTES:
        raise UpdatePackageError("release asset exceeds the size limit")
    with zipfile.ZipFile(zip_path) as archive:
        infos = archive.infolist()
        if not infos or len(infos) > MAX_RELEASE_FILES:
            raise UpdatePackageError("release archive file count is invalid")
        normalized: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
        roots: set[str] = set()
        total_size = 0
        for info in infos:
            name = info.filename.replace("\\", "/")
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or ":" in name or not pure.parts:
                raise UpdatePackageError("release archive contains an unsafe path")
            roots.add(pure.parts[0])
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise UpdatePackageError("release archive contains a symbolic link")
            total_size += int(info.file_size)
            if total_size > MAX_WINDOWS_RELEASE_BYTES:
                raise UpdatePackageError("release archive expands beyond the size limit")
            normalized.append((info, pure))
        if roots != {PRODUCT_NAME}:
            raise UpdatePackageError("release archive must contain one PersonalityRAG root directory")
        for info, pure in normalized:
            relative_parts = pure.parts[1:]
            if not relative_parts or info.is_dir():
                continue
            relative = PurePosixPath(*relative_parts).as_posix()
            if relative != MANIFEST_NAME:
                _validated_relative_path(relative)
            target = destination.joinpath(*pure.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
    root = destination / PRODUCT_NAME
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise UpdatePackageError("release manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UpdatePackageError("release manifest is invalid") from exc
    return root, validate_manifest(manifest, expected_tag=expected_tag, candidate_root=root)


def current_release_contract() -> dict[str, Any]:
    return {
        "product": PRODUCT_NAME,
        "platform": PLATFORM_NAME,
        "version": VERSION,
        "tag_name": TAG_NAME,
        "python": f"{sys.implementation.name} {sys.version_info.major}.{sys.version_info.minor}",
        "persistent_compatibility": {
            "config_schema": CONFIG_SCHEMA_VERSION,
            "control_schema": CONTROL_SCHEMA_VERSION,
            "database_layout": DATABASE_LAYOUT_VERSION,
        },
    }
