from __future__ import annotations

import asyncio
import codecs
import mimetypes
import os
import shutil
import time
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from fastapi import HTTPException, UploadFile

from .auth import AuthManager
from .io_utils import run_blocking
from .logger import logger


FILE_PREVIEW_LIMIT_BYTES = 1024 * 1024
FILE_EDIT_LIMIT_BYTES = 1024 * 1024
FILE_UPLOAD_CHUNK_BYTES = 1024 * 1024
FILE_UPLOAD_LIMIT_BYTES = 100 * 1024 * 1024
FILE_UPLOAD_MAX_FILES = 20

IMAGE_PREVIEW_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".webp",
}
IMAGE_PREVIEW_MEDIA_TYPES = {
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
BINARY_PREVIEW_EXTENSIONS = {
    ".7z",
    ".bin",
    ".bmp",
    ".db",
    ".dll",
    ".doc",
    ".docx",
    ".exe",
    ".faiss",
    ".gif",
    ".ico",
    ".index",
    ".jpeg",
    ".jpg",
    ".mp3",
    ".mp4",
    ".npy",
    ".npz",
    ".pdf",
    ".png",
    ".pyc",
    ".rar",
    ".so",
    ".sqlite",
    ".sqlite3",
    ".webm",
    ".wav",
    ".webp",
    ".zip",
}

HIDDEN_EXACT_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".test-runtime",
    "__pycache__",
    "node_modules",
}
HIDDEN_PREFIXES = (".venv",)
SOURCE_READONLY_ROOTS = (
    "personalityrag",
    "static",
    "assets",
    "docker",
    "tools",
)
SOURCE_READONLY_FILES = {
    ".dockerignore",
    ".env.example",
    "Dockerfile",
    "docker-compose.local.yml",
    "docker-compose.yml",
    "launcher.bat",
    "pyproject.toml",
    "run.py",
}


@dataclass(frozen=True, slots=True)
class FileRoot:
    id: str
    path: Path


class FileManager:
    def __init__(
        self,
        source_root: Path,
        state_root: Path,
        auth: AuthManager,
    ) -> None:
        self.source_root = source_root.expanduser().resolve()
        self.state_root = state_root.expanduser().resolve()
        self.auth = auth
        self.archive_dir = self.state_root / "data" / "file_manager_archives"
        self._roots = self._build_roots()
        self._failed_attempts: dict[str, list[float]] = {}
        self._attempt_lock = asyncio.Lock()

    def _build_roots(self) -> dict[str, FileRoot]:
        source = self.source_root
        state = self.state_root
        if source == state or state.is_relative_to(source):
            return {"project": FileRoot("project", source)}
        if source.is_relative_to(state):
            return {"state": FileRoot("state", state)}
        return {
            "project": FileRoot("project", source),
            "state": FileRoot("state", state),
        }

    def roots_payload(self) -> list[dict[str, str]]:
        return [
            {"id": root.id, "path": str(root.path)}
            for root in self._roots.values()
        ]

    def default_root_id(self) -> str:
        return next(iter(self._roots))

    def root(self, root_id: str | None) -> FileRoot:
        effective_id = str(root_id or self.default_root_id()).strip().lower()
        root = self._roots.get(effective_id)
        if root is None:
            raise HTTPException(status_code=400, detail="未知的文件管理根目录")
        return root

    @staticmethod
    def _hidden_name(name: str) -> bool:
        normalized = str(name or "").lower()
        return normalized in HIDDEN_EXACT_NAMES or normalized.startswith(
            HIDDEN_PREFIXES
        )

    def _reject_hidden_parts(self, parts: Iterable[str]) -> None:
        if any(self._hidden_name(part) for part in parts):
            raise HTTPException(status_code=404, detail="文件或目录不存在")

    def resolve(self, root_id: str | None, raw_path: str) -> tuple[FileRoot, Path]:
        root = self.root(root_id)
        cleaned = str(raw_path or "").strip().replace("\\", "/")
        if "\x00" in cleaned:
            raise HTTPException(status_code=400, detail="文件路径无效")
        if (
            cleaned.startswith("/")
            or Path(cleaned).drive
            or (len(cleaned) >= 2 and cleaned[1] == ":")
        ):
            raise HTTPException(
                status_code=400,
                detail="文件路径必须是根目录内的相对路径",
            )
        parts = [part for part in cleaned.split("/") if part and part != "."]
        if any(part == ".." for part in parts):
            raise HTTPException(status_code=403, detail="文件路径不能越过根目录")
        self._reject_hidden_parts(parts)
        candidate = (root.path / "/".join(parts)).resolve()
        if candidate != root.path and not candidate.is_relative_to(root.path):
            raise HTTPException(status_code=403, detail="文件路径不能越过根目录")
        resolved_parts = candidate.relative_to(root.path).parts
        self._reject_hidden_parts(resolved_parts)
        return root, candidate

    @staticmethod
    def normalize_name(raw_name: str) -> str:
        name = str(raw_name or "").strip()
        if not name or "\x00" in name:
            raise HTTPException(status_code=400, detail="名称不能为空")
        if name in {".", ".."} or "/" in name or "\\" in name:
            raise HTTPException(status_code=400, detail="名称不能包含目录层级")
        if Path(name).drive or (len(name) >= 2 and name[1] == ":"):
            raise HTTPException(status_code=400, detail="名称不能包含盘符")
        if any(character in name for character in '<>:"|?*'):
            raise HTTPException(status_code=400, detail="名称包含非法字符")
        if FileManager._hidden_name(name):
            raise HTTPException(status_code=403, detail="不能创建内部保留目录")
        return name

    @staticmethod
    def normalize_upload_name(raw_name: str) -> str:
        cleaned = str(raw_name or "").strip().replace("\\", "/")
        if not cleaned or "\x00" in cleaned:
            raise HTTPException(status_code=400, detail="上传文件名无效")
        if (
            cleaned.startswith("/")
            or Path(cleaned).drive
            or (len(cleaned) >= 2 and cleaned[1] == ":")
        ):
            raise HTTPException(status_code=400, detail="上传文件名必须是相对路径")
        parts = [part for part in cleaned.split("/") if part and part != "."]
        if not parts or any(part == ".." for part in parts):
            raise HTTPException(status_code=400, detail="上传文件名不能包含路径穿越")
        if any(any(character in part for character in '<>:"|?*') for part in parts):
            raise HTTPException(status_code=400, detail="上传文件名包含非法字符")
        if any(FileManager._hidden_name(part) for part in parts):
            raise HTTPException(status_code=403, detail="不能上传到内部保留目录")
        return "/".join(parts)

    def relative_path(self, root: FileRoot, target: Path) -> str:
        if target == root.path:
            return ""
        return target.relative_to(root.path).as_posix()

    @staticmethod
    def _inside(target: Path, parent: Path) -> bool:
        return target == parent or target.is_relative_to(parent)

    @staticmethod
    def _ancestor_of(target: Path, child: Path) -> bool:
        return target == child or child.is_relative_to(target)

    def _source_readonly_paths(self) -> tuple[set[Path], set[Path]]:
        exact = {self.source_root / name for name in SOURCE_READONLY_FILES}
        exact.update(
            path
            for path in self.source_root.glob("requirements*")
            if path.is_file()
        )
        roots = {self.source_root / name for name in SOURCE_READONLY_ROOTS}
        return exact, roots

    def _state_managed_roots(self) -> set[Path]:
        return {
            self.source_root / "config",
            self.source_root / "data",
            self.state_root / "config",
            self.state_root / "data",
        }

    def is_readonly(self, target: Path) -> bool:
        exact, roots = self._source_readonly_paths()
        if target in exact:
            return True
        return any(self._inside(target, root) for root in roots | self._state_managed_roots())

    def contains_readonly(self, target: Path) -> bool:
        exact, roots = self._source_readonly_paths()
        protected = exact | roots | self._state_managed_roots()
        return any(self._ancestor_of(target, item) for item in protected)

    def assert_mutable(
        self,
        root: FileRoot,
        target: Path,
        *,
        allow_root_container: bool = False,
    ) -> None:
        if (target == root.path and not allow_root_container) or self.is_readonly(target):
            raise HTTPException(
                status_code=403,
                detail="该路径由 PersonalityRAG 管理，只允许查看和下载",
            )
        if (
            not allow_root_container
            and target.exists()
            and target.is_dir()
            and self.contains_readonly(target)
        ):
            raise HTTPException(
                status_code=403,
                detail="该目录包含 PersonalityRAG 受保护内容，不能修改",
            )

    def is_sensitive(self, target: Path) -> bool:
        return any(self._inside(target, root) for root in self._state_managed_roots())

    def contains_sensitive(self, target: Path) -> bool:
        return self.is_sensitive(target) or any(
            self._ancestor_of(target, root) for root in self._state_managed_roots()
        )

    async def verify_sensitive_access(
        self,
        client_ip: str,
        password: str,
        targets: Iterable[Path],
    ) -> None:
        if not any(self.contains_sensitive(target) for target in targets):
            return
        if not self.auth.password_enabled:
            raise HTTPException(
                status_code=403,
                detail="请先在基础设置中设置 WebUI 登录/文件管理验证密码",
            )
        now = time.monotonic()
        async with self._attempt_lock:
            attempts = [
                attempt
                for attempt in self._failed_attempts.get(client_ip, [])
                if now - attempt < 300
            ]
            if attempts:
                self._failed_attempts[client_ip] = attempts
            else:
                self._failed_attempts.pop(client_ip, None)
            if len(attempts) >= 5:
                raise HTTPException(
                    status_code=429,
                    detail="尝试次数过多，请 5 分钟后再试",
                )
        if not await asyncio.to_thread(self.auth.verify_login_secret, password):
            async with self._attempt_lock:
                self._failed_attempts.setdefault(client_ip, []).append(
                    time.monotonic()
                )
            await asyncio.sleep(0.8)
            raise HTTPException(status_code=401, detail="文件管理验证密码错误")

    async def list_directory(self, root_id: str | None, path: str) -> dict[str, Any]:
        root, target = self.resolve(root_id, path)
        if not target.exists():
            raise HTTPException(status_code=404, detail="文件或目录不存在")
        if not target.is_dir():
            raise HTTPException(status_code=400, detail="目标路径不是目录")
        try:
            items = await run_blocking(self._list_directory_sync, root, target)
        except OSError as exc:
            logger.warning("文件管理目录读取失败：path=%s err=%r", target, exc)
            raise HTTPException(status_code=500, detail="读取目录失败") from exc
        relative = self.relative_path(root, target)
        parent = ""
        if target != root.path:
            parent = self.relative_path(root, target.parent)
        return {
            "roots": self.roots_payload(),
            "root": root.id,
            "root_path": str(root.path),
            "path": relative,
            "parent_path": parent,
            "can_go_up": target != root.path,
            "items": items,
        }

    def _list_directory_sync(self, root: FileRoot, target: Path) -> list[dict[str, Any]]:
        children = sorted(
            target.iterdir(),
            key=lambda item: (not item.is_dir(), item.name.casefold()),
        )
        items: list[dict[str, Any]] = []
        for child in children:
            if self._hidden_name(child.name):
                continue
            try:
                resolved = child.resolve()
                if resolved != root.path and not resolved.is_relative_to(root.path):
                    continue
                self._reject_hidden_parts(resolved.relative_to(root.path).parts)
                stat_result = child.stat()
            except (OSError, HTTPException):
                continue
            items.append(self.serialize_item(root, child, stat_result))
        return items

    def serialize_item(
        self,
        root: FileRoot,
        target: Path,
        stat_result: os.stat_result,
    ) -> dict[str, Any]:
        is_directory = target.is_dir()
        extension = "" if is_directory else target.suffix.lower()
        preview_type = "directory" if is_directory else "text"
        if extension in IMAGE_PREVIEW_EXTENSIONS:
            preview_type = "image"
        elif extension in BINARY_PREVIEW_EXTENSIONS:
            preview_type = "binary"
        readonly = self.is_readonly(target)
        can_edit = (
            preview_type == "text"
            and not readonly
            and stat_result.st_size <= FILE_EDIT_LIMIT_BYTES
        )
        return {
            "name": target.name,
            "path": self.relative_path(root, target),
            "is_directory": is_directory,
            "size": 0 if is_directory else stat_result.st_size,
            "mtime": datetime.fromtimestamp(stat_result.st_mtime).isoformat(
                timespec="seconds"
            ),
            "extension": extension,
            "preview_type": preview_type,
            "requires_password": self.is_sensitive(target),
            "is_protected": readonly,
            "can_edit": can_edit,
        }

    async def read_text(
        self,
        root_id: str,
        path: str,
        *,
        client_ip: str,
        password: str,
    ) -> dict[str, Any]:
        root, target = self.resolve(root_id, path)
        if not target.exists():
            raise HTTPException(status_code=404, detail="文件不存在")
        if not target.is_file():
            raise HTTPException(status_code=400, detail="目标路径不是文件")
        await self.verify_sensitive_access(client_ip, password, [target])
        if target.suffix.lower() in BINARY_PREVIEW_EXTENSIONS:
            raise HTTPException(status_code=415, detail="当前文件不支持文本预览")
        try:
            preview_bytes, has_more, stat_result = await run_blocking(
                self._read_preview_sync,
                target,
            )
        except OSError as exc:
            logger.warning("文件管理读取文件失败：path=%s err=%r", target, exc)
            raise HTTPException(status_code=500, detail="读取文件失败") from exc
        if self.looks_binary(preview_bytes):
            raise HTTPException(status_code=415, detail="当前文件不支持文本预览")
        try:
            decoder = codecs.getincrementaldecoder("utf-8-sig")("strict")
            content = decoder.decode(preview_bytes, final=not has_more)
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=415,
                detail="当前仅支持 UTF-8 文本文件预览",
            ) from exc
        return {
            "path": self.relative_path(root, target),
            "name": target.name,
            "size": stat_result.st_size,
            "mtime": datetime.fromtimestamp(stat_result.st_mtime).isoformat(
                timespec="seconds"
            ),
            "content": content,
            "encoding": "utf-8",
            "truncated": has_more,
            "requires_password": self.is_sensitive(target),
            "is_protected": self.is_readonly(target),
            "can_edit": (
                not has_more
                and not self.is_readonly(target)
                and stat_result.st_size <= FILE_EDIT_LIMIT_BYTES
            ),
        }

    @staticmethod
    def _read_preview_sync(target: Path) -> tuple[bytes, bool, os.stat_result]:
        with target.open("rb") as handle:
            payload = handle.read(FILE_PREVIEW_LIMIT_BYTES)
            has_more = bool(handle.read(1))
        return payload, has_more, target.stat()

    async def write_text(
        self,
        root_id: str,
        path: str,
        content: str,
        *,
        client_ip: str,
        password: str,
    ) -> dict[str, Any]:
        root, target = self.resolve(root_id, path)
        if not target.exists():
            raise HTTPException(status_code=404, detail="文件不存在")
        if not target.is_file():
            raise HTTPException(status_code=400, detail="目标路径不是文件")
        self.assert_mutable(root, target)
        await self.verify_sensitive_access(client_ip, password, [target])
        encoded = content.encode("utf-8")
        if len(encoded) > FILE_EDIT_LIMIT_BYTES:
            raise HTTPException(
                status_code=413,
                detail="在线编辑内容不能超过 1 MiB",
            )
        existing, has_more, _ = await run_blocking(self._read_preview_sync, target)
        if has_more or self.looks_binary(existing):
            raise HTTPException(status_code=415, detail="当前文件不支持在线编辑")
        try:
            existing.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=415,
                detail="当前仅支持 UTF-8 文本文件编辑",
            ) from exc
        try:
            stat_result = await run_blocking(self._atomic_write_sync, target, encoded)
        except OSError as exc:
            logger.warning("文件管理保存文件失败：path=%s err=%r", target, exc)
            raise HTTPException(status_code=500, detail="保存文件失败") from exc
        logger.info("文件管理保存文本：root=%s path=%s size=%s", root.id, self.relative_path(root, target), len(encoded))
        return {
            "ok": True,
            "item": self.serialize_item(root, target, stat_result),
        }

    @staticmethod
    def _atomic_write_sync(target: Path, payload: bytes) -> os.stat_result:
        temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            temp.write_bytes(payload)
            os.replace(temp, target)
            return target.stat()
        finally:
            temp.unlink(missing_ok=True)

    async def create_item(
        self,
        root_id: str,
        path: str,
        item_type: str,
    ) -> dict[str, Any]:
        root, target = self.resolve(root_id, path)
        normalized_type = str(item_type or "file").strip().lower()
        if normalized_type not in {"file", "directory"}:
            raise HTTPException(status_code=400, detail="新建类型必须是 file 或 directory")
        self.assert_mutable(root, target)
        if target.exists():
            raise HTTPException(status_code=409, detail="同名文件或目录已存在")
        parent = target.parent.resolve()
        if parent != root.path and not parent.is_relative_to(root.path):
            raise HTTPException(status_code=403, detail="文件路径不能越过根目录")
        self.assert_mutable(root, parent, allow_root_container=True)
        if normalized_type == "file" and not parent.exists():
            raise HTTPException(status_code=400, detail="父目录不存在")
        try:
            if normalized_type == "directory":
                await run_blocking(target.mkdir, 0o777, True, False)
            else:
                await run_blocking(target.write_text, "", "utf-8")
            stat_result = target.stat()
        except OSError as exc:
            logger.warning("文件管理新建失败：path=%s err=%r", target, exc)
            raise HTTPException(status_code=500, detail="新建失败") from exc
        logger.info("文件管理新建：root=%s path=%s type=%s", root.id, self.relative_path(root, target), normalized_type)
        return {"ok": True, "item": self.serialize_item(root, target, stat_result)}

    async def upload_files(
        self,
        root_id: str,
        path: str,
        files: list[UploadFile],
    ) -> dict[str, Any]:
        root, directory = self.resolve(root_id, path)
        if not directory.exists() or not directory.is_dir():
            raise HTTPException(status_code=400, detail="上传目标不是有效目录")
        self.assert_mutable(root, directory, allow_root_container=True)
        if not files:
            raise HTTPException(status_code=400, detail="请选择要上传的文件")
        if len(files) > FILE_UPLOAD_MAX_FILES:
            raise HTTPException(status_code=413, detail="单次最多上传 20 个文件")
        uploaded: list[dict[str, Any]] = []
        try:
            for upload in files:
                relative_name = self.normalize_upload_name(upload.filename or "")
                _, requested = self.resolve(
                    root.id,
                    "/".join(
                        item
                        for item in [self.relative_path(root, directory), relative_name]
                        if item
                    ),
                )
                self.assert_mutable(root, requested)
                parent = requested.parent.resolve()
                self.assert_mutable(root, parent, allow_root_container=True)
                await run_blocking(parent.mkdir, 0o777, True, True)
                destination = self._deduplicate_path(requested)
                await self._write_upload(upload, destination)
                uploaded.append(self.serialize_item(root, destination, destination.stat()))
        finally:
            for upload in files:
                await upload.close()
        logger.info("文件管理上传：root=%s path=%s count=%s", root.id, self.relative_path(root, directory), len(uploaded))
        return {"ok": True, "uploaded": len(uploaded), "items": uploaded}

    @staticmethod
    def _deduplicate_path(target: Path) -> Path:
        if not target.exists():
            return target
        suffix = target.suffix
        stem = target.stem if suffix else target.name
        for _ in range(20):
            candidate = target.with_name(f"{stem}-{uuid.uuid4().hex[:8]}{suffix}")
            if not candidate.exists():
                return candidate
        raise HTTPException(status_code=409, detail="无法生成不冲突的上传文件名")

    @staticmethod
    async def _write_upload(upload: UploadFile, destination: Path) -> None:
        await run_blocking(
            FileManager._write_upload_sync,
            upload.file,
            destination,
        )

    @staticmethod
    def _write_upload_sync(source: Any, destination: Path) -> None:
        written = 0
        try:
            with destination.open("wb") as handle:
                while True:
                    chunk = source.read(FILE_UPLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > FILE_UPLOAD_LIMIT_BYTES:
                        raise HTTPException(status_code=413, detail="单个文件不能超过 100 MiB")
                    handle.write(chunk)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise

    def resolve_targets(
        self,
        root_id: str,
        raw_paths: Iterable[str],
        *,
        operation: str,
    ) -> tuple[FileRoot, list[Path]]:
        root = self.root(root_id)
        targets: list[Path] = []
        seen: set[Path] = set()
        for raw_path in raw_paths:
            _, target = self.resolve(root.id, raw_path)
            if target == root.path:
                raise HTTPException(status_code=400, detail="不能操作文件管理根目录")
            if not target.exists():
                raise HTTPException(status_code=404, detail=f"文件或目录不存在：{raw_path}")
            if target not in seen:
                seen.add(target)
                targets.append(target)
        targets.sort(key=lambda item: len(item.parts), reverse=operation == "delete")
        return root, targets

    async def delete_items(self, root_id: str, paths: list[str]) -> dict[str, Any]:
        root, targets = self.resolve_targets(root_id, paths, operation="delete")
        if not targets:
            raise HTTPException(status_code=400, detail="请选择要删除的项目")
        for target in targets:
            self.assert_mutable(root, target)
        try:
            await run_blocking(self._delete_sync, targets)
        except OSError as exc:
            logger.warning("文件管理删除失败：err=%r", exc)
            raise HTTPException(status_code=500, detail="删除失败") from exc
        logger.info("文件管理删除：root=%s count=%s", root.id, len(targets))
        return {"ok": True, "deleted": len(targets)}

    @staticmethod
    def _delete_sync(targets: list[Path]) -> None:
        for target in targets:
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()

    async def move_items(
        self,
        root_id: str,
        paths: list[str],
        target_path: str,
    ) -> dict[str, Any]:
        root, sources = self.resolve_targets(root_id, paths, operation="move")
        _, directory = self.resolve(root.id, target_path)
        if not directory.exists() or not directory.is_dir():
            raise HTTPException(status_code=400, detail="移动目标不是目录")
        self.assert_mutable(root, directory, allow_root_container=True)
        plan: list[tuple[Path, Path]] = []
        destinations: set[Path] = set()
        for source in sources:
            self.assert_mutable(root, source)
            destination = (directory / source.name).resolve()
            self.assert_mutable(root, destination)
            if source == directory or (source.is_dir() and directory.is_relative_to(source)):
                raise HTTPException(status_code=400, detail="不能将目录移动到自身内部")
            if destination.exists() or destination in destinations:
                raise HTTPException(status_code=409, detail=f"目标已存在同名项目：{source.name}")
            destinations.add(destination)
            plan.append((source, destination))
        try:
            await run_blocking(self._move_sync, plan)
        except OSError as exc:
            logger.warning("文件管理移动失败：err=%r", exc)
            raise HTTPException(status_code=500, detail="移动失败") from exc
        items = [self.serialize_item(root, target, target.stat()) for _, target in plan]
        logger.info("文件管理移动：root=%s count=%s target=%s", root.id, len(items), self.relative_path(root, directory))
        return {
            "ok": True,
            "moved": len(items),
            "items": items,
            "target_path": self.relative_path(root, directory),
        }

    @staticmethod
    def _move_sync(plan: list[tuple[Path, Path]]) -> None:
        for source, destination in plan:
            shutil.move(str(source), str(destination))

    async def rename_item(
        self,
        root_id: str,
        path: str,
        name: str,
    ) -> dict[str, Any]:
        root, source = self.resolve(root_id, path)
        if not source.exists():
            raise HTTPException(status_code=404, detail="文件或目录不存在")
        self.assert_mutable(root, source)
        destination = (source.parent / self.normalize_name(name)).resolve()
        if destination != root.path and not destination.is_relative_to(root.path):
            raise HTTPException(status_code=403, detail="重命名目标不能越过根目录")
        self.assert_mutable(root, destination)
        if destination.exists():
            raise HTTPException(status_code=409, detail="同名文件或目录已存在")
        try:
            await run_blocking(source.rename, destination)
        except OSError as exc:
            logger.warning("文件管理重命名失败：path=%s err=%r", source, exc)
            raise HTTPException(status_code=500, detail="重命名失败") from exc
        logger.info("文件管理重命名：root=%s from=%s to=%s", root.id, self.relative_path(root, source), self.relative_path(root, destination))
        return {"ok": True, "item": self.serialize_item(root, destination, destination.stat())}

    def preview_file(self, root_id: str, path: str) -> tuple[Path, str]:
        _, target = self.resolve(root_id, path)
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="文件不存在")
        if self.is_sensitive(target):
            raise HTTPException(status_code=403, detail="敏感图片需要验证后查看")
        extension = target.suffix.lower()
        if extension not in IMAGE_PREVIEW_EXTENSIONS:
            raise HTTPException(status_code=415, detail="当前仅支持图片文件预览")
        media_type = IMAGE_PREVIEW_MEDIA_TYPES.get(extension)
        return target, media_type or mimetypes.guess_type(target.name)[0] or "application/octet-stream"

    async def prepare_download(
        self,
        root_id: str,
        paths: list[str],
        *,
        client_ip: str,
        password: str,
    ) -> tuple[Path, str, bool]:
        root, targets = self.resolve_targets(root_id, paths, operation="download")
        if not targets:
            raise HTTPException(status_code=400, detail="请选择要下载的项目")
        await self.verify_sensitive_access(client_ip, password, targets)
        if len(targets) == 1 and targets[0].is_file():
            return targets[0], targets[0].name, False
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        archive = self.archive_dir / f"files-{uuid.uuid4().hex}.zip"
        try:
            await run_blocking(self._build_archive_sync, root, targets, archive)
        except OSError as exc:
            archive.unlink(missing_ok=True)
            logger.warning("文件管理打包下载失败：err=%r", exc)
            raise HTTPException(status_code=500, detail="下载打包失败") from exc
        name = f"{targets[0].name}.zip" if len(targets) == 1 else "files.zip"
        return archive, name, True

    def _build_archive_sync(
        self,
        root: FileRoot,
        targets: list[Path],
        archive: Path,
    ) -> None:
        with zipfile.ZipFile(
            archive,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            allowZip64=True,
        ) as handle:
            written: set[str] = set()
            for target in targets:
                self._write_archive_entry(root, handle, target, target.name, written)

    def _write_archive_entry(
        self,
        root: FileRoot,
        handle: zipfile.ZipFile,
        target: Path,
        archive_name: str,
        written: set[str],
    ) -> None:
        safe_name = str(archive_name or target.name).replace("\\", "/").strip("/")
        if not safe_name or safe_name.startswith("../") or "/../" in safe_name:
            raise HTTPException(status_code=400, detail="压缩包路径无效")
        if any(self._hidden_name(part) for part in safe_name.split("/")):
            return
        resolved = target.resolve()
        if resolved != root.path and not resolved.is_relative_to(root.path):
            return
        if target.is_dir() and not target.is_symlink():
            directory_name = f"{safe_name}/"
            if directory_name not in written:
                handle.writestr(directory_name, b"")
                written.add(directory_name)
            for child in sorted(target.iterdir(), key=lambda item: item.name.casefold()):
                if self._hidden_name(child.name):
                    continue
                self._write_archive_entry(
                    root,
                    handle,
                    child,
                    f"{safe_name}/{child.name}",
                    written,
                )
            return
        if safe_name not in written:
            handle.write(target, safe_name)
            written.add(safe_name)

    @staticmethod
    def cleanup_archive(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("文件管理临时压缩包清理失败：path=%s", path)

    @staticmethod
    def looks_binary(payload: bytes) -> bool:
        if not payload:
            return False
        if b"\x00" in payload:
            return True
        sample = payload[:8192]
        controls = sum(
            byte < 32 and byte not in {9, 10, 12, 13}
            for byte in sample
        )
        return controls / max(1, len(sample)) > 0.08
