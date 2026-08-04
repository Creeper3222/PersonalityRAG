from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .database_types import (
    LIVINGMEMORY_V8_TYPE,
    TEXT_MEDIA_V1_TYPE,
    database_type_registry,
)
from .io_utils import atomic_write_json
from .logger import logger


DATABASE_LAYOUT_VERSION = 2


class DatabaseLayoutError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _MoveOperation:
    name: str
    source: Path
    target: Path


def _is_junction(path: Path) -> bool:
    checker = getattr(path, "is_junction", None)
    return bool(checker and checker())


def _is_redirect(path: Path) -> bool:
    return path.is_symlink() or _is_junction(path)


def _remove_redirect(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    elif _is_junction(path):
        os.rmdir(path)
    else:
        raise DatabaseLayoutError(f"not a storage layout redirect: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_manifest(root: Path) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    if not root.exists():
        return manifest
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().lower()):
        if _is_redirect(path) or path.is_symlink():
            raise DatabaseLayoutError(f"storage layout migration refuses redirects: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        manifest[relative] = {
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return manifest


def _sqlite_integrity(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise DatabaseLayoutError(f"SQLite integrity check failed for {path}: {integrity}")
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_errors:
            raise DatabaseLayoutError(f"SQLite foreign key check failed for {path}")
    finally:
        connection.close()


def _validate_sqlite_tree(root: Path) -> None:
    for path in root.rglob("*.db"):
        if path.name.endswith(("-wal", "-shm", "-journal")):
            continue
        _sqlite_integrity(path)


def _directory_is_empty(path: Path) -> bool:
    return path.is_dir() and not any(path.iterdir())


def _create_directory_redirect(source: Path, target: Path) -> None:
    if source.exists() or _is_redirect(source):
        if _is_redirect(source):
            _remove_redirect(source)
        elif _directory_is_empty(source):
            source.rmdir()
        else:
            raise DatabaseLayoutError(f"cannot create redirect over existing path: {source}")
    source.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(source), str(target)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if completed.returncode != 0:
            raise DatabaseLayoutError(f"failed to create directory junction: {source} -> {target}")
    else:
        os.symlink(target, source, target_is_directory=True)


class DatabaseStorageLayout:
    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.databases_root = self.data_root / "databases"
        self.marker_path = self.databases_root / ".layout.json"
        self.journal_path = self.data_root / ".database-layout-v2-journal.json"
        self._finalizer: threading.Thread | None = None

    def prepare(self) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)
        operations = self._operations()
        if self.journal_path.exists():
            journal = self._read_journal()
            status = str(journal.get("status") or "")
            if status in {"moved", "redirects"}:
                if not self._update_transaction_active():
                    self._finalize_success()
                return
            if status == "moving":
                self._rollback_from_journal(journal)
            else:
                raise DatabaseLayoutError(f"unsupported database layout journal status: {status}")

        self._remove_stale_redirects_if_possible(operations)
        legacy_sources = [
            operation
            for operation in operations
            if operation.source.exists() and not _is_redirect(operation.source)
        ]
        if not legacy_sources:
            self._ensure_current_roots()
            if self._update_transaction_active() and any(
                _is_redirect(operation.source) for operation in operations
            ):
                self._start_update_finalizer()
            else:
                self._write_marker()
            return

        self._preflight_operations(legacy_sources)
        journal = {
            "version": DATABASE_LAYOUT_VERSION,
            "status": "moving",
            "created_at": time.time(),
            "operations": [
                {
                    "name": operation.name,
                    "source": str(operation.source),
                    "target": str(operation.target),
                    "status": "pending",
                    "manifest": _tree_manifest(operation.source),
                }
                for operation in legacy_sources
            ],
        }
        self._write_journal(journal)
        try:
            self._move_operations(journal)
            journal["status"] = "moved"
            self._write_journal(journal)
        except Exception:
            logger.exception("database storage layout migration failed; rolling back moved roots")
            self._rollback_from_journal(journal)
            raise

        if self._update_transaction_active():
            self._create_update_redirects(journal)
            journal["status"] = "redirects"
            self._write_journal(journal)
            self._start_update_finalizer()
        else:
            self._finalize_success()

    def close(self) -> None:
        if self._finalizer and self._finalizer.is_alive():
            self._finalizer.join(timeout=1)

    def _operations(self) -> list[_MoveOperation]:
        return [
            _MoveOperation(
                "livingmemory_v8_libraries",
                self.data_root / "libraries",
                database_type_registry.type_root(self.data_root, LIVINGMEMORY_V8_TYPE),
            ),
            _MoveOperation(
                "text_media_v1_databases",
                self.data_root / "databases" / TEXT_MEDIA_V1_TYPE,
                database_type_registry.type_root(self.data_root, TEXT_MEDIA_V1_TYPE),
            ),
        ]

    def _ensure_current_roots(self) -> None:
        for database_type in (LIVINGMEMORY_V8_TYPE, TEXT_MEDIA_V1_TYPE):
            database_type_registry.type_root(self.data_root, database_type).mkdir(
                parents=True, exist_ok=True
            )

    def _preflight_operations(self, operations: list[_MoveOperation]) -> None:
        for operation in operations:
            if not operation.source.exists():
                continue
            if _is_redirect(operation.source):
                continue
            if not operation.source.is_dir():
                raise DatabaseLayoutError(f"legacy database root is not a directory: {operation.source}")
            if operation.target.exists():
                if _directory_is_empty(operation.target):
                    operation.target.rmdir()
                else:
                    raise DatabaseLayoutError(
                        f"legacy and new database roots both exist; refusing to merge: "
                        f"{operation.source} -> {operation.target}"
                    )
            _validate_sqlite_tree(operation.source)

    def _move_operations(self, journal: dict[str, Any]) -> None:
        for item in journal.get("operations") or []:
            if item.get("status") == "moved":
                continue
            source = Path(str(item["source"]))
            target = Path(str(item["target"]))
            if not source.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
            if _tree_manifest(target) != item.get("manifest"):
                raise DatabaseLayoutError(f"database layout migration hash mismatch: {target}")
            item["status"] = "moved"
            item["moved_at"] = time.time()
            self._write_journal(journal)

    def _rollback_from_journal(self, journal: dict[str, Any]) -> None:
        for item in reversed(list(journal.get("operations") or [])):
            source = Path(str(item["source"]))
            target = Path(str(item["target"]))
            if _is_redirect(source):
                _remove_redirect(source)
            if item.get("status") == "moved" and target.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                target.replace(source)
                item["status"] = "rolled_back"
                self._write_journal(journal)
        self.journal_path.unlink(missing_ok=True)

    def _remove_stale_redirects_if_possible(self, operations: list[_MoveOperation]) -> None:
        if self._update_transaction_active():
            return
        for operation in operations:
            if _is_redirect(operation.source):
                _remove_redirect(operation.source)

    def _create_update_redirects(self, journal: dict[str, Any]) -> None:
        redirects = []
        for item in journal.get("operations") or []:
            source = Path(str(item["source"]))
            target = Path(str(item["target"]))
            if not target.exists():
                continue
            _create_directory_redirect(source, target)
            redirects.append({"source": str(source), "target": str(target)})
        journal["redirects"] = redirects

    def _finalize_success(self) -> None:
        journal = self._read_journal() if self.journal_path.exists() else {}
        for item in journal.get("redirects") or []:
            source = Path(str(item["source"]))
            if _is_redirect(source):
                _remove_redirect(source)
        for operation in self._operations():
            if _is_redirect(operation.source):
                _remove_redirect(operation.source)
            elif _directory_is_empty(operation.source):
                operation.source.rmdir()
        self._ensure_current_roots()
        self._write_marker()
        self.journal_path.unlink(missing_ok=True)

    def _write_marker(self) -> None:
        self.databases_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            self.marker_path,
            {
                "version": DATABASE_LAYOUT_VERSION,
                "layout": {
                    "memory": "databases/memory_stores/<database_type>/<id>",
                    "knowledge": "databases/knowledge_bases/<database_type>/<id>",
                },
                "updated_at": time.time(),
            },
        )

    def _read_journal(self) -> dict[str, Any]:
        try:
            return json.loads(self.journal_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatabaseLayoutError("database layout journal is unreadable") from exc

    def _write_journal(self, payload: dict[str, Any]) -> None:
        atomic_write_json(self.journal_path, payload)

    def _update_transaction_active(self) -> bool:
        transaction = self._transaction_payload()
        if not transaction:
            return False
        status = str(transaction.get("status") or "")
        stage = str(transaction.get("stage") or "")
        return status not in {"completed", "rolled_back", "failed", "recovery_required"} and stage != "completed"

    def _transaction_payload(self) -> dict[str, Any] | None:
        transaction_id = os.environ.get("PERSONALITYRAG_UPDATE_TRANSACTION", "").strip()
        if not transaction_id:
            return None
        path = self.data_root / "update" / "transactions" / transaction_id / "transaction.json"
        if not path.is_file():
            return {"transaction_id": transaction_id, "status": "unknown"}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"transaction_id": transaction_id, "status": "unknown"}

    def _start_update_finalizer(self) -> None:
        if self._finalizer and self._finalizer.is_alive():
            return

        def wait_and_finalize() -> None:
            deadline = time.time() + 600
            while time.time() < deadline:
                payload = self._transaction_payload()
                status = str((payload or {}).get("status") or "")
                stage = str((payload or {}).get("stage") or "")
                if status == "completed" or stage == "completed":
                    try:
                        self._finalize_success()
                    except Exception:
                        logger.exception("database storage layout finalization failed")
                    return
                if status in {"rolling_back", "rolled_back", "failed", "recovery_required"}:
                    return
                time.sleep(1)

        self._finalizer = threading.Thread(
            target=wait_and_finalize,
            name="database-layout-finalizer",
            daemon=True,
        )
        self._finalizer.start()


DatabaseLayout = DatabaseStorageLayout
