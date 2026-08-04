from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from personalityrag.io_utils import read_ab_checkpoint, write_ab_checkpoint
from personalityrag.library_types.livingmemory_v8.manager import (
    LivingMemoryV8Manager as LibraryManager,
)
from personalityrag.resumable_tasks import ResumableLibraryTasks


def _create_database(path: Path, value: str) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE sample(value TEXT NOT NULL)")
        connection.execute("INSERT INTO sample(value) VALUES(?)", (value,))
        connection.commit()
    finally:
        connection.close()


def test_ab_checkpoint_recovers_from_one_corrupt_slot(tmp_path: Path) -> None:
    write_ab_checkpoint(tmp_path, {"phase": "first", "completed": 1})
    write_ab_checkpoint(tmp_path, {"phase": "second", "completed": 2})
    assert read_ab_checkpoint(tmp_path) == {"phase": "second", "completed": 2}

    slots = list(tmp_path.glob("checkpoint.*.json"))
    newest = max(
        slots,
        key=lambda path: json.loads(path.read_text(encoding="utf-8"))["sequence"],
    )
    newest.write_text("{corrupt", encoding="utf-8")
    assert read_ab_checkpoint(tmp_path) == {"phase": "first", "completed": 1}

    for slot in slots:
        slot.write_text("{corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="all checkpoint slots are corrupt"):
        read_ab_checkpoint(tmp_path)


def test_resumable_task_database_write_sets_are_minimal() -> None:
    assert ResumableLibraryTasks.database_write_set("index_rebuild") == ()
    assert ResumableLibraryTasks.database_write_set("graph_rebuild") == (
        "livingmemory.db",
    )
    assert ResumableLibraryTasks.database_write_set(
        "livingmemory_import", {"conversations_db": None}
    ) == ("livingmemory.db",)
    assert ResumableLibraryTasks.database_write_set(
        "livingmemory_import", {"conversations_db": "upload.db"}
    ) == ("livingmemory.db", "conversations.db")
    assert ResumableLibraryTasks.database_write_set("livingmemory_migration") == (
        "livingmemory.db",
        "conversations.db",
    )


def test_library_copy_uses_positive_file_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    _create_database(source / "livingmemory.db", "memory")
    _create_database(source / "conversations.db", "conversation")
    (source / "decay_state.json").write_text('{"cursor": 3}', encoding="utf-8")
    (source / ".plugin_version").write_text("1", encoding="utf-8")
    stopwords = source / "stopwords"
    stopwords.mkdir()
    (stopwords / "custom.txt").write_text("ignored-word", encoding="utf-8")

    indexes = source / "indexes"
    current_generation = indexes / "gen-current"
    stale_generation = indexes / "gen-stale"
    current_generation.mkdir(parents=True)
    stale_generation.mkdir()
    (indexes / "CURRENT").write_text("gen-current", encoding="utf-8")
    (current_generation / "manifest.json").write_text(
        '{"library_id": "source"}', encoding="utf-8"
    )
    (current_generation / "documents.index").write_bytes(b"current")
    (stale_generation / "manifest.json").write_text("{}", encoding="utf-8")

    for name in ("backups", "imports", "reports", "task_checkpoints"):
        directory = source / name
        directory.mkdir()
        (directory / "must-not-copy.txt").write_text(name, encoding="utf-8")

    LibraryManager._copy_library_directory(source, target)

    assert sorted(path.name for path in target.iterdir()) == [
        ".plugin_version",
        "conversations.db",
        "decay_state.json",
        "indexes",
        "livingmemory.db",
        "stopwords",
    ]
    assert (target / "indexes" / "gen-current" / "documents.index").read_bytes() == b"current"
    assert not (target / "indexes" / "gen-stale").exists()
    assert not (target / "backups").exists()
    with sqlite3.connect(target / "livingmemory.db") as connection:
        assert connection.execute("SELECT value FROM sample").fetchone()[0] == "memory"


@pytest.mark.asyncio
async def test_import_reuses_validation_report_after_fingerprint_check(
    tmp_path: Path,
) -> None:
    upload = tmp_path / "livingmemory.db"
    upload.write_bytes(b"validated-upload")
    report = {
        "size": upload.stat().st_size,
        "sha256": hashlib.sha256(upload.read_bytes()).hexdigest(),
        "integrity": "ok",
    }

    def unexpected_validation(_path: Path) -> dict:
        raise AssertionError("SQLite validation should not run twice")

    manager = LibraryManager.__new__(LibraryManager)
    reused = await manager._reuse_upload_validation(
        upload,
        report,
        unexpected_validation,
    )
    assert reused == report

    upload.write_bytes(b"changed-upload!!")
    with pytest.raises(ValueError, match="fingerprint changed"):
        await manager._reuse_upload_validation(
            upload,
            report,
            unexpected_validation,
        )
