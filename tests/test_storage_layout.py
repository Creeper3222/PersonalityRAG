from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from personalityrag import library_types as _registered_database_types  # noqa: F401
from personalityrag.database_types import (
    LIVINGMEMORY_V8_TYPE,
    TEXT_MEDIA_V1_TYPE,
    DatabaseRef,
    database_type_registry,
)
from personalityrag.storage_layout import (
    DATABASE_LAYOUT_VERSION,
    DatabaseLayoutError,
    DatabaseStorageLayout,
)


def _sqlite_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE fixture(id INTEGER PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO fixture(value) VALUES('ok')")
        connection.commit()
    finally:
        connection.close()


def test_storage_layout_migrates_legacy_memory_and_knowledge_roots(tmp_path: Path):
    data_root = tmp_path / "data"
    memory_source = data_root / "libraries" / "alpha"
    knowledge_source = data_root / "databases" / TEXT_MEDIA_V1_TYPE / "alpha"
    _sqlite_fixture(memory_source / "livingmemory.db")
    (memory_source / "indexes" / "CURRENT").parent.mkdir(parents=True)
    (memory_source / "indexes" / "CURRENT").write_text("g1", encoding="utf-8")
    _sqlite_fixture(knowledge_source / "textmediaknowledge.db")
    (knowledge_source / "assets" / "images").mkdir(parents=True)
    (knowledge_source / "assets" / "images" / "image.webp").write_bytes(b"webp")

    DatabaseStorageLayout(data_root).prepare()

    memory_target = database_type_registry.data_dir(
        data_root, DatabaseRef(LIVINGMEMORY_V8_TYPE, "alpha")
    )
    knowledge_target = database_type_registry.data_dir(
        data_root, DatabaseRef(TEXT_MEDIA_V1_TYPE, "alpha")
    )
    assert not (data_root / "libraries").exists()
    assert not (data_root / "databases" / TEXT_MEDIA_V1_TYPE).exists()
    assert (memory_target / "livingmemory.db").is_file()
    assert (memory_target / "indexes" / "CURRENT").read_text(encoding="utf-8") == "g1"
    assert (knowledge_target / "textmediaknowledge.db").is_file()
    assert (knowledge_target / "assets" / "images" / "image.webp").read_bytes() == b"webp"
    marker = json.loads((data_root / "databases" / ".layout.json").read_text(encoding="utf-8"))
    assert marker["version"] == DATABASE_LAYOUT_VERSION

    DatabaseStorageLayout(data_root).prepare()
    assert (memory_target / "livingmemory.db").is_file()
    assert (knowledge_target / "textmediaknowledge.db").is_file()


def test_storage_layout_rejects_legacy_and_new_root_conflict(tmp_path: Path):
    data_root = tmp_path / "data"
    source = data_root / "libraries" / "alpha"
    target = database_type_registry.data_dir(
        data_root, DatabaseRef(LIVINGMEMORY_V8_TYPE, "alpha")
    )
    _sqlite_fixture(source / "livingmemory.db")
    _sqlite_fixture(target / "livingmemory.db")

    with pytest.raises(DatabaseLayoutError, match="refusing to merge"):
        DatabaseStorageLayout(data_root).prepare()

    assert (source / "livingmemory.db").is_file()
    assert (target / "livingmemory.db").is_file()
