from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from personalityrag.migration import LivingMemoryMigrator, table_fingerprint


def make_source(root: Path):
    for name in ("livingmemory.db", "conversations.db"):
        con = sqlite3.connect(root / name)
        con.execute("CREATE TABLE sample(id INTEGER PRIMARY KEY,value TEXT)")
        con.executemany(
            "INSERT INTO sample(value) VALUES(?)", [("alpha",), ("贝雷特",)]
        )
        con.commit()
        con.close()
    (root / "decay_state.json").write_text('{"ok":true}', encoding="utf-8")
    (root / ".plugin_version").write_text("2.3.5", encoding="utf-8")


@pytest.mark.asyncio
async def test_migration_is_independent_and_hash_exact(tmp_path: Path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    make_source(source)
    result = await LivingMemoryMigrator(target).migrate(
        source, mode="rehearsal"
    )
    assert result["status"] == "verified"
    assert table_fingerprint(source / "livingmemory.db")["tables"]["sample"][
        "rows_sha256"
    ] == table_fingerprint(target / "livingmemory.db")["tables"]["sample"][
        "rows_sha256"
    ]
    assert (source / "livingmemory.db").exists()

