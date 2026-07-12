from __future__ import annotations

from types import SimpleNamespace

import pytest

from personalityrag.config import AppConfig
from personalityrag.libraries import LibraryManager


class ValidationStorage:
    async def integrity_report(self):
        healthy = {"exists": True, "integrity": "ok", "foreign_key_errors": 0}
        return {"livingmemory": healthy, "conversations": healthy}

    async def statistics(self):
        return {"total_memories": 2, "graph_entries": 3}

    async def document_ids(self):
        return [11, 12]

    async def graph_entry_ids(self):
        return [21, 22, 23]


class ValidationIndexes:
    def status(self):
        return {"document_vectors": 2, "graph_vectors": 3}

    def indexed_ids(self):
        return {11, 12}, {21, 22, 23}

    async def search_documents(self, *_args, **_kwargs):
        raise AssertionError("startup validation must not call the provider")


@pytest.mark.asyncio
async def test_startup_validation_is_local_and_provider_independent(
    tmp_path,
) -> None:
    manager = LibraryManager(tmp_path, AppConfig())
    runtime = SimpleNamespace(
        storage=ValidationStorage(),
        indexes=ValidationIndexes(),
    )

    await manager._validate_runtime(runtime)


@pytest.mark.asyncio
async def test_startup_validation_rejects_exact_faiss_id_drift(tmp_path) -> None:
    manager = LibraryManager(tmp_path, AppConfig())
    indexes = ValidationIndexes()
    indexes.indexed_ids = lambda: ({11, 99}, {21, 22, 23})
    runtime = SimpleNamespace(storage=ValidationStorage(), indexes=indexes)

    with pytest.raises(RuntimeError, match="文档 FAISS ID 数量"):
        await manager._validate_runtime(runtime)
