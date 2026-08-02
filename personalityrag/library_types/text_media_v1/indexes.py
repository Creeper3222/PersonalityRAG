from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from ...faiss_runtime import load_faiss
from ...io_utils import run_blocking
from .storage import TextMediaStorage


faiss = load_faiss()


class TextMediaIndex:
    def __init__(self, root: Path):
        self.root = Path(root) / "derived" / "indexes"
        self._generation: str | None = None
        self._index: Any = None
        self._media_generation: str | None = None
        self._media_index: Any = None

    @staticmethod
    def _write_generation(
        temp: Path,
        final: Path,
        generation: str,
        ids: list[int],
        vectors: np.ndarray,
        index_filename: str = "chunks.faiss",
    ) -> None:
        temp.mkdir(parents=True, exist_ok=False)
        dimensions = int(vectors.shape[1]) if vectors.size else 0
        if not dimensions or not ids:
            raise ValueError("Cannot write an empty FAISS generation")
        base = faiss.IndexFlatIP(dimensions)
        index = faiss.IndexIDMap2(base)
        index.add_with_ids(
            np.ascontiguousarray(vectors, dtype=np.float32),
            np.asarray(ids, dtype=np.int64),
        )
        faiss.write_index(index, str(temp / index_filename))
        (temp / "manifest.json").write_text(
            json.dumps(
                {
                    "generation": generation,
                    "dimensions": dimensions,
                    "vector_count": len(ids),
                    "metric": "cosine_ip",
                    "chunk_ids": ids,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        os.replace(temp, final)

    async def rebuild(self, storage: TextMediaStorage) -> dict[str, Any]:
        generation, ids, vectors = await storage.active_vectors()
        self.root.mkdir(parents=True, exist_ok=True)
        if not generation or not ids:
            self._generation = None
            self._index = None
            (self.root / "CURRENT").unlink(missing_ok=True)
        else:
            final = self.root / generation
            temp = self.root / f".{generation}.{os.getpid()}.tmp"
            if temp.exists():
                await run_blocking(shutil.rmtree, temp, True)
            if not final.exists():
                await run_blocking(
                    self._write_generation,
                    temp,
                    final,
                    generation,
                    ids,
                    vectors,
                )
            pointer_temp = self.root / f".CURRENT.{os.getpid()}.tmp"
            await run_blocking(pointer_temp.write_text, generation, encoding="utf-8")
            await run_blocking(os.replace, pointer_temp, self.root / "CURRENT")
            await self.load()
        media = await self._rebuild_media(storage)
        return {
            "generation": generation,
            "vector_count": len(ids),
            "dimensions": int(vectors.shape[1]) if vectors.size else 0,
            **media,
        }

    async def _rebuild_media(self, storage: TextMediaStorage) -> dict[str, Any]:
        generation, ids, vectors = await storage.active_media_vectors()
        media_root = self.root / "media"
        media_root.mkdir(parents=True, exist_ok=True)
        pointer = media_root / "CURRENT"
        if not generation or not ids:
            self._media_generation = None
            self._media_index = None
            pointer.unlink(missing_ok=True)
            return {
                "media_generation": None,
                "media_vector_count": 0,
                "media_dimensions": 0,
            }
        final = media_root / generation
        temp = media_root / f".{generation}.{os.getpid()}.tmp"
        if temp.exists():
            await run_blocking(shutil.rmtree, temp, True)
        if not final.exists():
            await run_blocking(
                self._write_generation,
                temp,
                final,
                generation,
                ids,
                vectors,
                "media.faiss",
            )
        pointer_temp = media_root / f".CURRENT.{os.getpid()}.tmp"
        await run_blocking(pointer_temp.write_text, generation, encoding="utf-8")
        await run_blocking(os.replace, pointer_temp, pointer)
        await self.load_media()
        return {
            "media_generation": generation,
            "media_vector_count": len(ids),
            "media_dimensions": int(vectors.shape[1]) if vectors.size else 0,
        }

    async def load(self) -> None:
        pointer = self.root / "CURRENT"
        if not pointer.exists():
            self._generation = None
            self._index = None
            return
        generation = await run_blocking(pointer.read_text, encoding="utf-8")
        generation = generation.strip()
        if not generation or generation == self._generation:
            return
        path = self.root / generation / "chunks.faiss"
        self._index = await run_blocking(faiss.read_index, str(path)) if path.exists() else None
        self._generation = generation

    async def load_media(self) -> None:
        media_root = self.root / "media"
        pointer = media_root / "CURRENT"
        if not pointer.exists():
            self._media_generation = None
            self._media_index = None
            return
        generation = (await run_blocking(pointer.read_text, encoding="utf-8")).strip()
        if not generation or generation == self._media_generation:
            return
        path = media_root / generation / "media.faiss"
        self._media_index = (
            await run_blocking(faiss.read_index, str(path)) if path.exists() else None
        )
        self._media_generation = generation

    async def search(self, vector: list[float], limit: int) -> list[tuple[int, float]]:
        await self.load()
        if self._index is None:
            return []
        query = np.asarray([vector], dtype=np.float32)
        faiss.normalize_L2(query)
        if query.shape[1] != int(self._index.d):
            raise ValueError("查询向量维度与知识库索引不一致")
        scores, ids = await run_blocking(
            self._index.search,
            np.ascontiguousarray(query),
            max(1, int(limit)),
        )
        return [
            (int(chunk_id), float(score))
            for chunk_id, score in zip(ids[0], scores[0], strict=True)
            if int(chunk_id) >= 0
        ]

    async def search_media(
        self, vector: list[float], limit: int
    ) -> list[tuple[int, float]]:
        await self.load_media()
        if self._media_index is None:
            return []
        query = np.asarray([vector], dtype=np.float32)
        faiss.normalize_L2(query)
        if query.shape[1] != int(self._media_index.d):
            raise ValueError("Query vector dimensions do not match the media index")
        scores, ids = await run_blocking(
            self._media_index.search,
            np.ascontiguousarray(query),
            max(1, int(limit)),
        )
        return [
            (int(row_id), float(score))
            for row_id, score in zip(ids[0], scores[0], strict=True)
            if int(row_id) >= 0
        ]

    @staticmethod
    def _score_subset(
        index: Any,
        query: np.ndarray,
        chunk_ids: list[int],
    ) -> list[tuple[int, float]]:
        ids = np.asarray(chunk_ids, dtype=np.int64)
        vectors = np.asarray(index.reconstruct_batch(ids), dtype=np.float32)
        scores = vectors @ query
        return sorted(
            (
                (int(chunk_id), float(score))
                for chunk_id, score in zip(ids, scores, strict=True)
            ),
            key=lambda item: (-item[1], item[0]),
        )

    async def score_subset(
        self,
        vector: list[float],
        chunk_ids: list[int],
    ) -> list[tuple[int, float]]:
        """Return exact cosine scores for an explicit active chunk subset."""
        await self.load()
        ids = sorted({int(value) for value in chunk_ids})
        if self._index is None or not ids:
            return []
        query = np.asarray(vector, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != int(self._index.d):
            raise ValueError("查询向量维度与知识库索引不一致")
        if not np.isfinite(query).all():
            raise ValueError("查询向量包含非有限值")
        norm = float(np.linalg.norm(query))
        if norm <= 0:
            raise ValueError("查询向量范数无效")
        query = np.ascontiguousarray(query / norm, dtype=np.float32)
        return await run_blocking(self._score_subset, self._index, query, ids)

    def status(self) -> dict[str, Any]:
        return {
            "generation": self._generation,
            "loaded": self._index is not None,
            "vector_count": int(self._index.ntotal) if self._index is not None else 0,
            "dimensions": int(self._index.d) if self._index is not None else 0,
            "media_generation": self._media_generation,
            "media_loaded": self._media_index is not None,
            "media_vector_count": (
                int(self._media_index.ntotal) if self._media_index is not None else 0
            ),
            "media_dimensions": (
                int(self._media_index.d) if self._media_index is not None else 0
            ),
        }
