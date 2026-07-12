from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from personalityrag.control import ControlStore  # noqa: E402
from personalityrag.providers import build_provider  # noqa: E402


def build_queries(db_path: Path, limit: int) -> list[str]:
    con = sqlite3.connect(db_path)
    try:
        queries: list[str] = []
        seen: set[str] = set()
        for text, raw_metadata in con.execute(
            "SELECT text,metadata FROM documents ORDER BY id"
        ):
            try:
                metadata = json.loads(raw_metadata or "{}")
            except json.JSONDecodeError:
                metadata = {}
            candidates = []
            candidates.extend(metadata.get("topics") or [])
            candidates.extend(metadata.get("participants") or [])
            candidates.extend(metadata.get("key_facts") or [])
            candidates.append(text)
            for raw in candidates:
                value = str(raw or "").strip()
                fingerprint = hashlib.sha256(value.encode()).hexdigest()
                if len(value) < 2 or fingerprint in seen:
                    continue
                seen.add(fingerprint)
                queries.append(value[:500])
                if len(queries) >= limit:
                    return queries
        return queries
    finally:
        con.close()


def ids(index, vector: np.ndarray, k: int) -> list[int]:
    distances, result_ids = index.search(vector, k)
    return [int(item) for item in result_ids[0] if int(item) >= 0]


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--queries", type=int, default=64)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    generation = (args.target / "indexes" / "CURRENT").read_text().strip()
    generation_dir = args.target / "indexes" / generation
    manifest = json.loads(
        (generation_dir / "manifest.json").read_text(encoding="utf-8")
    )
    old_document = faiss.read_index(str(args.source / "livingmemory.index"))
    old_graph = faiss.read_index(str(args.source / "livingmemory_graph.index"))
    new_document = faiss.read_index(str(generation_dir / "documents.index"))
    new_graph = faiss.read_index(str(generation_dir / "graph.index"))
    queries = build_queries(args.target / "livingmemory.db", args.queries)
    system_path = args.target.parents[1] / "personalityrag_system.db"
    control = ControlStore(system_path)
    provider_record = await control.get_provider(
        str(manifest.get("provider_id") or "vllm_embedding"),
        int(manifest.get("provider_revision") or 1),
    )
    if not provider_record:
        raise RuntimeError("无法解析目标 generation 使用的 Provider revision")
    provider = build_provider(provider_record.config)
    doc_exact = graph_exact = 0
    doc_overlap = graph_overlap = 0.0
    try:
        for start in range(0, len(queries), 64):
            vectors = await provider.get_embeddings(queries[start : start + 64])
            for raw_vector in vectors:
                vector = np.asarray([raw_vector], dtype=np.float32)
                faiss.normalize_L2(vector)
                old_doc_ids = ids(old_document, vector, args.k)
                new_doc_ids = ids(new_document, vector, args.k)
                old_graph_ids = ids(old_graph, vector, args.k)
                new_graph_ids = ids(new_graph, vector, args.k)
                doc_exact += old_doc_ids == new_doc_ids
                graph_exact += old_graph_ids == new_graph_ids
                doc_overlap += len(set(old_doc_ids) & set(new_doc_ids)) / max(
                    1, args.k
                )
                graph_overlap += len(
                    set(old_graph_ids) & set(new_graph_ids)
                ) / max(1, args.k)
    finally:
        await provider.close()

    total = max(1, len(queries))
    report = {
        "queries": len(queries),
        "top_k": args.k,
        "document_exact_order_ratio": doc_exact / total,
        "document_set_overlap_ratio": doc_overlap / total,
        "graph_exact_order_ratio": graph_exact / total,
        "graph_set_overlap_ratio": graph_overlap / total,
        "source_document_count": old_document.ntotal,
        "target_document_count": new_document.ntotal,
        "source_graph_count": old_graph.ntotal,
        "target_graph_count": new_graph.ntotal,
        "query_text_in_report": False,
    }
    report_path = args.target / "reports" / (
        "index-compatibility-" + generation + ".json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if (
        report["document_set_overlap_ratio"] >= 0.99
        and report["graph_set_overlap_ratio"] >= 0.99
    ) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
