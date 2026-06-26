from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from personalityrag.migration import table_fingerprint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    result = {}
    failed = False
    derived_tables = {
        "documents_fts",
        "livingmemory_memories_fts",
        "livingmemory_graph_entries_fts",
        "memory_atoms_fts",
    }
    for name in ("livingmemory.db", "conversations.db"):
        source = table_fingerprint(args.source / name)
        target = table_fingerprint(args.target / name)
        mismatches = []
        for table, details in source["tables"].items():
            if table in derived_tables:
                continue
            actual = target["tables"].get(table)
            if not actual or details["count"] != actual["count"] or details[
                "rows_sha256"
            ] != actual["rows_sha256"]:
                mismatches.append(table)
        result[name] = {
            "source": source,
            "target": target,
            "mismatches": mismatches,
            "ignored_derived_tables": sorted(derived_tables),
        }
        failed = failed or bool(mismatches)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
