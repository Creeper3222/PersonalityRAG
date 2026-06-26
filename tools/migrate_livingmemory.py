from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from personalityrag.config import load_config
from personalityrag.libraries import LibraryManager


DEFAULT_SOURCE = Path(
    r"D:\astrbot\astrbot\AstrBotLauncher-0.2.0\AstrBot\data"
    r"\plugin_data\astrbot_plugin_livingmemory"
)


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an independent, verified LivingMemory snapshot."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--mode", choices=("rehearsal", "formal"), default="rehearsal"
    )
    parser.add_argument("--library", default="beileite")
    parser.add_argument(
        "--skip-index-rebuild", action="store_true"
    )
    args = parser.parse_args()

    config = load_config(ROOT / "config" / "config.json")
    manager = LibraryManager(ROOT, config)
    await manager.initialize()
    service = await manager.get_runtime(args.library)

    async def progress(value: float, message: str):
        print(f"[{value * 100:6.2f}%] {message}", flush=True)

    try:
        migration = await service.migrator.migrate(
            args.source, mode=args.mode, progress=progress
        )
        await service.storage.initialize()
        rebuild = None
        if not args.skip_index_rebuild:
            rebuild = await manager.rebuild_library(
                args.library, None, progress
            )
        report = {
            "migration": migration,
            "rebuild": rebuild,
            "integrity": await service.storage.integrity_report(),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    finally:
        await manager.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
