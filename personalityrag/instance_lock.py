from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


class SingleInstanceError(RuntimeError):
    def __init__(self, info: dict[str, object]):
        self.info = info
        pid = info.get("pid") or "unknown"
        started_at = info.get("started_at") or "unknown"
        super().__init__(f"PersonalityRAG 已在运行：pid={pid} started_at={started_at}")


@dataclass(slots=True)
class InstanceLock:
    path: Path
    handle: BinaryIO | None = None
    acquired: bool = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            self._lock(handle)
        except OSError as exc:
            info = self._read_info(handle)
            handle.close()
            raise SingleInstanceError(info) from exc
        info = {"pid": os.getpid(), "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        handle.seek(0)
        handle.truncate()
        handle.write(b"\0" + json.dumps(info, ensure_ascii=False).encode("utf-8"))
        handle.flush()
        self.handle = handle
        self.acquired = True

    def release(self) -> None:
        if not self.handle:
            return
        try:
            self._unlock(self.handle)
        finally:
            self.handle.close()
            self.handle = None
            self.acquired = False

    @staticmethod
    def _read_info(handle: BinaryIO) -> dict[str, object]:
        try:
            handle.seek(1)
            raw = handle.read().decode("utf-8", errors="replace").strip()
            if raw:
                return json.loads(raw)
        except Exception:
            pass
        return {"pid": "unknown", "started_at": "unknown"}

    @staticmethod
    def _lock(handle: BinaryIO) -> None:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _unlock(handle: BinaryIO) -> None:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
