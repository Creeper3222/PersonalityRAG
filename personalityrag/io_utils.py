from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import time
from typing import Any, Callable
import uuid


FILE_CHUNK_BYTES = 1024 * 1024


class UploadSizeLimitError(ValueError):
    pass


async def run_blocking(function: Callable, /, *args, **kwargs):
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await asyncio.gather(task, return_exceptions=True)
        raise


def atomic_write_json(
    path: Path,
    payload: dict[str, Any],
    *,
    retries: int = 8,
    base_delay: float = 0.05,
) -> None:
    """Write JSON through a unique temp file and atomically replace the target.

    Windows can transiently deny replacing a recently written checkpoint file
    when another reader, indexer, or security scanner still has a handle open.
    Retrying the final replace keeps resumable task checkpoints deterministic
    without changing the persisted payload.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        attempts = max(1, int(retries))
        for attempt in range(attempts):
            try:
                os.replace(temp, path)
                return
            except PermissionError:
                if attempt >= attempts - 1:
                    raise
                time.sleep(max(0.0, float(base_delay)) * (2**attempt))
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


async def save_upload_file(
    upload: Any,
    target: Path,
    *,
    max_bytes: int | None = None,
) -> int:
    await run_blocking(target.parent.mkdir, parents=True, exist_ok=True)
    handle = await run_blocking(target.open, "wb")
    total = 0
    try:
        while chunk := await upload.read(FILE_CHUNK_BYTES):
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                raise UploadSizeLimitError("uploaded file exceeds size limit")
            await run_blocking(handle.write, chunk)
        await run_blocking(handle.flush)
        return total
    except BaseException:
        await run_blocking(handle.close)
        await run_blocking(target.unlink, missing_ok=True)
        raise
    finally:
        if not handle.closed:
            await run_blocking(handle.close)
        await upload.close()
