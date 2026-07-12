from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable


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
