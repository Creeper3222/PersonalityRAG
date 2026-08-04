from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable
import uuid

from .performance import measure_phase
from .resource_quotas import io_slot


FILE_CHUNK_BYTES = 1024 * 1024
CHECKPOINT_SLOT_NAMES = ("checkpoint.a.json", "checkpoint.b.json")


class UploadSizeLimitError(ValueError):
    pass


async def run_blocking(function: Callable, /, *args, **kwargs):
    async def invoke():
        async with io_slot():
            return await asyncio.to_thread(function, *args, **kwargs)

    task = asyncio.create_task(invoke())
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


def _checkpoint_checksum(sequence: int, payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        {"sequence": int(sequence), "payload": payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_checkpoint_slot(path: Path) -> tuple[int, dict[str, Any]]:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(envelope, dict):
        raise ValueError(f"invalid checkpoint envelope: {path.name}")
    sequence = int(envelope.get("sequence") or 0)
    payload = envelope.get("payload")
    checksum = str(envelope.get("checksum") or "")
    if sequence <= 0 or not isinstance(payload, dict):
        raise ValueError(f"invalid checkpoint payload: {path.name}")
    if checksum != _checkpoint_checksum(sequence, payload):
        raise ValueError(f"checkpoint checksum mismatch: {path.name}")
    return sequence, payload


def read_ab_checkpoint(checkpoint_dir: Path) -> dict[str, Any] | None:
    """Return the newest valid checkpoint, with legacy file compatibility."""

    checkpoint_dir = Path(checkpoint_dir)
    valid: list[tuple[int, dict[str, Any]]] = []
    slots_present = False
    for name in CHECKPOINT_SLOT_NAMES:
        path = checkpoint_dir / name
        if not path.exists():
            continue
        slots_present = True
        try:
            valid.append(_read_checkpoint_slot(path))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    if valid:
        return max(valid, key=lambda item: item[0])[1]
    if slots_present:
        raise ValueError("all checkpoint slots are corrupt")

    for name in ("checkpoint.json", "checkpoint.prev.json"):
        path = checkpoint_dir / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"invalid legacy checkpoint: {path.name}")
        return payload
    return None


def write_ab_checkpoint(
    checkpoint_dir: Path,
    payload: dict[str, Any],
) -> Path:
    """Persist a checksummed checkpoint by alternating between two slots."""

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sequences: list[int] = []
    for name in CHECKPOINT_SLOT_NAMES:
        path = checkpoint_dir / name
        if not path.exists():
            continue
        try:
            sequences.append(_read_checkpoint_slot(path)[0])
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    sequence = max(sequences, default=0) + 1
    slot_index = (sequence - 1) % len(CHECKPOINT_SLOT_NAMES)
    target = checkpoint_dir / CHECKPOINT_SLOT_NAMES[slot_index]
    completed_rows = int(payload.get("completed_documents") or 0) + int(
        payload.get("completed_graph_entries") or 0
    )
    with measure_phase(
        "resumable_task",
        "checkpoint_write",
        rows=completed_rows,
    ):
        atomic_write_json(
            target,
            {
                "version": 1,
                "sequence": sequence,
                "checksum": _checkpoint_checksum(sequence, payload),
                "payload": payload,
            },
        )
    return target


async def save_upload_file(
    upload: Any,
    target: Path,
    *,
    max_bytes: int | None = None,
    hasher: Any | None = None,
) -> int:
    source = getattr(upload, "file", None)
    try:
        if source is not None and callable(getattr(source, "read", None)):
            return await run_blocking(
                _copy_upload_stream,
                source,
                target,
                max_bytes,
                hasher,
            )
        return await _copy_async_upload_stream(
            upload,
            target,
            max_bytes=max_bytes,
            hasher=hasher,
        )
    finally:
        await upload.close()


def _copy_upload_stream(
    source: Any,
    target: Path,
    max_bytes: int | None,
    hasher: Any | None = None,
) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        with target.open("wb") as handle:
            while chunk := source.read(FILE_CHUNK_BYTES):
                total += len(chunk)
                if max_bytes is not None and total > max_bytes:
                    raise UploadSizeLimitError("uploaded file exceeds size limit")
                if hasher is not None:
                    hasher.update(chunk)
                handle.write(chunk)
            handle.flush()
        return total
    except BaseException:
        target.unlink(missing_ok=True)
        raise


async def _copy_async_upload_stream(
    upload: Any,
    target: Path,
    *,
    max_bytes: int | None,
    hasher: Any | None = None,
) -> int:
    await run_blocking(target.parent.mkdir, parents=True, exist_ok=True)
    handle = await run_blocking(target.open, "wb")
    total = 0
    try:
        while chunk := await upload.read(FILE_CHUNK_BYTES):
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                raise UploadSizeLimitError("uploaded file exceeds size limit")
            if hasher is not None:
                hasher.update(chunk)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(FILE_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
