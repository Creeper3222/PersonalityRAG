from __future__ import annotations

from contextlib import contextmanager
import time
from typing import Iterator

from .logger import logger


@contextmanager
def measure_phase(
    operation: str,
    phase: str,
    *,
    rows: int | None = None,
    bytes_count: int | None = None,
) -> Iterator[None]:
    """Log coarse operation timing without including payload or secret data."""

    started = time.perf_counter()
    try:
        yield
    finally:
        logger.info(
            "性能阶段：operation=%s phase=%s rows=%s bytes=%s elapsed_ms=%.2f",
            str(operation),
            str(phase),
            "" if rows is None else max(0, int(rows)),
            "" if bytes_count is None else max(0, int(bytes_count)),
            (time.perf_counter() - started) * 1000,
        )
