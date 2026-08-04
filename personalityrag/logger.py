from __future__ import annotations

import asyncio
import atexit
import logging
import queue
import re
import threading
from collections import deque
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from typing import Any


LOGGER_NAME = "personalityrag"
logger = logging.getLogger(LOGGER_NAME)


_SECRET_PATTERNS = (
    re.compile(r"(prag_[A-Za-z0-9_\-]{12,})"),
    re.compile(r"(?i)\b(?:psk|pkb)-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"(?i)(api[_-]?key|authorization|bearer|token|cookie)(\s*[=:]\s*)([^,\s}]+)"),
    re.compile(r"(?i)data:image/[^;\s]+;base64,[A-Za-z0-9+/=]+"),
    re.compile(
        r"(?i)https?://[^\s]+/api/v1/knowledge-libraries/[^\s?]+/"
        r"assets/[^\s?]+/(?:content|thumbnail)(?:\?[^\s]*)?"
    ),
)


def sanitize_log_message(message: str, *, max_chars: int = 2000) -> str:
    text = str(message)
    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 3:
            text = pattern.sub(r"\1\2[redacted]", text)
        else:
            text = pattern.sub("[redacted]", text)
    if len(text) > max_chars:
        text = text[:max_chars] + "…[truncated]"
    return text


def safe_summary(value: Any, *, max_chars: int = 120) -> str:
    text = " ".join(str(value or "").split())
    text = sanitize_log_message(text, max_chars=max_chars)
    return text


class RedactingFilter(logging.Filter):
    """Sanitize every record before it reaches WebUI, file, or console sinks."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = sanitize_log_message(record.getMessage())
        record.args = ()
        return True


class WebLogBuffer:
    def __init__(
        self,
        max_entries: int = 2000,
        *,
        max_bytes: int = 4 * 1024 * 1024,
        max_entry_bytes: int = 32 * 1024,
    ):
        self.max_entries = max(1, int(max_entries))
        self.max_bytes = max(1024, int(max_bytes))
        self.max_entry_bytes = max(256, int(max_entry_bytes))
        self._entries: deque[dict[str, Any]] = deque()
        self._total_bytes = 0
        self._lock = threading.Lock()
        self._next_id = 1
        self._version = 0
        self._changed = asyncio.Event()
        self._event_loop: asyncio.AbstractEventLoop | None = None

    def append_record(self, record: logging.LogRecord) -> None:
        message = sanitize_log_message(record.getMessage())
        encoded = message.encode("utf-8", errors="replace")
        if len(encoded) > self.max_entry_bytes:
            encoded = encoded[: self.max_entry_bytes]
            message = encoded.decode("utf-8", errors="ignore") + "…[truncated]"
        level = record.levelname.upper()
        if level == "WARNING":
            level = "WARN"
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        timestamp += f".{int(record.msecs):03d}"
        entry = {
            "id": self._next_id,
            "timestamp": timestamp,
            "level": level,
            "logger": record.name,
            "message": message,
        }
        entry_bytes = (
            len(timestamp)
            + len(level)
            + len(record.name)
            + len(message.encode("utf-8", errors="replace"))
            + 640
        )
        entry["_bytes"] = entry_bytes

        with self._lock:
            self._entries.append(entry)
            self._total_bytes += entry_bytes
            while len(self._entries) > self.max_entries or self._total_bytes > self.max_bytes:
                removed = self._entries.popleft()
                self._total_bytes -= int(removed.get("_bytes") or 0)
            self._next_id += 1
            self._version += 1
        self._notify_changed()

    def get_entries(self, *, after_id: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            result: list[dict[str, Any]] = []
            for entry in self._entries:
                if int(entry["id"]) <= int(after_id):
                    continue
                item = {key: value for key, value in entry.items() if key != "_bytes"}
                item["line"] = (
                    f"[{item['timestamp']}] [{item['level']}] "
                    f"[{item['logger']}] {item['message']}"
                )
                result.append(item)
            return result

    def summary(self) -> dict[str, int]:
        with self._lock:
            return {
                "entry_count": len(self._entries),
                "max_entries": self.max_entries,
                "bytes": self._total_bytes,
                "max_bytes": self.max_bytes,
            }

    def latest_id(self) -> int:
        with self._lock:
            if self._entries:
                return int(self._entries[-1]["id"])
            return max(0, self._next_id - 1)

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    async def wait_for_change(self, version: int, *, timeout: float) -> None:
        self._event_loop = asyncio.get_running_loop()
        if self.version != version:
            return
        try:
            await asyncio.wait_for(self._changed.wait(), timeout=max(0.0, float(timeout)))
        except asyncio.TimeoutError:
            return
        finally:
            self._changed.clear()

    def clear(self) -> int:
        with self._lock:
            cleared = len(self._entries)
            self._entries.clear()
            self._total_bytes = 0
            self._next_id = 1
            self._version += 1
        self._notify_changed()
        return cleared

    def _notify_changed(self) -> None:
        loop = self._event_loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._changed.set)


class WebLogHandler(logging.Handler):
    def __init__(self, buffer: WebLogBuffer):
        super().__init__()
        self.buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.buffer.append_record(record)
        except Exception:
            self.handleError(record)


_buffer = WebLogBuffer()
_log_queue: queue.Queue[logging.LogRecord] | None = None
_queue_listener: QueueListener | None = None
_persistent_handlers: tuple[logging.Handler, ...] = ()


class BoundedQueueHandler(QueueHandler):
    def __init__(
        self,
        record_queue: queue.Queue[logging.LogRecord],
        fallback_handlers: tuple[logging.Handler, ...],
    ):
        super().__init__(record_queue)
        self.fallback_handlers = fallback_handlers

    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            if record.levelno >= logging.WARNING:
                for handler in self.fallback_handlers:
                    if record.levelno >= handler.level:
                        handler.handle(record)


def get_log_buffer() -> WebLogBuffer:
    return _buffer


def flush_logging() -> None:
    record_queue = _log_queue
    if record_queue is not None:
        record_queue.join()
    for handler in _persistent_handlers:
        try:
            handler.flush()
        except (OSError, ValueError):
            pass


def shutdown_logging() -> None:
    global _log_queue, _queue_listener, _persistent_handlers
    listener = _queue_listener
    if listener is not None:
        flush_logging()
        listener.stop()
    for handler in _persistent_handlers:
        try:
            handler.close()
        except Exception:
            pass
    _log_queue = None
    _queue_listener = None
    _persistent_handlers = ()


def configure_logging(
    log_file: Path,
    *,
    level_name: str = "INFO",
    file_max_bytes: int = 10 * 1024 * 1024,
    file_backup_count: int = 3,
    web_max_entries: int = 2000,
    web_max_bytes: int = 4 * 1024 * 1024,
    web_max_entry_bytes: int = 32 * 1024,
    file_enabled: bool = True,
) -> None:
    global _buffer, _log_queue, _queue_listener, _persistent_handlers
    shutdown_logging()
    level = getattr(logging, str(level_name or "INFO").upper(), logging.INFO)
    if file_enabled:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    _buffer = WebLogBuffer(
        max_entries=web_max_entries,
        max_bytes=web_max_bytes,
        max_entry_bytes=web_max_entry_bytes,
    )

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    app_logger = logging.getLogger(LOGGER_NAME)
    # Avoid constructing and redacting records that every configured sink
    # would discard. DEBUG remains available when the configured level is
    # explicitly DEBUG.
    app_logger.setLevel(level)
    app_logger.propagate = False
    for handler in list(app_logger.handlers):
        app_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    web_handler = WebLogHandler(_buffer)
    web_handler.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    record_queue: queue.Queue[logging.LogRecord] = queue.Queue(maxsize=8192)
    persistent_handlers_list: list[logging.Handler] = []
    if file_enabled:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max(1, int(file_max_bytes)),
            backupCount=max(0, int(file_backup_count)),
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        persistent_handlers_list.append(file_handler)
    persistent_handlers_list.append(stream_handler)
    persistent_handlers = tuple(persistent_handlers_list)
    queue_handler = BoundedQueueHandler(record_queue, persistent_handlers)
    queue_handler.setLevel(level)
    redacting_filter = RedactingFilter()
    queue_handler.addFilter(redacting_filter)
    web_handler.addFilter(redacting_filter)
    listener = QueueListener(
        record_queue,
        *persistent_handlers,
        respect_handler_level=True,
    )
    listener.start()

    _log_queue = record_queue
    _queue_listener = listener
    _persistent_handlers = persistent_handlers

    app_logger.addHandler(queue_handler)
    app_logger.addHandler(web_handler)


atexit.register(shutdown_logging)
