from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable


class JobControlSignal(Exception):
    """Base class for cooperative task-control transitions."""


class JobPauseRequested(JobControlSignal):
    def __init__(self, reason: str = "manual"):
        super().__init__(reason)
        self.reason = reason


class JobStopRequested(JobControlSignal):
    pass


class JobInterrupted(JobControlSignal):
    def __init__(self, reason: str, message: str, *, error: str | None = None):
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.error = error or message


ProgressCallback = Callable[[float, str], Awaitable[None]]


@dataclass(slots=True)
class ResolvedJobOperation:
    run: Callable[["JobExecutionContext"], Awaitable[Any]]
    rollback: Callable[["JobExecutionContext"], Awaitable[None]]
    cancel_queued: Callable[["JobExecutionContext"], Awaitable[None]] | None = None
    finalize_completed: Callable[["JobExecutionContext"], Awaitable[None]] | None = None


class JobExecutionContext:
    def __init__(self, manager: Any, job_id: str):
        self.manager = manager
        self.job_id = job_id

    async def job(self) -> dict[str, Any]:
        payload = await self.manager.get(self.job_id, internal=True)
        if payload is None:
            raise KeyError(self.job_id)
        return payload

    async def progress(self, value: float, message: str) -> None:
        await self.manager._update(
            self.job_id,
            status="running",
            progress=max(0.0, min(1.0, float(value))),
            message=message,
        )

    async def checkpoint(
        self,
        checkpoint: dict[str, Any],
        *,
        progress: float | None = None,
        message: str | None = None,
    ) -> None:
        fields: dict[str, Any] = {"checkpoint": checkpoint}
        if progress is not None:
            fields["progress"] = max(0.0, min(1.0, float(progress)))
        if message is not None:
            fields["message"] = message
        await self.manager._update(self.job_id, **fields)
        await self.control_point()

    async def control_point(self) -> None:
        request = self.manager.control_request(self.job_id)
        if request == "stop":
            raise JobStopRequested()
        if request in {"pause", "shutdown"}:
            raise JobPauseRequested(request)
