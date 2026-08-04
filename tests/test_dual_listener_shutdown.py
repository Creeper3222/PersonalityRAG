from __future__ import annotations

import asyncio
from typing import Any

import pytest

import run as launcher


@pytest.mark.asyncio
async def test_first_listener_exit_stops_sibling_without_task_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    servers: list[FakeServer] = []

    class FakeConfig:
        def __init__(self, app: Any, **kwargs: Any) -> None:
            self.app = app
            self.kwargs = kwargs

    class FakeServer:
        def __init__(self, config: FakeConfig) -> None:
            self.config = config
            self.should_exit = False
            self.started = False
            self.cancelled = False
            servers.append(self)

        async def serve(self) -> None:
            self.started = True
            if len(servers) == 2 and self is servers[1]:
                return
            try:
                while not self.should_exit:
                    await asyncio.sleep(0)
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    callbacks: list[Any] = []
    monkeypatch.setattr(launcher.uvicorn, "Config", FakeConfig)
    monkeypatch.setattr(launcher.uvicorn, "Server", FakeServer)
    monkeypatch.setattr(
        launcher.app_module,
        "set_process_shutdown_callback",
        callbacks.append,
    )

    await launcher.serve_dual_ports("127.0.0.1", 8875, 8876)

    assert len(servers) == 2
    assert servers[0].should_exit is True
    assert servers[0].cancelled is False
    assert callbacks[-1] is None
