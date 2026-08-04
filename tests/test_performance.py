from __future__ import annotations

import logging

import pytest

from personalityrag import performance


def test_measure_phase_logs_only_operational_dimensions(monkeypatch):
    messages: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(
        performance.logger,
        "info",
        lambda message, *args: messages.append((message, args)),
    )
    clock = iter([10.0, 10.025])
    monkeypatch.setattr(performance.time, "perf_counter", lambda: next(clock))

    with performance.measure_phase(
        "library_copy",
        "sqlite_snapshot",
        rows=7,
        bytes_count=2048,
    ):
        logging.getLogger("test").debug("payload is deliberately unrelated")

    assert len(messages) == 1
    assert messages[0][1][:-1] == ("library_copy", "sqlite_snapshot", 7, 2048)
    assert messages[0][1][-1] == pytest.approx(25.0)
    assert "payload" not in messages[0][0]
