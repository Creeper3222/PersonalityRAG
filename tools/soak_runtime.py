from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, math.ceil((percentile / 100.0) * len(ordered)) - 1)
    return ordered[min(rank, len(ordered) - 1)]


def _process_resources(pid: int, temp_root: Path) -> dict[str, int | float]:
    proc_root = Path(f"/proc/{pid}")
    if proc_root.is_dir():
        status: dict[str, str] = {}
        for line in (proc_root / "status").read_text("utf-8").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                status[key] = value.strip()
        rss_bytes = int(status.get("VmRSS", "0 kB").split()[0]) * 1024
        threads = int(status.get("Threads", "0"))
        fd_count = len(tuple((proc_root / "fd").iterdir()))
    elif os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        class ThreadEntry32(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ThreadID", wintypes.DWORD),
                ("th32OwnerProcessID", wintypes.DWORD),
                ("tpBasePri", wintypes.LONG),
                ("tpDeltaPri", wintypes.LONG),
                ("dwFlags", wintypes.DWORD),
            ]

        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        process = kernel32.OpenProcess(0x1000 | 0x0010, False, pid)
        if not process:
            raise OSError(ctypes.get_last_error(), f"cannot open process {pid}")
        try:
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            if not ctypes.windll.psapi.GetProcessMemoryInfo(
                process, ctypes.byref(counters), counters.cb
            ):
                raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
            handle_count = wintypes.DWORD()
            if not kernel32.GetProcessHandleCount(
                process, ctypes.byref(handle_count)
            ):
                raise OSError(ctypes.get_last_error(), "GetProcessHandleCount failed")
            rss_bytes = int(counters.WorkingSetSize)
            fd_count = int(handle_count.value)
        finally:
            kernel32.CloseHandle(process)

        snapshot = kernel32.CreateToolhelp32Snapshot(0x00000004, 0)
        invalid_handle = ctypes.c_void_p(-1).value
        if snapshot == invalid_handle:
            raise OSError(ctypes.get_last_error(), "thread snapshot failed")
        threads = 0
        try:
            entry = ThreadEntry32()
            entry.dwSize = ctypes.sizeof(entry)
            available = kernel32.Thread32First(snapshot, ctypes.byref(entry))
            while available:
                if int(entry.th32OwnerProcessID) == pid:
                    threads += 1
                available = kernel32.Thread32Next(snapshot, ctypes.byref(entry))
        finally:
            kernel32.CloseHandle(snapshot)
    else:
        raise RuntimeError("resource sampling supports Linux and Windows")
    temp_files = sum(1 for item in temp_root.rglob("*") if item.is_file())
    return {
        "rss_mib": round(rss_bytes / (1024.0 * 1024.0), 3),
        "threads": threads,
        "file_descriptors": fd_count,
        "temporary_files": temp_files,
    }


def _growth_percent(before: float, after: float) -> float:
    if before <= 0:
        return 0.0 if after <= 0 else 100.0
    return round(((after - before) / before) * 100.0, 3)


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(
        (Path(args.state_root) / "config" / "config.json").read_text("utf-8")
    )
    authorization = {"Authorization": f"Bearer {config['api_key']}"}
    process_id = args.process_id or (
        1 if Path("/proc/1/status").is_file() else os.getpid()
    )
    temp_root = Path(args.temp_root or tempfile.gettempdir())
    endpoints = (
        (args.web_url, "/api/v1/health", None),
        (args.access_url, "/api/v1/health", None),
        (args.web_url, "/api/v1/databases?stats_mode=summary", authorization),
        (
            args.web_url,
            "/api/v1/jobs?limit=20&include_details=false",
            authorization,
        ),
        (args.web_url, "/api/v1/providers", authorization),
        (args.web_url, "/api/v1/settings", authorization),
        (args.web_url, "/api/v1/database-types", authorization),
    )
    limits = httpx.Limits(
        max_connections=max(4, args.concurrency + 2),
        max_keepalive_connections=max(2, args.concurrency),
        keepalive_expiry=30.0,
    )
    status_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    latencies_ms: list[float] = []
    event_loop_lag_ms: list[float] = []
    request_index = 0

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(15.0),
        limits=limits,
        follow_redirects=False,
        trust_env=False,
    ) as client:

        async def request_once() -> None:
            nonlocal request_index
            current = request_index
            request_index += 1
            base_url, path, headers = endpoints[current % len(endpoints)]
            started = time.perf_counter()
            try:
                response = await client.get(
                    f"{base_url.rstrip('/')}{path}", headers=headers
                )
                status_counts[str(response.status_code)] += 1
                await response.aread()
            except Exception as exc:  # pragma: no cover - acceptance diagnostics
                error_counts[type(exc).__name__] += 1
            finally:
                latencies_ms.append((time.perf_counter() - started) * 1000.0)

        async def run_fixed(total: int, concurrency: int) -> None:
            queue: asyncio.Queue[None] = asyncio.Queue()
            for _ in range(total):
                queue.put_nowait(None)

            async def worker() -> None:
                while True:
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    try:
                        await request_once()
                    finally:
                        queue.task_done()

            await asyncio.gather(
                *(worker() for _ in range(max(1, min(concurrency, total))))
            )

        await run_fixed(args.warmup_requests, args.concurrency)
        baseline = _process_resources(process_id, temp_root)
        burst_latency_start = len(latencies_ms)
        await run_fixed(args.burst_requests, args.concurrency)
        after_burst = _process_resources(process_id, temp_root)
        burst_latencies = latencies_ms[burst_latency_start:]

        stop_lag_monitor = asyncio.Event()

        async def monitor_lag() -> None:
            interval = 0.1
            expected = time.perf_counter() + interval
            while not stop_lag_monitor.is_set():
                await asyncio.sleep(interval)
                now = time.perf_counter()
                event_loop_lag_ms.append(max(0.0, now - expected) * 1000.0)
                expected = now + interval

        monitor = asyncio.create_task(monitor_lag())
        steady_started = time.monotonic()
        steady_deadline = steady_started + args.duration_seconds
        interval = 1.0 / max(0.1, args.steady_rps)
        pending: set[asyncio.Task[None]] = set()
        try:
            next_request_at = time.monotonic()
            while time.monotonic() < steady_deadline:
                pending = {task for task in pending if not task.done()}
                if len(pending) < args.concurrency:
                    pending.add(asyncio.create_task(request_once()))
                next_request_at += interval
                await asyncio.sleep(max(0.0, next_request_at - time.monotonic()))
            if pending:
                await asyncio.gather(*pending)
        finally:
            stop_lag_monitor.set()
            await monitor

    final = _process_resources(process_id, temp_root)
    leak_growth = {
        key: _growth_percent(float(baseline[key]), float(final[key]))
        for key in ("threads", "file_descriptors", "temporary_files")
    }
    non_success = sum(
        count for status, count in status_counts.items() if status != "200"
    )
    passed = (
        not error_counts
        and non_success == 0
        and float(after_burst["rss_mib"]) <= 300.0
        and float(final["rss_mib"]) <= 300.0
        and all(value <= 5.0 for value in leak_growth.values())
        and _percentile(event_loop_lag_ms, 99.0) <= 50.0
    )
    return {
        "schema_version": 1,
        "passed": passed,
        "duration_seconds": round(time.monotonic() - steady_started, 3),
        "warmup_requests": args.warmup_requests,
        "burst_requests": args.burst_requests,
        "steady_requests": max(0, request_index - args.warmup_requests - args.burst_requests),
        "total_requests": request_index,
        "concurrency": args.concurrency,
        "steady_rps": args.steady_rps,
        "status_counts": dict(status_counts),
        "error_counts": dict(error_counts),
        "latency_ms": {
            "burst_p50": round(_percentile(burst_latencies, 50.0), 3),
            "burst_p95": round(_percentile(burst_latencies, 95.0), 3),
            "overall_p95": round(_percentile(latencies_ms, 95.0), 3),
        },
        "event_loop_lag_ms": {
            "p99": round(_percentile(event_loop_lag_ms, 99.0), 3),
            "maximum": round(max(event_loop_lag_ms, default=0.0), 3),
        },
        "resources": {
            "baseline": baseline,
            "after_burst": after_burst,
            "final": final,
            "leak_growth_percent": leak_growth,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run bounded PersonalityRAG control-plane leak and soak validation."
    )
    parser.add_argument("--web-url", default="http://127.0.0.1:8765")
    parser.add_argument("--access-url", default="http://127.0.0.1:8766")
    parser.add_argument("--state-root", default=os.environ.get("PERSONALITYRAG_STATE_ROOT", "/app/state"))
    parser.add_argument("--process-id", type=int)
    parser.add_argument("--temp-root")
    parser.add_argument("--warmup-requests", type=int, default=200)
    parser.add_argument("--burst-requests", type=int, default=5000)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--steady-rps", type=float, default=4.0)
    parser.add_argument("--duration-seconds", type=float, default=7200.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = asyncio.run(_run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
