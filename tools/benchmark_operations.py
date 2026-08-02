from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENDPOINTS = (
    ("health", "/api/v1/health", False),
    ("databases_summary", "/api/v1/databases?stats_mode=summary", True),
    ("stats", "/api/v1/stats", True),
    ("jobs_active", "/api/v1/jobs?status=active", True),
)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def _default_credential(state_root: Path) -> str:
    config_path = state_root / "config" / "config.json"
    if not config_path.exists():
        return ""
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return str(payload.get("api_key") or "")


def _request(
    base_url: str,
    path: str,
    *,
    credential: str,
    authenticated: bool,
    timeout: float,
) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    if authenticated and credential:
        headers["Authorization"] = f"Bearer {credential}"
    request = urllib.request.Request(base_url.rstrip("/") + path, headers=headers)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        body = exc.read()
        status = int(exc.code)
    return {
        "elapsed_ms": (time.perf_counter() - started) * 1000,
        "status": status,
        "bytes": len(body),
    }


def _summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    timings = [float(item["elapsed_ms"]) for item in samples]
    return {
        "requests": len(samples),
        "statuses": sorted({int(item["status"]) for item in samples}),
        "mean_ms": round(statistics.fmean(timings), 2),
        "median_ms": round(statistics.median(timings), 2),
        "p95_ms": round(_percentile(timings, 0.95), 2),
        "max_ms": round(max(timings), 2),
        "mean_bytes": round(statistics.fmean(int(item["bytes"]) for item in samples), 2),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    state_root = args.state_root.resolve()
    credential = args.credential or _default_credential(state_root)
    report: dict[str, Any] = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "state_root": str(state_root),
        "rounds": args.rounds,
        "concurrency": args.concurrency,
        "process_id": os.getpid(),
        "sequential": {},
        "concurrent": {},
    }
    for name, path, authenticated in DEFAULT_ENDPOINTS:
        samples = [
            _request(
                args.base_url,
                path,
                credential=credential,
                authenticated=authenticated,
                timeout=args.timeout,
            )
            for _ in range(args.rounds)
        ]
        report["sequential"][name] = _summary(samples)

    concurrent_path = "/api/v1/databases?stats_mode=summary"
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        samples = list(
            executor.map(
                lambda _: _request(
                    args.base_url,
                    concurrent_path,
                    credential=credential,
                    authenticated=True,
                    timeout=args.timeout,
                ),
                range(args.concurrency),
            )
        )
    report["concurrent"]["databases_summary"] = {
        **_summary(samples),
        "wall_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproducible, read-only PersonalityRAG operational benchmark."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--state-root", type=Path, default=ROOT)
    parser.add_argument("--credential", default="")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.rounds = max(1, int(args.rounds))
    args.concurrency = max(1, int(args.concurrency))
    report = run(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
