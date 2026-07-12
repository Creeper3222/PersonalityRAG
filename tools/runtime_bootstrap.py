from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from typing import Sequence


REQUIRED_PYTHON = (3, 12)
MARKER_FORMAT = 1


def python_supported(
    version_info: Sequence[int] | None = None,
    pointer_bits: int | None = None,
    implementation: str | None = None,
) -> bool:
    version = tuple(version_info or sys.version_info)
    bits = pointer_bits if pointer_bits is not None else struct.calcsize("P") * 8
    runtime = implementation or sys.implementation.name
    return (
        runtime == "cpython"
        and version[:2] == REQUIRED_PYTHON
        and bits == 64
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dependency_fingerprint(
    *,
    lock_path: Path,
    requirements_path: Path,
) -> str:
    payload = {
        "format": MARKER_FORMAT,
        "python": {
            "implementation": sys.implementation.name,
            "version": list(sys.version_info[:3]),
            "pointer_bits": struct.calcsize("P") * 8,
        },
        "requirements_sha256": _sha256(requirements_path),
        "runtime_lock_sha256": _sha256(lock_path),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def marker_is_current(marker_path: Path, fingerprint: str) -> bool:
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    if not isinstance(payload, dict):
        return False
    return bool(
        payload.get("format") == MARKER_FORMAT
        and payload.get("fingerprint") == fingerprint
    )


def write_marker(marker_path: Path, fingerprint: str) -> None:
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker_path.with_name(f"{marker_path.name}.{os.getpid()}.tmp")
    payload = {
        "format": MARKER_FORMAT,
        "fingerprint": fingerprint,
        "python": ".".join(str(item) for item in sys.version_info[:3]),
    }
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, marker_path)


def _fingerprint(args: argparse.Namespace) -> str:
    return dependency_fingerprint(
        lock_path=args.lock.resolve(),
        requirements_path=args.requirements.resolve(),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    for name in ("check", "mark"):
        command = subparsers.add_parser(name)
        command.add_argument("--marker", type=Path, required=True)
        command.add_argument("--lock", type=Path, required=True)
        command.add_argument("--requirements", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not python_supported():
        print(
            "PersonalityRAG requires CPython 3.12 x64; "
            f"current runtime is {sys.version.split()[0]} "
            f"({struct.calcsize('P') * 8}-bit).",
            file=sys.stderr,
        )
        return 1
    if args.command == "validate":
        return 0
    try:
        fingerprint = _fingerprint(args)
        if args.command == "check":
            return 0 if marker_is_current(args.marker, fingerprint) else 1
        write_marker(args.marker, fingerprint)
        return 0
    except OSError as exc:
        print(f"Dependency fingerprint error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
