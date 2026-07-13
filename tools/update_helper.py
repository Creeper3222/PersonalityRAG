from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


MANAGED_DIRECTORIES = ("personalityrag", "static")
MANAGED_FILES = (
    "launcher.bat",
    "run.py",
    "requirements.txt",
    "requirements-runtime.lock",
    "tools/runtime_bootstrap.py",
    "tools/update_helper.py",
)
INCOMPLETE_STAGES = frozenset(
    {"prepared", "helper_started", "backing_up", "replacing", "starting_target", "checking_target"}
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any], **changes: Any) -> dict[str, Any]:
    payload.update(changes, updated_at=time.time())
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return payload


def _wait_for_pid(pid: int, timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return True
        time.sleep(0.25)
    return False


def _copy_path(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _remove_managed(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _backup(source_root: Path, backup_root: Path) -> None:
    backup_root.mkdir(parents=True, exist_ok=False)
    presence: dict[str, bool] = {}
    for relative in (*MANAGED_DIRECTORIES, *MANAGED_FILES):
        source = source_root / Path(relative)
        presence[relative] = source.exists()
        if source.exists():
            _copy_path(source, backup_root / Path(relative))
    (backup_root / "presence.json").write_text(
        json.dumps(presence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _install(source_root: Path, candidate_root: Path) -> None:
    for relative in (*MANAGED_DIRECTORIES, *MANAGED_FILES):
        target = source_root / Path(relative)
        candidate = candidate_root / Path(relative)
        if not candidate.exists():
            raise RuntimeError(f"candidate managed path is missing: {relative}")
        _remove_managed(target)
        _copy_path(candidate, target)


def _restore(source_root: Path, backup_root: Path) -> None:
    presence = _read(backup_root / "presence.json")
    for relative in (*MANAGED_DIRECTORIES, *MANAGED_FILES):
        target = source_root / Path(relative)
        _remove_managed(target)
        if presence.get(relative):
            _copy_path(backup_root / Path(relative), target)


def _service_command(root: Path) -> list[str]:
    launcher = root / "launcher.bat"
    if os.name == "nt" and launcher.is_file():
        return ["cmd.exe", "/c", str(launcher)]
    return [sys.executable, str(root / "run.py")]


def _start_service(root: Path, transaction_id: str) -> subprocess.Popen[Any]:
    env = {
        **os.environ,
        "PERSONALITYRAG_SUPPRESS_BROWSER": "1",
        "PERSONALITYRAG_UPDATE_TRANSACTION": transaction_id,
    }
    kwargs: dict[str, Any] = {"cwd": str(root), "env": env, "close_fds": True}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    else:
        kwargs.update(
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    return subprocess.Popen(_service_command(root), **kwargs)


def _terminate_tree(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()


def _health_matches(urls: list[str], expected_version: str, timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        for url in urls:
            try:
                with urllib.request.urlopen(url.rstrip("/") + "/api/v1/health", timeout=2) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if payload.get("status") == "ok" and payload.get("version") == expected_version:
                    return True
            except (OSError, ValueError, urllib.error.URLError):
                pass
        time.sleep(1)
    return False


def apply_transaction(transaction_file: Path) -> int:
    payload = _read(transaction_file)
    source_root = Path(payload["source_root"]).resolve()
    candidate_root = Path(payload["candidate_root"]).resolve()
    backup_root = transaction_file.parent / "backup"
    old_version = str(payload["current_version"])
    target_version = str(payload["target_version"])
    transaction_id = str(payload["transaction_id"])
    old_pid = int(payload["service_pid"])
    urls = [str(url) for url in payload.get("health_urls") or []]
    _write(transaction_file, payload, status="running", stage="helper_started", helper_pid=os.getpid())
    if not _wait_for_pid(old_pid):
        _write(transaction_file, payload, status="failed", stage="waiting_for_shutdown", error="service did not stop")
        return 3
    target_process: subprocess.Popen[Any] | None = None
    try:
        _write(transaction_file, payload, stage="backing_up")
        _backup(source_root, backup_root)
        _write(transaction_file, payload, stage="replacing")
        _install(source_root, candidate_root)
        _write(transaction_file, payload, stage="starting_target")
        target_process = _start_service(source_root, transaction_id)
        _write(transaction_file, payload, stage="checking_target", target_pid=target_process.pid)
        if not _health_matches(urls, target_version):
            raise RuntimeError("target service failed the version health check")
        _write(transaction_file, payload, status="completed", stage="completed", completed_at=time.time())
        return 0
    except Exception as exc:
        _write(transaction_file, payload, status="rolling_back", stage="rolling_back", error=str(exc))
        if target_process is not None:
            _terminate_tree(target_process)
        try:
            _restore(source_root, backup_root)
            rollback = _start_service(source_root, transaction_id)
            if not _health_matches(urls, old_version):
                _write(
                    transaction_file,
                    payload,
                    status="recovery_required",
                    stage="rollback_health_failed",
                    rollback_pid=rollback.pid,
                )
                return 5
            _write(
                transaction_file,
                payload,
                status="rolled_back",
                stage="rolled_back",
                rollback_pid=rollback.pid,
                completed_at=time.time(),
            )
            return 4
        except Exception as rollback_exc:
            _write(
                transaction_file,
                payload,
                status="recovery_required",
                stage="rollback_failed",
                rollback_error=str(rollback_exc),
            )
            return 6


def recover_transactions(state_root: Path, active_transaction: str = "") -> int:
    transactions_root = state_root.resolve() / "data" / "update" / "transactions"
    if not transactions_root.is_dir():
        return 0
    recovered = 0
    for transaction_file in sorted(transactions_root.glob("*/transaction.json")):
        try:
            payload = _read(transaction_file)
            if payload.get("transaction_id") == active_transaction:
                continue
            if payload.get("status") not in {"running", "rolling_back"} and payload.get("stage") not in INCOMPLETE_STAGES:
                continue
            backup_root = transaction_file.parent / "backup"
            source_root = Path(payload["source_root"]).resolve()
            if not (backup_root / "presence.json").is_file():
                if payload.get("stage") in {"prepared", "helper_started", "backing_up"}:
                    _write(transaction_file, payload, status="failed", stage="recovery_not_required", error="update stopped before code replacement")
                    continue
                _write(transaction_file, payload, status="recovery_required", stage="recovery_backup_missing", error="recovery backup is missing")
                raise RuntimeError("an interrupted update has no valid recovery backup")
            _restore(source_root, backup_root)
            _write(transaction_file, payload, status="rolled_back", stage="startup_recovered", completed_at=time.time())
            recovered += 1
        except Exception as exc:
            try:
                payload = _read(transaction_file)
                _write(transaction_file, payload, status="recovery_required", stage="startup_recovery_failed", error=str(exc))
            except Exception:
                pass
    return recovered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("transaction_file")
    recover_parser = subparsers.add_parser("recover")
    recover_parser.add_argument("state_root")
    recover_parser.add_argument("--active-transaction", default="")
    args = parser.parse_args(argv)
    if args.command == "apply":
        return apply_transaction(Path(args.transaction_file).resolve())
    recover_transactions(Path(args.state_root), args.active_transaction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
