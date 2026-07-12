from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from .instance_lock import InstanceLock, SingleInstanceError


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _start_detached_process(command: list[str], cwd: Path) -> None:
    kwargs = {
        "cwd": str(cwd),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
        "env": {
            **os.environ,
            "PERSONALITYRAG_SUPPRESS_BROWSER": "1",
        },
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(command, **kwargs)


def _restart_service_command(root: Path) -> list[str]:
    launcher = root / "launcher.bat"
    local_python = root / ".venv" / "Scripts" / "python.exe"
    if os.name == "nt" and launcher.exists() and local_python.exists():
        return ["cmd.exe", "/c", str(launcher)]
    return [sys.executable, str(root / "run.py")]


def _start_service_process(command: list[str], cwd: Path) -> None:
    env = {
        **os.environ,
        "PERSONALITYRAG_SUPPRESS_BROWSER": "1",
    }
    if os.name == "nt":
        subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            close_fds=True,
            creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
        )
        return
    subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )


def _instance_lock_path(root: Path) -> Path:
    return root / "data" / "logs" / "personalityrag.instance.lock"


def _instance_lock_available(root: Path) -> bool:
    probe = InstanceLock(_instance_lock_path(root))
    try:
        probe.acquire()
    except SingleInstanceError:
        return False
    finally:
        probe.release()
    return True


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv
    if len(argv) != 3:
        return 2
    int(argv[1])
    root = Path(argv[2]).resolve()
    deadline = time.time() + 90
    while time.time() < deadline:
        if _instance_lock_available(root):
            break
        time.sleep(0.25)
    if not _instance_lock_available(root):
        return 3
    _start_service_process(_restart_service_command(root), root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
