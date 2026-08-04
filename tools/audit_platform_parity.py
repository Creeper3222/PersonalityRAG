from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_ALLOWED = {
    ".dockerignore",
    ".env.example",
    ".gitattributes",
    ".gitignore",
    "Dockerfile",
    "DELIVERY.md",
    "IMPLEMENTATION_AUDIT.md",
    "README.md",
    "docker-compose.local.yml",
    "docker-compose.yml",
    "docker/entrypoint.sh",
    "docs/REMOTE_ADAPTER_DEPLOYMENT.md",
    "docs/examples/Caddyfile",
    "docs/examples/nginx.personalityrag.conf",
    "docs/operations/compatibility-inventory.md",
    "docs/operations/livingmemory-2.5-alignment.md",
    "docs/operations/livingmemory-2.5.3-alignment.md",
    "docs/operations/performance-baseline-v0.1.1.md",
    "launcher.bat",
    "personalityrag/application_context.py",
    "personalityrag/backup_migration.py",
    "personalityrag/config.py",
    "personalityrag/compat.py",
    "personalityrag/docker_engine.py",
    "personalityrag/docker_update_helper.py",
    "personalityrag/file_manager.py",
    "personalityrag/logger.py",
    "personalityrag/routes/updates.py",
    "personalityrag/update_manifest.py",
    "personalityrag/updates.py",
    "personalityrag/version.py",
    "personalityrag/routes/auth_settings.py",
    "requirements-runtime.lock",
    "static/locales/en.js",
    "static/locales/ru.js",
    "static/locales/zh.js",
    "static/app.js",
    "static/modules/api.js",
    "static/modules/settings.js",
    "static/styles.css",
    "tests/test_docker_deployment.py",
    "tests/test_core.py",
    "tests/test_operational_contract.py",
    "tests/test_text_integrity.py",
    "tests/test_runtime_bootstrap.py",
    "tests/test_updates.py",
    "tests/test_webui_modules.py",
    "tools/audit_platform_parity.py",
    "tools/build_linux_release.py",
    "tools/build_windows_release.py",
    "tools/runtime_bootstrap.py",
    "tools/update_helper.py",
}

IGNORED_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".test-runtime",
    ".venv",
    "__pycache__",
    "config",
    "data",
    "reports",
    "test-results",
}

IGNORED_ROOT_PREFIXES = (
    ".codex-",
    ".venv",
    "lm_recall_compare_",
)


def source_files(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if (
            any(part in IGNORED_PARTS for part in relative.parts)
            or relative.parts[0].startswith(IGNORED_ROOT_PREFIXES)
        ):
            continue
        result[relative.as_posix()] = path
    return result


def normalized(path: Path) -> bytes:
    payload = path.read_bytes()
    if b"\0" in payload:
        return payload
    return payload.replace(b"\r\n", b"\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--linux", type=Path, required=True)
    parser.add_argument("--allow", action="append", default=[])
    args = parser.parse_args()

    windows = source_files(args.windows.resolve())
    linux = source_files(args.linux.resolve())
    allowed = DEFAULT_ALLOWED | {item.replace("\\", "/") for item in args.allow}
    unexpected: list[str] = []
    for relative in sorted(set(windows) | set(linux)):
        if relative in allowed:
            continue
        if relative not in windows:
            unexpected.append(f"linux-only: {relative}")
        elif relative not in linux:
            unexpected.append(f"missing-in-linux: {relative}")
        elif normalized(windows[relative]) != normalized(linux[relative]):
            unexpected.append(f"content-diff: {relative}")

    if unexpected:
        print("Unexpected platform differences:")
        for item in unexpected:
            print(f"- {item}")
        return 1
    print(f"Parity audit passed; documented differences={len(allowed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
