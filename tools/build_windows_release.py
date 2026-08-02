from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the minimal PersonalityRAG Windows update artifact")
    parser.add_argument("--source", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    version = args.tag.removeprefix("v")
    expected_name = f"PersonalityRAG-{args.tag}.zip"
    if output.name != expected_name:
        raise SystemExit(f"output filename must be {expected_name}")

    import sys

    sys.path.insert(0, str(source))
    from personalityrag.update_manifest import (  # noqa: PLC0415
        MANAGED_DIRECTORIES,
        MANAGED_FILES,
        MANIFEST_NAME,
        PRODUCT_NAME,
        audit_release_source_contract,
        inspect_and_extract_zip,
        write_manifest,
    )

    tracked_files = audit_release_source_contract(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="personalityrag-windows-release-") as temporary:
        staging = Path(temporary) / PRODUCT_NAME
        staging.mkdir()
        for relative in MANAGED_DIRECTORIES:
            shutil.copytree(
                source / relative,
                staging / relative,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache"),
            )
        for relative in MANAGED_FILES:
            target = staging / Path(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / Path(relative), target)
        write_manifest(staging, version=version, tag_name=args.tag)

        temporary_zip = output.with_suffix(".tmp")
        with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(staging.rglob("*"), key=lambda item: item.as_posix().lower()):
                if path.is_file():
                    archive.write(path, path.relative_to(staging.parent).as_posix())
        temporary_zip.replace(output)

        verify = Path(temporary) / "verify"
        _, manifest = inspect_and_extract_zip(output, verify, expected_tag=args.tag)
        print(json.dumps({
            "output": str(output),
            "tag": args.tag,
            "files": len(manifest["files"]),
            "tracked_source_files": len(tracked_files),
            "manifest": MANIFEST_NAME,
        }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
