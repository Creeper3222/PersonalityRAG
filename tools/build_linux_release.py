from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))

from personalityrag.update_manifest import (  # noqa: E402
    MANIFEST_NAME,
    RELEASE_DIRECTORIES,
    RELEASE_FILES,
    inspect_and_extract_zip,
    write_manifest,
)
from personalityrag.version import RELEASE_ROOT_NAME, release_asset_name  # noqa: E402

RELEASE_COPY_IGNORE = shutil.ignore_patterns(
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".pytest_cache",
    ".ruff_cache",
)


def build(args: argparse.Namespace) -> Path:
    source = args.source.resolve()
    output = args.output.resolve()
    if output.name != release_asset_name(args.tag):
        raise ValueError(f"output filename must be {release_asset_name(args.tag)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="personalityrag-linux-release-") as temporary:
        staging = Path(temporary) / RELEASE_ROOT_NAME
        staging.mkdir()
        for directory in RELEASE_DIRECTORIES:
            shutil.copytree(
                source / directory,
                staging / directory,
                ignore=RELEASE_COPY_IGNORE,
            )
        for relative in RELEASE_FILES:
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, target)
        write_manifest(
            staging,
            version=args.tag.removeprefix("v"),
            tag_name=args.tag,
            source_commit=args.source_commit,
            index_digest=args.index_digest,
            amd64_digest=args.amd64_digest,
            arm64_digest=args.arm64_digest,
        )
        if not (staging / MANIFEST_NAME).is_file():
            raise RuntimeError("update manifest was not generated")
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(staging.rglob("*"), key=lambda item: item.as_posix().lower()):
                if path.is_file():
                    archive.write(path, Path(RELEASE_ROOT_NAME) / path.relative_to(staging))
        verify_root = Path(temporary) / "verify"
        inspect_and_extract_zip(output, verify_root, expected_tag=args.tag)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", default="v0.1.2")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--index-digest", required=True)
    parser.add_argument("--amd64-digest", required=True)
    parser.add_argument("--arm64-digest", required=True)
    output = build(parser.parse_args())
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
