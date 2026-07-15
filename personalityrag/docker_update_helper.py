from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .docker_engine import DockerEngineClient, clone_container_body


IMAGE_IDENTITY_LABELS = (
    "org.opencontainers.image.title",
    "org.opencontainers.image.version",
    "org.opencontainers.image.revision",
    "org.opencontainers.image.source",
    "io.personalityrag.platform",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any], **changes: Any) -> None:
    payload.update(changes, updated_at=time.time())
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _target_container_body(
    engine: DockerEngineClient, old: dict[str, Any], target_ref: str
) -> dict[str, Any]:
    body = clone_container_body(old, target_ref)
    image_labels = (
        (engine.inspect_image(target_ref).get("Config") or {}).get("Labels") or {}
    )
    labels = dict(body.get("Labels") or {})
    for key in IMAGE_IDENTITY_LABELS:
        if key in image_labels:
            labels[key] = image_labels[key]
    body["Labels"] = labels
    return body


def apply(transaction_file: Path) -> int:
    payload = _read(transaction_file)
    engine = DockerEngineClient()
    old_id = str(payload["source_container_id"])
    target_ref = str(payload["target_image_ref"])
    original_name = str(payload["source_container_name"])
    rollback_name = f"{original_name}-rollback-{payload['transaction_id'][:8]}"
    target_id = ""
    old = engine.inspect_container(old_id)
    old_image = str(old["Image"])
    old_stopped = False
    old_renamed = False
    try:
        _write(transaction_file, payload, status="running", stage="stopping_container", helper_container_id=os.environ.get("HOSTNAME", ""))
        engine.stop(old_id, timeout=120)
        old_stopped = True
        engine.rename(old_id, rollback_name)
        old_renamed = True
        _write(transaction_file, payload, stage="creating_target")
        target_id = engine.create_container(
            original_name,
            _target_container_body(engine, old, target_ref),
        )
        engine.start(target_id)
        _write(transaction_file, payload, stage="checking_target", target_container_id=target_id)
        if not engine.wait_healthy(target_id, expected_version=str(payload["target_tag"]), timeout=180):
            raise RuntimeError("target container did not become healthy with the expected version")
        engine.remove_container(old_id)
        if old_image != str(engine.inspect_container(target_id)["Image"]):
            try:
                engine.remove_image(old_image)
            except Exception:
                pass
        _write(transaction_file, payload, status="completed", stage="completed", completed_at=time.time())
        return 0
    except Exception as exc:
        _write(transaction_file, payload, status="rolling_back", stage="rolling_back", error=str(exc))
        try:
            if target_id:
                engine.remove_container(target_id, force=True)
            if old_renamed:
                engine.rename(old_id, original_name)
            if old_stopped:
                engine.start(old_id)
            if not engine.wait_healthy(old_id, expected_version=str(payload["current_tag"]), timeout=180):
                raise RuntimeError("rollback container did not become healthy")
            _write(transaction_file, payload, status="rolled_back", stage="rolled_back", completed_at=time.time())
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


def main(argv: list[str] | None = None) -> int:
    values = list(argv or sys.argv[1:])
    if len(values) != 1:
        raise SystemExit("usage: python -m personalityrag.docker_update_helper TRANSACTION")
    return apply(Path(values[0]).resolve())


if __name__ == "__main__":
    raise SystemExit(main())
