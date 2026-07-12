from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest
from fastapi import HTTPException, UploadFile
from httpx import ASGITransport, AsyncClient

from personalityrag.application import create_app
from personalityrag.application_context import ApplicationContext
from personalityrag.auth import AuthManager, hash_password
from personalityrag.config import AppConfig
from personalityrag.file_manager import FILE_PREVIEW_LIMIT_BYTES, FileManager


def _manager(source: Path, state: Path, password: str = "secret") -> FileManager:
    source.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    return FileManager(
        source,
        state,
        AuthManager(
            "fixture-api-key",
            "fixture-session",
            password_hash=hash_password(password),
        ),
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.asyncio
async def test_file_manager_roots_and_path_boundary(tmp_path: Path) -> None:
    source = tmp_path / "project"
    state = tmp_path / "state"
    manager = _manager(source, state)
    (source / "docs").mkdir()
    (source / "docs" / "note.txt").write_text("hello", encoding="utf-8")
    (source / ".git").mkdir()
    (source / ".git" / "config").write_text("secret", encoding="utf-8")

    payload = await manager.list_directory("project", "")
    assert [item["id"] for item in payload["roots"]] == ["project", "state"]
    assert [item["name"] for item in payload["items"]] == ["docs"]

    for invalid in ("../outside", "/etc/passwd", "C:/Windows/System32", ".git/config"):
        with pytest.raises(HTTPException) as raised:
            manager.resolve("project", invalid)
        assert raised.value.status_code in {400, 403, 404}

    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = source / "outside-link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pass
    else:
        listed = await manager.list_directory("project", "")
        assert "outside-link.txt" not in {item["name"] for item in listed["items"]}
        with pytest.raises(HTTPException, match="不能越过根目录"):
            manager.resolve("project", "outside-link.txt")


def test_file_manager_merges_overlapping_roots(tmp_path: Path) -> None:
    source = tmp_path / "project"
    state = source / "runtime"
    manager = _manager(source, state)
    assert manager.roots_payload() == [{"id": "project", "path": str(source.resolve())}]


@pytest.mark.asyncio
async def test_managed_files_are_readonly_but_regular_files_are_mutable(
    tmp_path: Path,
) -> None:
    source = tmp_path / "project"
    state = tmp_path / "state"
    manager = _manager(source, state)
    (source / "personalityrag").mkdir()
    source_file = source / "personalityrag" / "core.py"
    source_file.write_text("print('core')", encoding="utf-8")
    (source / "config").mkdir()
    source_config = source / "config" / "legacy.json"
    source_config.write_text('{"legacy":true}', encoding="utf-8")
    (state / "config").mkdir()
    config_file = state / "config" / "config.json"
    config_file.write_text('{"api_key":"secret"}', encoding="utf-8")
    source_hash = _digest(source_file)
    config_hash = _digest(config_file)

    for root, path in (
        ("project", "personalityrag/core.py"),
        ("project", "config/legacy.json"),
        ("state", "config/config.json"),
    ):
        with pytest.raises(HTTPException) as raised:
            await manager.delete_items(root, [path])
        assert raised.value.status_code == 403
    assert _digest(source_file) == source_hash
    assert _digest(config_file) == config_hash

    root_item = await manager.create_item("project", "root-note.txt", "file")
    assert root_item["item"]["path"] == "root-note.txt"
    (source / "docs").mkdir()
    created = await manager.create_item("project", "docs/note.txt", "file")
    assert created["item"]["can_edit"] is True
    await manager.write_text(
        "project",
        "docs/note.txt",
        "updated",
        client_ip="127.0.0.1",
        password="",
    )
    renamed = await manager.rename_item("project", "docs/note.txt", "renamed.txt")
    assert renamed["item"]["name"] == "renamed.txt"
    (source / "target").mkdir()
    moved = await manager.move_items("project", ["docs/renamed.txt"], "target")
    assert moved["moved"] == 1
    await manager.delete_items("project", ["target/renamed.txt"])
    assert not (source / "target" / "renamed.txt").exists()


@pytest.mark.asyncio
async def test_text_preview_encoding_and_size_rules(tmp_path: Path) -> None:
    source = tmp_path / "project"
    state = tmp_path / "state"
    manager = _manager(source, state)
    (source / "docs").mkdir()
    (source / "docs" / "bom.txt").write_bytes(b"\xef\xbb\xbfhello")
    (source / "docs" / "binary.txt").write_bytes(b"hello\x00world")
    (source / "docs" / "legacy.txt").write_bytes("中文".encode("gbk"))
    (source / "docs" / "large.txt").write_bytes(b"a" * (FILE_PREVIEW_LIMIT_BYTES + 1))

    preview = await manager.read_text(
        "project", "docs/bom.txt", client_ip="127.0.0.1", password=""
    )
    assert preview["content"] == "hello"
    assert preview["truncated"] is False

    for path in ("docs/binary.txt", "docs/legacy.txt"):
        with pytest.raises(HTTPException) as raised:
            await manager.read_text("project", path, client_ip="127.0.0.1", password="")
        assert raised.value.status_code == 415

    large = await manager.read_text(
        "project", "docs/large.txt", client_ip="127.0.0.1", password=""
    )
    assert large["truncated"] is True
    assert large["can_edit"] is False


@pytest.mark.asyncio
async def test_upload_conflicts_and_streamed_archive(tmp_path: Path) -> None:
    source = tmp_path / "project"
    state = tmp_path / "state"
    manager = _manager(source, state)
    (source / "uploads").mkdir()
    (source / "uploads" / "same.txt").write_text("first", encoding="utf-8")
    (source / "uploads" / ".pytest_cache").mkdir()
    (source / "uploads" / ".pytest_cache" / "hidden.txt").write_text(
        "hidden", encoding="utf-8"
    )
    upload = UploadFile(filename="same.txt", file=io.BytesIO(b"second"))

    result = await manager.upload_files("project", "uploads", [upload])
    assert result["uploaded"] == 1
    assert result["items"][0]["name"].startswith("same-")

    archive, name, temporary = await manager.prepare_download(
        "project",
        ["uploads"],
        client_ip="127.0.0.1",
        password="",
    )
    try:
        assert name == "uploads.zip"
        assert temporary is True
        with zipfile.ZipFile(archive) as handle:
            names = set(handle.namelist())
        assert "uploads/same.txt" in names
        assert not any(".pytest_cache" in item for item in names)
    finally:
        manager.cleanup_archive(archive)
    assert not archive.exists()


@pytest.mark.asyncio
async def test_file_routes_require_admin_and_reauthenticate_sensitive_reads(
    tmp_path: Path,
) -> None:
    source = tmp_path / "project"
    state = tmp_path / "state"
    (source / "static").mkdir(parents=True)
    (state / "config").mkdir(parents=True)
    config_file = state / "config" / "config.json"
    config_file.write_text('{"private":true}', encoding="utf-8")
    config = AppConfig(
        api_key="fixture-api-key",
        session_secret="fixture-session",
        webui_password_hash=hash_password("file-password"),
    )
    context = ApplicationContext.create(
        source_root=source,
        state_root=state,
        config=config,
        configure_logs=False,
    )
    application = create_app(context)
    headers = {"Authorization": "Bearer fixture-api-key"}
    async with AsyncClient(
        transport=ASGITransport(app=application),
        base_url="http://test:8765",
    ) as client:
        unauthenticated = await client.get("/api/v1/files")
        listed = await client.get("/api/v1/files?root=state", headers=headers)
        missing_password = await client.post(
            "/api/v1/files/read",
            headers=headers,
            json={"root": "state", "path": "config/config.json"},
        )
        valid = await client.post(
            "/api/v1/files/read",
            headers=headers,
            json={
                "root": "state",
                "path": "config/config.json",
                "password": "file-password",
            },
        )
        protected_delete = await client.post(
            "/api/v1/files/delete",
            headers=headers,
            json={"root": "state", "paths": ["config/config.json"]},
        )
        downloaded = await client.post(
            "/api/v1/files/download",
            headers=headers,
            json={
                "root": "state",
                "paths": ["config/config.json"],
                "password": "file-password",
            },
        )

    assert unauthenticated.status_code == 401
    assert listed.status_code == 200
    assert missing_password.status_code == 401
    assert valid.status_code == 200
    assert valid.json()["content"] == '{"private":true}'
    assert protected_delete.status_code == 403
    assert downloaded.status_code == 200
    assert downloaded.content == config_file.read_bytes()
    assert context.manager.runtimes == {}
