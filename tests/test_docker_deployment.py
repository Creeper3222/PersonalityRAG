from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import BackgroundTasks, HTTPException

from personalityrag.backup_migration import export_prag_package, import_prag_package
from personalityrag.application_context import ApplicationContext, set_default_context
from personalityrag.config import (
    DOCKER_ACCESS_PORT,
    DOCKER_HOST,
    DOCKER_WEBUI_PORT,
    AppConfig,
    ProviderConfig,
    load_config,
)
from personalityrag.libraries import LibraryManager
from personalityrag.routes import auth_settings
from personalityrag.schemas import SettingsUpdate


def test_docker_load_config_enforces_network_and_initial_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PERSONALITYRAG_DEPLOYMENT", "docker")
    monkeypatch.setenv(
        "PERSONALITYRAG_DEFAULT_EMBEDDING_API_BASE",
        "http://host.docker.internal:9001/v1",
    )
    config = load_config(tmp_path / "config" / "config.json")

    assert config.host == DOCKER_HOST
    assert config.port == DOCKER_WEBUI_PORT
    assert config.access_port == DOCKER_ACCESS_PORT
    assert config.provider.api_base == "http://host.docker.internal:9001/v1"


def test_docker_existing_provider_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config" / "config.json"
    monkeypatch.delenv("PERSONALITYRAG_DEPLOYMENT", raising=False)
    original = AppConfig(
        host="127.0.0.1",
        port=9000,
        access_port=9001,
        provider=ProviderConfig(api_base="http://embedding:8001/v1"),
    )
    from personalityrag.config import save_config

    save_config(config_path, original)
    monkeypatch.setenv("PERSONALITYRAG_DEPLOYMENT", "docker")
    loaded = load_config(config_path)

    assert loaded.host == DOCKER_HOST
    assert loaded.port == DOCKER_WEBUI_PORT
    assert loaded.access_port == DOCKER_ACCESS_PORT
    assert loaded.provider.api_base == "http://embedding:8001/v1"


@pytest.mark.asyncio
async def test_docker_settings_reject_managed_port_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONALITYRAG_DEPLOYMENT", "docker")
    set_default_context(
        ApplicationContext.create(state_root=tmp_path, configure_logs=False)
    )
    with pytest.raises(HTTPException) as caught:
        await auth_settings.update_settings(SettingsUpdate(port=9876))

    assert caught.value.status_code == 409
    assert caught.value.detail["code"] == "docker_managed_settings"
    assert caught.value.detail["fields"] == ["port"]


@pytest.mark.asyncio
async def test_docker_restart_uses_container_strategy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONALITYRAG_DEPLOYMENT", "docker")
    set_default_context(
        ApplicationContext.create(state_root=tmp_path, configure_logs=False)
    )

    def unexpected_helper() -> None:
        raise AssertionError("Docker restart must not launch a child helper")

    monkeypatch.setattr(auth_settings, "_launch_restart_helper", unexpected_helper)
    context = auth_settings.current_context()
    context.restart_in_progress = False
    payload = await auth_settings.restart_service(BackgroundTasks())
    context.restart_in_progress = False

    assert payload["restart_strategy"] == "container"
    assert payload["deployment_mode"] == "docker"
    assert payload["managed_settings"] == ["port", "access_port"]


@pytest.mark.asyncio
async def test_windows_package_import_preserves_docker_network(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "windows"
    source_config = AppConfig(
        host="127.0.0.1",
        port=19065,
        access_port=19066,
        api_key="prag_windows",
    )
    source_manager = LibraryManager(source_root, source_config)
    await source_manager.initialize()
    package = tmp_path / "windows.prag"
    try:
        await export_prag_package(
            root=source_root,
            config=source_config,
            manager=source_manager,
            target=package,
            password="migration-password",
            include_libraries=False,
            include_providers=False,
        )
    finally:
        await source_manager.close()

    target_root = tmp_path / "docker"
    target_config = AppConfig(
        host=DOCKER_HOST,
        port=DOCKER_WEBUI_PORT,
        access_port=DOCKER_ACCESS_PORT,
        api_key="prag_docker",
    )
    target_manager = LibraryManager(target_root, target_config)
    await target_manager.initialize()
    try:
        next_config, result = await import_prag_package(
            root=target_root,
            config_path=target_root / "config" / "config.json",
            config=target_config,
            manager=target_manager,
            package_path=package,
            password="migration-password",
            preserve_managed_network=True,
        )
    finally:
        await target_manager.close()

    assert next_config.host == DOCKER_HOST
    assert next_config.port == DOCKER_WEBUI_PORT
    assert next_config.access_port == DOCKER_ACCESS_PORT
    assert next_config.api_key == "prag_windows"
    assert result["ignored_config_fields"] == ["host", "port", "access_port"]
