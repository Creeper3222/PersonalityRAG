from __future__ import annotations

import json
import stat
import zipfile
from pathlib import Path

import pytest

from personalityrag import docker_update_helper
from personalityrag.docker_engine import clone_container_body, state_bind_source
from personalityrag.update_manifest import (
    MANIFEST_NAME,
    RELEASE_DIRECTORIES,
    RELEASE_FILES,
    UpdatePackageError,
    compare_tags,
    inspect_and_extract_zip,
    sha256_file,
    validate_manifest,
    write_manifest,
)
from personalityrag.updates import UpdateService
from personalityrag.version import PLATFORM_NAME, RELEASE_ROOT_NAME, release_asset_name


DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64
DIGEST_C = "sha256:" + "c" * 64
SOURCE_COMMIT = "d" * 40


def _candidate(tmp_path: Path, *, version: str = "0.1.0") -> Path:
    root = tmp_path / RELEASE_ROOT_NAME
    for directory in RELEASE_DIRECTORIES:
        path = root / directory
        path.mkdir(parents=True)
        (path / "module.txt").write_text(directory, encoding="utf-8")
    for relative in RELEASE_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    (root / "personalityrag" / "version.py").write_text(f'VERSION = "{version}"\n', encoding="utf-8")
    write_manifest(
        root,
        version=version,
        tag_name=f"v{version}",
        source_commit=SOURCE_COMMIT,
        index_digest=DIGEST_A,
        amd64_digest=DIGEST_B,
        arm64_digest=DIGEST_C,
    )
    return root


def _zip(root: Path, target: Path) -> Path:
    with zipfile.ZipFile(target, "w") as archive:
        for path in root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(root.parent).as_posix())
    return target


def test_semver_comparison_includes_prereleases() -> None:
    assert compare_tags("v0.1.1", "v0.1.0") > 0
    assert compare_tags("v0.1.1-rc.2", "v0.1.1-rc.1") > 0
    assert compare_tags("v0.1.1", "v0.1.1-rc.2") > 0
    assert compare_tags("v0.1.0", "v0.1.0") == 0
    with pytest.raises(UpdatePackageError):
        compare_tags("latest", "v0.1.0")


def test_runtime_version_and_linux_asset_are_centralized() -> None:
    root = Path(__file__).resolve().parents[1]
    from personalityrag.version import VERSION

    assert PLATFORM_NAME == "linux-docker"
    assert release_asset_name() == "PersonalityRAG-linux-v0.1.2.zip"
    assert "v0.1.0" not in (root / "run.py").read_text(encoding="utf-8")
    assert 'id="sidebar-version"' in (root / "static" / "index.html").read_text(encoding="utf-8")
    assert f'version = "{VERSION}"' in (root / "pyproject.toml").read_text(encoding="utf-8")


def test_manifest_validates_files_and_docker_contract(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    validate_manifest(manifest, expected_tag="v0.1.0", candidate_root=root)
    assert manifest["root_directory"] == "PersonalityRAG-linux"
    assert manifest["docker"]["platforms"]["linux/amd64"] == DIGEST_B
    (root / "static" / "module.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(UpdatePackageError, match="(size|hash)"):
        validate_manifest(manifest, expected_tag="v0.1.0", candidate_root=root)


def test_manifest_rejects_wrong_platform_or_docker_digest(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["platform"] = "windows"
    with pytest.raises(UpdatePackageError, match="platform"):
        validate_manifest(manifest, expected_tag="v0.1.0")
    manifest["platform"] = PLATFORM_NAME
    manifest["docker"]["platforms"]["linux/arm64"] = "latest"
    with pytest.raises(UpdatePackageError, match="digest"):
        validate_manifest(manifest, expected_tag="v0.1.0")


def test_manifest_blocks_incompatible_persistent_schema(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["persistent_compatibility"]["livingmemory_database"] = {"minimum": 9, "maximum": 9}
    with pytest.raises(UpdatePackageError, match="incompatible with livingmemory_database"):
        validate_manifest(manifest, expected_tag="v0.1.0")


def test_manifest_rejects_internal_runtime_version_mismatch(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    version_file = root / "personalityrag" / "version.py"
    version_file.write_text('VERSION = "9.9.9"\n', encoding="utf-8")
    for entry in manifest["files"]:
        if entry["path"] == "personalityrag/version.py":
            entry["size"] = version_file.stat().st_size
            entry["sha256"] = sha256_file(version_file)
    with pytest.raises(UpdatePackageError, match="internal runtime version"):
        validate_manifest(manifest, expected_tag="v0.1.0", candidate_root=root)


def test_zip_security_and_round_trip(tmp_path: Path) -> None:
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("PersonalityRAG-linux/../config/config.json", "secret")
    with pytest.raises(UpdatePackageError, match="unsafe path"):
        inspect_and_extract_zip(bad, tmp_path / "bad-out", expected_tag="v0.1.0")

    symlink = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("PersonalityRAG-linux/personalityrag/link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(symlink, "w") as archive:
        archive.writestr(info, "target")
    with pytest.raises(UpdatePackageError, match="symbolic link"):
        inspect_and_extract_zip(symlink, tmp_path / "link-out", expected_tag="v0.1.0")

    archive = _zip(_candidate(tmp_path / "build"), tmp_path / release_asset_name())
    root, manifest = inspect_and_extract_zip(archive, tmp_path / "extract", expected_tag="v0.1.0")
    assert root.name == RELEASE_ROOT_NAME
    assert manifest["version"] == "0.1.0"


def test_release_filter_requires_exact_linux_asset() -> None:
    asset_name = "PersonalityRAG-linux-v0.1.1.zip"
    base = {
        "tag_name": "v0.1.1",
        "draft": False,
        "prerelease": True,
        "assets": [{
            "name": asset_name,
            "browser_download_url": f"https://github.com/Creeper3222/PersonalityRAG/releases/download/v0.1.1/{asset_name}",
            "size": 12,
            "digest": DIGEST_A,
        }],
    }
    assert UpdateService._normalize_release(base)["tag_name"] == "v0.1.1"
    windows = {**base, "assets": [{**base["assets"][0], "name": "PersonalityRAG-v0.1.1.zip"}]}
    assert UpdateService._normalize_release(windows) is None
    assert UpdateService._normalize_release({**base, "draft": True}) is None
    poisoned = {**base, "assets": [{**base["assets"][0], "browser_download_url": "https://example.test/file.zip"}]}
    assert UpdateService._normalize_release(poisoned) is None


@pytest.mark.asyncio
async def test_release_check_keeps_cache_when_network_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = UpdateService(tmp_path / "source", tmp_path / "state")
    service._write_cache({"checked_at": 1.0, "stale": False, "error": "", "releases": []})
    monkeypatch.setattr(service, "_request_json", lambda _: (_ for _ in ()).throw(OSError("offline")))
    monkeypatch.setattr(service, "_fallback_release_feed", lambda: (_ for _ in ()).throw(OSError("offline")))
    payload = await service.releases(refresh=True)
    assert payload["stale"] is True
    assert payload["releases"] == []


def test_transaction_public_payload_does_not_leak_paths(tmp_path: Path) -> None:
    service = UpdateService(tmp_path / "source", tmp_path / "state")
    public = service._public_transaction({
        "transaction_id": "abc", "status": "prepared", "source_root": "/secret", "target_image_ref": "secret"
    })
    assert public == {"transaction_id": "abc", "status": "prepared"}


def _container_fixture() -> dict:
    return {
        "Id": "1" * 64,
        "Image": "sha256:" + "1" * 64,
        "Name": "/PersonalityRAG",
        "Config": {
            "Env": ["PERSONALITYRAG_STATE_ROOT=/app/state"],
            "Labels": {"com.docker.compose.service": "personalityrag"},
            "ExposedPorts": {"8765/tcp": {}},
            "Healthcheck": {"Test": ["CMD", "true"]},
            "WorkingDir": "/app",
            "Entrypoint": ["/usr/local/bin/personalityrag-entrypoint.sh"],
            "Cmd": [],
        },
        "HostConfig": {
            "Binds": ["/host/state:/app/state:rw", "/var/run/docker.sock:/var/run/docker.sock:rw"],
            "PortBindings": {"8765/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8765"}]},
            "RestartPolicy": {"Name": "unless-stopped"},
            "NetworkMode": "personalityrag_default",
        },
        "Mounts": [{"Destination": "/app/state", "Source": "/host/state", "RW": True}],
    }


def test_container_clone_preserves_runtime_contract() -> None:
    current = _container_fixture()
    assert state_bind_source(current) == "/host/state"
    body = clone_container_body(current, "repo@" + DIGEST_B)
    assert body["Image"] == "repo@" + DIGEST_B
    assert body["HostConfig"]["PortBindings"] == current["HostConfig"]["PortBindings"]
    assert body["HostConfig"]["Binds"] == current["HostConfig"]["Binds"]
    assert "Id" not in body


class _FakeEngine:
    instances: list["_FakeEngine"] = []

    def __init__(self) -> None:
        self.old = _container_fixture()
        self.calls: list[tuple] = []
        self.created_bodies: list[dict] = []
        self.fail_target = False
        self.__class__.instances.append(self)

    def inspect_container(self, container: str) -> dict:
        if container == "target":
            return {"Image": "sha256:" + "2" * 64}
        return self.old

    def inspect_image(self, image: str) -> dict:
        return {
            "Config": {
                "Labels": {
                    "org.opencontainers.image.version": "v0.1.1",
                    "org.opencontainers.image.revision": SOURCE_COMMIT,
                    "io.personalityrag.platform": "linux-docker",
                }
            }
        }

    def stop(self, container: str, timeout: int = 120) -> None:
        self.calls.append(("stop", container))

    def rename(self, container: str, name: str) -> None:
        self.calls.append(("rename", container, name))

    def create_container(self, name: str, body: dict) -> str:
        self.calls.append(("create", name))
        self.created_bodies.append(body)
        return "target"

    def start(self, container: str) -> None:
        self.calls.append(("start", container))

    def wait_healthy(self, container: str, *, expected_version: str, timeout: float = 180) -> bool:
        return not self.fail_target or container != "target"

    def remove_container(self, container: str, *, force: bool = False) -> None:
        self.calls.append(("remove", container))

    def remove_image(self, image: str) -> None:
        self.calls.append(("remove_image", image))


def test_docker_helper_completes_and_records_transaction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docker_update_helper, "DockerEngineClient", _FakeEngine)
    transaction = tmp_path / "transaction.json"
    transaction.write_text(json.dumps({
        "transaction_id": "abcdef012345", "source_container_id": "old", "source_container_name": "PersonalityRAG",
        "target_image_ref": "repo@" + DIGEST_B, "target_tag": "v0.1.1", "current_tag": "v0.1.0",
    }), encoding="utf-8")
    assert docker_update_helper.apply(transaction) == 0
    created = _FakeEngine.instances[-1].created_bodies[0]
    assert created["Labels"]["com.docker.compose.service"] == "personalityrag"
    assert created["Labels"]["org.opencontainers.image.version"] == "v0.1.1"
    assert created["Labels"]["org.opencontainers.image.revision"] == SOURCE_COMMIT
    assert created["Labels"]["io.personalityrag.platform"] == "linux-docker"
    assert json.loads(transaction.read_text(encoding="utf-8"))["status"] == "completed"
