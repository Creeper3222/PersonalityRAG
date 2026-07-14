from __future__ import annotations

import http.client
import json
import os
import socket
import time
import urllib.parse
from pathlib import Path
from typing import Any

from .update_manifest import UpdatePackageError


DOCKER_SOCKET = Path(os.environ.get("PERSONALITYRAG_DOCKER_SOCKET", "/var/run/docker.sock"))
DOCKER_API_VERSION = "v1.45"


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: Path, timeout: float = 60.0):
        super().__init__("localhost", timeout=timeout)
        self.socket_path = str(socket_path)

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.socket_path)


class DockerEngineClient:
    def __init__(self, socket_path: Path = DOCKER_SOCKET):
        self.socket_path = socket_path

    @property
    def available(self) -> bool:
        return self.socket_path.is_socket()

    def request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        expected: tuple[int, ...] = (200, 201, 204),
        timeout: float = 120.0,
    ) -> Any:
        if not self.available:
            raise UpdatePackageError("Docker Engine socket is unavailable")
        connection = _UnixHTTPConnection(self.socket_path, timeout=timeout)
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        try:
            connection.request(method, f"/{DOCKER_API_VERSION}{path}", body=payload, headers=headers)
            response = connection.getresponse()
            content = response.read()
        except OSError as exc:
            raise UpdatePackageError(f"Docker Engine request failed: {exc}") from exc
        finally:
            connection.close()
        if response.status not in expected:
            message = content.decode("utf-8", "replace")[:1000]
            raise UpdatePackageError(f"Docker Engine returned {response.status}: {message}")
        if not content:
            return None
        text = content.decode("utf-8", "replace")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def version(self) -> dict[str, Any]:
        return dict(self.request("GET", "/version"))

    def inspect_container(self, container: str) -> dict[str, Any]:
        identifier = urllib.parse.quote(container, safe="")
        return dict(self.request("GET", f"/containers/{identifier}/json"))

    def inspect_image(self, image: str) -> dict[str, Any]:
        identifier = urllib.parse.quote(image, safe="")
        return dict(self.request("GET", f"/images/{identifier}/json"))

    def pull(self, repository: str, digest: str) -> None:
        query = urllib.parse.urlencode({"fromImage": repository, "tag": digest})
        self.request("POST", f"/images/create?{query}", expected=(200,), timeout=1800)

    def create_container(self, name: str, body: dict[str, Any]) -> str:
        query = urllib.parse.urlencode({"name": name})
        result = self.request("POST", f"/containers/create?{query}", body=body, expected=(201,))
        return str(result["Id"])

    def start(self, container: str) -> None:
        identifier = urllib.parse.quote(container, safe="")
        self.request("POST", f"/containers/{identifier}/start", expected=(204, 304))

    def stop(self, container: str, timeout: int = 120) -> None:
        identifier = urllib.parse.quote(container, safe="")
        self.request("POST", f"/containers/{identifier}/stop?t={timeout}", expected=(204, 304))

    def rename(self, container: str, name: str) -> None:
        identifier = urllib.parse.quote(container, safe="")
        query = urllib.parse.urlencode({"name": name})
        self.request("POST", f"/containers/{identifier}/rename?{query}", expected=(204,))

    def remove_container(self, container: str, *, force: bool = False) -> None:
        identifier = urllib.parse.quote(container, safe="")
        query = urllib.parse.urlencode({"force": str(force).lower(), "v": "false"})
        self.request("DELETE", f"/containers/{identifier}?{query}", expected=(204, 404))

    def remove_image(self, image: str) -> None:
        identifier = urllib.parse.quote(image, safe="")
        self.request("DELETE", f"/images/{identifier}?force=false&noprune=false", expected=(200, 404))

    def wait_healthy(self, container: str, *, expected_version: str, timeout: float = 180.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                payload = self.inspect_container(container)
            except UpdatePackageError:
                time.sleep(1)
                continue
            state = payload.get("State") or {}
            health = state.get("Health") or {}
            labels = (payload.get("Config") or {}).get("Labels") or {}
            if state.get("Running") and health.get("Status") == "healthy":
                return labels.get("org.opencontainers.image.version") == expected_version
            if state.get("Status") in {"dead", "exited"}:
                return False
            time.sleep(1)
        return False


def current_container_id() -> str:
    value = os.environ.get("PERSONALITYRAG_CONTAINER_ID") or os.environ.get("HOSTNAME") or ""
    if not value or len(value) < 12:
        raise UpdatePackageError("current Docker container identity is unavailable")
    return value


def state_bind_source(container: dict[str, Any]) -> str:
    for mount in container.get("Mounts") or []:
        if mount.get("Destination") == "/app/state" and mount.get("RW"):
            source = str(mount.get("Source") or "")
            if source:
                return source
    raise UpdatePackageError("/app/state is not a writable persistent mount")


def clone_container_body(container: dict[str, Any], target_image: str) -> dict[str, Any]:
    config = container.get("Config") or {}
    host = container.get("HostConfig") or {}
    body: dict[str, Any] = {
        "Image": target_image,
        "Env": list(config.get("Env") or []),
        "Labels": dict(config.get("Labels") or {}),
        "ExposedPorts": config.get("ExposedPorts") or {},
        "Healthcheck": config.get("Healthcheck"),
        "WorkingDir": config.get("WorkingDir") or "",
        "Entrypoint": config.get("Entrypoint"),
        "Cmd": config.get("Cmd"),
        "User": config.get("User") or "",
        "HostConfig": {
            "Binds": list(host.get("Binds") or []),
            "PortBindings": host.get("PortBindings") or {},
            "RestartPolicy": host.get("RestartPolicy") or {"Name": "unless-stopped"},
            "ExtraHosts": list(host.get("ExtraHosts") or []),
            "NetworkMode": host.get("NetworkMode") or "default",
            "LogConfig": host.get("LogConfig") or {"Type": "json-file", "Config": {}},
        },
    }
    return {key: value for key, value in body.items() if value is not None}


def helper_container_body(container: dict[str, Any], transaction_file: str) -> dict[str, Any]:
    state_source = state_bind_source(container)
    socket_source = str(DOCKER_SOCKET)
    return {
        "Image": container["Image"],
        "Entrypoint": ["python", "-m", "personalityrag.docker_update_helper"],
        "Cmd": [transaction_file],
        "Env": [
            "PYTHONUNBUFFERED=1",
            "PERSONALITYRAG_STATE_ROOT=/app/state",
            f"PERSONALITYRAG_DOCKER_SOCKET={DOCKER_SOCKET}",
        ],
        "HostConfig": {
            "AutoRemove": True,
            "Binds": [f"{state_source}:/app/state:rw", f"{socket_source}:{socket_source}:rw"],
            "NetworkMode": "none",
            "RestartPolicy": {"Name": "no"},
        },
    }
