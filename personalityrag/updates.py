from __future__ import annotations

import asyncio
import hashlib
import html
import json
import os
import re
import secrets
import shutil
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .docker_engine import (
    DockerEngineClient,
    current_container_id,
    helper_container_body,
    state_bind_source,
)
from .logger import logger
from .update_manifest import (
    MAX_RELEASE_BYTES,
    UpdatePackageError,
    compare_tags,
    current_architecture,
    inspect_and_extract_zip,
    parse_tag,
)
from .version import DOCKER_REPOSITORY, TAG_NAME, VERSION, release_asset_name


GITHUB_RELEASES_URL = "https://api.github.com/repos/Creeper3222/PersonalityRAG/releases?per_page=100"
GITHUB_RELEASES_ATOM_URL = "https://github.com/Creeper3222/PersonalityRAG/releases.atom"
CACHE_SECONDS = 600
MANUAL_REFRESH_SECONDS = 60
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
TERMINAL_TRANSACTION_STATUSES = frozenset(
    {"completed", "failed", "rolled_back", "recovery_required"}
)


class UpdateService:
    def __init__(self, source_root: Path, state_root: Path):
        self.source_root = source_root.resolve()
        self.state_root = state_root.resolve()
        self.update_root = self.state_root / "data" / "update"
        self.transactions_root = self.update_root / "transactions"
        self.cache_path = self.update_root / "releases-cache.json"
        self._cache: dict[str, Any] | None = None
        self._lock = asyncio.Lock()
        self._last_manual_refresh = 0.0

    def _read_cache(self) -> dict[str, Any] | None:
        if self._cache is not None:
            return self._cache
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if isinstance(payload, dict) and isinstance(payload.get("releases"), list):
            self._cache = payload
            return payload
        return None

    def _write_cache(self, payload: dict[str, Any]) -> None:
        self.update_root.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.cache_path)
        self._cache = payload

    @staticmethod
    def _request_json(url: str) -> Any:
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": f"PersonalityRAG/{VERSION}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _request_text(url: str) -> str:
        request = urllib.request.Request(url, headers={"User-Agent": f"PersonalityRAG/{VERSION}"})
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.read().decode("utf-8", "replace")

    @classmethod
    def _fallback_release_feed(cls) -> list[dict[str, Any]]:
        """Use GitHub's public HTML/Atom endpoints when anonymous API quota is exhausted."""
        document = ET.fromstring(cls._request_text(GITHUB_RELEASES_ATOM_URL))
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        releases: list[dict[str, Any]] = []
        for entry in document.findall("atom:entry", namespace):
            link = entry.find("atom:link[@rel='alternate']", namespace)
            html_url = str((link.attrib if link is not None else {}).get("href") or "")
            tag = html_url.rstrip("/").rsplit("/", 1)[-1]
            try:
                parse_tag(tag)
            except UpdatePackageError:
                continue
            expected_name = release_asset_name(tag)
            assets_url = f"https://github.com/Creeper3222/PersonalityRAG/releases/expanded_assets/{tag}"
            assets_html = cls._request_text(assets_url)
            marker = f"/releases/download/{tag}/{expected_name}"
            marker_index = assets_html.find(marker)
            if marker_index < 0:
                continue
            digest_match = re.search(r"sha256:[0-9a-fA-F]{64}", assets_html[marker_index : marker_index + 5000])
            if not digest_match:
                continue
            page_html = cls._request_text(html_url)
            content = entry.findtext("atom:content", default="", namespaces=namespace)
            notes = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(content))).strip()
            releases.append(
                {
                    "tag_name": tag,
                    "version": tag.removeprefix("v"),
                    "name": entry.findtext("atom:title", default=tag, namespaces=namespace),
                    "published_at": entry.findtext("atom:updated", default="", namespaces=namespace),
                    "prerelease": bool(re.search(r">\s*Pre-release\s*<", page_html, re.IGNORECASE)),
                    "notes": notes[:20_000],
                    "html_url": html_url,
                    "asset": {
                        "name": expected_name,
                        "url": f"https://github.com/Creeper3222/PersonalityRAG/releases/download/{tag}/{expected_name}",
                        "size": 0,
                        "digest": digest_match.group(0).lower(),
                    },
                }
            )
        return releases

    @staticmethod
    def _normalize_release(raw: dict[str, Any]) -> dict[str, Any] | None:
        tag = str(raw.get("tag_name") or "")
        try:
            parse_tag(tag)
        except UpdatePackageError:
            return None
        if raw.get("draft"):
            return None
        expected_asset = release_asset_name(tag)
        expected_url = f"https://github.com/Creeper3222/PersonalityRAG/releases/download/{tag}/{expected_asset}"
        asset = next(
            (
                item
                for item in raw.get("assets") or []
                if isinstance(item, dict) and item.get("name") == expected_asset
            ),
            None,
        )
        digest = str((asset or {}).get("digest") or "")
        if (
            not asset
            or asset.get("browser_download_url") != expected_url
            or not digest.startswith("sha256:")
            or len(digest) != len("sha256:") + 64
        ):
            return None
        return {
            "tag_name": tag,
            "version": tag.removeprefix("v"),
            "name": str(raw.get("name") or tag),
            "published_at": raw.get("published_at") or raw.get("created_at"),
            "prerelease": bool(raw.get("prerelease")),
            "notes": str(raw.get("body") or "")[:20_000],
            "html_url": str(raw.get("html_url") or ""),
            "asset": {
                "name": expected_asset,
                "url": expected_url,
                "size": int(asset.get("size") or 0),
                "digest": digest,
            },
        }

    async def releases(self, *, refresh: bool = False) -> dict[str, Any]:
        async with self._lock:
            now = time.time()
            cached = self._read_cache()
            if refresh and now - self._last_manual_refresh < MANUAL_REFRESH_SECONDS:
                if cached:
                    return {**cached, "refresh_limited": True}
                raise RuntimeError("update check refresh is rate limited")
            if not refresh and cached and now - float(cached.get("checked_at") or 0) < CACHE_SECONDS:
                return dict(cached)
            if refresh:
                self._last_manual_refresh = now
            try:
                try:
                    raw_releases = await asyncio.to_thread(self._request_json, GITHUB_RELEASES_URL)
                    normalized = [
                        release
                        for item in (raw_releases if isinstance(raw_releases, list) else [])
                        if isinstance(item, dict)
                        if (release := self._normalize_release(item)) is not None
                    ]
                except (OSError, ValueError, urllib.error.URLError):
                    normalized = await asyncio.to_thread(self._fallback_release_feed)
                normalized.sort(key=lambda item: parse_tag(item["tag_name"]), reverse=True)
                payload = {
                    "checked_at": now,
                    "stale": False,
                    "error": "",
                    "releases": normalized,
                }
                self._write_cache(payload)
                return dict(payload)
            except (OSError, ValueError, urllib.error.URLError) as exc:
                logger.debug("GitHub update check unavailable: %s", exc)
                if cached:
                    return {**cached, "stale": True, "error": "update service unavailable"}
                return {
                    "checked_at": now,
                    "stale": True,
                    "error": "update service unavailable",
                    "releases": [],
                }

    async def status(self, *, refresh: bool = False) -> dict[str, Any]:
        payload = await self.releases(refresh=refresh)
        newer = [
            release
            for release in payload["releases"]
            if compare_tags(release["tag_name"], TAG_NAME) > 0
        ]
        latest = payload["releases"][0] if payload["releases"] else None
        active = self.active_transaction()
        switch_available = DockerEngineClient().available
        return {
            "current_version": VERSION,
            "current_tag": TAG_NAME,
            "latest_version": latest["version"] if latest else VERSION,
            "latest_tag": latest["tag_name"] if latest else TAG_NAME,
            "update_available": bool(newer),
            "checked_at": payload["checked_at"],
            "stale": payload.get("stale", False),
            "error": payload.get("error", ""),
            "refresh_limited": payload.get("refresh_limited", False),
            "active_transaction": active,
            "deployment_mode": "docker",
            "switch_available": switch_available,
            "switch_unavailable_reason": "" if switch_available else "docker_engine_socket_unavailable",
        }

    def transaction(self, transaction_id: str) -> dict[str, Any] | None:
        if not transaction_id or any(character not in "0123456789abcdef" for character in transaction_id):
            return None
        path = self.transactions_root / transaction_id / "transaction.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return self._public_transaction(payload)

    def active_transaction(self) -> dict[str, Any] | None:
        if not self.transactions_root.is_dir():
            return None
        candidates: list[dict[str, Any]] = []
        for path in self.transactions_root.glob("*/transaction.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("status") not in TERMINAL_TRANSACTION_STATUSES:
                candidates.append(payload)
        if not candidates:
            return None
        return self._public_transaction(max(candidates, key=lambda item: float(item.get("created_at") or 0)))

    @staticmethod
    def _public_transaction(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            key: payload.get(key)
            for key in (
                "transaction_id",
                "status",
                "stage",
                "current_version",
                "target_version",
                "target_tag",
                "action",
                "created_at",
                "updated_at",
                "completed_at",
                "error",
                "rollback_error",
            )
            if key in payload
        }

    @staticmethod
    def _download(url: str, destination: Path, expected_digest: str = "") -> str:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": f"PersonalityRAG/{VERSION}"},
        )
        digest = hashlib.sha256()
        total = 0
        with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as output:
            final_url = response.geturl()
            if not final_url.lower().startswith("https://"):
                raise UpdatePackageError("release download did not use TLS")
            while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                total += len(chunk)
                if total > MAX_RELEASE_BYTES:
                    raise UpdatePackageError("release asset exceeds the size limit")
                digest.update(chunk)
                output.write(chunk)
        value = digest.hexdigest()
        if expected_digest.startswith("sha256:") and value != expected_digest.split(":", 1)[1].lower():
            raise UpdatePackageError("GitHub release asset digest mismatch")
        return value

    async def prepare_switch(
        self,
        *,
        tag_name: str,
        service_pid: int,
        health_urls: list[str],
    ) -> dict[str, Any]:
        parse_tag(tag_name)
        releases_payload = await self.releases(refresh=False)
        async with self._lock:
            if self.active_transaction():
                raise UpdatePackageError("another version transaction is already active")
            engine = DockerEngineClient()
            if not engine.available:
                raise UpdatePackageError(
                    "Docker Engine socket is unavailable; use the official Compose deployment to switch versions"
                )
            release = next(
                (item for item in releases_payload["releases"] if item["tag_name"] == tag_name),
                None,
            )
            if release is None:
                raise UpdatePackageError("target release is unavailable or lacks the Linux/Docker asset")
            expected_url = (
                f"https://github.com/Creeper3222/PersonalityRAG/releases/download/"
                f"{tag_name}/{release_asset_name(tag_name)}"
            )
            if release["asset"].get("url") != expected_url:
                raise UpdatePackageError("target release asset URL is not the official repository URL")
            transaction_id = secrets.token_hex(12)
            transaction_root = self.transactions_root / transaction_id
            transaction_root.mkdir(parents=True, exist_ok=False)
            archive = transaction_root / release["asset"]["name"]
            extract_root = transaction_root / "candidate"
            try:
                asset_hash = await asyncio.to_thread(
                    self._download,
                    release["asset"]["url"],
                    archive,
                    release["asset"].get("digest") or "",
                )
                candidate_root, manifest = await asyncio.to_thread(
                    inspect_and_extract_zip,
                    archive,
                    extract_root,
                    expected_tag=tag_name,
                )
                architecture = current_architecture()
                docker_contract = manifest["docker"]
                platform_digest = str(docker_contract["platforms"][architecture])
                target_image_ref = f"{DOCKER_REPOSITORY}@{platform_digest}"
                source_container_id = current_container_id()
                source_container = await asyncio.to_thread(engine.inspect_container, source_container_id)
                state_bind_source(source_container)
                await asyncio.to_thread(engine.pull, DOCKER_REPOSITORY, platform_digest)
                target_image = await asyncio.to_thread(engine.inspect_image, target_image_ref)
                labels = (target_image.get("Config") or {}).get("Labels") or {}
                expected_commit = str((manifest.get("source") or {}).get("commit") or "")
                if labels.get("org.opencontainers.image.version") != tag_name:
                    raise UpdatePackageError("target Docker image version label does not match")
                if labels.get("org.opencontainers.image.revision") != expected_commit:
                    raise UpdatePackageError("target Docker image source revision does not match")
                action = "reinstall"
                comparison = compare_tags(tag_name, TAG_NAME)
                if comparison > 0:
                    action = "update"
                elif comparison < 0:
                    action = "rollback"
                payload = {
                    "transaction_id": transaction_id,
                    "status": "prepared",
                    "stage": "prepared",
                    "action": action,
                    "current_version": VERSION,
                    "current_tag": TAG_NAME,
                    "target_version": manifest["version"],
                    "target_tag": tag_name,
                    "asset_name": release["asset"]["name"],
                    "asset_sha256": asset_hash,
                    "source_root": str(self.source_root),
                    "state_root": str(self.state_root),
                    "candidate_root": str(candidate_root),
                    "service_pid": service_pid,
                    "health_urls": health_urls,
                    "architecture": architecture,
                    "source_commit": expected_commit,
                    "source_container_id": str(source_container.get("Id") or source_container_id),
                    "source_container_name": str(source_container.get("Name") or "PersonalityRAG").lstrip("/"),
                    "source_image_id": str(source_container.get("Image") or ""),
                    "target_image_ref": target_image_ref,
                    "target_image_id": str(target_image.get("Id") or ""),
                    "created_at": time.time(),
                    "updated_at": time.time(),
                }
                transaction_file = transaction_root / "transaction.json"
                transaction_file.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                helper_name = f"personalityrag-update-{transaction_id[:12]}"
                helper_id = await asyncio.to_thread(
                    engine.create_container,
                    helper_name,
                    helper_container_body(source_container, str(transaction_file)),
                )
                payload["helper_container_id"] = helper_id
                transaction_file.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                await asyncio.to_thread(engine.start, helper_id)
                return self._public_transaction(payload)
            except Exception:
                shutil.rmtree(transaction_root, ignore_errors=True)
                raise
