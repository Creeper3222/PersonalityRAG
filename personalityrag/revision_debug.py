from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import secrets
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from fastapi import Request

from .auth import AuthManager, COOKIE_NAME


DEBUG_COOKIE_NAME = "personalityrag_revision_debug"
DEBUG_SESSION_TTL_SECONDS = 15 * 60
DEBUG_FAILURE_LIMIT = 5
DEBUG_FAILURE_WINDOW_SECONDS = 5 * 60


def is_loopback_client(request: Request) -> bool:
    host = str(request.client.host if request.client else "").strip().strip("[]")
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "") or "")


def admin_credential_subject(request: Request, auth: AuthManager) -> str | None:
    authorization = str(request.headers.get("authorization") or "")
    if authorization.lower().startswith("bearer "):
        bearer = authorization[7:].strip()
        if auth.verify_api_key(bearer):
            return "bearer:" + hashlib.sha256(bearer.encode()).hexdigest()
    session = request.cookies.get(COOKIE_NAME)
    if auth.verify_session(session):
        return "session:" + hashlib.sha256(str(session).encode()).hexdigest()
    return None


@dataclass(slots=True)
class RevisionDebugSessionManager:
    ttl_seconds: int = DEBUG_SESSION_TTL_SECONDS
    boot_nonce: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    _active: dict[str, tuple[str, float]] = field(default_factory=dict)
    _failures: dict[str, deque[float]] = field(
        default_factory=lambda: defaultdict(deque)
    )
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @staticmethod
    def _password_subject(auth: AuthManager) -> str:
        return hashlib.sha256(str(auth.password_hash or "").encode()).hexdigest()

    @staticmethod
    def _encode(payload: dict[str, Any], secret: str) -> str:
        body = base64.urlsafe_b64encode(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        ).rstrip(b"=")
        signature = hmac.new(secret.encode(), body, hashlib.sha256).digest()
        return (
            body.decode()
            + "."
            + base64.urlsafe_b64encode(signature).rstrip(b"=").decode()
        )

    @staticmethod
    def _decode(token: str | None, secret: str) -> dict[str, Any] | None:
        if not token or "." not in token:
            return None
        body_text, signature_text = token.split(".", 1)
        body = body_text.encode()
        try:
            supplied = base64.urlsafe_b64decode(
                signature_text + "=" * (-len(signature_text) % 4)
            )
            expected = hmac.new(secret.encode(), body, hashlib.sha256).digest()
            if not hmac.compare_digest(supplied, expected):
                return None
            decoded = base64.urlsafe_b64decode(
                body_text + "=" * (-len(body_text) % 4)
            )
            payload = json.loads(decoded)
            return payload if isinstance(payload, dict) else None
        except (ValueError, TypeError, json.JSONDecodeError):
            return None

    def _purge_locked(self, now: float) -> None:
        expired = [
            nonce for nonce, (_subject, exp) in self._active.items() if exp <= now
        ]
        for nonce in expired:
            self._active.pop(nonce, None)
        cutoff = now - DEBUG_FAILURE_WINDOW_SECONDS
        for key in list(self._failures):
            attempts = self._failures[key]
            while attempts and attempts[0] <= cutoff:
                attempts.popleft()
            if not attempts:
                self._failures.pop(key, None)

    def failure_status(self, client_key: str, *, now: float | None = None) -> dict[str, Any]:
        current = float(time.time() if now is None else now)
        with self._lock:
            self._purge_locked(current)
            attempts = self._failures.get(client_key, deque())
            retry_after = 0
            if len(attempts) >= DEBUG_FAILURE_LIMIT:
                retry_after = max(
                    1,
                    int(DEBUG_FAILURE_WINDOW_SECONDS - (current - attempts[0])) + 1,
                )
            return {
                "limited": retry_after > 0,
                "retry_after": retry_after,
                "remaining_attempts": max(0, DEBUG_FAILURE_LIMIT - len(attempts)),
            }

    def record_failure(self, client_key: str, *, now: float | None = None) -> None:
        current = float(time.time() if now is None else now)
        with self._lock:
            self._purge_locked(current)
            self._failures[client_key].append(current)

    def clear_failures(self, client_key: str) -> None:
        with self._lock:
            self._failures.pop(client_key, None)

    def issue(
        self,
        request: Request,
        auth: AuthManager,
        *,
        now: float | None = None,
    ) -> tuple[str, dict[str, Any]]:
        subject = admin_credential_subject(request, auth)
        if not subject:
            raise ValueError("administrator credential is unavailable")
        current = float(time.time() if now is None else now)
        expires_at = current + max(1, int(self.ttl_seconds))
        nonce = secrets.token_urlsafe(18)
        payload = {
            "iat": int(current),
            "exp": int(expires_at),
            "nonce": nonce,
            "boot": self.boot_nonce,
            "subject": subject,
            "password": self._password_subject(auth),
        }
        with self._lock:
            self._purge_locked(current)
            self._active[nonce] = (subject, expires_at)
        token = self._encode(payload, auth.session_secret)
        return token, self.status(token, request, auth, now=current)

    def status(
        self,
        token: str | None,
        request: Request,
        auth: AuthManager,
        *,
        now: float | None = None,
    ) -> dict[str, Any]:
        current = float(time.time() if now is None else now)
        base = {
            "password_configured": auth.password_enabled,
            "unlocked": False,
            "expires_at": None,
            "remaining_seconds": 0,
            "fixed_ttl_seconds": int(self.ttl_seconds),
        }
        if not auth.password_enabled:
            return base
        payload = self._decode(token, auth.session_secret)
        if not payload:
            return base
        subject = admin_credential_subject(request, auth)
        try:
            nonce = str(payload["nonce"])
            expires_at = float(payload["exp"])
        except (KeyError, TypeError, ValueError):
            return base
        if (
            not subject
            or payload.get("boot") != self.boot_nonce
            or payload.get("subject") != subject
            or payload.get("password") != self._password_subject(auth)
            or expires_at <= current
        ):
            return base
        with self._lock:
            self._purge_locked(current)
            active = self._active.get(nonce)
        if not active or active[0] != subject or active[1] <= current:
            return base
        return {
            **base,
            "unlocked": True,
            "expires_at": expires_at,
            "remaining_seconds": max(0, int(expires_at - current)),
        }

    def revoke(self, token: str | None, auth: AuthManager) -> None:
        payload = self._decode(token, auth.session_secret)
        nonce = str((payload or {}).get("nonce") or "")
        if nonce:
            with self._lock:
                self._active.pop(nonce, None)

    def revoke_all(self) -> None:
        with self._lock:
            self._active.clear()
