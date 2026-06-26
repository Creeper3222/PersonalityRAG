from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass


COOKIE_NAME = "personalityrag_session"
PASSWORD_ALGORITHM = "pbkdf2_sha256"
PASSWORD_ITERATIONS = 260_000


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
    )
    return "$".join(
        [
            PASSWORD_ALGORITHM,
            str(PASSWORD_ITERATIONS),
            base64.urlsafe_b64encode(salt).decode().rstrip("="),
            base64.urlsafe_b64encode(digest).decode().rstrip("="),
        ]
    )


def verify_password(password: str | None, encoded: str | None) -> bool:
    if not password or not encoded:
        return False
    try:
        algorithm, iterations_text, salt_text, digest_text = encoded.split("$", 3)
        if algorithm != PASSWORD_ALGORITHM:
            return False
        iterations = int(iterations_text)
        salt = base64.urlsafe_b64decode(salt_text + "=" * (-len(salt_text) % 4))
        supplied = base64.urlsafe_b64decode(
            digest_text + "=" * (-len(digest_text) % 4)
        )
        expected = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations,
        )
        return hmac.compare_digest(supplied, expected)
    except (ValueError, TypeError):
        return False


@dataclass(slots=True)
class AuthManager:
    api_key: str
    session_secret: str
    session_ttl_seconds: int = 86400
    password_hash: str = ""

    def verify_api_key(self, candidate: str | None) -> bool:
        return bool(candidate) and hmac.compare_digest(candidate, self.api_key)

    @property
    def password_enabled(self) -> bool:
        return bool(self.password_hash)

    def verify_login_secret(self, candidate: str | None) -> bool:
        if self.password_enabled:
            return verify_password(candidate, self.password_hash)
        return self.verify_api_key(candidate)

    def issue_session(self) -> str:
        payload = {
            "iat": int(time.time()),
            "exp": int(time.time()) + self.session_ttl_seconds,
            "nonce": hashlib.sha256(
                f"{time.time_ns()}:{self.api_key}".encode()
            ).hexdigest()[:16],
        }
        body = base64.urlsafe_b64encode(
            json.dumps(payload, separators=(",", ":")).encode()
        ).rstrip(b"=")
        signature = hmac.new(
            self.session_secret.encode(), body, hashlib.sha256
        ).digest()
        return (
            body.decode()
            + "."
            + base64.urlsafe_b64encode(signature).rstrip(b"=").decode()
        )

    def verify_session(self, token: str | None) -> bool:
        if not token or "." not in token:
            return False
        body_text, sig_text = token.split(".", 1)
        body = body_text.encode()
        try:
            supplied = base64.urlsafe_b64decode(sig_text + "=" * (-len(sig_text) % 4))
            expected = hmac.new(
                self.session_secret.encode(), body, hashlib.sha256
            ).digest()
            if not hmac.compare_digest(supplied, expected):
                return False
            payload = json.loads(
                base64.urlsafe_b64decode(
                    body_text + "=" * (-len(body_text) % 4)
                )
            )
            return int(payload.get("exp", 0)) >= int(time.time())
        except (ValueError, TypeError, json.JSONDecodeError):
            return False
