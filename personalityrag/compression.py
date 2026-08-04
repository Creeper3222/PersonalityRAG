from __future__ import annotations

from typing import Any

from starlette.datastructures import Headers
from starlette.middleware.gzip import GZipMiddleware, GZipResponder, IdentityResponder


def _compressible(headers: Headers) -> bool:
    if "content-encoding" in headers or "content-range" in headers:
        return False
    disposition = headers.get("content-disposition", "").lower()
    if "attachment" in disposition:
        return False
    content_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if not content_type or content_type == "text/event-stream":
        return False
    if content_type.startswith("text/"):
        return True
    if content_type == "image/svg+xml":
        return True
    return (
        content_type in {
            "application/json",
            "application/javascript",
            "application/xml",
        }
        or content_type.endswith("+json")
        or content_type.endswith("+xml")
    )


class SelectiveGZipResponder(GZipResponder):
    async def send_with_compression(self, message: dict[str, Any]) -> None:
        if message.get("type") == "http.response.start":
            await super().send_with_compression(message)
            headers = Headers(raw=self.initial_message["headers"])
            self.content_type_is_excluded = not _compressible(headers)
            return
        await super().send_with_compression(message)


class SelectiveGZipMiddleware(GZipMiddleware):
    """Compress sizeable textual payloads, never media or package streams."""

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        if "gzip" in headers.get("Accept-Encoding", ""):
            responder = SelectiveGZipResponder(
                self.app,
                self.minimum_size,
                compresslevel=self.compresslevel,
            )
        else:
            responder = IdentityResponder(self.app, self.minimum_size)
        await responder(scope, receive, send)
