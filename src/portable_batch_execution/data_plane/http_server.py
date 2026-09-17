"""Loopback HTTP front-end for the authenticated private data plane service."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .base import ArtifactContentStream
from .service import PrivateDataPlaneService

_LOOPBACK_BIND_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_CONFIG_ERROR = "private data plane server environment is not configured"


def require_loopback_bind_host(host: str) -> str:
    """Reject non-loopback bind addresses; external HTTPS terminates before this service."""
    normalized = host.strip().lower()
    if normalized not in _LOOPBACK_BIND_HOSTS:
        raise ValueError("private data plane bind host must be loopback")
    return host.strip()


def _read_nonempty_utf8_secret_file(path: str) -> str:
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError:
        raise ValueError(_CONFIG_ERROR) from None
    token = raw.strip()
    if not token:
        raise ValueError(_CONFIG_ERROR)
    return token


def _resolve_bearer_token_from_environment() -> str:
    import os

    literal = os.environ.get("PBE_PRIVATE_DATA_PLANE_BEARER_TOKEN")
    token_file = os.environ.get("PBE_PRIVATE_DATA_PLANE_BEARER_TOKEN_FILE")
    literal_set = bool(literal and literal.strip())
    file_set = bool(token_file and token_file.strip())
    if literal_set and file_set:
        raise ValueError(_CONFIG_ERROR)
    if file_set:
        return _read_nonempty_utf8_secret_file(token_file.strip())
    if literal_set:
        return literal.strip()
    raise ValueError(_CONFIG_ERROR)


def _response_bytes(body: bytes | None) -> bytes:
    return body if body is not None else b""


class _PrivateDataPlaneHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def _dispatch(self, method: str) -> None:
        if not self.service.authorize(self.headers.get("Authorization")):
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if method != "HEAD":
                self.wfile.write(b'{"error":"unauthorized"}')
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length else b""
        status, headers, payload = self.service.dispatch(
            method,
            self.path,
            authorization=self.headers.get("Authorization"),
            headers={key: value for key, value in self.headers.items()},
            body=body,
        )
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        if not isinstance(payload, ArtifactContentStream):
            data = _response_bytes(payload)
            if data:
                self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if method == "HEAD":
            return
        if isinstance(payload, ArtifactContentStream):
            for chunk in payload.chunks:
                if chunk:
                    self.wfile.write(chunk)
            return
        data = _response_bytes(payload)
        if data:
            self.wfile.write(data)

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_HEAD(self) -> None:
        self._dispatch("HEAD")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_PUT(self) -> None:
        self._dispatch("PUT")


def serve_private_data_plane(
    state_root: Path,
    bearer_token: str,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
) -> ThreadingHTTPServer:
    host = require_loopback_bind_host(host)
    service = PrivateDataPlaneService(state_root, bearer_token)

    class Handler(_PrivateDataPlaneHandler):
        @property
        def service(self) -> PrivateDataPlaneService:
            return service

    server = ThreadingHTTPServer((host, port), Handler)
    return server


def serve_private_data_plane_from_environment() -> ThreadingHTTPServer:
    import os

    state_root = os.environ.get("PBE_PRIVATE_DATA_PLANE_STATE_ROOT")
    bearer_token = _resolve_bearer_token_from_environment()
    host = os.environ.get("PBE_PRIVATE_DATA_PLANE_BIND_HOST", "127.0.0.1")
    port = int(os.environ.get("PBE_PRIVATE_DATA_PLANE_BIND_PORT", "8765"))
    if not state_root:
        raise ValueError(_CONFIG_ERROR)
    return serve_private_data_plane(
        Path(state_root),
        bearer_token,
        host=require_loopback_bind_host(host),
        port=port,
    )
