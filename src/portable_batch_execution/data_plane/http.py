"""HTTP client for a controller-owned private data plane."""
from __future__ import annotations

import os
from collections.abc import Iterator
from hashlib import sha256
from typing import Any
from urllib.parse import quote, urlsplit

import httpx

from portable_batch_execution.contracts import (
    ArtifactRef,
    RunManifest,
    ShardAttemptRecord,
)

from .base import ArtifactContentStream, RevisionConflictError


class PrivateDataPlaneError(RuntimeError):
    """An error intentionally containing operation/status, never response data."""


_HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0)
_ARTIFACT_CHUNK_BYTES = 64 * 1024


class HttpPrivateDataPlane:
    def __init__(self, base_url: str, bearer_token: str, *, client: httpx.Client | None = None):
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("private data plane base URL must be an HTTPS origin URL")
        if not bearer_token:
            raise ValueError("private data plane bearer token is required")
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=_HTTP_TIMEOUT)
        self._client.headers.setdefault("Authorization", f"Bearer {bearer_token}")

    @classmethod
    def from_environment(cls) -> HttpPrivateDataPlane:
        base_url, token = os.environ.get("PBE_PRIVATE_DATA_PLANE_BASE_URL"), os.environ.get("PBE_PRIVATE_DATA_PLANE_BEARER_TOKEN")
        if not base_url or not token:
            raise ValueError("private data plane environment is not configured")
        return cls(base_url, token)

    @staticmethod
    def _part(value: str, name: str) -> str:
        if not isinstance(value, str) or not value or len(value) > 128 or any(c in value for c in "/\\?#"):
            raise ValueError(f"{name} must be an opaque identifier")
        return quote(value, safe="")

    def _request(self, operation: str, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._client.request(method, f"{self.base_url}{path}", **kwargs)
        except httpx.HTTPError as exc:
            raise PrivateDataPlaneError(f"private data plane {operation} failed") from exc
        if response.status_code >= 400:
            raise PrivateDataPlaneError(f"private data plane {operation} failed with HTTP {response.status_code}")
        return response

    @staticmethod
    def _json(response: httpx.Response, operation: str) -> Any:
        try: return response.json()
        except ValueError as exc: raise PrivateDataPlaneError(f"private data plane {operation} returned invalid JSON") from exc

    def resolve_wave(self, run_id: str, wave_id: str) -> dict[str, Any]:
        payload = self._json(self._request("resolve wave", "GET", f"/v1/runs/{self._part(run_id, 'run_id')}/waves/{self._part(wave_id, 'wave_id')}"), "resolve wave")
        if not isinstance(payload, dict):
            raise PrivateDataPlaneError("private data plane resolve wave returned invalid JSON")
        return payload

    def read(self, ref: ArtifactRef) -> bytes:
        return self._request("read artifact", "GET", f"/v1/artifacts/{self._part(ref.object_id, 'artifact object_id')}/content").content

    @staticmethod
    def _resolve_bounded_stream_size(
        ref: ArtifactRef, response: httpx.Response
    ) -> int:
        header_value = response.headers.get("content-length")
        header_size: int | None = None
        if header_value is not None:
            try:
                header_size = int(header_value)
            except ValueError as exc:
                raise PrivateDataPlaneError(
                    "private data plane read artifact failed"
                ) from exc
            if header_size < 0:
                raise PrivateDataPlaneError("private data plane read artifact failed")
        if (
            header_size is not None
            and ref.size_bytes is not None
            and header_size != ref.size_bytes
        ):
            raise PrivateDataPlaneError("private data plane read artifact failed")
        if header_size is not None:
            return header_size
        if ref.size_bytes is not None:
            return ref.size_bytes
        raise PrivateDataPlaneError("private data plane read artifact failed")

    @staticmethod
    def _iter_response_chunks(response: httpx.Response) -> Iterator[bytes]:
        try:
            try:
                for chunk in response.iter_bytes(_ARTIFACT_CHUNK_BYTES):
                    if chunk:
                        yield chunk
            except httpx.HTTPError as exc:
                raise PrivateDataPlaneError(
                    "private data plane read artifact failed"
                ) from exc
        finally:
            response.close()

    def open_content(self, ref: ArtifactRef) -> ArtifactContentStream:
        """Stream artifact bytes from the opaque object-id endpoint."""
        path = f"/v1/artifacts/{self._part(ref.object_id, 'artifact object_id')}/content"
        request = self._client.build_request("GET", f"{self.base_url}{path}")
        try:
            response = self._client.send(request, stream=True)
        except httpx.HTTPError as exc:
            raise PrivateDataPlaneError("private data plane read artifact failed") from exc
        if response.status_code >= 400:
            response.close()
            raise PrivateDataPlaneError(
                f"private data plane read artifact failed with HTTP {response.status_code}"
            )
        try:
            size_bytes = self._resolve_bounded_stream_size(ref, response)
        except PrivateDataPlaneError:
            response.close()
            raise
        return ArtifactContentStream(
            size_bytes=size_bytes,
            chunks=self._iter_response_chunks(response),
        )

    def write(self, data: bytes, media_type: str | None = None) -> ArtifactRef:
        payload = self._json(self._request("write artifact", "POST", "/v1/artifacts", content=data, headers={"Content-Type": media_type or "application/octet-stream"}), "write artifact")
        try: return ArtifactRef.model_validate(payload)
        except ValueError as exc: raise PrivateDataPlaneError("private data plane write artifact returned invalid reference") from exc

    def exists(self, ref: ArtifactRef) -> bool:
        try: response = self._client.request("HEAD", f"{self.base_url}/v1/artifacts/{self._part(ref.object_id, 'artifact object_id')}/content")
        except httpx.HTTPError: return False
        return response.status_code < 400

    def verify(self, ref: ArtifactRef) -> bool:
        try: data = self.read(ref)
        except PrivateDataPlaneError: return False
        return f"sha256:{sha256(data).hexdigest()}" == ref.sha256 and (ref.size_bytes is None or len(data) == ref.size_bytes)

    def append_attempt(self, record: ShardAttemptRecord) -> None:
        self._request("append attempt", "POST", f"/v1/runs/{self._part(record.logical_run_id, 'run_id')}/attempts", json=record.model_dump(mode="json"))

    def read_attempts(self, run_id: str) -> tuple[ShardAttemptRecord, ...]:
        payload = self._json(self._request("read attempts", "GET", f"/v1/runs/{self._part(run_id, 'run_id')}/attempts"), "read attempts")
        try: return tuple(ShardAttemptRecord.model_validate(item) for item in payload)
        except (TypeError, ValueError) as exc: raise PrivateDataPlaneError("private data plane read attempts returned invalid contracts") from exc

    def read_manifest(self, run_id: str) -> RunManifest | None:
        try: response = self._client.request("GET", f"{self.base_url}/v1/runs/{self._part(run_id, 'run_id')}/manifest")
        except httpx.HTTPError as exc: raise PrivateDataPlaneError("private data plane read manifest failed") from exc
        if response.status_code == 404: return None
        if response.status_code >= 400: raise PrivateDataPlaneError(f"private data plane read manifest failed with HTTP {response.status_code}")
        try: return RunManifest.model_validate(response.json())
        except ValueError as exc: raise PrivateDataPlaneError("private data plane read manifest returned invalid contracts") from exc

    def write_next_manifest(self, manifest: RunManifest, expected_revision: int) -> RunManifest:
        try: response = self._client.request("PUT", f"{self.base_url}/v1/runs/{self._part(manifest.logical_run_id, 'run_id')}/manifest", json=manifest.model_dump(mode="json"), headers={"If-Match": str(expected_revision)})
        except httpx.HTTPError as exc: raise PrivateDataPlaneError("private data plane write manifest failed") from exc
        if response.status_code == 409: raise RevisionConflictError("private data plane manifest revision conflict")
        if response.status_code >= 400: raise PrivateDataPlaneError(f"private data plane write manifest failed with HTTP {response.status_code}")
        try: return RunManifest.model_validate(response.json())
        except ValueError as exc: raise PrivateDataPlaneError("private data plane write manifest returned invalid contracts") from exc

