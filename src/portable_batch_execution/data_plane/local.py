from __future__ import annotations

from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path
from threading import RLock
from urllib.parse import urlparse
from urllib.request import url2pathname

from portable_batch_execution.contracts import (
    ArtifactRef,
    RunManifest,
    ShardAttemptRecord,
)
from portable_batch_execution.controller.closed_wave_registry import safe_file_component

from .base import ArtifactContentStream, RevisionConflictError

_ARTIFACT_CHUNK_BYTES = 64 * 1024


class LocalFilesystemDataPlane:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    @property
    def _artifacts(self) -> Path:
        path = self.root / "artifacts"
        path.mkdir(exist_ok=True)
        return path

    @property
    def _runs(self) -> Path:
        path = self.root / "runs"
        path.mkdir(exist_ok=True)
        return path

    def write(self, data: bytes, media_type: str | None = None) -> ArtifactRef:
        digest = sha256(data).hexdigest()
        p = self._artifacts / digest
        with self._lock:
            if not p.exists():
                temp = p.with_suffix(".tmp")
                temp.write_bytes(data)
                temp.replace(p)
        return ArtifactRef(
            object_id=digest,
            uri=p.as_uri(),
            sha256=f"sha256:{digest}",
            media_type=media_type,
            size_bytes=len(data),
        )

    @staticmethod
    def _artifact_path(ref: ArtifactRef) -> Path:
        parsed = urlparse(ref.uri)
        if parsed.scheme != "file" or parsed.netloc:
            raise ValueError("artifact ref must be a local file URI")
        return Path(url2pathname(parsed.path))

    def read(self, ref: ArtifactRef) -> bytes:
        path = self._artifact_path(ref)
        if path.parent != self._artifacts or path.name != ref.object_id:
            raise ValueError("artifact ref is outside this data plane")
        return path.read_bytes()

    def open_content(self, ref: ArtifactRef) -> ArtifactContentStream:
        """Stream artifact bytes in bounded chunks without whole-object reads."""
        path = self._artifact_path(ref)
        if path.parent != self._artifacts or path.name != ref.object_id:
            raise ValueError("artifact ref is outside this data plane")
        return ArtifactContentStream(
            size_bytes=path.stat().st_size,
            chunks=self._iter_artifact_chunks(path),
        )

    @staticmethod
    def _iter_artifact_chunks(path: Path) -> Iterator[bytes]:
        with path.open("rb") as handle:
            while chunk := handle.read(_ARTIFACT_CHUNK_BYTES):
                yield chunk

    def exists(self, ref: ArtifactRef) -> bool:
        try:
            path = self._artifact_path(ref)
            return (
                path.parent == self._artifacts
                and path.name == ref.object_id
                and path.is_file()
            )
        except ValueError:
            return False

    def verify(self, ref: ArtifactRef) -> bool:
        try:
            data = self.read(ref)
        except (OSError, ValueError):
            return False
        return f"sha256:{sha256(data).hexdigest()}" == ref.sha256 and (
            ref.size_bytes is None or len(data) == ref.size_bytes
        )

    def _run_directory(self, run_id: str) -> Path:
        safe_file_component(run_id, "run_id")
        path = self._runs / run_id
        path.mkdir(exist_ok=True)
        return path

    def append_attempt(self, record: ShardAttemptRecord) -> None:
        """Persist an immutable attempt record; duplicate IDs are rejected."""
        with self._lock:
            run = self._run_directory(record.logical_run_id)
            safe_file_component(record.attempt_id, "attempt_id")
            attempts = run / "attempts"
            attempts.mkdir(exist_ok=True)
            path = attempts / f"{record.attempt_id}.json"
            if path.exists():
                existing = ShardAttemptRecord.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
                if existing != record:
                    raise ValueError("attempt_id already belongs to a different record")
                return
            temporary = path.with_suffix(".tmp")
            temporary.write_text(record.model_dump_json() + "\n", encoding="utf-8")
            temporary.replace(path)

    def read_attempts(self, run_id: str) -> tuple[ShardAttemptRecord, ...]:
        run = self._run_directory(run_id)
        attempts = run / "attempts"
        if not attempts.exists():
            return ()
        records = [
            ShardAttemptRecord.model_validate_json(path.read_text(encoding="utf-8"))
            for path in attempts.glob("*.json")
        ]
        return tuple(
            sorted(
                records,
                key=lambda record: (
                    record.finished_at,
                    record.started_at,
                    record.attempt_id,
                ),
            )
        )

    def read_manifest(self, run_id: str) -> RunManifest | None:
        path = self._run_directory(run_id) / "latest.json"
        return (
            RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
            if path.exists()
            else None
        )

    def write_next_manifest(
        self, manifest: RunManifest, expected_revision: int
    ) -> RunManifest:
        """Compare-and-swap latest manifest and retain every immutable revision."""
        with self._lock:
            manifest = RunManifest.model_validate(manifest.model_dump())
            run = self._run_directory(manifest.logical_run_id)
            current = self.read_manifest(manifest.logical_run_id)
            current_revision = current.revision if current else -1
            if current_revision != expected_revision:
                raise RevisionConflictError(
                    f"expected revision {expected_revision}, found {current_revision}"
                )
            if manifest.revision != expected_revision + 1:
                raise ValueError("next manifest revision must increment by one")
            history = run / "manifests"
            history.mkdir(exist_ok=True)
            revision_path = history / f"{manifest.revision:020d}.json"
            if revision_path.exists():
                raise RevisionConflictError("manifest revision already exists")
            encoded = manifest.model_dump_json() + "\n"
            temporary = revision_path.with_suffix(".tmp")
            temporary.write_text(encoded, encoding="utf-8")
            temporary.replace(revision_path)
            latest = run / "latest.json"
            latest_temp = latest.with_suffix(".tmp")
            latest_temp.write_text(encoded, encoding="utf-8")
            latest_temp.replace(latest)
            return manifest
