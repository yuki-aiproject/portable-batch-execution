from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from portable_batch_execution.contracts import (
    ArtifactRef,
    RunManifest,
    ShardAttemptRecord,
)


@dataclass(frozen=True)
class ArtifactContentStream:
    """Bounded incremental reader over immutable artifact content."""

    size_bytes: int
    chunks: Iterator[bytes]


class ArtifactStore(Protocol):
    def read(self, ref: ArtifactRef) -> bytes: ...
    def write(self, data: bytes, media_type: str | None = None) -> ArtifactRef: ...
    def exists(self, ref: ArtifactRef) -> bool: ...
    def verify(self, ref: ArtifactRef) -> bool: ...
    def open_content(self, ref: ArtifactRef) -> ArtifactContentStream: ...


class RunStateStore(Protocol):
    def append_attempt(self, record: ShardAttemptRecord) -> None: ...
    def read_attempts(self, run_id: str) -> tuple[ShardAttemptRecord, ...]: ...
    def read_manifest(self, run_id: str) -> RunManifest | None: ...
    def write_next_manifest(
        self, manifest: RunManifest, expected_revision: int
    ) -> RunManifest: ...


class RevisionConflictError(RuntimeError):
    """The persisted manifest revision did not match the caller's CAS value."""
