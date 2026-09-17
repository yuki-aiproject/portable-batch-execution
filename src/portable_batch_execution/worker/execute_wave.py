"""Closed public-synthetic worker entry point for one already planned wave."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from contextlib import nullcontext
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from portable_batch_execution.contracts import (
    ArtifactRef,
    JobSpec,
    ShardAttemptRecord,
    ShardSpec,
    WaveSpec,
)
from portable_batch_execution.data_plane import LocalFilesystemDataPlane
from portable_batch_execution.data_plane.base import ArtifactContentStream
from portable_batch_execution.packs import MediaPack, ReplayReductionPack, TabularPack
from portable_batch_execution.packs.ml.char_wb_tfidf_logistic_score import (
    execute_char_wb_tfidf_logistic_score,
)
from portable_batch_execution.packs.ml.cosine_similarity_matrix import (
    execute_cosine_similarity_matrix,
)
from portable_batch_execution.packs.ml.distilbert_pair_binary_scores import (
    execute_distilbert_pair_binary_scores,
)
from portable_batch_execution.packs.replay_reduction.canonicalize import (
    BUCKET_MEDIA_TYPE,
    StructuralCanonicalizeError,
    attach_bucket_refs,
    decode_state_from_bucket_payloads,
    iter_state_bucket_payloads,
    state_summary,
)
from portable_batch_execution.packs.replay_reduction.models import (
    BUCKET_COUNT_MAX,
)

_WAVE_ID = re.compile(r"wave-[0-9]{4}")
_PUBLIC_WAVES = frozenset({"wave-0000"})
_PRIVATE_TABULAR_SINGLE_INPUT_OPS = frozenset(
    {
        "tabular.normalize",
        "tabular.cast",
        "tabular.sort",
        "tabular.dedup",
        "tabular.window",
        "tabular.rolling",
        "tabular.statistics",
    }
)
_PRIVATE_TABULAR_TWO_TABLE_OPS = frozenset(
    {
        "tabular.join",
        "tabular.pit_join",
    }
)
_PRIVATE_TABULAR_UNAVAILABLE_MULTI_INPUT_OPS = frozenset({"tabular.format_migration"})
_PRIVATE_TABULAR_PARQUET_MEDIA_TYPES = frozenset(
    {
        "application/vnd.apache.parquet",
        "application/x-parquet",
    }
)
_PRIVATE_ML_SINGLE_INPUT_OPS = frozenset(
    {
        "ml.char_wb_tfidf_logistic_score",
        "ml.cosine_similarity_matrix",
    }
)
_PRIVATE_ML_FIVE_INPUT_OPS = frozenset({"ml.distilbert_pair_binary_scores"})
_PRIVATE_MEDIA_SINGLE_INPUT_OPS = frozenset({"media.asr_normalize_flac"})
_PRIVATE_REPLAY_BATCH_OPS = frozenset(
    {
        "replay.structural_canonicalize",
        "replay.structural_canonicalize_merge",
        "replay.event_window_extract",
        "replay.causal_grid_extract",
    }
)
_MAX_REPLAY_PARQUET_INPUTS = 64
_PRIVATE_REPLAY_PARQUET_MEDIA_TYPES = _PRIVATE_TABULAR_PARQUET_MEDIA_TYPES


class PrivateWaveExecutionError(RuntimeError):
    """One or more shards failed during private wave execution."""

    def __init__(self, attempts: tuple[ShardAttemptRecord, ...]) -> None:
        self.attempts = attempts
        super().__init__("private wave execution completed with shard failures")


class _ShardStageFailure(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _matching_current_attempts(
    prior: tuple[ShardAttemptRecord, ...], shard: ShardSpec
) -> list[ShardAttemptRecord]:
    return [
        item
        for item in prior
        if item.shard_id == shard.shard_id
        and item.input_digest == shard.input_digest
        and item.execution_fingerprint == shard.execution_fingerprint
    ]


def _private_generation_discriminator(
    input_digest: str, execution_fingerprint: str
) -> str:
    material = f"{input_digest}\x1f{execution_fingerprint}".encode()
    return sha256(material).hexdigest()[:12]


def _private_attempt_id(
    wave_id: str,
    shard_id: str,
    *,
    input_digest: str,
    execution_fingerprint: str,
    current_attempt_count: int,
) -> str:
    generation = _private_generation_discriminator(
        input_digest, execution_fingerprint
    )
    ordinal = current_attempt_count + 1
    return f"{wave_id}-{shard_id}-{generation}-{ordinal}"


def _parse_private_tabular_two_table_envelope(
    parsed_input: object,
) -> tuple[list[dict], list[dict]]:
    if not isinstance(parsed_input, dict):
        raise TypeError("tabular two-table input must be an object")
    if frozenset(parsed_input) != frozenset({"left", "right"}):
        raise TypeError("tabular two-table input must contain only left and right")
    left = parsed_input["left"]
    right = parsed_input["right"]
    if not isinstance(left, list) or not isinstance(right, list):
        raise TypeError("left and right must be record arrays")
    if not all(isinstance(row, dict) for row in left) or not all(
        isinstance(row, dict) for row in right
    ):
        raise TypeError("left and right must be arrays of objects")
    return left, right


def _artifact_ref_matches_bytes(data: bytes, ref: ArtifactRef) -> bool:
    if f"sha256:{sha256(data).hexdigest()}" != ref.sha256:
        return False
    return ref.size_bytes is None or len(data) == ref.size_bytes


def _read_verified_artifact_bytes(plane, ref: ArtifactRef) -> bytes:
    try:
        payload = plane.read(ref)
    except Exception as exc:  # noqa: BLE001
        raise _ShardStageFailure(
            _execution_failure_code(exc, stage="input_read")
        ) from None
    if not _artifact_ref_matches_bytes(payload, ref):
        raise _ShardStageFailure("input_artifact_mismatch")
    return payload


def _open_artifact_content_stream(plane, ref: ArtifactRef) -> ArtifactContentStream:
    try:
        stream = plane.open_content(ref)
    except Exception as exc:  # noqa: BLE001
        raise _ShardStageFailure(
            _execution_failure_code(exc, stage="input_read")
        ) from None
    if ref.size_bytes is not None and stream.size_bytes != ref.size_bytes:
        raise _ShardStageFailure("input_artifact_mismatch")
    return stream


def _artifact_ref_is_parquet(ref: ArtifactRef) -> bool:
    return ref.media_type in _PRIVATE_REPLAY_PARQUET_MEDIA_TYPES


def _materialize_verified_parquet_inputs(
    plane, refs: tuple[ArtifactRef, ...], directory: Path
) -> list[Path]:
    if not refs:
        raise _ShardStageFailure("input_artifact_invalid")
    if len(refs) > _MAX_REPLAY_PARQUET_INPUTS:
        raise _ShardStageFailure("input_artifact_invalid")
    paths: list[Path] = []
    for index, ref in enumerate(refs):
        if not _artifact_ref_is_parquet(ref):
            raise _ShardStageFailure("input_artifact_invalid")
        destination = directory / f"input-{index}.parquet"
        _materialize_verified_artifact(plane, ref, destination)
        paths.append(destination)
    return paths


def _materialize_verified_artifact(
    plane, ref: ArtifactRef, destination: Path
) -> None:
    stream = _open_artifact_content_stream(plane, ref)
    expected_size = stream.size_bytes
    digest = sha256()
    total = 0
    try:
        with destination.open("wb") as handle:
            for chunk in stream.chunks:
                chunk_len = len(chunk)
                if total + chunk_len > expected_size:
                    raise _ShardStageFailure("input_artifact_mismatch")
                digest.update(chunk)
                total += chunk_len
                handle.write(chunk)
    except _ShardStageFailure:
        destination.unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001
        destination.unlink(missing_ok=True)
        raise _ShardStageFailure(
            _execution_failure_code(exc, stage="input_read")
        ) from None
    if total != expected_size:
        destination.unlink(missing_ok=True)
        raise _ShardStageFailure("input_artifact_mismatch")
    if ref.size_bytes is not None and total != ref.size_bytes:
        destination.unlink(missing_ok=True)
        raise _ShardStageFailure("input_artifact_mismatch")
    if f"sha256:{digest.hexdigest()}" != ref.sha256:
        destination.unlink(missing_ok=True)
        raise _ShardStageFailure("input_artifact_mismatch")


def _parquet_row_count(path: Path) -> int:
    import polars as pl

    return int(pl.scan_parquet(str(path)).select(pl.len()).collect().item())


def _write_canonicalize_state(
    plane, state
) -> tuple[bytes, tuple[ArtifactRef, ...]]:
    """Publish one summary JSON plus bucket artifacts in deterministic bucket order."""
    bucket_refs: list[ArtifactRef] = []
    for payload in iter_state_bucket_payloads(state):
        ref = plane.write(payload, BUCKET_MEDIA_TYPE)
        if not _artifact_ref_matches_bytes(payload, ref):
            raise _ShardStageFailure("output_artifact_mismatch")
        bucket_refs.append(ref)
    summary = attach_bucket_refs(state_summary(state), tuple(bucket_refs))
    summary_bytes = json.dumps(summary, sort_keys=True).encode("utf-8")
    summary_ref = plane.write(summary_bytes, "application/json")
    if not _artifact_ref_matches_bytes(summary_bytes, summary_ref):
        raise _ShardStageFailure("output_artifact_mismatch")
    return summary_bytes, (summary_ref, *bucket_refs)


def _read_canonicalize_state(plane, summary_ref: ArtifactRef):
    """Read and validate one complete canonicalization state from its artifacts."""
    payload = _read_verified_artifact_bytes(plane, summary_ref)
    try:
        summary = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise _ShardStageFailure(
            _execution_failure_code(exc, stage="input_parse")
        ) from None
    if not isinstance(summary, dict):
        raise _ShardStageFailure("input_artifact_invalid")
    raw_refs = summary.get("bucket_refs")
    if not isinstance(raw_refs, list) or not 1 <= len(raw_refs) <= BUCKET_COUNT_MAX:
        raise _ShardStageFailure("input_artifact_invalid")
    try:
        bucket_refs = tuple(ArtifactRef.model_validate(item) for item in raw_refs)
    except ValueError:
        raise _ShardStageFailure("input_artifact_invalid") from None
    if any(ref.media_type != BUCKET_MEDIA_TYPE for ref in bucket_refs):
        raise _ShardStageFailure("input_artifact_invalid")
    def _iter_verified_bucket_payloads():
        for ref in bucket_refs:
            yield _read_verified_artifact_bytes(plane, ref)

    try:
        return decode_state_from_bucket_payloads(summary, _iter_verified_bucket_payloads())
    except StructuralCanonicalizeError:
        raise _ShardStageFailure("input_artifact_invalid") from None


def _private_tabular_single_input_is_parquet(
    job, input_ref: ArtifactRef
) -> bool:
    return (
        job.pack == "tabular-batch"
        and job.operation in _PRIVATE_TABULAR_SINGLE_INPUT_OPS
        and input_ref.media_type in _PRIVATE_TABULAR_PARQUET_MEDIA_TYPES
    )

def _execution_failure_code(exc: BaseException, *, stage: str) -> str:
    if stage == "input_read":
        return "input_artifact_read_failed"
    if stage == "input_decode":
        return "input_artifact_decode_failed"
    if stage == "input_parse":
        return "input_artifact_invalid"
    if stage == "pack":
        return "shard_pack_execution_failed"
    if stage == "output":
        return "output_artifact_write_failed"
    return "shard_execution_failed"


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _plan_path(wave_id: str, repository_root: Path) -> Path:
    if not isinstance(wave_id, str) or not _WAVE_ID.fullmatch(wave_id):
        raise ValueError("wave_id must be a closed planned wave identifier")
    if wave_id not in _PUBLIC_WAVES:
        raise ValueError("wave_id is not an approved public synthetic wave")
    return repository_root / "fixtures" / "public" / "synthetic" / f"{wave_id}.json"


def _load_plan(wave_id: str, repository_root: Path) -> dict:
    path = _plan_path(wave_id, repository_root)
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("public synthetic wave plan is invalid") from exc
    if not isinstance(plan, dict):
        raise TypeError("public synthetic wave plan must be an object")
    return plan


def execute_public_wave(
    wave_id: str,
    *,
    repository_root: Path | None = None,
    state_root: Path | None = None,
) -> tuple[ShardAttemptRecord, ...]:
    """Execute a committed public plan and append attempts without touching manifests."""
    root = (repository_root or _repository_root()).resolve()
    plan = _load_plan(wave_id, root)
    fixture_root = (root / "fixtures" / "public").resolve()
    data_name = plan.get("input_fixture")
    if not isinstance(data_name, str) or Path(data_name).name != data_name:
        raise ValueError("public synthetic input fixture is invalid")
    data_path = (fixture_root / data_name).resolve()
    if fixture_root not in data_path.parents or not data_path.is_file():
        raise ValueError("public synthetic input fixture is unavailable")

    context = tempfile.TemporaryDirectory() if state_root is None else nullcontext()
    with context as temporary:
        plane = LocalFilesystemDataPlane(state_root or Path(temporary))
        input_ref = plane.write(data_path.read_bytes(), "application/json")
        job = JobSpec.model_validate(
            {**plan["job"], "input_manifest_ref": input_ref.model_dump(mode="json")}
        )
        wave = WaveSpec.model_validate(plan["wave"])
        if wave.wave_id != wave_id or wave.logical_run_id != job.logical_run_id:
            raise ValueError("public synthetic wave does not match its job")
        shards = tuple(
            ShardSpec.model_validate(
                {
                    **item,
                    "logical_run_id": job.logical_run_id,
                    "correctness": job.sharding.model_dump(mode="json"),
                    "input_refs": [input_ref.model_dump(mode="json")],
                }
            )
            for item in plan["shards"]
        )
        if tuple(shard.shard_id for shard in shards) != wave.shard_ids:
            raise ValueError("public synthetic wave shard plan is invalid")
        if job.pack != "tabular-batch" or job.operation != "tabular.rolling":
            raise ValueError("public synthetic plan uses an unsupported pack operation")
        rows = json.loads(data_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise TypeError("public synthetic input must be a record array")
        pack = TabularPack()
        attempts: list[ShardAttemptRecord] = []
        for shard in shards:
            started_at = datetime.now(UTC)
            result = pack.execute(job, shard, job.operation_params, {"data": rows})
            output = json.dumps(result.to_dicts(), sort_keys=True).encode("utf-8")
            output_ref = plane.write(output, "application/json")
            attempt = ShardAttemptRecord(
                logical_run_id=job.logical_run_id,
                shard_id=shard.shard_id,
                attempt_id=f"{wave.wave_id}-{shard.shard_id}",
                status="succeeded",
                input_digest=shard.input_digest,
                execution_fingerprint=shard.execution_fingerprint,
                started_at=started_at,
                finished_at=datetime.now(UTC),
                wave_id=wave.wave_id,
                output_refs=(output_ref,),
                output_digest=sha256(output).hexdigest(),
                counts={"input_rows": len(rows), "output_rows": result.height},
            )
            plane.append_attempt(attempt)
            attempts.append(attempt)
        return tuple(attempts)


def execute_private_wave(
    run_id: str, wave_id: str, *, plane=None
) -> tuple[ShardAttemptRecord, ...]:
    """Execute an externally resolved closed tabular wave through the private plane."""
    from portable_batch_execution.data_plane import HttpPrivateDataPlane

    plane = plane or HttpPrivateDataPlane.from_environment()
    payload = plane.resolve_wave(run_id, wave_id)
    try:
        job = JobSpec.model_validate(payload["job"])
        wave = WaveSpec.model_validate(payload["wave"])
        shards = tuple(ShardSpec.model_validate(item) for item in payload["shards"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("private data plane returned invalid closed wave contracts") from exc
    if job.logical_run_id != run_id or wave.logical_run_id != run_id or wave.wave_id != wave_id:
        raise ValueError("private data plane resolved a different run or wave")
    if tuple(shard.shard_id for shard in shards) != wave.shard_ids or any(shard.logical_run_id != run_id for shard in shards):
        raise ValueError("private data plane returned mismatched shards")
    if job.pack == "tabular-batch":
        if job.operation in _PRIVATE_TABULAR_UNAVAILABLE_MULTI_INPUT_OPS:
            raise ValueError("private wave operation requires a typed multi-input contract")
        if (
            job.operation not in _PRIVATE_TABULAR_SINGLE_INPUT_OPS
            and job.operation not in _PRIVATE_TABULAR_TWO_TABLE_OPS
        ):
            raise ValueError("private wave operation is not available on the public runner")
        private_pack = "tabular-batch"
    elif job.pack == "ml-batch":
        if (
            job.operation not in _PRIVATE_ML_SINGLE_INPUT_OPS
            and job.operation not in _PRIVATE_ML_FIVE_INPUT_OPS
        ):
            raise ValueError("private wave operation is not available on the public runner")
        if job.operation in _PRIVATE_ML_FIVE_INPUT_OPS and job.operation_params:
            raise ValueError("closed operation parameters")
        private_pack = "ml-batch"
    elif job.pack == "media-batch":
        if job.operation not in _PRIVATE_MEDIA_SINGLE_INPUT_OPS:
            raise ValueError("private wave operation is not available on the public runner")
        if job.operation_params:
            raise ValueError("closed operation parameters")
        private_pack = "media-batch"
    elif job.pack == "replay-batch":
        if job.operation not in _PRIVATE_REPLAY_BATCH_OPS:
            raise ValueError("private wave operation is not available on the public runner")
        private_pack = "replay-batch"
    else:
        raise ValueError("private wave operation is not available on the public runner")
    prior = plane.read_attempts(run_id)
    attempts: list[ShardAttemptRecord] = []
    pack = TabularPack()
    media_pack = MediaPack()
    replay_pack = ReplayReductionPack()
    wave_failures = 0
    for shard in shards:
        current = _matching_current_attempts(prior, shard)
        if any(item.status == "succeeded" for item in current):
            continue
        if len(current) >= job.execution.max_attempts_per_shard:
            continue
        if not shard.input_refs:
            raise ValueError("private shard has no input artifact")
        started_at = datetime.now(UTC)
        attempt_id = _private_attempt_id(
            wave_id,
            shard.shard_id,
            input_digest=shard.input_digest,
            execution_fingerprint=shard.execution_fingerprint,
            current_attempt_count=len(current),
        )
        output_refs: tuple[ArtifactRef, ...] | None = None
        output_digest_value: str | None = None
        try:
            if (
                private_pack == "ml-batch"
                and job.operation == "ml.distilbert_pair_binary_scores"
            ):
                if len(shard.input_refs) != 5:
                    raise _ShardStageFailure("input_artifact_invalid")
                refs = shard.input_refs
                tensor_bytes = _read_verified_artifact_bytes(plane, refs[0])
                model_a_config = _read_verified_artifact_bytes(plane, refs[1])
                model_a_weights = _read_verified_artifact_bytes(plane, refs[2])
                model_b_config = _read_verified_artifact_bytes(plane, refs[3])
                model_b_weights = _read_verified_artifact_bytes(plane, refs[4])
                try:
                    text = tensor_bytes.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise _ShardStageFailure(
                        _execution_failure_code(exc, stage="input_decode")
                    ) from None
                try:
                    parsed_input = json.loads(text)
                except ValueError as exc:
                    raise _ShardStageFailure(
                        _execution_failure_code(exc, stage="input_parse")
                    ) from None
                try:
                    result_payload = execute_distilbert_pair_binary_scores(
                        parsed_input,
                        model_a_config=model_a_config,
                        model_a_weights=model_a_weights,
                        model_b_config=model_b_config,
                        model_b_weights=model_b_weights,
                    )
                except Exception as exc:  # noqa: BLE001
                    raise _ShardStageFailure(
                        _execution_failure_code(exc, stage="pack")
                    ) from None
                input_rows = len(result_payload["row_ids"])
                output_rows = input_rows
                output = json.dumps(result_payload, sort_keys=True).encode("utf-8")
                output_media_type = "application/json"
            elif private_pack == "media-batch":
                input_ref = shard.input_refs[0]
                payload = _read_verified_artifact_bytes(plane, input_ref)
                with tempfile.TemporaryDirectory() as temporary:
                    input_path = Path(temporary) / "input.bin"
                    output_path = Path(temporary) / "output.flac"
                    input_path.write_bytes(payload)
                    try:
                        media_pack.execute(
                            job,
                            shard,
                            job.operation_params,
                            {
                                "input": str(input_path),
                                "output": str(output_path),
                            },
                        )
                    except Exception as exc:  # noqa: BLE001
                        raise _ShardStageFailure(
                            _execution_failure_code(exc, stage="pack")
                        ) from None
                    try:
                        output = output_path.read_bytes()
                    except OSError as exc:
                        raise _ShardStageFailure(
                            _execution_failure_code(exc, stage="pack")
                        ) from None
                media_counts = {
                    "input_bytes": len(payload),
                    "output_bytes": len(output),
                }
                output_media_type = "audio/flac"
            elif private_pack == "replay-batch":
                publish_single_output = True
                if job.operation == "replay.structural_canonicalize":
                    with tempfile.TemporaryDirectory() as temporary:
                        paths = _materialize_verified_parquet_inputs(
                            plane, tuple(shard.input_refs), Path(temporary)
                        )
                        input_rows = sum(_parquet_row_count(path) for path in paths)
                        try:
                            state = replay_pack.execute(
                                job,
                                shard,
                                job.operation_params,
                                {"parquet_paths": paths, "operation": job.operation},
                            )
                        except StructuralCanonicalizeError as exc:
                            raise _ShardStageFailure(
                                _execution_failure_code(exc, stage="pack")
                            ) from None
                        except Exception as exc:  # noqa: BLE001
                            raise _ShardStageFailure(
                                _execution_failure_code(exc, stage="pack")
                            ) from None
                        output, output_refs = _write_canonicalize_state(plane, state)
                        output_digest_value = sha256(output).hexdigest()
                        output_rows = state.positive_row_count + state.witness_row_count
                        publish_single_output = False
                elif job.operation == "replay.structural_canonicalize_merge":
                    if len(shard.input_refs) != 2:
                        raise _ShardStageFailure("input_artifact_invalid")
                    left_state = _read_canonicalize_state(plane, shard.input_refs[0])
                    right_state = _read_canonicalize_state(plane, shard.input_refs[1])
                    try:
                        merged_state = replay_pack.execute(
                            job,
                            shard,
                            job.operation_params,
                            {
                                "left_state": left_state,
                                "right_state": right_state,
                                "operation": job.operation,
                            },
                        )
                    except StructuralCanonicalizeError as exc:
                        raise _ShardStageFailure(
                            _execution_failure_code(exc, stage="pack")
                        ) from None
                    except Exception as exc:  # noqa: BLE001
                        raise _ShardStageFailure(
                            _execution_failure_code(exc, stage="pack")
                        ) from None
                    input_rows = (
                        left_state.positive_row_count + right_state.positive_row_count
                    )
                    output, output_refs = _write_canonicalize_state(plane, merged_state)
                    output_digest_value = sha256(output).hexdigest()
                    output_rows = (
                        merged_state.positive_row_count + merged_state.witness_row_count
                    )
                    publish_single_output = False
                elif job.operation == "replay.event_window_extract":
                    if len(shard.input_refs) < 2:
                        raise _ShardStageFailure("input_artifact_invalid")
                    parquet_refs = shard.input_refs[:-1]
                    request_ref = shard.input_refs[-1]
                    if not all(_artifact_ref_is_parquet(ref) for ref in parquet_refs):
                        raise _ShardStageFailure("input_artifact_invalid")
                    request_payload = _read_verified_artifact_bytes(plane, request_ref)
                    try:
                        parsed_request = json.loads(request_payload.decode("utf-8"))
                    except (UnicodeDecodeError, ValueError) as exc:
                        raise _ShardStageFailure(
                            _execution_failure_code(exc, stage="input_parse")
                        ) from None
                    with tempfile.TemporaryDirectory() as temporary:
                        paths = _materialize_verified_parquet_inputs(
                            plane, tuple(parquet_refs), Path(temporary)
                        )
                        input_rows = sum(_parquet_row_count(path) for path in paths)
                        try:
                            result_payload = replay_pack.execute(
                                job,
                                shard,
                                job.operation_params,
                                {
                                    "parquet_paths": paths,
                                    "request": parsed_request,
                                    "operation": job.operation,
                                },
                            )
                        except Exception as exc:  # noqa: BLE001
                            raise _ShardStageFailure(
                                _execution_failure_code(exc, stage="pack")
                            ) from None
                        output_rows = len(result_payload["facts"])
                elif job.operation == "replay.causal_grid_extract":
                    if len(shard.input_refs) < 2:
                        raise _ShardStageFailure("input_artifact_invalid")
                    parquet_refs = shard.input_refs[:-1]
                    request_ref = shard.input_refs[-1]
                    if not all(_artifact_ref_is_parquet(ref) for ref in parquet_refs):
                        raise _ShardStageFailure("input_artifact_invalid")
                    request_payload = _read_verified_artifact_bytes(plane, request_ref)
                    try:
                        parsed_request = json.loads(request_payload.decode("utf-8"))
                    except (UnicodeDecodeError, ValueError) as exc:
                        raise _ShardStageFailure(
                            _execution_failure_code(exc, stage="input_parse")
                        ) from None
                    with tempfile.TemporaryDirectory() as temporary:
                        paths = _materialize_verified_parquet_inputs(
                            plane, tuple(parquet_refs), Path(temporary)
                        )
                        input_rows = sum(_parquet_row_count(path) for path in paths)
                        try:
                            result_payload = replay_pack.execute(
                                job,
                                shard,
                                job.operation_params,
                                {
                                    "parquet_paths": paths,
                                    "request": parsed_request,
                                    "operation": job.operation,
                                },
                            )
                        except Exception as exc:  # noqa: BLE001
                            raise _ShardStageFailure(
                                _execution_failure_code(exc, stage="pack")
                            ) from None
                        output_rows = len(result_payload["rows"])
                else:
                    raise _ShardStageFailure(
                        _execution_failure_code(ValueError(), stage="pack")
                    )
                if publish_single_output:
                    output = json.dumps(result_payload, sort_keys=True).encode("utf-8")
                    output_media_type = "application/json"
            else:
                input_ref = shard.input_refs[0]
                if _private_tabular_single_input_is_parquet(job, input_ref):
                    with tempfile.TemporaryDirectory() as temporary:
                        input_path = Path(temporary) / "input.parquet"
                        _materialize_verified_artifact(plane, input_ref, input_path)
                        try:
                            input_rows = _parquet_row_count(input_path)
                            result = pack.execute(
                                job,
                                shard,
                                job.operation_params,
                                {"data": str(input_path)},
                            )
                        except Exception as exc:  # noqa: BLE001
                            raise _ShardStageFailure(
                                _execution_failure_code(exc, stage="pack")
                            ) from None
                        output = json.dumps(
                            result.to_dicts(), sort_keys=True
                        ).encode("utf-8")
                        output_rows = result.height
                    output_media_type = "application/json"
                else:
                    payload = _read_verified_artifact_bytes(plane, input_ref)
                    try:
                        text = payload.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise _ShardStageFailure(
                            _execution_failure_code(exc, stage="input_decode")
                        ) from None
                    try:
                        parsed_input = json.loads(text)
                    except ValueError as exc:
                        raise _ShardStageFailure(
                            _execution_failure_code(exc, stage="input_parse")
                        ) from None
                    output_media_type = "application/json"
                    if private_pack == "tabular-batch":
                        if job.operation in _PRIVATE_TABULAR_TWO_TABLE_OPS:
                            try:
                                left, right = _parse_private_tabular_two_table_envelope(
                                    parsed_input
                                )
                            except TypeError as exc:
                                raise _ShardStageFailure(
                                    _execution_failure_code(exc, stage="input_parse")
                                ) from None
                            try:
                                result = pack.execute(
                                    job,
                                    shard,
                                    job.operation_params,
                                    {"data": left, "right": right},
                                )
                            except Exception as exc:  # noqa: BLE001
                                raise _ShardStageFailure(
                                    _execution_failure_code(exc, stage="pack")
                                ) from None
                            input_rows = len(left) + len(right)
                        else:
                            if not isinstance(parsed_input, list):
                                raise _ShardStageFailure(
                                    _execution_failure_code(TypeError(), stage="input_parse")
                                )
                            try:
                                result = pack.execute(
                                    job, shard, job.operation_params, {"data": parsed_input}
                                )
                            except Exception as exc:  # noqa: BLE001
                                raise _ShardStageFailure(
                                    _execution_failure_code(exc, stage="pack")
                                ) from None
                            input_rows = len(parsed_input)
                        output = json.dumps(result.to_dicts(), sort_keys=True).encode(
                            "utf-8"
                        )
                        output_rows = result.height
                    else:
                        if not isinstance(parsed_input, dict):
                            raise _ShardStageFailure(
                                _execution_failure_code(TypeError(), stage="input_parse")
                            )
                        try:
                            if job.operation == "ml.cosine_similarity_matrix":
                                result_payload = execute_cosine_similarity_matrix(
                                    parsed_input
                                )
                                input_rows = len(parsed_input["left"]) + len(
                                    parsed_input["right"]
                                )
                                output_rows = len(result_payload["scores"]) * len(
                                    result_payload["scores"][0]
                                )
                            elif job.operation == "ml.char_wb_tfidf_logistic_score":
                                result_payload = execute_char_wb_tfidf_logistic_score(
                                    parsed_input
                                )
                                input_rows = len(parsed_input.get("rows", ()))
                                output_rows = len(result_payload["rows"])
                            else:
                                raise _ShardStageFailure(
                                    _execution_failure_code(ValueError(), stage="pack")
                                )
                        except Exception as exc:  # noqa: BLE001
                            raise _ShardStageFailure(
                                _execution_failure_code(exc, stage="pack")
                            ) from None
                        output = json.dumps(result_payload, sort_keys=True).encode("utf-8")
            if output_refs is None:
                try:
                    single_ref = plane.write(output, output_media_type)
                except Exception as exc:  # noqa: BLE001
                    raise _ShardStageFailure(
                        _execution_failure_code(exc, stage="output")
                    ) from None
                if not _artifact_ref_matches_bytes(output, single_ref):
                    raise _ShardStageFailure("output_artifact_mismatch")
                output_refs = (single_ref,)
                output_digest_value = sha256(output).hexdigest()
            attempt = ShardAttemptRecord(
                logical_run_id=run_id,
                shard_id=shard.shard_id,
                attempt_id=attempt_id,
                status="succeeded",
                input_digest=shard.input_digest,
                execution_fingerprint=shard.execution_fingerprint,
                started_at=started_at,
                finished_at=datetime.now(UTC),
                wave_id=wave_id,
                output_refs=output_refs,
                output_digest=output_digest_value,
                counts=(
                    media_counts
                    if private_pack == "media-batch"
                    else {"input_rows": input_rows, "output_rows": output_rows}
                ),
            )
        except _ShardStageFailure as failed:
            attempt = ShardAttemptRecord(
                logical_run_id=run_id,
                shard_id=shard.shard_id,
                attempt_id=attempt_id,
                status="failed",
                input_digest=shard.input_digest,
                execution_fingerprint=shard.execution_fingerprint,
                started_at=started_at,
                finished_at=datetime.now(UTC),
                wave_id=wave_id,
                failure=failed.code,
            )
            wave_failures += 1
        plane.append_attempt(attempt)
        attempts.append(attempt)
    if wave_failures:
        raise PrivateWaveExecutionError(tuple(attempts))
    return tuple(attempts)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute one approved wave.")
    parser.add_argument("--wave-id", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--mode", choices=("public", "private"), default="public")
    args = parser.parse_args(argv)
    if args.mode == "private" and not args.run_id:
        parser.error("--mode private requires --run-id")
    try:
        attempts = (
            execute_private_wave(args.run_id, args.wave_id)
            if args.mode == "private"
            else execute_public_wave(args.wave_id)
        )
    except PrivateWaveExecutionError as exc:
        attempts = exc.attempts
        print(
            json.dumps(
                {
                    "wave_id": args.wave_id,
                    "attempt_ids": [item.attempt_id for item in attempts],
                }
            )
        )
        return 1
    print(json.dumps({"wave_id": args.wave_id, "attempt_ids": [item.attempt_id for item in attempts]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
