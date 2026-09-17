"""Closed public-synthetic worker entry point for one already planned wave.

This module imports no domain pack at import time.  The tabular, ML, and media
implementations -- and therefore polars, scikit-learn, torch, transformers, and
the FFmpeg-backed media pack -- are imported only inside the execution path that
needs them, so a tabular job bootstraps without the unrelated heavy stacks.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import tempfile
from contextlib import nullcontext
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from portable_batch_execution.contracts import (
    ArtifactRef,
    JobSpec,
    ShardAttemptRecord,
    ShardSpec,
    WaveSpec,
)
from portable_batch_execution.data_plane import LocalFilesystemDataPlane
from portable_batch_execution.worker import runtime_profile as _runtime_profile

_PRIVATE_TABULAR_SINGLE_INPUT_OPS = _runtime_profile.TABULAR_SINGLE_INPUT_OPERATIONS
_PRIVATE_TABULAR_TWO_TABLE_OPS = _runtime_profile.TABULAR_TWO_TABLE_OPERATIONS
_PRIVATE_TABULAR_FEATURE_REQUEST_OPS = _runtime_profile.TABULAR_FEATURE_REQUEST_OPERATIONS
_PRIVATE_TABULAR_UNAVAILABLE_MULTI_INPUT_OPS = (
    _runtime_profile.TABULAR_UNAVAILABLE_OPERATIONS
)
_PRIVATE_ML_SINGLE_INPUT_OPS = _runtime_profile.ML_SINGLE_INPUT_OPERATIONS
_PRIVATE_ML_FIVE_INPUT_OPS = _runtime_profile.ML_FIVE_INPUT_OPERATIONS
_PRIVATE_MEDIA_SINGLE_INPUT_OPS = _runtime_profile.MEDIA_SINGLE_INPUT_OPERATIONS

_WAVE_ID = re.compile(r"wave-[0-9]{4}")
_PUBLIC_WAVES = frozenset({"wave-0000"})

_LAZY_PACK_EXPORTS = {
    "TabularPack": ("portable_batch_execution.packs.tabular", "TabularPack"),
    "MediaPack": ("portable_batch_execution.packs.media", "MediaPack"),
}


def __getattr__(name: str) -> Any:
    """Resolve the tabular/media packs on demand for backward-compatible callers.

    Importing this module must not import any domain pack.  Attribute access such
    as ``execute_wave.TabularPack`` still resolves lazily so existing imports and
    ``unittest.mock.patch`` targets keep working.
    """
    try:
        module_name, attribute = _LAZY_PACK_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


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


def _parse_private_tabular_feature_request_inputs(
    feature_rows: object, request_rows: object,
) -> dict[str, list[dict]]:
    if not all(
        isinstance(rows, list) and all(isinstance(row, dict) for row in rows)
        for rows in (feature_rows, request_rows)
    ):
        raise TypeError("feature rows and request rows must be arrays of objects")
    return {"features": feature_rows, "requests": request_rows}


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
        from portable_batch_execution.packs.tabular import TabularPack

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
            and job.operation not in _PRIVATE_TABULAR_FEATURE_REQUEST_OPS
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
    else:
        raise ValueError("private wave operation is not available on the public runner")
    prior = plane.read_attempts(run_id)
    attempts: list[ShardAttemptRecord] = []
    pack = None
    media_pack = None
    if private_pack == "tabular-batch":
        from portable_batch_execution.packs.tabular import TabularPack

        pack = TabularPack()
    elif private_pack == "media-batch":
        from portable_batch_execution.packs.media import MediaPack

        media_pack = MediaPack()
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
                    from portable_batch_execution.packs.ml.distilbert_pair_binary_scores import (
                        execute_distilbert_pair_binary_scores,
                    )

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
            else:
                input_ref = shard.input_refs[0]
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
                    if job.operation in _PRIVATE_TABULAR_FEATURE_REQUEST_OPS:
                        try:
                            if len(shard.input_refs) != 2:
                                raise TypeError("trailing sparse window aggregate requires exactly two input artifacts")
                            request_payload = _read_verified_artifact_bytes(
                                plane, shard.input_refs[1]
                            )
                            request_rows = json.loads(request_payload.decode("utf-8"))
                            feature_request_input = _parse_private_tabular_feature_request_inputs(
                                parsed_input, request_rows
                            )
                            result = pack.execute(job, shard, job.operation_params, {"data": feature_request_input})
                        except Exception as exc:  # noqa: BLE001
                            raise _ShardStageFailure(_execution_failure_code(exc, stage="pack")) from None
                        input_rows = len(feature_request_input["features"]) + len(feature_request_input["requests"])
                    elif job.operation in _PRIVATE_TABULAR_TWO_TABLE_OPS:
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
                        if job.operation == "tabular.text_event_features.v1" and len(shard.input_refs) != 1:
                            raise _ShardStageFailure("input_artifact_invalid")
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
                            from portable_batch_execution.packs.ml.cosine_similarity_matrix import (
                                execute_cosine_similarity_matrix,
                            )

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
                            from portable_batch_execution.packs.ml.char_wb_tfidf_logistic_score import (
                                execute_char_wb_tfidf_logistic_score,
                            )

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
            try:
                output_ref = plane.write(output, output_media_type)
            except Exception as exc:  # noqa: BLE001
                raise _ShardStageFailure(
                    _execution_failure_code(exc, stage="output")
                ) from None
            if not _artifact_ref_matches_bytes(output, output_ref):
                raise _ShardStageFailure("output_artifact_mismatch")
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
                output_refs=(output_ref,),
                output_digest=sha256(output).hexdigest(),
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
