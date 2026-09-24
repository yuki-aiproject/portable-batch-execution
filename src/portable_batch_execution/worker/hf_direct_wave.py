"""HF-direct multi-shard wave descriptor, worker execution, and result manifests."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from portable_batch_execution.contracts import JobSpec, ShardSpec, WaveSpec
from portable_batch_execution.packs.replay_reduction.pack import ReplayReductionPack
from portable_batch_execution.packs.replay_reduction.trade_path_scenario_evaluate_fixed_set import (
    FIXED_SET_OPERATION,
)
from portable_batch_execution.transport.hf_bucket import (
    HF_BUCKET_ID,
    HF_BUCKET_REF_SCHEMA,
    HfBucketTransport,
    artifact_ref_to_hf_bucket_ref,
    build_hf_api_storage_from_token,
    parse_hf_object_uri,
    validate_hf_bucket_ref,
)
from portable_batch_execution.worker.hf_dataset_resolve import (
    artifact_ref_is_pinned_hf_dataset_resolve,
)
from portable_batch_execution.worker.hf_direct import (
    HF_DIRECT_OPERATIONS,
    PAIRED_FILL_LEDGER_CANONICAL_FINALIZE_OPERATION,
    PAIRED_FILL_OPERATION,
    HfDirectExecutionError,
    assert_no_private_data_plane_dependency,
    derive_result_object_path_from_input_object_path,
    artifact_ref_is_hf_bucket,
    execute_hf_direct_paired_fill_ledger_canonical_finalize_shard,
    execute_hf_direct_paired_fill_reduce_shard,
    execute_hf_direct_trade_path_fixed_set_shard,
    hf_direct_enabled,
    paired_fill_ledger_canonical_finalize_shard_input_digest,
    paired_fill_shard_input_digest,
    validate_hf_direct_transport_profile,
)

HF_WAVE_DESCRIPTOR_SCHEMA = "pbe.hf-direct.wave-descriptor.v1"
HF_WAVE_RESULT_MANIFEST_SCHEMA = "pbe.hf-direct.wave-result-manifest.v1"


class HfDirectWaveError(RuntimeError):
    """Fail-closed HF-direct wave error."""


def hf_direct_mode_from_environment() -> bool:
    if os.environ.get("PBE_MODE", "").strip() == "hf-direct":
        return True
    return hf_direct_enabled()


def wave_descriptor_object_path(
    *,
    bucket_prefix: str,
    public_revision: str,
    manifest_digest: str,
    wave_id: str,
) -> str:
    prefix = bucket_prefix.rstrip("/")
    revision = public_revision.strip()
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise HfDirectWaveError("public_revision must be a git commit sha")
    digest = manifest_digest.strip()
    if not digest:
        raise HfDirectWaveError("manifest_digest required")
    wave = wave_id.strip()
    if not wave:
        raise HfDirectWaveError("wave_id required")
    tag = sha256(f"{revision}|{digest}|{wave}|descriptor".encode()).hexdigest()[:16]
    return f"{prefix}/waves/{revision}/{digest}/{wave}-descriptor-{tag}.json".lstrip(
        "/"
    )


def wave_result_manifest_object_path(
    *,
    bucket_prefix: str,
    public_revision: str,
    manifest_digest: str,
    wave_id: str,
) -> str:
    prefix = bucket_prefix.rstrip("/")
    revision = public_revision.strip()
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise HfDirectWaveError("public_revision must be a git commit sha")
    digest = manifest_digest.strip()
    if not digest:
        raise HfDirectWaveError("manifest_digest required")
    wave = wave_id.strip()
    if not wave:
        raise HfDirectWaveError("wave_id required")
    tag = sha256(f"{revision}|{digest}|{wave}|manifest".encode()).hexdigest()[:16]
    return (
        f"{prefix}/wave-results/{revision}/{digest}/{wave}-manifest-{tag}.json".lstrip(
            "/"
        )
    )


def validate_wave_descriptor_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != HF_WAVE_DESCRIPTOR_SCHEMA:
        raise HfDirectWaveError("wave descriptor schema mismatch")
    public_revision = str(payload.get("public_revision") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", public_revision):
        raise HfDirectWaveError("public_revision must be a git commit sha")
    logical_run_id = str(payload.get("logical_run_id") or "")
    wave_id = str(payload.get("wave_id") or "")
    if not logical_run_id or not wave_id:
        raise HfDirectWaveError("logical_run_id and wave_id required")
    manifest_digest = str(payload.get("manifest_digest") or "")
    if not manifest_digest:
        raise HfDirectWaveError("manifest_digest required")
    job = JobSpec.model_validate(payload["job"])
    wave = WaveSpec.model_validate(payload["wave"])
    shards_raw = payload.get("shards")
    if not isinstance(shards_raw, list) or not (1 <= len(shards_raw) <= 16):
        raise HfDirectWaveError("wave descriptor requires 1..16 shards")
    shards = tuple(ShardSpec.model_validate(item) for item in shards_raw)
    if job.logical_run_id != logical_run_id or wave.logical_run_id != logical_run_id:
        raise HfDirectWaveError("descriptor run identity mismatch")
    if wave.wave_id != wave_id:
        raise HfDirectWaveError("descriptor wave identity mismatch")
    if tuple(shard.shard_id for shard in shards) != wave.shard_ids:
        raise HfDirectWaveError("descriptor shard_ids mismatch")
    if job.operation not in HF_DIRECT_OPERATIONS:
        raise HfDirectWaveError("unsupported HF-direct operation")
    try:
        validate_hf_direct_transport_profile(job)
    except HfDirectExecutionError as exc:
        raise HfDirectWaveError(str(exc)) from exc
    return {
        "schema_version": HF_WAVE_DESCRIPTOR_SCHEMA,
        "public_revision": public_revision,
        "logical_run_id": logical_run_id,
        "wave_id": wave_id,
        "manifest_digest": manifest_digest,
        "job": job,
        "wave": wave,
        "shards": shards,
    }


def load_wave_descriptor_from_ref(
    transport: HfBucketTransport, ref: Mapping[str, Any]
) -> dict[str, Any]:
    validated_ref = validate_hf_bucket_ref(ref)
    raw = transport.read_verified(validated_ref)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise HfDirectWaveError("wave descriptor is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise HfDirectWaveError("wave descriptor must be a JSON object")
    descriptor = validate_wave_descriptor_payload(payload)
    material = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    if sha256(material).hexdigest() != validated_ref["sha256"]:
        raise HfDirectWaveError("wave descriptor hash mismatch")
    if len(material) != validated_ref["size_bytes"]:
        raise HfDirectWaveError("wave descriptor size mismatch")
    return descriptor


def build_wave_result_manifest(
    *,
    descriptor: Mapping[str, Any],
    shard_results: list[dict[str, Any]],
    public_revision: str,
) -> dict[str, Any]:
    job: JobSpec = descriptor["job"]
    wave: WaveSpec = descriptor["wave"]
    return {
        "schema_version": HF_WAVE_RESULT_MANIFEST_SCHEMA,
        "public_revision": public_revision,
        "logical_run_id": wave.logical_run_id,
        "wave_id": wave.wave_id,
        "manifest_digest": descriptor["manifest_digest"],
        "pack": job.pack,
        "operation": job.operation,
        "finished_at": "1970-01-01T00:00:00+00:00",
        "shards": shard_results,
    }


def _shard_result_row(
    shard: ShardSpec,
    *,
    status: str,
    output_hf_ref: dict[str, Any] | None,
    input_object_path: str,
    failure: str | None = None,
    output_hf_refs: list[dict[str, Any]] | None = None,
    input_digest: str | None = None,
) -> dict[str, Any]:
    row = {
        "shard_id": shard.shard_id,
        "ordinal": shard.ordinal,
        "input_object_path": input_object_path,
        "status": status,
    }
    if output_hf_ref is not None:
        row["output_hf_ref"] = validate_hf_bucket_ref(output_hf_ref)
    if output_hf_refs is not None:
        row["output_hf_refs"] = [
            validate_hf_bucket_ref(item) for item in output_hf_refs
        ]
    if input_digest is not None:
        row["input_digest"] = input_digest
    if failure is not None:
        row["failure"] = failure
    return row


def execute_hf_direct_wave(
    *,
    wave_descriptor_ref: Mapping[str, Any],
    expected_manifest_path: str | None = None,
    transport: HfBucketTransport | None = None,
) -> dict[str, Any]:
    assert_no_private_data_plane_dependency()
    if not hf_direct_mode_from_environment():
        raise HfDirectExecutionError("HF-direct mode is not enabled")
    token = HfBucketTransport.token_from_environment()
    transport = transport or HfBucketTransport(build_hf_api_storage_from_token(token))
    descriptor = load_wave_descriptor_from_ref(transport, wave_descriptor_ref)
    public_revision = str(descriptor["public_revision"])
    expected_revision = expected_public_revision_from_environment()
    if public_revision != expected_revision:
        raise HfDirectWaveError(
            "descriptor public_revision mismatch with checked-out revision"
        )
    job: JobSpec = descriptor["job"]
    wave: WaveSpec = descriptor["wave"]
    shards: tuple[ShardSpec, ...] = descriptor["shards"]
    manifest_digest = str(descriptor["manifest_digest"])
    bucket_prefix = str(job.operation_params.get("bucket_prefix") or "").strip()
    if not bucket_prefix and shards:
        first_ref = shards[0].input_refs[0]
        if job.operation == PAIRED_FILL_OPERATION:
            request_path = parse_hf_object_uri(shards[0].input_refs[-1].uri)
            match = re.match(r"^(?P<prefix>.+/)paired-fill-requests/", request_path)
            if match is None:
                raise HfDirectWaveError("cannot derive bucket_prefix from request path")
            bucket_prefix = match.group("prefix")
        else:
            first_path = parse_hf_object_uri(first_ref.uri)
            match = re.match(r"^(?P<prefix>.+/)normalized/", first_path)
            if match is None:
                raise HfDirectWaveError(
                    "cannot derive bucket_prefix from shard input path"
                )
            bucket_prefix = match.group("prefix")
    bucket_prefix = bucket_prefix.rstrip("/") + "/"
    manifest_path = wave_result_manifest_object_path(
        bucket_prefix=bucket_prefix,
        public_revision=public_revision,
        manifest_digest=manifest_digest,
        wave_id=wave.wave_id,
    )
    if expected_manifest_path is not None:
        normalized_expected = expected_manifest_path.lstrip("/")
        if normalized_expected != manifest_path:
            raise HfDirectWaveError("result manifest path binding mismatch")
    replay_pack = ReplayReductionPack()
    max_parallel = max(1, min(int(wave.max_parallel or 1), len(shards)))

    def run_shard(shard: ShardSpec) -> dict[str, Any]:
        if job.operation == FIXED_SET_OPERATION:
            if len(shard.input_refs) != 1:
                raise HfDirectExecutionError("shard requires one HF input ref")
            input_ref = shard.input_refs[0]
            input_object_path = parse_hf_object_uri(input_ref.uri)
            output_path = derive_result_object_path_from_input_object_path(
                input_object_path,
                public_revision=public_revision,
            )
            out_ref, _summary = execute_hf_direct_trade_path_fixed_set_shard(
                transport=transport,
                replay_pack=replay_pack,
                job=job,
                shard=shard,
                input_ref=input_ref,
                output_object_path=output_path,
            )
            return _shard_result_row(
                shard,
                status="succeeded",
                output_hf_ref=validate_hf_bucket_ref(
                    {
                        "schema_version": HF_BUCKET_REF_SCHEMA,
                        "bucket_id": HF_BUCKET_ID,
                        "object_path": parse_hf_object_uri(out_ref.uri),
                        "sha256": out_ref.sha256.removeprefix("sha256:"),
                        "size_bytes": int(out_ref.size_bytes or 0),
                        "media_type": str(out_ref.media_type or "application/json"),
                    }
                ),
                input_object_path=input_object_path,
            )
        if job.operation == PAIRED_FILL_OPERATION:
            if len(shard.input_refs) < 2:
                raise HfDirectExecutionError("paired-fill shard input layout invalid")
            for ref in shard.input_refs[:-1]:
                if not artifact_ref_is_pinned_hf_dataset_resolve(ref):
                    raise HfDirectExecutionError(
                        "parquet input must be pinned HF resolve ref"
                    )
            request_ref = shard.input_refs[-1]
            input_object_path = parse_hf_object_uri(request_ref.uri)
            metadata_ref, output_hf_refs, _summary = (
                execute_hf_direct_paired_fill_reduce_shard(
                    transport=transport,
                    replay_pack=replay_pack,
                    job=job,
                    shard=shard,
                    bucket_prefix=bucket_prefix,
                    public_revision=public_revision,
                    manifest_digest=manifest_digest,
                    wave_id=wave.wave_id,
                )
            )
            return _shard_result_row(
                shard,
                status="succeeded",
                output_hf_ref=artifact_ref_to_hf_bucket_ref(metadata_ref),
                output_hf_refs=output_hf_refs,
                input_object_path=input_object_path,
                input_digest=paired_fill_shard_input_digest(shard),
            )
        if job.operation == PAIRED_FILL_LEDGER_CANONICAL_FINALIZE_OPERATION:
            if len(shard.input_refs) != 3:
                raise HfDirectExecutionError(
                    "paired-ledger canonical finalize shard input layout invalid"
                )
            for ref in shard.input_refs:
                if not artifact_ref_is_hf_bucket(ref):
                    raise HfDirectExecutionError(
                        "paired-ledger canonical finalize inputs must be HF bucket refs"
                    )
            request_ref = shard.input_refs[2]
            input_object_path = parse_hf_object_uri(request_ref.uri)
            metadata_ref, output_hf_refs, _summary = (
                execute_hf_direct_paired_fill_ledger_canonical_finalize_shard(
                    transport=transport,
                    replay_pack=replay_pack,
                    job=job,
                    shard=shard,
                    bucket_prefix=bucket_prefix,
                    public_revision=public_revision,
                    manifest_digest=manifest_digest,
                    wave_id=wave.wave_id,
                )
            )
            return _shard_result_row(
                shard,
                status="succeeded",
                output_hf_ref=artifact_ref_to_hf_bucket_ref(metadata_ref),
                output_hf_refs=output_hf_refs,
                input_object_path=input_object_path,
                input_digest=paired_fill_ledger_canonical_finalize_shard_input_digest(
                    shard
                ),
            )
        raise HfDirectExecutionError("unsupported HF-direct operation")

    shard_results: list[dict[str, Any]] = []
    failures: list[str] = []
    if max_parallel <= 1:
        for shard in shards:
            try:
                shard_results.append(run_shard(shard))
            except HfDirectExecutionError as exc:
                failures.append(str(exc))
                input_path = parse_hf_object_uri(shard.input_refs[0].uri)
                shard_results.append(
                    _shard_result_row(
                        shard,
                        status="failed",
                        output_hf_ref=None,
                        input_object_path=input_path,
                        failure=str(exc),
                    )
                )
    else:
        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            future_map = {pool.submit(run_shard, shard): shard for shard in shards}
            for future in as_completed(future_map):
                shard = future_map[future]
                try:
                    shard_results.append(future.result())
                except HfDirectExecutionError as exc:
                    failures.append(str(exc))
                    input_path = parse_hf_object_uri(shard.input_refs[0].uri)
                    shard_results.append(
                        _shard_result_row(
                            shard,
                            status="failed",
                            output_hf_ref=None,
                            input_object_path=input_path,
                            failure=str(exc),
                        )
                    )
    shard_results.sort(key=lambda row: int(row["ordinal"]))
    manifest_body = build_wave_result_manifest(
        descriptor=descriptor,
        shard_results=shard_results,
        public_revision=public_revision,
    )
    manifest_bytes = json.dumps(
        manifest_body, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    uploaded = transport.write_verified_append_only(
        object_path=manifest_path,
        data=manifest_bytes,
        media_type="application/json",
    )
    if failures:
        raise HfDirectWaveError(
            f"wave completed with shard failures: {', '.join(sorted(set(failures)))}"
        )
    return {
        "manifest": manifest_body,
        "result_manifest_hf_ref": uploaded,
        "attempt_count": len(shards),
    }


def wave_descriptor_ref_from_environment() -> dict[str, Any]:
    raw = os.environ.get("PBE_HF_WAVE_DESCRIPTOR_REF", "").strip()
    if not raw:
        raise HfDirectWaveError("PBE_HF_WAVE_DESCRIPTOR_REF is required")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HfDirectWaveError("PBE_HF_WAVE_DESCRIPTOR_REF must be JSON") from exc
    if not isinstance(payload, dict):
        raise HfDirectWaveError("PBE_HF_WAVE_DESCRIPTOR_REF must be a JSON object")
    return validate_hf_bucket_ref(payload)


def expected_manifest_path_from_environment() -> str | None:
    raw = os.environ.get("PBE_HF_WAVE_RESULT_MANIFEST_OBJECT_PATH", "").strip()
    return raw or None


def expected_public_revision_from_environment() -> str:
    raw = os.environ.get("PBE_EXPECTED_PUBLIC_REVISION", "").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", raw):
        raise HfDirectWaveError(
            "PBE_EXPECTED_PUBLIC_REVISION must be a 40-char git sha"
        )
    return raw


@dataclass(frozen=True)
class SerializedWaveDescriptor:
    body: dict[str, Any]
    bytes: bytes
    hf_ref: dict[str, Any]

    @classmethod
    def build(
        cls,
        *,
        logical_run_id: str,
        wave_id: str,
        manifest_digest: str,
        public_revision: str,
        bucket_prefix: str,
        job: JobSpec,
        wave: WaveSpec,
        shards: tuple[ShardSpec, ...],
    ) -> SerializedWaveDescriptor:
        body = {
            "schema_version": HF_WAVE_DESCRIPTOR_SCHEMA,
            "public_revision": public_revision,
            "logical_run_id": logical_run_id,
            "wave_id": wave_id,
            "manifest_digest": manifest_digest,
            "job": job.model_dump(mode="json"),
            "wave": wave.model_dump(mode="json"),
            "shards": [shard.model_dump(mode="json") for shard in shards],
        }
        validate_wave_descriptor_payload(body)
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        object_path = wave_descriptor_object_path(
            bucket_prefix=bucket_prefix,
            public_revision=public_revision,
            manifest_digest=manifest_digest,
            wave_id=wave_id,
        )
        digest = sha256(encoded).hexdigest()
        hf_ref = validate_hf_bucket_ref(
            {
                "schema_version": HF_BUCKET_REF_SCHEMA,
                "bucket_id": HF_BUCKET_ID,
                "object_path": object_path,
                "sha256": digest,
                "size_bytes": len(encoded),
                "media_type": "application/vnd.pbe.hf-direct-wave-descriptor.v1",
            }
        )
        return cls(body=body, bytes=encoded, hf_ref=hf_ref)
