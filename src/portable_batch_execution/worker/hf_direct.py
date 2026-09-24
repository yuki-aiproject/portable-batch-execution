"""HF-direct wave execution helpers (no Private Data Plane byte I/O)."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

from portable_batch_execution.contracts import ArtifactRef
from portable_batch_execution.packs.replay_reduction.paired_fill_ledger_canonical_finalize import (
    PairedFillLedgerCanonicalizeCrossShardState,
    publish_paired_fill_ledger_canonical_finalize_artifacts,
)
from portable_batch_execution.packs.replay_reduction.paired_fill_reduce import (
    publish_paired_fill_reduce_artifacts,
)
from portable_batch_execution.packs.replay_reduction.trade_path_scenario_evaluate import (
    REQUEST_SCHEMA_VERSION,
)
from portable_batch_execution.packs.replay_reduction.trade_path_scenario_evaluate_fixed_set import (
    FIXED_SET_OPERATION,
    FIXED_SET_RESULT_SCHEMA_VERSION,
)
from portable_batch_execution.transport.hf_bucket import (
    HfBucketTransport,
    HfBucketTransportError,
    artifact_ref_from_hf_object,
    artifact_ref_to_hf_bucket_ref,
    parse_hf_object_uri,
    validate_hf_bucket_ref,
)
from portable_batch_execution.worker.hf_dataset_resolve import (
    HfDatasetResolveError,
    artifact_ref_is_pinned_hf_dataset_resolve,
    download_verified_pinned_resolve,
    pinned_resolve_input_digest,
    validate_parquet_hf_resolve_ref,
)


class HfDirectExecutionError(RuntimeError):
    """Fail-closed HF-direct execution error."""


PAIRED_FILL_OPERATION = "replay.paired_fill_reduce"
PAIRED_FILL_LEDGER_CANONICAL_FINALIZE_OPERATION = (
    "replay.paired_fill_ledger_canonical_finalize"
)
HF_DIRECT_OPERATIONS = frozenset(
    {
        FIXED_SET_OPERATION,
        PAIRED_FILL_OPERATION,
        PAIRED_FILL_LEDGER_CANONICAL_FINALIZE_OPERATION,
    }
)
HF_DIRECT_TRANSPORT_PROFILES = frozenset({"hf_bucket_direct", "hf_direct"})


def jpx_hf_direct_enabled() -> bool:
    return os.environ.get("PBE_JPX_HF_DIRECT", "").strip() == "1"


def hf_direct_enabled() -> bool:
    if os.environ.get("PBE_HF_DIRECT", "").strip() == "1":
        return True
    return jpx_hf_direct_enabled()


def assert_no_private_data_plane_dependency() -> None:
    if os.environ.get("PBE_PRIVATE_DATA_PLANE_BASE_URL"):
        raise HfDirectExecutionError(
            "JPX HF-direct mode must not depend on Private Data Plane URLs"
        )
    if os.environ.get("PBE_PRIVATE_DATA_PLANE_BEARER_TOKEN"):
        raise HfDirectExecutionError(
            "JPX HF-direct mode must not depend on Private Data Plane credentials"
        )


def artifact_ref_is_hf_bucket(ref: ArtifactRef) -> bool:
    try:
        parse_hf_object_uri(ref.uri)
        return True
    except HfBucketTransportError:
        return False


def hf_bucket_ref_from_artifact(ref: ArtifactRef) -> dict[str, Any]:
    return validate_hf_bucket_ref(
        {
            "schema_version": "pbe.hf-bucket-artifact-ref.v1",
            "bucket_id": "yamauchiJP/system-trading-data",
            "object_path": parse_hf_object_uri(ref.uri),
            "sha256": ref.sha256.removeprefix("sha256:"),
            "size_bytes": int(ref.size_bytes or 0),
            "media_type": str(ref.media_type or "application/json"),
        }
    )


def read_verified_input_batch(
    transport: HfBucketTransport, ref: ArtifactRef
) -> tuple[bytes, dict[str, Any]]:
    payload = transport.read_verified(hf_bucket_ref_from_artifact(ref))
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise HfDirectExecutionError("input batch is not valid JSON") from exc
    if parsed.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise HfDirectExecutionError("input batch schema mismatch")
    return payload, parsed


def fixed_set_result_object_path(
    *,
    bucket_prefix: str,
    public_revision: str,
    manifest_digest: str,
    batch_id: str,
) -> str:
    revision = public_revision.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise HfDirectExecutionError("public_revision must be a git commit sha")
    tag = sha256(
        f"{revision}|{manifest_digest}|{batch_id}|fixed-set".encode()
    ).hexdigest()[:16]
    prefix = bucket_prefix.rstrip("/")
    return (
        f"{prefix}/results/{revision}/{manifest_digest}/{batch_id}-{tag}.json"
    ).lstrip("/")


def derive_result_object_path_from_input_object_path(
    input_object_path: str, *, public_revision: str
) -> str:
    pattern = re.compile(
        r"^(?P<prefix>.+/normalized/(?P<digest>[^/]+)/)(?P<batch_id>batch-\d{5})-(?P<tag>[0-9a-f]{16})\.json$"
    )
    match = pattern.match(input_object_path.lstrip("/"))
    if match is None:
        raise HfDirectExecutionError("cannot derive result path from input object path")
    prefix = match.group("prefix").split("/normalized/")[0].rstrip("/") + "/"
    return fixed_set_result_object_path(
        bucket_prefix=prefix,
        public_revision=public_revision,
        manifest_digest=match.group("digest"),
        batch_id=match.group("batch_id"),
    )


def execute_hf_direct_trade_path_fixed_set_shard(
    *,
    transport: HfBucketTransport,
    replay_pack,
    job,
    shard,
    input_ref: ArtifactRef,
    output_object_path: str,
) -> tuple[ArtifactRef, dict[str, Any]]:
    if job.operation != FIXED_SET_OPERATION:
        raise HfDirectExecutionError("unsupported HF-direct operation")
    if job.operation_params.get("transport_profile") != "hf_bucket_direct":
        raise HfDirectExecutionError("transport_profile mismatch")
    assert_no_private_data_plane_dependency()

    batch_bytes, parsed_batch = read_verified_input_batch(transport, input_ref)
    resolved_output_path = output_object_path.lstrip("/")
    if not resolved_output_path:
        public_revision = str(job.provenance.revision or "")
        resolved_output_path = derive_result_object_path_from_input_object_path(
            parse_hf_object_uri(input_ref.uri),
            public_revision=public_revision,
        )
    result_payload = replay_pack.execute(
        job,
        shard,
        job.operation_params,
        {
            "batch": parsed_batch,
            "encoded_size": len(batch_bytes),
            "operation": job.operation,
        },
    )
    output = bytes(result_payload.get("result_json_bytes") or b"")
    if not output:
        raise HfDirectExecutionError("fixed set result bytes missing")
    if result_payload.get("schema_version") != FIXED_SET_RESULT_SCHEMA_VERSION:
        raise HfDirectExecutionError("fixed set result schema mismatch")

    uploaded = transport.write_verified_append_only(
        object_path=resolved_output_path,
        data=output,
        media_type="application/json",
    )
    out_ref = artifact_ref_from_hf_object(
        object_path=uploaded["object_path"],
        sha256_hex=uploaded["sha256"],
        size_bytes=uploaded["size_bytes"],
        media_type=uploaded["media_type"],
    )
    summary = dict(result_payload.get("summary") or {})
    summary["output_hf_ref"] = artifact_ref_to_hf_bucket_ref(out_ref)
    return out_ref, summary


def broker_input_digest_from_hf_ref(ref: Mapping[str, Any]) -> str:
    validated = validate_hf_bucket_ref(ref)
    material = json.dumps(validated, sort_keys=True, separators=(",", ":"))
    return sha256(material.encode("utf-8")).hexdigest()


def validate_hf_direct_transport_profile(job) -> None:
    profile = job.operation_params.get("transport_profile")
    if job.operation == FIXED_SET_OPERATION:
        if profile != "hf_bucket_direct":
            raise HfDirectExecutionError("transport_profile mismatch")
        return
    if job.operation == PAIRED_FILL_OPERATION:
        if profile not in HF_DIRECT_TRANSPORT_PROFILES:
            raise HfDirectExecutionError("transport_profile mismatch")
        return
    if job.operation == PAIRED_FILL_LEDGER_CANONICAL_FINALIZE_OPERATION:
        if profile not in HF_DIRECT_TRANSPORT_PROFILES:
            raise HfDirectExecutionError("transport_profile mismatch")
        return
    raise HfDirectExecutionError("unsupported HF-direct operation")


def paired_fill_shard_output_object_path(
    *,
    bucket_prefix: str,
    public_revision: str,
    manifest_digest: str,
    wave_id: str,
    shard_id: str,
    role: str,
    extension: str,
) -> str:
    revision = public_revision.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise HfDirectExecutionError("public_revision must be a git commit sha")
    tag = sha256(
        f"{revision}|{manifest_digest}|{wave_id}|{shard_id}|{role}|paired-fill".encode()
    ).hexdigest()[:16]
    prefix = bucket_prefix.rstrip("/")
    return (
        f"{prefix}/paired-fill-results/{revision}/{manifest_digest}/"
        f"{wave_id}/{shard_id}-{role}-{tag}.{extension}"
    ).lstrip("/")


def paired_fill_ledger_canonical_finalize_shard_output_object_path(
    *,
    bucket_prefix: str,
    public_revision: str,
    manifest_digest: str,
    wave_id: str,
    shard_id: str,
    role: str,
    extension: str,
) -> str:
    revision = public_revision.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise HfDirectExecutionError("public_revision must be a git commit sha")
    tag = sha256(
        f"{revision}|{manifest_digest}|{wave_id}|{shard_id}|{role}|paired-ledger-canonical".encode()
    ).hexdigest()[:16]
    prefix = bucket_prefix.rstrip("/")
    return (
        f"{prefix}/paired-ledger-canonical-results/{revision}/{manifest_digest}/"
        f"{wave_id}/{shard_id}-{role}-{tag}.{extension}"
    ).lstrip("/")


def paired_fill_ledger_canonical_finalize_shard_input_digest(shard) -> str:
    if len(shard.input_refs) != 3:
        raise HfDirectExecutionError(
            "paired-ledger canonical finalize shard requires ledger, metadata, and request"
        )
    ledger_ref, metadata_ref, request_ref = shard.input_refs
    material = {
        "ledger_sha256": ledger_ref.sha256,
        "ledger_size_bytes": ledger_ref.size_bytes,
        "metadata_sha256": metadata_ref.sha256,
        "metadata_size_bytes": metadata_ref.size_bytes,
        "request_sha256": request_ref.sha256,
        "request_size_bytes": request_ref.size_bytes,
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return sha256(encoded).hexdigest()


def paired_fill_shard_input_digest(shard) -> str:
    parquet_refs = shard.input_refs[:-1]
    request_ref = shard.input_refs[-1]
    material = {
        "parquet": pinned_resolve_input_digest(parquet_refs),
        "request_sha256": request_ref.sha256,
        "request_size_bytes": request_ref.size_bytes,
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return sha256(encoded).hexdigest()


class _SequentialHfBucketPlane:
    def __init__(self, transport: HfBucketTransport, object_paths: list[str]) -> None:
        self._transport = transport
        self._object_paths = object_paths
        self._index = 0

    def write(self, data: bytes, media_type: str) -> ArtifactRef:
        if self._index >= len(self._object_paths):
            raise HfDirectExecutionError("unexpected paired-fill output slot")
        object_path = self._object_paths[self._index]
        self._index += 1
        uploaded = self._transport.write_verified_append_only(
            object_path=object_path,
            data=data,
            media_type=media_type,
        )
        return artifact_ref_from_hf_object(
            object_path=uploaded["object_path"],
            sha256_hex=uploaded["sha256"],
            size_bytes=uploaded["size_bytes"],
            media_type=uploaded["media_type"],
        )


def _artifact_ref_matches_bytes(data: bytes, ref: ArtifactRef) -> bool:
    digest = sha256(data).hexdigest()
    return ref.sha256 == f"sha256:{digest}" and int(ref.size_bytes or -1) == len(data)


def execute_hf_direct_paired_fill_reduce_shard(
    *,
    transport: HfBucketTransport,
    replay_pack,
    job,
    shard,
    bucket_prefix: str,
    public_revision: str,
    manifest_digest: str,
    wave_id: str,
) -> tuple[ArtifactRef, list[dict[str, Any]], dict[str, Any]]:
    if job.operation != PAIRED_FILL_OPERATION:
        raise HfDirectExecutionError("unsupported HF-direct operation")
    validate_hf_direct_transport_profile(job)
    assert_no_private_data_plane_dependency()
    if len(shard.input_refs) < 2:
        raise HfDirectExecutionError(
            "paired-fill shard requires parquet inputs and request"
        )
    parquet_refs = shard.input_refs[:-1]
    request_ref = shard.input_refs[-1]
    if not artifact_ref_is_hf_bucket(request_ref):
        raise HfDirectExecutionError("paired-fill request must be an HF bucket ref")
    for ref in parquet_refs:
        if not artifact_ref_is_pinned_hf_dataset_resolve(ref):
            try:
                validate_parquet_hf_resolve_ref(ref)
            except HfDatasetResolveError as exc:
                raise HfDirectExecutionError(str(exc)) from exc

    request_payload = transport.read_verified(hf_bucket_ref_from_artifact(request_ref))
    try:
        parsed_request = json.loads(request_payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise HfDirectExecutionError("paired-fill request is not valid JSON") from exc

    output_paths: list[str] = []
    with tempfile.TemporaryDirectory() as temporary:
        paths: list[Path] = []
        for index, ref in enumerate(parquet_refs):
            dest = Path(temporary) / f"input-{index}.parquet"
            try:
                download_verified_pinned_resolve(ref, dest)
            except HfDatasetResolveError as exc:
                raise HfDirectExecutionError(str(exc)) from exc
            paths.append(dest)
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
        ledger_bytes = bytes(result_payload.get("ledger_parquet_bytes") or b"")
        if ledger_bytes:
            output_paths.append(
                paired_fill_shard_output_object_path(
                    bucket_prefix=bucket_prefix,
                    public_revision=public_revision,
                    manifest_digest=manifest_digest,
                    wave_id=wave_id,
                    shard_id=shard.shard_id,
                    role="ledger",
                    extension="parquet",
                )
            )
        output_paths.append(
            paired_fill_shard_output_object_path(
                bucket_prefix=bucket_prefix,
                public_revision=public_revision,
                manifest_digest=manifest_digest,
                wave_id=wave_id,
                shard_id=shard.shard_id,
                role="metadata",
                extension="json",
            )
        )
        plane = _SequentialHfBucketPlane(transport, output_paths)
        _metadata_bytes, published_refs = publish_paired_fill_reduce_artifacts(
            plane,
            result_payload,
            artifact_ref_matches_bytes=_artifact_ref_matches_bytes,
            shard_stage_failure=HfDirectExecutionError,
        )

    metadata_ref = published_refs[0]
    output_hf_refs = [artifact_ref_to_hf_bucket_ref(ref) for ref in published_refs]
    summary = dict(result_payload.get("summary") or {})
    summary["output_hf_ref"] = artifact_ref_to_hf_bucket_ref(metadata_ref)
    summary["output_hf_refs"] = output_hf_refs
    return metadata_ref, output_hf_refs, summary


def execute_hf_direct_paired_fill_ledger_canonical_finalize_shard(
    *,
    transport: HfBucketTransport,
    replay_pack,
    job,
    shard,
    bucket_prefix: str,
    public_revision: str,
    manifest_digest: str,
    wave_id: str,
) -> tuple[ArtifactRef, list[dict[str, Any]], dict[str, Any]]:
    if job.operation != PAIRED_FILL_LEDGER_CANONICAL_FINALIZE_OPERATION:
        raise HfDirectExecutionError("unsupported HF-direct operation")
    validate_hf_direct_transport_profile(job)
    assert_no_private_data_plane_dependency()
    if len(shard.input_refs) != 3:
        raise HfDirectExecutionError(
            "paired-ledger canonical finalize shard requires ledger, metadata, and request"
        )
    ledger_ref, metadata_ref, request_ref = shard.input_refs
    for ref in (ledger_ref, metadata_ref, request_ref):
        if not artifact_ref_is_hf_bucket(ref):
            raise HfDirectExecutionError(
                "paired-ledger canonical finalize inputs must be HF bucket refs"
            )

    ledger_bytes = transport.read_verified(hf_bucket_ref_from_artifact(ledger_ref))
    reducer_metadata_bytes = transport.read_verified(
        hf_bucket_ref_from_artifact(metadata_ref)
    )
    request_payload = transport.read_verified(hf_bucket_ref_from_artifact(request_ref))
    try:
        parsed_request = json.loads(request_payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise HfDirectExecutionError(
            "paired-ledger canonical finalize request is not valid JSON"
        ) from exc
    if not isinstance(parsed_request, dict):
        raise HfDirectExecutionError(
            "paired-ledger canonical finalize request must be a JSON object"
        )
    cross_shard_state = PairedFillLedgerCanonicalizeCrossShardState.from_dict(
        parsed_request.pop("cross_shard_state_in", None)
    )

    result_payload = replay_pack.execute(
        job,
        shard,
        job.operation_params,
        {
            "ledger_bytes": ledger_bytes,
            "reducer_metadata_bytes": reducer_metadata_bytes,
            "request": parsed_request,
            "cross_shard_state": cross_shard_state,
            "operation": job.operation,
        },
    )

    canonical_ledger_bytes = bytes(result_payload.get("canonical_ledger_bytes") or b"")
    output_paths: list[str] = []
    if canonical_ledger_bytes:
        output_paths.append(
            paired_fill_ledger_canonical_finalize_shard_output_object_path(
                bucket_prefix=bucket_prefix,
                public_revision=public_revision,
                manifest_digest=manifest_digest,
                wave_id=wave_id,
                shard_id=shard.shard_id,
                role="ledger",
                extension="parquet",
            )
        )
    output_paths.append(
        paired_fill_ledger_canonical_finalize_shard_output_object_path(
            bucket_prefix=bucket_prefix,
            public_revision=public_revision,
            manifest_digest=manifest_digest,
            wave_id=wave_id,
            shard_id=shard.shard_id,
            role="metadata",
            extension="json",
        )
    )
    plane = _SequentialHfBucketPlane(transport, output_paths)
    _metadata_bytes, published_refs = publish_paired_fill_ledger_canonical_finalize_artifacts(
        plane,
        result_payload,
        artifact_ref_matches_bytes=_artifact_ref_matches_bytes,
        shard_stage_failure=HfDirectExecutionError,
    )

    metadata_ref_out = published_refs[0]
    output_hf_refs = [artifact_ref_to_hf_bucket_ref(ref) for ref in published_refs]
    summary = dict(result_payload.get("summary") or {})
    summary["output_hf_ref"] = artifact_ref_to_hf_bucket_ref(metadata_ref_out)
    summary["output_hf_refs"] = output_hf_refs
    return metadata_ref_out, output_hf_refs, summary
