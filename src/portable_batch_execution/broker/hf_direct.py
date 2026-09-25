"""HF-direct Unix broker request/response (wave-level metadata-only artifact refs)."""

from __future__ import annotations

import json
import re
from hashlib import sha256
from typing import Any, Literal

from pydantic import Field, field_validator

from portable_batch_execution.broker.config import opaque_request_id
from portable_batch_execution.broker.protocol import _StrictModel, _walk_forbidden
from portable_batch_execution.transport.hf_bucket import validate_hf_bucket_ref

_HF_DIRECT_REQUEST_SCHEMA = "pbe.a1-unix-broker.hf-direct-request.v1"
_HF_DIRECT_RESPONSE_SCHEMA = "pbe.a1-unix-broker.hf-direct-response.v1"
_FORBIDDEN_BYTE_KEYS = frozenset({"input_b64", "output_b64", "result_b64"})


class BrokerHfDirectExecuteRequest(_StrictModel):
    schema_version: Literal["pbe.a1-unix-broker.hf-direct-request.v1"] = (
        _HF_DIRECT_REQUEST_SCHEMA
    )
    request_id: str
    pack: Literal["replay-batch"]
    operation: Literal[
        "replay.trade_path_scenario_evaluate_fixed_set",
        "replay.paired_fill_reduce",
        "replay.paired_fill_ledger_canonical_finalize",
    ]
    operation_params: dict[str, Any] = Field(default_factory=dict)
    logical_run_id: str
    wave_id: str
    public_revision: str
    wave_descriptor_hf_ref: dict[str, Any]
    result_manifest_object_path: str
    shard_count: int = Field(ge=1, le=16)
    max_parallel: int = Field(ge=1, le=16)

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        return opaque_request_id(value)

    @field_validator("logical_run_id", "wave_id")
    @classmethod
    def validate_opaque_identity(cls, value: str) -> str:
        if not value or len(value) > 128 or any(c in value for c in "/\\?#"):
            raise ValueError("identity must be opaque")
        return value

    @field_validator("public_revision")
    @classmethod
    def validate_public_revision(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError("public_revision must be a git commit sha")
        return value

    @field_validator("result_manifest_object_path")
    @classmethod
    def validate_manifest_path(cls, value: str) -> str:
        normalized = value.lstrip("/")
        if not normalized or ".." in normalized.split("/"):
            raise ValueError("invalid result manifest object path")
        return normalized

    @field_validator("operation_params")
    @classmethod
    def validate_operation_params(cls, value: dict[str, Any]) -> dict[str, Any]:
        _walk_forbidden(value)
        return value

    @field_validator("wave_descriptor_hf_ref")
    @classmethod
    def validate_wave_descriptor_hf_ref(cls, value: dict[str, Any]) -> dict[str, Any]:
        return validate_hf_bucket_ref(value)


class BrokerHfDirectExecuteResponse(_StrictModel):
    schema_version: Literal["pbe.a1-unix-broker.hf-direct-response.v1"] = (
        _HF_DIRECT_RESPONSE_SCHEMA
    )
    request_id: str
    status: Literal["succeeded", "exhausted", "failed"]
    error_code: str | None = None
    logical_run_id: str | None = None
    wave_id: str | None = None
    execution_id: str | None = None
    input_digest: str | None = None
    execution_fingerprint: str | None = None
    result_manifest_object_path: str | None = None
    result_manifest_hf_ref: dict[str, Any] | None = None


def parse_hf_direct_request(payload: object) -> BrokerHfDirectExecuteRequest:
    if not isinstance(payload, dict):
        raise TypeError("request must be a JSON object")
    if payload.get("schema_version") != _HF_DIRECT_REQUEST_SCHEMA:
        raise TypeError("not an HF-direct broker request")
    for key in _FORBIDDEN_BYTE_KEYS:
        if key in payload:
            raise ValueError(f"{key} forbidden in HF-direct mode")
    if "input_hf_ref" in payload or "artifact_url" in payload:
        raise ValueError("legacy per-batch HF-direct fields are forbidden")
    return BrokerHfDirectExecuteRequest.model_validate(payload)


def reject_legacy_byte_fields(payload: object) -> None:
    if not isinstance(payload, dict):
        return
    if payload.get("transport_profile") == "hf_bucket_direct" or payload.get(
        "schema_version"
    ) in {_HF_DIRECT_REQUEST_SCHEMA, _HF_DIRECT_RESPONSE_SCHEMA}:
        for key in _FORBIDDEN_BYTE_KEYS.union({"input_hf_ref", "artifact_url"}):
            if key in payload:
                raise ValueError(f"{key} forbidden in HF-direct mode")


def hf_direct_response_to_json(response: BrokerHfDirectExecuteResponse) -> bytes:
    return (response.model_dump_json(exclude_none=True) + "\n").encode("utf-8")


def hf_direct_wave_input_digest(
    *,
    wave_descriptor_hf_ref: dict[str, Any],
    result_manifest_object_path: str,
    logical_run_id: str,
    wave_id: str,
    public_revision: str,
    shard_count: int,
) -> str:
    material = json.dumps(
        {
            "wave_descriptor_hf_ref": validate_hf_bucket_ref(wave_descriptor_hf_ref),
            "result_manifest_object_path": result_manifest_object_path.lstrip("/"),
            "logical_run_id": logical_run_id,
            "wave_id": wave_id,
            "public_revision": public_revision,
            "shard_count": shard_count,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(material.encode("utf-8")).hexdigest()
