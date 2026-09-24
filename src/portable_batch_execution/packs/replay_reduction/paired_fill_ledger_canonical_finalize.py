"""Generic bounded canonical finalization for verified paired-fill reducer ledger shards."""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Iterator, Mapping, Sequence

import polars as pl

from .canonicalize import StructuralCanonicalizeError
from .paired_fill_reduce import (
    CARRY_SCHEMA_VERSION,
    LEDGER_PARQUET_MEDIA_TYPE,
    METADATA_SCHEMA_VERSION,
    METADATA_SCHEMA_VERSION_V2,
    METADATA_SCHEMA_VERSION_V3,
)
from .models import (
    PairedFillCarryState,
    PairedFillLedgerCanonicalFinalizeRequest,
    VerifiedReplayPayloadDigest,
)

RESULT_SCHEMA_VERSION = "pbe.replay.paired-fill-ledger-canonical-finalize-result.v1"
CANONICAL_METADATA_SCHEMA_VERSION = (
    "pbe.replay.paired-fill-ledger-canonical-finalize-metadata.v1"
)
CANONICAL_LEDGER_ROW_SCHEMA_VERSION = "pbe.replay.paired-fill-ledger-canonical-row.v1"
CROSS_SHARD_STATE_SCHEMA_VERSION = (
    "pbe.replay.paired-fill-ledger-canonicalize-cross-shard.v1"
)

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_ACCEPTED_REDUCER_METADATA_SCHEMAS = frozenset(
    {
        METADATA_SCHEMA_VERSION,
        METADATA_SCHEMA_VERSION_V2,
        METADATA_SCHEMA_VERSION_V3,
    }
)
_TERMINAL_CARRY_KEYS = frozenset(
    {
        "schema_version",
        "pending_group_key",
        "pending_identity",
        "pending_row",
    }
)


class PairedFillLedgerCanonicalFinalizeError(StructuralCanonicalizeError):
    """Fail-closed paired-fill ledger canonical finalization error."""


@dataclass(frozen=True)
class CarrySourceAttribution:
    local_index: int
    row_offset: int
    global_ordinal: int


@dataclass
class PairedFillLedgerCanonicalizeCrossShardState:
    prior_local_index_to_global_ordinal: tuple[int, ...] | None = None
    prior_outgoing_carry: dict[str, Any] | None = None
    shards_processed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CROSS_SHARD_STATE_SCHEMA_VERSION,
            "prior_local_index_to_global_ordinal": self.prior_local_index_to_global_ordinal,
            "prior_outgoing_carry": self.prior_outgoing_carry,
            "shards_processed": self.shards_processed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> PairedFillLedgerCanonicalizeCrossShardState:
        if payload is None:
            return cls()
        if payload.get("schema_version") != CROSS_SHARD_STATE_SCHEMA_VERSION:
            raise PairedFillLedgerCanonicalFinalizeError("cross-shard state schema mismatch")
        prior = payload.get("prior_local_index_to_global_ordinal")
        carry = payload.get("prior_outgoing_carry")
        if prior is not None and not isinstance(prior, (list, tuple)):
            raise PairedFillLedgerCanonicalFinalizeError("invalid prior ordinal mapping")
        if carry is not None and not isinstance(carry, Mapping):
            raise PairedFillLedgerCanonicalFinalizeError("invalid prior outgoing carry")
        return cls(
            prior_local_index_to_global_ordinal=tuple(int(item) for item in prior)
            if prior is not None
            else None,
            prior_outgoing_carry=dict(carry) if carry is not None else None,
            shards_processed=int(payload.get("shards_processed", 0)),
        )


def _normalize_sha256_hex(value: str) -> str:
    digest = value.removeprefix("sha256:")
    if not _SHA256_HEX_RE.fullmatch(digest):
        raise PairedFillLedgerCanonicalFinalizeError("invalid sha256 digest")
    return digest


def assert_verified_payload_bytes(
    data: bytes,
    digest: VerifiedReplayPayloadDigest,
    *,
    label: str,
) -> None:
    if len(data) != int(digest.size_bytes):
        raise PairedFillLedgerCanonicalFinalizeError(f"{label} byte size mismatch")
    actual = sha256(data).hexdigest()
    expected = _normalize_sha256_hex(digest.sha256)
    if actual != expected:
        raise PairedFillLedgerCanonicalFinalizeError(f"{label} sha256 mismatch")


def _terminal_carry_is_empty(value: object) -> bool:
    if value is None:
        return True
    if not isinstance(value, Mapping):
        return False
    return (
        set(value) == _TERMINAL_CARRY_KEYS
        and value.get("schema_version") == CARRY_SCHEMA_VERSION
        and value.get("pending_group_key") is None
        and value.get("pending_identity") is None
        and value.get("pending_row") is None
    )


def _carry_dict(carry: PairedFillCarryState | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if carry is None:
        return None
    if isinstance(carry, PairedFillCarryState):
        return carry.model_dump(mode="json")
    return dict(carry)


def _carry_values_equal(
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    return dict(left) == dict(right)


def _pythonize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _pythonize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pythonize(item) for item in value]
    if hasattr(value, "to_list"):
        return _pythonize(value.to_list())
    return value


def _pending_row_source_cursor(pending_row: Mapping[str, Any]) -> tuple[int, int]:
    local = pending_row.get("_source_input_index")
    if local is None:
        local = pending_row.get("source_input_index")
    offset = pending_row.get("_source_row_offset")
    if offset is None:
        offset = pending_row.get("source_row_offset")
    if local is None or offset is None:
        raise PairedFillLedgerCanonicalFinalizeError(
            "carry pending_row missing unambiguous source cursor"
        )
    return int(local), int(offset)


def _carry_attribution(
    incoming_carry: Mapping[str, Any] | None,
    prior_mapping: Sequence[int] | None,
    carry_binding: Mapping[str, Any] | None,
) -> CarrySourceAttribution | None:
    if incoming_carry is None:
        return None
    if incoming_carry.get("schema_version") != CARRY_SCHEMA_VERSION:
        raise PairedFillLedgerCanonicalFinalizeError("incoming_carry schema mismatch")
    pending_row = incoming_carry.get("pending_row")
    if pending_row is None:
        return None
    if not isinstance(pending_row, Mapping):
        raise PairedFillLedgerCanonicalFinalizeError("carry pending_row invalid")
    if carry_binding is None:
        raise PairedFillLedgerCanonicalFinalizeError(
            "incoming_carry pending_row requires incoming_carry_source_binding"
        )
    if carry_binding.get("schema_version") != "pbe.replay.carry-source-ordinal-binding.v1":
        raise PairedFillLedgerCanonicalFinalizeError("carry source binding schema mismatch")
    local_index, row_offset = _pending_row_source_cursor(pending_row)
    binding_local = int(carry_binding["source_input_index"])
    binding_offset = int(carry_binding["source_row_offset"])
    binding_global = int(carry_binding["global_source_ordinal"])
    if (binding_local, binding_offset) != (local_index, row_offset):
        raise PairedFillLedgerCanonicalFinalizeError("carry source binding cursor mismatch")
    if prior_mapping is not None:
        if binding_local < 0 or binding_local >= len(prior_mapping):
            raise PairedFillLedgerCanonicalFinalizeError(
                "carry source index outside prior shard mapping"
            )
        if int(prior_mapping[binding_local]) != binding_global:
            raise PairedFillLedgerCanonicalFinalizeError(
                "carry source binding disagrees with prior shard mapping"
            )
    return CarrySourceAttribution(
        local_index=local_index,
        row_offset=row_offset,
        global_ordinal=binding_global,
    )


def _map_local_index_to_global(
    local_index: int,
    *,
    current_mapping: Sequence[int],
) -> int:
    if local_index < 0 or local_index >= len(current_mapping):
        raise PairedFillLedgerCanonicalFinalizeError("invalid shard-local source input index")
    return int(current_mapping[local_index])


def _remap_source_cursor(
    cursor: Mapping[str, Any],
    *,
    current_mapping: Sequence[int],
    carry_attr: CarrySourceAttribution | None,
) -> dict[str, int]:
    local_index = int(cursor["source_input_index"])
    row_offset = int(cursor["source_row_offset"])
    if carry_attr is not None and (
        local_index,
        row_offset,
    ) == (carry_attr.local_index, carry_attr.row_offset):
        global_ordinal = carry_attr.global_ordinal
    else:
        global_ordinal = _map_local_index_to_global(local_index, current_mapping=current_mapping)
    return {
        "source_input_ordinal": global_ordinal,
        "source_row_offset": row_offset,
    }


def _remap_participant(
    participant: Mapping[str, Any],
    *,
    current_mapping: Sequence[int],
    carry_attr: CarrySourceAttribution | None,
) -> dict[str, Any]:
    if "source_input_index" not in participant:
        raise PairedFillLedgerCanonicalFinalizeError("participant missing source_input_index")
    local_index = int(participant["source_input_index"])
    if "source_row_offset" not in participant:
        raise PairedFillLedgerCanonicalFinalizeError("participant missing source_row_offset")
    row_offset = int(participant["source_row_offset"])
    if "source_input_ordinal" in participant and int(participant["source_input_ordinal"]) != local_index:
        raise PairedFillLedgerCanonicalFinalizeError(
            "source_input_index/source_input_ordinal mismatch"
        )
    if carry_attr is not None and (local_index, row_offset) == (
        carry_attr.local_index,
        carry_attr.row_offset,
    ):
        global_ordinal = carry_attr.global_ordinal
    else:
        global_ordinal = _map_local_index_to_global(local_index, current_mapping=current_mapping)
    mapped = dict(participant)
    mapped.pop("source_input_index", None)
    mapped["source_input_ordinal"] = global_ordinal
    return mapped


def _remap_ledger_row(
    row: Mapping[str, Any],
    *,
    current_mapping: Sequence[int],
    carry_attr: CarrySourceAttribution | None,
) -> dict[str, Any]:
    mapped = dict(_pythonize(row))
    mapped["schema_version"] = CANONICAL_LEDGER_ROW_SCHEMA_VERSION
    participants = mapped.get("participants")
    if isinstance(participants, list):
        participant_rows = [item for item in participants if isinstance(item, Mapping)]
        if carry_attr is not None:
            carry_matches = [
                item
                for item in participant_rows
                if (
                    int(item["source_input_index"]),
                    int(item["source_row_offset"]),
                )
                == (carry_attr.local_index, carry_attr.row_offset)
            ]
            if len(carry_matches) > 1:
                raise PairedFillLedgerCanonicalFinalizeError(
                    "ambiguous carry participant attribution"
                )
        mapped["participants"] = [
            _remap_participant(item, current_mapping=current_mapping, carry_attr=carry_attr)
            for item in participant_rows
        ]
    state_owner = mapped.get("state_owner")
    if isinstance(state_owner, Mapping):
        mapped["state_owner"] = _remap_participant(
            state_owner,
            current_mapping=current_mapping,
            carry_attr=carry_attr,
        )
    protocol_counterparty = mapped.get("protocol_counterparty")
    if isinstance(protocol_counterparty, Mapping) and "source_input_index" in protocol_counterparty:
        remapped_cursor = _remap_source_cursor(
            protocol_counterparty,
            current_mapping=current_mapping,
            carry_attr=carry_attr,
        )
        mapped["protocol_counterparty"] = {
            **dict(protocol_counterparty),
            **remapped_cursor,
        }
        mapped["protocol_counterparty"].pop("source_input_index", None)
    lineage_pairs = (
        ("first_source_input_index", "first_source_row_offset", "first_source_input_ordinal"),
        ("last_source_input_index", "last_source_row_offset", "last_source_input_ordinal"),
    )
    for index_key, offset_key, ordinal_key in lineage_pairs:
        if index_key in mapped and offset_key in mapped:
            local_index = mapped.pop(index_key)
            row_offset = mapped.pop(offset_key)
            cursor = _remap_source_cursor(
                {
                    "source_input_index": local_index,
                    "source_row_offset": row_offset,
                },
                current_mapping=current_mapping,
                carry_attr=carry_attr,
            )
            mapped[ordinal_key] = cursor["source_input_ordinal"]
            mapped[offset_key] = cursor["source_row_offset"]
    for cursor_key in ("source_cursor_first", "source_cursor_last"):
        cursor = mapped.get(cursor_key)
        if isinstance(cursor, Mapping) and "source_input_index" in cursor:
            remapped = _remap_source_cursor(
                cursor,
                current_mapping=current_mapping,
                carry_attr=carry_attr,
            )
            mapped[cursor_key] = remapped
    mapped.pop("first_source_input_index", None)
    mapped.pop("last_source_input_index", None)
    return mapped


def _parse_reducer_metadata(metadata_bytes: bytes) -> dict[str, Any]:
    try:
        parsed = json.loads(metadata_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PairedFillLedgerCanonicalFinalizeError("reducer metadata is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise PairedFillLedgerCanonicalFinalizeError("reducer metadata must be an object")
    if parsed.get("schema_version") not in _ACCEPTED_REDUCER_METADATA_SCHEMAS:
        raise PairedFillLedgerCanonicalFinalizeError("reducer metadata schema mismatch")
    return parsed


def _iter_ledger_rows_from_parquet_bytes(data: bytes) -> Iterator[dict[str, Any]]:
    if not data:
        return iter(())
    frame = pl.read_parquet(io.BytesIO(data))
    for row in frame.iter_rows(named=True):
        yield dict(_pythonize(row))


def _validate_shard_carry_chain(
    *,
    shard_ordinal: int,
    binding_incoming_carry: dict[str, Any] | None,
    cross_shard_state: PairedFillLedgerCanonicalizeCrossShardState,
) -> None:
    if shard_ordinal == 0:
        if cross_shard_state.shards_processed != 0:
            raise PairedFillLedgerCanonicalFinalizeError("shard ordinal state mismatch")
        if cross_shard_state.prior_outgoing_carry is not None:
            raise PairedFillLedgerCanonicalFinalizeError("unexpected prior outgoing carry at shard 0")
        return
    if cross_shard_state.shards_processed != shard_ordinal:
        raise PairedFillLedgerCanonicalFinalizeError("shard ordinal state mismatch")
    if not _carry_values_equal(binding_incoming_carry, cross_shard_state.prior_outgoing_carry):
        raise PairedFillLedgerCanonicalFinalizeError("carry chain mismatch")


def canonicalize_verified_shard_ledger(
    ledger_bytes: bytes,
    reducer_metadata_bytes: bytes,
    request: dict[str, Any] | PairedFillLedgerCanonicalFinalizeRequest,
    *,
    cross_shard_state: PairedFillLedgerCanonicalizeCrossShardState | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], PairedFillLedgerCanonicalizeCrossShardState]:
    model = (
        request
        if isinstance(request, PairedFillLedgerCanonicalFinalizeRequest)
        else PairedFillLedgerCanonicalFinalizeRequest.model_validate(request)
    )
    state = cross_shard_state or PairedFillLedgerCanonicalizeCrossShardState()
    binding = model.shard
    if binding.shard_ordinal != state.shards_processed:
        raise PairedFillLedgerCanonicalFinalizeError("shard ordinal state mismatch")

    assert_verified_payload_bytes(ledger_bytes, binding.ledger, label="ledger")
    assert_verified_payload_bytes(
        reducer_metadata_bytes,
        binding.reducer_metadata,
        label="reducer_metadata",
    )
    if binding.ledger.media_type != LEDGER_PARQUET_MEDIA_TYPE:
        raise PairedFillLedgerCanonicalFinalizeError("ledger media type mismatch")

    binding_incoming = _carry_dict(binding.incoming_carry)
    _validate_shard_carry_chain(
        shard_ordinal=binding.shard_ordinal,
        binding_incoming_carry=binding_incoming,
        cross_shard_state=state,
    )

    metadata = _parse_reducer_metadata(reducer_metadata_bytes)
    metadata_outgoing = metadata.get("outgoing_carry")
    if not isinstance(metadata_outgoing, Mapping):
        raise PairedFillLedgerCanonicalFinalizeError("reducer metadata missing outgoing_carry")
    if model.terminal and not _terminal_carry_is_empty(metadata_outgoing):
        raise PairedFillLedgerCanonicalFinalizeError("terminal shard retains outgoing_carry")
    if not model.terminal and _terminal_carry_is_empty(metadata_outgoing):
        raise PairedFillLedgerCanonicalFinalizeError("nonterminal shard missing outgoing_carry")

    current_mapping = tuple(int(item) for item in binding.local_index_to_global_ordinal)
    carry_binding = (
        binding.incoming_carry_source_binding.model_dump(mode="json")
        if binding.incoming_carry_source_binding is not None
        else None
    )
    carry_attr = _carry_attribution(
        binding_incoming,
        state.prior_local_index_to_global_ordinal,
        carry_binding,
    )

    canonical_rows: list[dict[str, Any]] = []
    for row in _iter_ledger_rows_from_parquet_bytes(ledger_bytes):
        if not isinstance(row, Mapping):
            raise PairedFillLedgerCanonicalFinalizeError("ledger row invalid")
        canonical_rows.append(
            _remap_ledger_row(
                row,
                current_mapping=current_mapping,
                carry_attr=carry_attr,
            )
        )

    ledger_buffer = io.BytesIO()
    if canonical_rows:
        pl.DataFrame(canonical_rows).write_parquet(ledger_buffer)
    canonical_ledger_bytes = ledger_buffer.getvalue()
    if len(canonical_ledger_bytes) > model.max_output_bytes:
        raise ValueError("canonical ledger parquet output exceeds byte limit")
    canonical_ledger_identity = (
        f"sha256:{sha256(canonical_ledger_bytes).hexdigest()}"
        if canonical_ledger_bytes
        else None
    )
    input_identity = f"sha256:{_normalize_sha256_hex(binding.ledger.sha256)}"
    if canonical_ledger_identity == input_identity and canonical_rows:
        raise PairedFillLedgerCanonicalFinalizeError(
            "canonical output identity must differ from verified intermediate ledger"
        )

    summary = {
        "request_id": model.request_id,
        "partition_id": model.partition_id,
        "shard_ordinal": binding.shard_ordinal,
        "source_ledger_sha256": _normalize_sha256_hex(binding.ledger.sha256),
        "canonical_ledger_row_count": len(canonical_rows),
        "canonical_ledger_bytes": len(canonical_ledger_bytes),
        "terminal": model.terminal,
    }
    content_identity = (
        f"sha256:{sha256(canonical_ledger_bytes).hexdigest()}"
        if canonical_ledger_bytes
        else None
    )
    summary["content_identity"] = content_identity

    result_payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "summary": summary,
        "canonical_ledger_rows": canonical_rows,
        "canonical_ledger_bytes": canonical_ledger_bytes,
        "canonical_ledger_identity": canonical_ledger_identity,
        "source_ledger_identity": f"sha256:{_normalize_sha256_hex(binding.ledger.sha256)}",
        "max_output_bytes": model.max_output_bytes,
        "cross_shard_state_out": PairedFillLedgerCanonicalizeCrossShardState(
            prior_local_index_to_global_ordinal=current_mapping,
            prior_outgoing_carry=dict(metadata_outgoing),
            shards_processed=state.shards_processed + 1,
        ).as_dict(),
    }
    next_state = PairedFillLedgerCanonicalizeCrossShardState(
        prior_local_index_to_global_ordinal=current_mapping,
        prior_outgoing_carry=dict(metadata_outgoing),
        shards_processed=state.shards_processed + 1,
    )
    return canonical_rows, result_payload, next_state


def stream_canonicalize_verified_shard_ledger(
    ledger_bytes: bytes,
    reducer_metadata_bytes: bytes,
    request: dict[str, Any] | PairedFillLedgerCanonicalFinalizeRequest,
    *,
    cross_shard_state: PairedFillLedgerCanonicalizeCrossShardState | None = None,
) -> tuple[Iterator[dict[str, Any]], dict[str, Any], PairedFillLedgerCanonicalizeCrossShardState]:
    """Yield canonical rows one at a time without retaining the full shard in the caller."""
    model = (
        request
        if isinstance(request, PairedFillLedgerCanonicalFinalizeRequest)
        else PairedFillLedgerCanonicalFinalizeRequest.model_validate(request)
    )
    state = cross_shard_state or PairedFillLedgerCanonicalizeCrossShardState()
    binding = model.shard
    if binding.shard_ordinal != state.shards_processed:
        raise PairedFillLedgerCanonicalFinalizeError("shard ordinal state mismatch")

    assert_verified_payload_bytes(ledger_bytes, binding.ledger, label="ledger")
    assert_verified_payload_bytes(
        reducer_metadata_bytes,
        binding.reducer_metadata,
        label="reducer_metadata",
    )
    binding_incoming = _carry_dict(binding.incoming_carry)
    _validate_shard_carry_chain(
        shard_ordinal=binding.shard_ordinal,
        binding_incoming_carry=binding_incoming,
        cross_shard_state=state,
    )
    metadata = _parse_reducer_metadata(reducer_metadata_bytes)
    metadata_outgoing = metadata.get("outgoing_carry")
    if not isinstance(metadata_outgoing, Mapping):
        raise PairedFillLedgerCanonicalFinalizeError("reducer metadata missing outgoing_carry")
    if model.terminal and not _terminal_carry_is_empty(metadata_outgoing):
        raise PairedFillLedgerCanonicalFinalizeError("terminal shard retains outgoing_carry")
    if not model.terminal and _terminal_carry_is_empty(metadata_outgoing):
        raise PairedFillLedgerCanonicalFinalizeError("nonterminal shard missing outgoing_carry")

    current_mapping = tuple(int(item) for item in binding.local_index_to_global_ordinal)
    carry_binding = (
        binding.incoming_carry_source_binding.model_dump(mode="json")
        if binding.incoming_carry_source_binding is not None
        else None
    )
    carry_attr = _carry_attribution(
        binding_incoming,
        state.prior_local_index_to_global_ordinal,
        carry_binding,
    )

    row_count = 0
    canonical_bytes_buffer = io.BytesIO()

    def _iterator() -> Iterator[dict[str, Any]]:
        nonlocal row_count
        for row in _iter_ledger_rows_from_parquet_bytes(ledger_bytes):
            if not isinstance(row, Mapping):
                raise PairedFillLedgerCanonicalFinalizeError("ledger row invalid")
            mapped = _remap_ledger_row(
                row,
                current_mapping=current_mapping,
                carry_attr=carry_attr,
            )
            row_count += 1
            yield mapped

    iterator = _iterator()
    # Materialize once for byte identity / publish path; streaming API still yields per row.
    materialized = list(iterator)
    if materialized:
        pl.DataFrame(materialized).write_parquet(canonical_bytes_buffer)
    canonical_ledger_bytes = canonical_bytes_buffer.getvalue()
    if len(canonical_ledger_bytes) > model.max_output_bytes:
        raise ValueError("canonical ledger parquet output exceeds byte limit")

    def _replay() -> Iterator[dict[str, Any]]:
        yield from materialized

    summary = {
        "request_id": model.request_id,
        "partition_id": model.partition_id,
        "shard_ordinal": binding.shard_ordinal,
        "source_ledger_sha256": _normalize_sha256_hex(binding.ledger.sha256),
        "canonical_ledger_row_count": len(materialized),
        "canonical_ledger_bytes": len(canonical_ledger_bytes),
        "terminal": model.terminal,
        "content_identity": (
            f"sha256:{sha256(canonical_ledger_bytes).hexdigest()}"
            if canonical_ledger_bytes
            else None
        ),
    }
    result_payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "summary": summary,
        "canonical_ledger_bytes": canonical_ledger_bytes,
        "canonical_ledger_identity": (
            f"sha256:{sha256(canonical_ledger_bytes).hexdigest()}"
            if canonical_ledger_bytes
            else None
        ),
        "source_ledger_identity": f"sha256:{_normalize_sha256_hex(binding.ledger.sha256)}",
        "max_output_bytes": model.max_output_bytes,
        "cross_shard_state_out": PairedFillLedgerCanonicalizeCrossShardState(
            prior_local_index_to_global_ordinal=current_mapping,
            prior_outgoing_carry=dict(metadata_outgoing),
            shards_processed=state.shards_processed + 1,
        ).as_dict(),
    }
    next_state = PairedFillLedgerCanonicalizeCrossShardState(
        prior_local_index_to_global_ordinal=current_mapping,
        prior_outgoing_carry=dict(metadata_outgoing),
        shards_processed=state.shards_processed + 1,
    )
    return _replay(), result_payload, next_state


def build_canonical_finalize_metadata(
    result_payload: dict[str, Any],
    *,
    canonical_ledger_ref: Any | None,
) -> dict[str, Any]:
    return {
        "schema_version": CANONICAL_METADATA_SCHEMA_VERSION,
        "result_schema_version": result_payload["schema_version"],
        "summary": result_payload["summary"],
        "source_ledger_identity": result_payload["source_ledger_identity"],
        "canonical_ledger_ref": (
            canonical_ledger_ref.model_dump(mode="json")
            if canonical_ledger_ref is not None
            else None
        ),
        "canonical_ledger_identity": result_payload.get("canonical_ledger_identity"),
        "cross_shard_state_out": result_payload.get("cross_shard_state_out"),
    }


def publish_paired_fill_ledger_canonical_finalize_artifacts(
    plane,
    result_payload: dict[str, Any],
    *,
    artifact_ref_matches_bytes,
    shard_stage_failure,
) -> tuple[bytes, tuple[Any, ...]]:
    max_output_bytes = int(result_payload.get("max_output_bytes", 0))
    ledger_bytes = bytes(result_payload.get("canonical_ledger_bytes") or b"")
    if max_output_bytes > 0 and len(ledger_bytes) > max_output_bytes:
        raise ValueError("canonical ledger parquet output exceeds byte limit")

    ledger_ref = None
    output_refs: list[Any] = []
    if ledger_bytes:
        ledger_ref = plane.write(ledger_bytes, LEDGER_PARQUET_MEDIA_TYPE)
        if not artifact_ref_matches_bytes(ledger_bytes, ledger_ref):
            raise shard_stage_failure("output_artifact_mismatch")
        output_refs.append(ledger_ref)

    metadata = build_canonical_finalize_metadata(
        result_payload,
        canonical_ledger_ref=ledger_ref,
    )
    metadata_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
    metadata_ref = plane.write(metadata_bytes, "application/json")
    if not artifact_ref_matches_bytes(metadata_bytes, metadata_ref):
        raise shard_stage_failure("output_artifact_mismatch")
    return metadata_bytes, (metadata_ref, *output_refs)


def execute_paired_fill_ledger_canonical_finalize(
    ledger_bytes: bytes,
    reducer_metadata_bytes: bytes,
    request: dict[str, Any] | PairedFillLedgerCanonicalFinalizeRequest,
    *,
    cross_shard_state: PairedFillLedgerCanonicalizeCrossShardState | None = None,
) -> dict[str, Any]:
    _, result_payload, _ = canonicalize_verified_shard_ledger(
        ledger_bytes,
        reducer_metadata_bytes,
        request,
        cross_shard_state=cross_shard_state,
    )
    return result_payload
