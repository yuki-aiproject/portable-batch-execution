"""Generic bounded paired-fill reduction with deterministic source order."""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import polars as pl

from .canonicalize import (
    _IDENTITY_SCAN_BATCH,
    StructuralCanonicalizeError,
    _require_columns,
)
from .event_window import _row_is_sentinel, _validate_positive_row
from .models import (
    PAIRED_FILL_MAX_PAIR_SIZE,
    PairedFillReduceRequest,
)
from .nullable_json_projection import apply_nullable_json_projections_to_lazy
from .row_invariants import invariant_columns, validate_row_invariants

RESULT_SCHEMA_VERSION = "pbe.replay.paired-fill-reduce-result.v1"
RESULT_SCHEMA_VERSION_V2 = "pbe.replay.paired-fill-reduce-result.v2"
METADATA_SCHEMA_VERSION = "pbe.replay.paired-fill-reduce-metadata.v1"
METADATA_SCHEMA_VERSION_V2 = "pbe.replay.paired-fill-reduce-metadata.v2"
CARRY_SCHEMA_VERSION = "pbe.replay.paired-fill-reduce-carry.v1"
LEDGER_PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"
_SOURCE_INPUT_INDEX = "_source_input_index"
_SOURCE_ROW_OFFSET = "_source_row_offset"


def _validated(
    request: dict[str, Any] | PairedFillReduceRequest,
) -> PairedFillReduceRequest:
    if isinstance(request, PairedFillReduceRequest):
        return request
    return PairedFillReduceRequest.model_validate(request)


def _group_key(row: dict[str, Any], model: PairedFillReduceRequest) -> tuple[Any, ...]:
    mapping = model.identity_mapping
    identity_col = mapping.identity_source_column
    identity = int(row[identity_col])
    return (*tuple(row[column] for column in mapping.namespace_columns), identity)


def _ledger_identity(group_key: tuple[Any, ...]) -> int:
    return int(group_key[-1])


def _identity_namespace(
    group_key: tuple[Any, ...], model: PairedFillReduceRequest
) -> dict[str, Any] | None:
    columns = model.identity_mapping.namespace_columns
    if not columns:
        return None
    return {column: group_key[index] for index, column in enumerate(columns)}


def _mechanical_quantities(
    start_position: float, signed_execution: float
) -> tuple[float, float, float]:
    pre = float(start_position)
    qty = float(signed_execution)
    if pre == 0.0 or qty == 0.0:
        closing = 0.0
    elif (pre > 0.0 and qty < 0.0) or (pre < 0.0 and qty > 0.0):
        closing = min(abs(pre), abs(qty))
    else:
        closing = 0.0
    opening = abs(qty) - closing
    post = pre + qty
    return opening, closing, post


def _resolve_signed_execution(row: dict[str, Any], pair_mapping) -> float:
    if pair_mapping.signed_execution_column is not None:
        raw = row.get(pair_mapping.signed_execution_column)
        if raw is None:
            raise StructuralCanonicalizeError("signed execution missing")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise StructuralCanonicalizeError("signed execution invalid")
        if not math.isfinite(value):
            raise StructuralCanonicalizeError("signed execution invalid")
        return value
    mapping = pair_mapping.signed_execution_mapping
    if mapping is not None:
        side = row.get(mapping.side_column)
        if side is None:
            raise StructuralCanonicalizeError("signed execution side missing")
        raw_qty = row.get(mapping.quantity_column)
        if raw_qty is None:
            raise StructuralCanonicalizeError("signed execution quantity missing")
        try:
            quantity = float(raw_qty)
        except (TypeError, ValueError):
            raise StructuralCanonicalizeError("signed execution quantity invalid")
        if not math.isfinite(quantity) or quantity <= 0.0:
            raise StructuralCanonicalizeError("signed execution quantity invalid")
        if side == mapping.positive_side_value:
            return quantity
        if side == mapping.negative_side_value:
            return -quantity
        raise StructuralCanonicalizeError("signed execution side invalid")
    spec = pair_mapping.side_size_signed_execution
    if spec is None:
        raise StructuralCanonicalizeError("signed execution mapping missing")
    side = row.get(spec.side_column)
    raw_size = row.get(spec.size_column)
    if raw_size is None:
        raise StructuralCanonicalizeError("signed execution size missing")
    try:
        size = float(raw_size)
    except (TypeError, ValueError):
        raise StructuralCanonicalizeError("signed execution size invalid")
    if not math.isfinite(size):
        raise StructuralCanonicalizeError("signed execution size invalid")
    magnitude = abs(size)
    if side == spec.buy_side_value:
        return magnitude
    if side == spec.sell_side_value:
        return -magnitude
    raise StructuralCanonicalizeError("signed execution side invalid")


def _role_name(
    row: dict[str, Any],
    *,
    pair_role_column: str,
    aggressor_value: Any,
    passive_value: Any,
) -> Literal["aggressor", "passive"] | None:
    actual = row.get(pair_role_column)
    if actual == aggressor_value:
        return "aggressor"
    if actual == passive_value:
        return "passive"
    return None


def _participant_record(
    row: dict[str, Any],
    *,
    role: Literal["aggressor", "passive"],
    pair_mapping,
) -> dict[str, Any]:
    start_position = float(row[pair_mapping.start_position_column])
    signed_execution = _resolve_signed_execution(row, pair_mapping)
    opening, closing, post = _mechanical_quantities(start_position, signed_execution)
    record: dict[str, Any] = {
        "role": role,
        "start_position": start_position,
        "signed_execution": signed_execution,
        "opening_quantity": opening,
        "closing_quantity": closing,
        "post_position": post,
        "source_input_index": int(row[_SOURCE_INPUT_INDEX]),
        "source_row_offset": int(row[_SOURCE_ROW_OFFSET]),
    }
    if pair_mapping.participant_passthrough_columns:
        passthrough: dict[str, Any] = {}
        for column in pair_mapping.participant_passthrough_columns:
            if column not in row:
                raise StructuralCanonicalizeError(
                    "participant passthrough column missing"
                )
            passthrough[column] = row[column]
        record["passthrough"] = passthrough
    for binding in pair_mapping.participant_field_bindings:
        record[binding.output_field] = row.get(binding.source_column)
    return record


def _measurement_core(row: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {field_name: row[field_name] for field_name in fields}


def _cores_match(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> bool:
    if not rows:
        return True
    reference = _measurement_core(rows[0], fields)
    for row in rows[1:]:
        for field_name in fields:
            if row.get(field_name) != reference[field_name]:
                return False
    return True


def _source_cursor(row: dict[str, Any]) -> dict[str, int]:
    return {
        "source_input_index": int(row[_SOURCE_INPUT_INDEX]),
        "source_row_offset": int(row[_SOURCE_ROW_OFFSET]),
    }


def _source_lineage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        rows,
        key=lambda row: (int(row[_SOURCE_INPUT_INDEX]), int(row[_SOURCE_ROW_OFFSET])),
    )
    first = ordered[0]
    last = ordered[-1]
    return {
        "first_source_input_index": int(first[_SOURCE_INPUT_INDEX]),
        "first_source_row_offset": int(first[_SOURCE_ROW_OFFSET]),
        "last_source_input_index": int(last[_SOURCE_INPUT_INDEX]),
        "last_source_row_offset": int(last[_SOURCE_ROW_OFFSET]),
        "source_cursor_first": _source_cursor(first),
        "source_cursor_last": _source_cursor(last),
    }


def _row_matches_transition_marker(row: dict[str, Any], model: PairedFillReduceRequest) -> bool:
    spec = model.state_transition_handling
    if spec is None:
        return False
    return all(row.get(column) == expected for column, expected in spec.marker_exact_match_fields.items())


def _validate_transition_row_invariants(row: dict[str, Any], model: PairedFillReduceRequest) -> None:
    spec = model.state_transition_handling
    if spec is None:
        validate_row_invariants(row, model.row_invariants)
        return
    bypass = set(spec.bypass_row_invariant_columns)
    for invariant in model.row_invariants:
        columns = set(invariant_columns((invariant,)))
        if columns & bypass:
            continue
        validate_row_invariants(row, (invariant,))


def _numeric_close(left: Any, right: float, *, atol: float, rtol: float) -> bool:
    try:
        value = float(left)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and math.isclose(value, right, abs_tol=atol, rel_tol=rtol)


def _state_transition_ledger_row(
    rows: list[dict[str, Any]],
    *,
    group_key: tuple[Any, ...],
    model: PairedFillReduceRequest,
) -> dict[str, Any]:
    spec = model.state_transition_handling
    if spec is None or len(rows) != PAIRED_FILL_MAX_PAIR_SIZE:
        raise StructuralCanonicalizeError("state transition group size invalid")
    if not all(_row_matches_transition_marker(row, model) for row in rows):
        raise StructuralCanonicalizeError("state transition marker mismatch")
    if not _cores_match(rows, model.pair_mapping.measurement_core_fields):
        raise StructuralCanonicalizeError("measurement core fields disagree")
    for field_name in spec.shared_fields:
        if rows[0].get(field_name) != rows[1].get(field_name):
            raise StructuralCanonicalizeError("state transition shared fields disagree")
    for row in rows:
        for field_name in spec.zero_numeric_fields:
            if not _numeric_close(
                row.get(field_name),
                0.0,
                atol=spec.absolute_tolerance,
                rtol=spec.relative_tolerance,
            ):
                raise StructuralCanonicalizeError("state transition zero field invalid")

    role_column = model.pair_mapping.pair_role_column
    by_role = {row.get(role_column): row for row in rows}
    if set(by_role) != {spec.state_owner_role_value, spec.protocol_counterparty_role_value}:
        raise StructuralCanonicalizeError("state transition roles invalid")
    owner = by_role[spec.state_owner_role_value]
    counterparty = by_role[spec.protocol_counterparty_role_value]
    start_column = model.pair_mapping.start_position_column
    owner_start = owner.get(start_column)
    counterparty_start = counterparty.get(start_column)
    if not _numeric_close(
        counterparty_start,
        0.0,
        atol=spec.absolute_tolerance,
        rtol=spec.relative_tolerance,
    ):
        raise StructuralCanonicalizeError("state transition counterparty start invalid")
    try:
        owner_start_num = float(owner_start)
    except (TypeError, ValueError):
        raise StructuralCanonicalizeError("state transition owner start invalid")
    if not math.isfinite(owner_start_num) or math.isclose(
        owner_start_num,
        0.0,
        abs_tol=spec.absolute_tolerance,
        rel_tol=spec.relative_tolerance,
    ):
        raise StructuralCanonicalizeError("state transition owner start invalid")
    owner_signed = _resolve_signed_execution(owner, model.pair_mapping)
    counterparty_signed = _resolve_signed_execution(counterparty, model.pair_mapping)
    if not math.isclose(
        owner_signed,
        -owner_start_num,
        abs_tol=spec.absolute_tolerance,
        rel_tol=spec.relative_tolerance,
    ):
        raise StructuralCanonicalizeError("state transition owner does not flatten")
    if not math.isclose(
        owner_start_num + owner_signed,
        0.0,
        abs_tol=spec.absolute_tolerance,
        rel_tol=spec.relative_tolerance,
    ):
        raise StructuralCanonicalizeError("state transition owner post position invalid")
    if not math.isclose(
        counterparty_signed,
        -owner_signed,
        abs_tol=spec.absolute_tolerance,
        rel_tol=spec.relative_tolerance,
    ):
        raise StructuralCanonicalizeError("state transition counterparty quantity invalid")

    owner_role = _role_name(
        owner,
        pair_role_column=model.pair_mapping.pair_role_column,
        aggressor_value=model.pair_mapping.aggressor_role_value,
        passive_value=model.pair_mapping.passive_role_value,
    )
    if owner_role is None:
        raise StructuralCanonicalizeError("state transition owner role invalid")
    owner_record = _participant_record(
        owner,
        role=owner_role,
        pair_mapping=model.pair_mapping,
    )
    owner_record["post_position"] = 0.0

    lineage = _source_lineage(rows)
    return {
        "identity_namespace": _identity_namespace(group_key, model),
        "ledger_identity": _ledger_identity(group_key),
        "classification": "state_transition",
        "measurement_core": _measurement_core(rows[0], model.pair_mapping.measurement_core_fields),
        **lineage,
        "state_owner": owner_record,
        "protocol_counterparty": {
            "inventory_effect": "none",
            **_source_cursor(counterparty),
        },
        "economic_fill": False,
        "economic_notional": 0.0,
    }


def _participants_same_identity(rows: list[dict[str, Any]], pair_mapping) -> bool | None:
    column = pair_mapping.participant_identity_column
    if column is None:
        return None
    left = rows[0].get(column)
    right = rows[1].get(column)
    if left is None or right is None:
        raise StructuralCanonicalizeError("participant identity missing")
    return left == right


def _ledger_row(
    rows: list[dict[str, Any]],
    *,
    classification: Literal["complete_pair", "singleton"],
    group_key: tuple[Any, ...],
    model: PairedFillReduceRequest,
) -> dict[str, Any]:
    pair_mapping = model.pair_mapping
    if not _cores_match(rows, pair_mapping.measurement_core_fields):
        raise StructuralCanonicalizeError("measurement core fields disagree")
    participants: list[dict[str, Any]] = []
    for row in rows:
        role = _role_name(
            row,
            pair_role_column=pair_mapping.pair_role_column,
            aggressor_value=pair_mapping.aggressor_role_value,
            passive_value=pair_mapping.passive_role_value,
        )
        if role is None:
            raise StructuralCanonicalizeError("pair role invalid")
        participants.append(_participant_record(row, role=role, pair_mapping=pair_mapping))
    if classification == "complete_pair":
        roles = {item["role"] for item in participants}
        if roles != {"aggressor", "passive"}:
            raise StructuralCanonicalizeError("pair role invalid")
    lineage = _source_lineage(rows)
    payload: dict[str, Any] = {
        "identity_namespace": _identity_namespace(group_key, model),
        "ledger_identity": _ledger_identity(group_key),
        "classification": classification,
        "measurement_core": _measurement_core(
            rows[0], pair_mapping.measurement_core_fields
        ),
        **lineage,
        "participants": participants,
    }
    if classification == "complete_pair":
        payload["participants_same_identity"] = _participants_same_identity(
            rows, pair_mapping
        )
    return payload


@dataclass
class _Reducer:
    model: PairedFillReduceRequest
    closed_group_keys: set[tuple[Any, ...]] = field(default_factory=set)
    pending: list[dict[str, Any]] = field(default_factory=list)
    pending_group_key: tuple[Any, ...] | None = None
    boundary_carry: dict[str, Any] | None = None
    boundary_carry_group_key: tuple[Any, ...] | None = None
    ledger_rows: list[dict[str, Any]] = field(default_factory=list)
    administrative_row_count: int = 0
    state_transition_count: int = 0

    def __post_init__(self) -> None:
        carry = self.model.partition.incoming_carry
        if carry is not None and carry.pending_row is not None:
            if carry.pending_group_key is not None:
                self.boundary_carry_group_key = tuple(carry.pending_group_key)
            elif carry.pending_identity is not None:
                self.boundary_carry_group_key = (carry.pending_identity,)
            else:
                raise StructuralCanonicalizeError("carry group key missing")
            self.boundary_carry = dict(carry.pending_row)

    def _check_output_bounds(self) -> None:
        if len(self.ledger_rows) > self.model.max_output_rows:
            raise ValueError("paired fill output row count exceeds limit")

    def _append_ledger(self, row: dict[str, Any]) -> None:
        self.ledger_rows.append(row)
        self._check_output_bounds()

    def _clear_pending(self) -> None:
        self.pending = []
        self.pending_group_key = None

    def _finalize_singleton(self, group_key: tuple[Any, ...]) -> None:
        if len(self.pending) != 1:
            raise StructuralCanonicalizeError("singleton group size invalid")
        if _row_matches_transition_marker(self.pending[0], self.model):
            raise StructuralCanonicalizeError("state transition group is incomplete")
        self._append_ledger(
            _ledger_row(
                self.pending,
                classification="singleton",
                group_key=group_key,
                model=self.model,
            )
        )
        self.closed_group_keys.add(group_key)
        self._clear_pending()

    def _finalize_pair(self, group_key: tuple[Any, ...]) -> None:
        if len(self.pending) != PAIRED_FILL_MAX_PAIR_SIZE:
            raise StructuralCanonicalizeError("pair group size invalid")
        transition_marked = any(
            _row_matches_transition_marker(row, self.model) for row in self.pending
        )
        if transition_marked:
            row = _state_transition_ledger_row(
                self.pending,
                group_key=group_key,
                model=self.model,
            )
            self.state_transition_count += 1
        else:
            row = _ledger_row(
                self.pending,
                classification="complete_pair",
                group_key=group_key,
                model=self.model,
            )
        self._append_ledger(row)
        self.closed_group_keys.add(group_key)
        self._clear_pending()

    def _ingest_economic_row(self, row: dict[str, Any], group_key: tuple[Any, ...]) -> None:
        if group_key in self.closed_group_keys:
            raise StructuralCanonicalizeError("identity is not contiguous")

        if self.boundary_carry is not None:
            carry = self.boundary_carry
            carry_key = self.boundary_carry_group_key
            self.boundary_carry = None
            self.boundary_carry_group_key = None
            if carry_key is None:
                raise StructuralCanonicalizeError("carry group key missing")
            if group_key == carry_key:
                self.pending = [carry, row]
                self.pending_group_key = group_key
                self._finalize_pair(group_key)
                return
            self.pending = [carry]
            self.pending_group_key = carry_key
            self._finalize_singleton(carry_key)

        if not self.pending:
            self.pending = [row]
            self.pending_group_key = group_key
            return

        if group_key != self.pending_group_key:
            if self.pending_group_key is None:
                raise StructuralCanonicalizeError("pending group key missing")
            self._finalize_singleton(self.pending_group_key)
            self.pending = [row]
            self.pending_group_key = group_key
            return

        self.pending.append(row)
        if len(self.pending) > PAIRED_FILL_MAX_PAIR_SIZE:
            raise StructuralCanonicalizeError(
                "adjacent identity group exceeds pair size"
            )
        if len(self.pending) == PAIRED_FILL_MAX_PAIR_SIZE:
            self._finalize_pair(group_key)

    def ingest(self, row: dict[str, Any]) -> None:
        admin = self.model.administrative_row_handling
        if admin is not None and _row_is_sentinel(row, admin.predicate):
            self.administrative_row_count += 1
            return

        identity_col = self.model.identity_mapping.identity_source_column
        normalized_col = self.model.identity_mapping.identity_normalized_column
        _validate_positive_row(
            row,
            identity_col=identity_col,
            normalized_col=normalized_col,
            core_fields=self.model.pair_mapping.measurement_core_fields,
        )
        if _row_matches_transition_marker(row, self.model):
            _validate_transition_row_invariants(row, self.model)
        else:
            validate_row_invariants(row, self.model.row_invariants)
        group_key = _group_key(row, self.model)
        self._ingest_economic_row(row, group_key)

    def end_input_boundary(self) -> None:
        if len(self.pending) == PAIRED_FILL_MAX_PAIR_SIZE:
            if self.pending_group_key is None:
                raise StructuralCanonicalizeError("pending group key missing")
            self._finalize_pair(self.pending_group_key)
        elif len(self.pending) == 1:
            if self.boundary_carry is not None:
                raise StructuralCanonicalizeError("carry overflow")
            if self.pending_group_key is None:
                raise StructuralCanonicalizeError("pending group key missing")
            self.boundary_carry = dict(self.pending[0])
            self.boundary_carry_group_key = self.pending_group_key
            self._clear_pending()
        elif len(self.pending) > PAIRED_FILL_MAX_PAIR_SIZE:
            raise StructuralCanonicalizeError(
                "adjacent identity group exceeds pair size"
            )

    def finish(self) -> dict[str, Any] | None:
        if self.boundary_carry is not None:
            if not self.model.partition.terminal:
                return {
                    "schema_version": CARRY_SCHEMA_VERSION,
                    "pending_row": self.boundary_carry,
                    "pending_identity": _ledger_identity(self.boundary_carry_group_key)
                    if self.boundary_carry_group_key
                    else None,
                    "pending_group_key": list(self.boundary_carry_group_key)
                    if self.boundary_carry_group_key is not None
                    else None,
                }
            if self.boundary_carry_group_key is None:
                raise StructuralCanonicalizeError("carry group key missing")
            self.pending = [self.boundary_carry]
            self.pending_group_key = self.boundary_carry_group_key
            self.boundary_carry = None
            self.boundary_carry_group_key = None

        if len(self.pending) == 1:
            if self.pending_group_key is None:
                raise StructuralCanonicalizeError("pending group key missing")
            self._finalize_singleton(self.pending_group_key)
        elif len(self.pending) == PAIRED_FILL_MAX_PAIR_SIZE:
            if self.pending_group_key is None:
                raise StructuralCanonicalizeError("pending group key missing")
            self._finalize_pair(self.pending_group_key)
        elif len(self.pending) > PAIRED_FILL_MAX_PAIR_SIZE:
            raise StructuralCanonicalizeError(
                "adjacent identity group exceeds pair size"
            )

        if self.boundary_carry is not None:
            raise StructuralCanonicalizeError("carry overflow")
        return None


def _required_columns(model: PairedFillReduceRequest) -> tuple[str, ...]:
    pair = model.pair_mapping
    identity = model.identity_mapping
    admin = model.administrative_row_handling
    admin_fields = ()
    if admin is not None:
        admin_fields = tuple(admin.predicate.exact_match_fields)
    signed_columns: list[str] = []
    if pair.signed_execution_column is not None:
        signed_columns.append(pair.signed_execution_column)
    if pair.signed_execution_mapping is not None:
        spec = pair.signed_execution_mapping
        signed_columns.extend((spec.quantity_column, spec.side_column))
    if pair.side_size_signed_execution is not None:
        spec = pair.side_size_signed_execution
        signed_columns.extend((spec.side_column, spec.size_column))
    passthrough = [
        *pair.participant_passthrough_columns,
        *(binding.source_column for binding in pair.participant_field_bindings),
    ]
    participant_identity = (
        (pair.participant_identity_column,)
        if pair.participant_identity_column is not None
        else ()
    )
    transition = model.state_transition_handling
    transition_columns: tuple[str, ...] = ()
    if transition is not None:
        transition_columns = tuple(
            dict.fromkeys(
                [
                    *transition.marker_exact_match_fields,
                    *transition.zero_numeric_fields,
                    *transition.shared_fields,
                    *transition.bypass_row_invariant_columns,
                ]
            )
        )
    return tuple(
        dict.fromkeys(
            [
                identity.identity_source_column,
                identity.identity_normalized_column,
                *identity.namespace_columns,
                pair.pair_role_column,
                pair.start_position_column,
                *signed_columns,
                *pair.measurement_core_fields,
                *passthrough,
                *participant_identity,
                *transition_columns,
                *invariant_columns(model.row_invariants),
                *admin_fields,
                *(
                    projection.source_column
                    for projection in model.nullable_json_scalar_projections
                ),
            ]
        )
    )


def _stream_rows(
    paths: list[str | Any],
    model: PairedFillReduceRequest,
) -> tuple[_Reducer, dict[str, Any] | None]:
    reducer = _Reducer(model)
    required = _required_columns(model)
    identity_col = model.identity_mapping.identity_source_column

    for input_index, path in enumerate(paths):
        lazy = pl.scan_parquet(str(path)).with_row_index(_SOURCE_ROW_OFFSET)
        lazy = apply_nullable_json_projections_to_lazy(
            lazy, model.nullable_json_scalar_projections
        )
        _require_columns(lazy.collect_schema(), required)
        row_count = int(lazy.select(pl.len()).collect().item())
        offset = 0
        while offset < row_count:
            batch_size = min(_IDENTITY_SCAN_BATCH, row_count - offset)
            batch = (
                lazy.slice(offset, batch_size)
                .with_columns(
                    pl.lit(input_index).cast(pl.Int64).alias(_SOURCE_INPUT_INDEX),
                    pl.col(_SOURCE_ROW_OFFSET).cast(pl.Int64),
                    pl.col(identity_col)
                    .cast(pl.Int64, strict=False)
                    .alias("_identity_int"),
                )
                .collect()
            )
            select_columns = list(
                dict.fromkeys(
                    [
                        *required,
                        _SOURCE_INPUT_INDEX,
                        _SOURCE_ROW_OFFSET,
                        "_identity_int",
                    ]
                )
            )
            for row in batch.select(select_columns).iter_rows(named=True):
                reducer.ingest(dict(row))
            offset += batch_size
        if input_index < len(paths) - 1:
            reducer.end_input_boundary()
    if not model.partition.terminal:
        reducer.end_input_boundary()

    outgoing_carry = reducer.finish()
    return reducer, outgoing_carry


def _content_identity(
    ledger_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    exceptions: list[dict[str, Any]],
) -> str:
    payload = {
        "ledger_rows": ledger_rows,
        "summary": summary,
        "exceptions": exceptions,
    }
    digest = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def _total_input_bytes(paths: list[str | Any]) -> int:
    return sum(Path(path).stat().st_size for path in paths)


def execute_paired_fill_reduce(
    paths: list[str | Any],
    request: dict[str, Any] | PairedFillReduceRequest,
) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one parquet input is required")
    model = _validated(request)
    if len(paths) > model.max_input_files:
        raise ValueError("replay parquet input count exceeds limit")
    input_bytes = _total_input_bytes(paths)
    if input_bytes > model.max_input_bytes:
        raise ValueError("replay parquet input bytes exceed limit")
    reducer, outgoing_carry = _stream_rows(paths, model)
    exceptions: list[dict[str, Any]] = []
    summary = {
        "request_id": model.request_id,
        "ledger_row_count": len(reducer.ledger_rows),
        "administrative_row_count": reducer.administrative_row_count,
        "state_transition_count": reducer.state_transition_count,
        "exception_row_count": len(exceptions),
        "input_bytes": input_bytes,
    }
    content_identity = _content_identity(reducer.ledger_rows, summary, exceptions)
    summary["content_identity"] = content_identity

    ledger_buffer = io.BytesIO()
    if reducer.ledger_rows:
        pl.DataFrame(reducer.ledger_rows).write_parquet(ledger_buffer)
    ledger_bytes = ledger_buffer.getvalue()
    if len(ledger_bytes) > model.max_output_bytes:
        raise ValueError("paired fill parquet output exceeds byte limit")
    ledger_parquet_identity = (
        f"sha256:{sha256(ledger_bytes).hexdigest()}" if ledger_bytes else None
    )

    result_schema_version = (
        RESULT_SCHEMA_VERSION_V2
        if model.state_transition_handling is not None
        else RESULT_SCHEMA_VERSION
    )
    return {
        "schema_version": result_schema_version,
        "summary": summary,
        "exceptions": exceptions,
        "ledger_rows": reducer.ledger_rows,
        "ledger_parquet_bytes": ledger_bytes,
        "ledger_parquet_identity": ledger_parquet_identity,
        "max_output_bytes": model.max_output_bytes,
        "outgoing_carry": outgoing_carry
        or {
            "schema_version": CARRY_SCHEMA_VERSION,
            "pending_row": None,
            "pending_identity": None,
            "pending_group_key": None,
        },
    }


def build_paired_fill_metadata(
    result_payload: dict[str, Any],
    *,
    ledger_parquet_ref: Any | None,
) -> dict[str, Any]:
    metadata_schema_version = (
        METADATA_SCHEMA_VERSION_V2
        if result_payload.get("schema_version") == RESULT_SCHEMA_VERSION_V2
        else METADATA_SCHEMA_VERSION
    )
    return {
        "schema_version": metadata_schema_version,
        "result_schema_version": result_payload["schema_version"],
        "summary": result_payload["summary"],
        "exceptions": result_payload["exceptions"],
        "outgoing_carry": result_payload["outgoing_carry"],
        "ledger_parquet_ref": (
            ledger_parquet_ref.model_dump(mode="json")
            if ledger_parquet_ref is not None
            else None
        ),
        "ledger_parquet_identity": result_payload.get("ledger_parquet_identity"),
    }


def publish_paired_fill_reduce_artifacts(
    plane,
    result_payload: dict[str, Any],
    *,
    artifact_ref_matches_bytes,
    shard_stage_failure,
) -> tuple[bytes, tuple[Any, ...]]:
    """Publish ledger Parquet (when non-empty) then deterministic metadata JSON."""
    max_output_bytes = int(result_payload.get("max_output_bytes", 0))
    ledger_bytes = bytes(result_payload.get("ledger_parquet_bytes") or b"")
    if max_output_bytes > 0 and len(ledger_bytes) > max_output_bytes:
        raise ValueError("paired fill parquet output exceeds byte limit")

    ledger_ref = None
    output_refs: list[Any] = []
    if ledger_bytes:
        ledger_ref = plane.write(ledger_bytes, LEDGER_PARQUET_MEDIA_TYPE)
        if not artifact_ref_matches_bytes(ledger_bytes, ledger_ref):
            raise shard_stage_failure("output_artifact_mismatch")
        output_refs.append(ledger_ref)

    metadata = build_paired_fill_metadata(result_payload, ledger_parquet_ref=ledger_ref)
    metadata_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
    metadata_ref = plane.write(metadata_bytes, "application/json")
    if not artifact_ref_matches_bytes(metadata_bytes, metadata_ref):
        raise shard_stage_failure("output_artifact_mismatch")
    return metadata_bytes, (metadata_ref, *output_refs)
