"""Closed event-window fact extraction using UTC timestamps and causal block guards."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from .canonicalize import (
    _IDENTITY_SCAN_BATCH,
    StructuralCanonicalizeError,
    _require_columns,
    execute_structural_canonicalize,
)
from .json_scalar_projection import (
    JsonScalarProjectionError,
    apply_json_scalar_projections_to_lazy,
)
from .models import EventWindowExtractRequest, SentinelPredicate

RESULT_SCHEMA_VERSION = "pbe.replay.event-window-extract-result.v2"


def _validated(request: dict[str, Any] | EventWindowExtractRequest) -> EventWindowExtractRequest:
    if isinstance(request, EventWindowExtractRequest):
        return request
    return EventWindowExtractRequest.model_validate(request)


def _tie_break_key(row: dict[str, Any], columns: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(row[column] for column in columns)


def _as_of_preferred(
    candidate: dict[str, Any],
    current: dict[str, Any] | None,
    tie_break: tuple[str, ...],
) -> bool:
    if current is None:
        return True
    return _tie_break_key(candidate, tie_break) > _tie_break_key(current, tie_break)


def _future_preferred(
    candidate: dict[str, Any],
    current: dict[str, Any] | None,
    tie_break: tuple[str, ...],
) -> bool:
    if current is None:
        return True
    return _tie_break_key(candidate, tie_break) < _tie_break_key(current, tie_break)


def _measurement_value(row: dict[str, Any], field_name: str) -> float:
    raw = row[field_name]
    if raw is None:
        return 0.0
    return float(raw)


@dataclass
class _FactAccumulators:
    trailing_sums: dict[str, float] = field(default_factory=dict)
    trailing_counts: dict[str, int] = field(default_factory=dict)
    as_of_best: dict[int, dict[str, Any] | None] = field(default_factory=dict)
    future_best: dict[str, dict[str, Any] | None] = field(default_factory=dict)

    def ingest(
        self,
        row: dict[str, Any],
        model: EventWindowExtractRequest,
    ) -> None:
        block = int(row["_block_int"])
        timestamp_ms = int(row["_timestamp_ms"])
        decision_ms = int(model.decision_timestamp_ms)
        causal = int(model.causal_cutoff_block)
        tie_break = model.tie_break_columns

        if block <= causal:
            for spec in model.trailing_windows:
                lower = decision_ms - int(spec.trailing_width_ms)
                if timestamp_ms > lower and timestamp_ms <= decision_ms:
                    key = spec.fact_id
                    self.trailing_sums[key] = (
                        self.trailing_sums.get(key, 0.0)
                        + _measurement_value(row, spec.measurement_field)
                    )
                    self.trailing_counts[key] = self.trailing_counts.get(key, 0) + 1
            for offset_ms in model.as_of_offsets_ms:
                target_ms = decision_ms - int(offset_ms)
                if timestamp_ms <= target_ms:
                    current = self.as_of_best.get(int(offset_ms))
                    if _as_of_preferred(row, current, tie_break):
                        self.as_of_best[int(offset_ms)] = row

        for spec in model.future_windows:
            start_ms = decision_ms + int(spec.start_offset_ms)
            end_ms = decision_ms + int(spec.end_offset_ms)
            if (
                timestamp_ms >= start_ms
                and timestamp_ms <= end_ms
                and block > int(spec.min_block_exclusive)
            ):
                current = self.future_best.get(spec.fact_id)
                if _future_preferred(row, current, tie_break):
                    self.future_best[spec.fact_id] = row

    def to_facts(self, model: EventWindowExtractRequest) -> list[dict[str, Any]]:
        facts: list[dict[str, Any]] = []
        for spec in model.trailing_windows:
            facts.append(
                {
                    "fact_id": f"{spec.fact_id}.sum",
                    "value": self.trailing_sums.get(spec.fact_id, 0.0),
                }
            )
            facts.append(
                {
                    "fact_id": f"{spec.fact_id}.count",
                    "value": int(self.trailing_counts.get(spec.fact_id, 0)),
                }
            )
        for offset_ms in model.as_of_offsets_ms:
            best = self.as_of_best.get(int(offset_ms))
            value = None if best is None else best.get(model.as_of_measurement_field)
            facts.append({"fact_id": f"as_of.{offset_ms}", "value": value})
        for spec in model.future_windows:
            best = self.future_best.get(spec.fact_id)
            value = None if best is None else best.get(spec.measurement_field)
            facts.append({"fact_id": spec.fact_id, "value": value})
        return facts


def _row_is_sentinel(row: dict[str, Any], sentinel: SentinelPredicate | None) -> bool:
    if sentinel is None:
        return False
    if row.get("_identity_int") != sentinel.identity_equals:
        return False
    for match_field, expected in sentinel.exact_match_fields.items():
        actual = row.get(match_field)
        if expected is None:
            if actual is not None:
                return False
        elif actual != expected:
            return False
    return True


def _validate_positive_row(
    row: dict[str, Any],
    *,
    identity_col: str,
    normalized_col: str,
    core_fields: tuple[str, ...],
) -> int:
    identity_raw = row.get(identity_col)
    if identity_raw is None:
        raise StructuralCanonicalizeError("identity is missing or not positive")
    try:
        identity = int(identity_raw)
    except (TypeError, ValueError) as exc:
        raise StructuralCanonicalizeError("identity is missing or not positive") from exc
    if identity <= 0:
        raise StructuralCanonicalizeError("identity is missing or not positive")
    normalized = row.get(normalized_col)
    if normalized is None or str(normalized) != str(identity):
        raise StructuralCanonicalizeError("identity normalized column mismatch")
    for field_name in core_fields:
        if field_name not in row or row[field_name] is None:
            raise StructuralCanonicalizeError("measurement core fields disagree")
    return identity


def _symbol_lazy_frame(path: str | Path, model: EventWindowExtractRequest) -> pl.LazyFrame:
    lazy = pl.scan_parquet(str(path))
    profile = model.canonical_trade_profile
    try:
        lazy = apply_json_scalar_projections_to_lazy(lazy, profile.json_scalar_projections)
    except JsonScalarProjectionError as exc:
        raise StructuralCanonicalizeError(str(exc)) from exc
    _require_columns(
        lazy.collect_schema(),
        (
            model.symbol_column,
            model.block_column,
            model.timestamp_column,
            model.canonical_trade_profile.identity_source_column,
            model.canonical_trade_profile.identity_normalized_column,
            *model.canonical_trade_profile.measurement_core_fields,
            *model.tie_break_columns,
            model.as_of_measurement_field,
            *(spec.measurement_field for spec in model.trailing_windows),
            *(spec.measurement_field for spec in model.future_windows),
        ),
    )
    sentinel = profile.sentinel
    required_sentinel = tuple(sentinel.exact_match_fields) if sentinel is not None else ()
    if required_sentinel:
        _require_columns(lazy.collect_schema(), required_sentinel)

    identity_col = profile.identity_source_column
    lazy = lazy.filter(pl.col(model.symbol_column) == model.symbol).with_columns(
        pl.col(model.block_column).cast(pl.Int64, strict=False).alias("_block_int"),
        pl.col(model.timestamp_column).cast(pl.Int64, strict=False).alias("_timestamp_ms"),
        pl.col(identity_col).cast(pl.Int64, strict=False).alias("_identity_int"),
    )
    return lazy


def _stream_canonical_rows(
    paths: list[str | Path],
    model: EventWindowExtractRequest,
    accumulators: _FactAccumulators,
) -> None:
    profile = model.canonical_trade_profile
    identity_col = profile.identity_source_column
    normalized_col = profile.identity_normalized_column
    core_fields = profile.measurement_core_fields
    sentinel = profile.sentinel

    select_columns = list(
        dict.fromkeys(
            [
                model.symbol_column,
                model.block_column,
                model.timestamp_column,
                identity_col,
                normalized_col,
                *core_fields,
                *model.tie_break_columns,
                model.as_of_measurement_field,
                *(spec.measurement_field for spec in model.trailing_windows),
                *(spec.measurement_field for spec in model.future_windows),
                *(sentinel.exact_match_fields if sentinel is not None else ()),
                "_block_int",
                "_timestamp_ms",
                "_identity_int",
            ]
        )
    )

    previous_identity: int | None = None
    group_core: dict[str, Any] | None = None
    representative: dict[str, Any] | None = None

    def finalize_group() -> None:
        nonlocal representative
        if representative is not None:
            accumulators.ingest(representative, model)
        representative = None

    for path in paths:
        lazy = _symbol_lazy_frame(path, model)
        row_count = int(lazy.select(pl.len()).collect().item())
        offset = 0
        while offset < row_count:
            batch_size = min(_IDENTITY_SCAN_BATCH, row_count - offset)
            batch = lazy.slice(offset, batch_size).select(select_columns).collect()
            for row in batch.iter_rows(named=True):
                if _row_is_sentinel(row, sentinel):
                    continue
                identity = _validate_positive_row(
                    row,
                    identity_col=identity_col,
                    normalized_col=normalized_col,
                    core_fields=core_fields,
                )
                if previous_identity is None or identity != previous_identity:
                    finalize_group()
                    previous_identity = identity
                    group_core = {field_name: row[field_name] for field_name in core_fields}
                elif group_core is None:
                    raise StructuralCanonicalizeError("measurement core fields disagree")
                else:
                    for field_name in core_fields:
                        if row[field_name] != group_core[field_name]:
                            raise StructuralCanonicalizeError(
                                "measurement core fields disagree"
                            )
                representative = dict(row)
            offset += batch_size

    if representative is not None:
        accumulators.ingest(representative, model)


def execute_event_window_extract(
    paths: list[str | Path],
    request: dict[str, Any] | EventWindowExtractRequest,
) -> dict[str, Any]:
    """Extract trailing, as-of, and future window facts from canonical economic rows."""
    if not paths:
        raise ValueError("at least one parquet input is required")
    model = _validated(request)
    canonical_params = model.canonical_trade_profile.structural_canonicalize_params()
    validation_state = execute_structural_canonicalize(paths, canonical_params)
    try:
        accumulators = _FactAccumulators()
        for offset_ms in model.as_of_offsets_ms:
            accumulators.as_of_best[int(offset_ms)] = None
        for spec in model.future_windows:
            accumulators.future_best[spec.fact_id] = None
        _stream_canonical_rows(paths, model, accumulators)
        facts = accumulators.to_facts(model)
    finally:
        shutil.rmtree(validation_state.spill_dir, ignore_errors=True)

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "facts": facts,
    }
