"""Typed closed parameters for generic replay reduction primitives."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from portable_batch_execution.contracts.models import Frozen

SentinelScalar = str | int | float | bool | None

BUCKET_COUNT_DEFAULT = 256
BUCKET_COUNT_MIN = 1
BUCKET_COUNT_MAX = 4096


class JsonScalarProjection(Frozen):
    schema_version: Literal["pbe.replay.json-scalar-projection.v1"]
    source_column: str = Field(min_length=1)
    key_path: tuple[str, ...] = Field(min_length=1)
    scalar_type: Literal["integer", "string"]
    output_column: str = Field(min_length=1)


class SentinelPredicate(Frozen):
    identity_equals: int
    exact_match_fields: dict[str, SentinelScalar] = Field(default_factory=dict)


class StructuralCanonicalizeParams(Frozen):
    schema_version: Literal["pbe.replay.structural-canonicalize.v1"]
    identity_source_column: str
    identity_normalized_column: str
    measurement_core_fields: tuple[str, ...] = Field(min_length=1)
    sentinel: SentinelPredicate | None = None
    bucket_count: int = BUCKET_COUNT_DEFAULT
    json_scalar_projections: tuple[JsonScalarProjection, ...] = ()

    @model_validator(mode="after")
    def _validate_json_scalar_projections(self) -> StructuralCanonicalizeParams:
        from .json_scalar_projection import validate_json_scalar_projection_bundle

        validate_json_scalar_projection_bundle(self.json_scalar_projections)
        return self

    @field_validator("bucket_count")
    @classmethod
    def _power_of_two_in_range(cls, value: int) -> int:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < BUCKET_COUNT_MIN
            or value > BUCKET_COUNT_MAX
            or value & (value - 1)
        ):
            raise ValueError(
                "bucket_count must be a power of two within the allowed range"
            )
        return value


class TrailingWindowSpec(Frozen):
    fact_id: str
    measurement_field: str
    trailing_width_ms: int = Field(ge=1)


class FutureWindowSpec(Frozen):
    fact_id: str
    measurement_field: str
    start_offset_ms: int = Field(ge=0)
    end_offset_ms: int = Field(ge=0)
    min_block_exclusive: int

    @model_validator(mode="after")
    def _end_not_before_start(self) -> FutureWindowSpec:
        if self.end_offset_ms < self.start_offset_ms:
            raise ValueError("end_offset_ms must be >= start_offset_ms")
        return self


class CanonicalTradeProfile(Frozen):
    schema_version: Literal["pbe.replay.canonical-trade-profile.v1"]
    identity_source_column: str
    identity_normalized_column: str
    measurement_core_fields: tuple[str, ...] = Field(min_length=1)
    sentinel: SentinelPredicate | None = None
    json_scalar_projections: tuple[JsonScalarProjection, ...] = ()

    @model_validator(mode="after")
    def _validate_json_scalar_projections(self) -> CanonicalTradeProfile:
        from .json_scalar_projection import validate_json_scalar_projection_bundle

        validate_json_scalar_projection_bundle(self.json_scalar_projections)
        return self

    def structural_canonicalize_params(self) -> StructuralCanonicalizeParams:
        return StructuralCanonicalizeParams(
            schema_version="pbe.replay.structural-canonicalize.v1",
            identity_source_column=self.identity_source_column,
            identity_normalized_column=self.identity_normalized_column,
            measurement_core_fields=self.measurement_core_fields,
            sentinel=self.sentinel,
            json_scalar_projections=self.json_scalar_projections,
        )


class EventWindowExtractJobParams(Frozen):
    schema_version: Literal["pbe.replay.event-window-extract-job.v1"]


class EventWindowExtractRequest(Frozen):
    schema_version: Literal["pbe.replay.event-window-extract.v3"]
    request_id: str
    symbol: str
    symbol_column: str
    block_column: str
    timestamp_column: str
    decision_timestamp_ms: int
    causal_cutoff_block: int
    as_of_measurement_field: str
    canonical_trade_profile: CanonicalTradeProfile
    as_of_offsets_ms: tuple[int, ...] = (0,)
    trailing_windows: tuple[TrailingWindowSpec, ...] = ()
    future_windows: tuple[FutureWindowSpec, ...] = ()
    tie_break_columns: tuple[str, ...] = ("timestamp_ms",)


class CausalWitnessMapping(Frozen):
    block_column: str
    timestamp_column: str


class CanonicalTradeInputMapping(Frozen):
    canonical_trade_profile: CanonicalTradeProfile
    symbol_column: str
    block_column: str
    timestamp_column: str
    price_field: str
    notional_field: str


class InputRoleBinding(Frozen):
    input_index: int = Field(ge=0)
    role: Literal["canonical_trade", "causal_witness"]


class EmitGridSpec(Frozen):
    start_timestamp_ms: int
    end_timestamp_ms: int
    step_ms: int = Field(ge=1)


class CausalGridCarryState(Frozen):
    schema_version: Literal[
        "pbe.replay.causal-grid-carry.v1",
        "pbe.replay.causal-grid-carry.v2",
        "pbe.replay.causal-grid-carry.v3",
        "pbe.replay.causal-grid-carry.v4",
    ]
    causal_observations: tuple[tuple[int, int, int], ...] = ()
    causal_block_first_ms: tuple[tuple[int, int, int], ...] = ()
    causal_segment_frontiers: tuple[tuple[int, int, int, int, int, bool], ...] = ()
    trade_rows: tuple[dict[str, Any], ...] = ()


class PartitionSpec(Frozen):
    emit_start_ms: int
    emit_end_ms: int
    overlap_ms: int = Field(ge=0)
    hard_gap_missing_dates: tuple[str, ...] = ()
    incoming_carry: CausalGridCarryState | None = None

    @model_validator(mode="after")
    def _emit_window_valid(self) -> PartitionSpec:
        if self.emit_end_ms < self.emit_start_ms:
            raise ValueError("partition emit_end_ms must be >= emit_start_ms")
        return self


class CausalGridExtractJobParams(Frozen):
    schema_version: Literal["pbe.replay.causal-grid-extract-job.v1"]


class CausalGridExtractRequest(Frozen):
    schema_version: Literal["pbe.replay.causal-grid-extract.v1"]
    request_id: str
    target_symbols: tuple[str, ...] = Field(min_length=1)
    input_roles: tuple[InputRoleBinding, ...] = Field(min_length=1)
    causal_witness_mapping: CausalWitnessMapping
    canonical_trade_mapping: CanonicalTradeInputMapping
    emit_grid: EmitGridSpec
    partition: PartitionSpec
    as_of_measurement_field: str
    as_of_offsets_ms: tuple[int, ...] = (0, 5000)
    trailing_windows: tuple[TrailingWindowSpec, ...] = ()
    tie_break_columns: tuple[str, ...] = ("timestamp_ms",)
    max_output_rows: int = Field(default=500_000, ge=1, le=10_000_000)

    @model_validator(mode="after")
    def _roles_cover_inputs(self) -> CausalGridExtractRequest:
        indices = [binding.input_index for binding in self.input_roles]
        if len(indices) != len(set(indices)):
            raise ValueError("duplicate input_index in input_roles")
        if not any(binding.role == "canonical_trade" for binding in self.input_roles):
            raise ValueError("at least one canonical_trade input role is required")
        return self


class StructuralCanonicalizeMergeParams(Frozen):
    schema_version: Literal["pbe.replay.structural-canonicalize-merge.v1"]


PARAM_MODELS = {
    "replay.structural_canonicalize": StructuralCanonicalizeParams,
    "replay.event_window_extract": EventWindowExtractJobParams,
    "replay.causal_grid_extract": CausalGridExtractJobParams,
    "replay.structural_canonicalize_merge": StructuralCanonicalizeMergeParams,
}
