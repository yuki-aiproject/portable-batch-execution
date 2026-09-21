"""Typed closed parameters for generic replay reduction primitives."""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

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


class ExactTextRowInvariant(Frozen):
    schema_version: Literal["pbe.replay.row-invariant.exact-text.v1"]
    column: str = Field(min_length=1)
    expected: str


class NumericRowInvariant(Frozen):
    schema_version: Literal["pbe.replay.row-invariant.numeric.v1"]
    left_column: str = Field(min_length=1)
    right_column: str = Field(min_length=1)


class TimestampMsEquivalenceRowInvariant(Frozen):
    schema_version: Literal["pbe.replay.row-invariant.timestamp-ms-equivalence.v1"]
    left_column: str = Field(min_length=1)
    right_column: str = Field(min_length=1)
    left_mode: Literal["integer_ms", "iso8601"] = "integer_ms"
    right_mode: Literal["integer_ms", "iso8601"] = "integer_ms"


class PositiveFiniteRowInvariant(Frozen):
    schema_version: Literal["pbe.replay.row-invariant.positive-finite.v1"]
    column: str = Field(min_length=1)


class ProductWithToleranceRowInvariant(Frozen):
    schema_version: Literal["pbe.replay.row-invariant.product-with-tolerance.v1"]
    factor_columns: tuple[str, ...] = Field(min_length=1)
    expected: float
    absolute_tolerance: float = Field(ge=0)
    relative_tolerance: float = Field(default=0.0, ge=0)


class ProductColumnWithToleranceRowInvariant(Frozen):
    schema_version: Literal["pbe.replay.row-invariant.product-column-with-tolerance.v1"]
    factor_columns: tuple[str, ...] = Field(min_length=1)
    expected_column: str = Field(min_length=1)
    absolute_tolerance: float = Field(ge=0)
    relative_tolerance: float = Field(default=0.0, ge=0)


class TextColumnEquivalenceRowInvariant(Frozen):
    schema_version: Literal["pbe.replay.row-invariant.text-column-equivalence.v1"]
    left_column: str = Field(min_length=1)
    right_column: str = Field(min_length=1)


class NonEmptyTextRowInvariant(Frozen):
    schema_version: Literal["pbe.replay.row-invariant.non-empty-text.v1"]
    column: str = Field(min_length=1)


class BooleanRowInvariant(Frozen):
    schema_version: Literal["pbe.replay.row-invariant.boolean.v1"]
    column: str = Field(min_length=1)


class FiniteNumericRowInvariant(Frozen):
    schema_version: Literal["pbe.replay.row-invariant.finite-numeric.v1"]
    column: str = Field(min_length=1)


RowInvariant = Annotated[
    ExactTextRowInvariant
    | NumericRowInvariant
    | TimestampMsEquivalenceRowInvariant
    | PositiveFiniteRowInvariant
    | ProductWithToleranceRowInvariant
    | ProductColumnWithToleranceRowInvariant
    | TextColumnEquivalenceRowInvariant
    | NonEmptyTextRowInvariant
    | BooleanRowInvariant
    | FiniteNumericRowInvariant,
    Field(discriminator="schema_version"),
]


class CanonicalTradeProfile(Frozen):
    schema_version: Literal["pbe.replay.canonical-trade-profile.v1"]
    identity_source_column: str
    identity_normalized_column: str
    measurement_core_fields: tuple[str, ...] = Field(min_length=1)
    sentinel: SentinelPredicate | None = None
    json_scalar_projections: tuple[JsonScalarProjection, ...] = ()
    row_invariants: tuple[RowInvariant, ...] = ()

    @model_validator(mode="after")
    def _validate_json_scalar_projections(self) -> CanonicalTradeProfile:
        from .json_scalar_projection import validate_json_scalar_projection_bundle

        validate_json_scalar_projection_bundle(self.json_scalar_projections)
        return self

    @model_validator(mode="after")
    def _validate_row_invariants(self) -> CanonicalTradeProfile:
        from .row_invariants import validate_row_invariant_bundle

        validate_row_invariant_bundle(self.row_invariants)
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
    source_order_tie_break: bool = False


class CausalWitnessMapping(Frozen):
    block_column: str
    timestamp_column: str
    timestamp_mode: Literal["integer_ms", "iso8601"] = "integer_ms"


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


class SparseEmitPoint(Frozen):
    timestamp_ms: int
    symbols: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _symbols_unique(self) -> SparseEmitPoint:
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("sparse emit point symbols must be unique")
        return self


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
    source_order_tie_break: bool = False
    sparse_emit_points: tuple[SparseEmitPoint, ...] = ()

    @model_validator(mode="after")
    def _roles_cover_inputs(self) -> CausalGridExtractRequest:
        pairs = [(binding.input_index, binding.role) for binding in self.input_roles]
        if len(pairs) != len(set(pairs)):
            raise ValueError("duplicate input_index and role pair in input_roles")
        if not any(binding.role == "canonical_trade" for binding in self.input_roles):
            raise ValueError("at least one canonical_trade input role is required")
        return self

    @model_validator(mode="after")
    def _sparse_emit_points_valid(self) -> CausalGridExtractRequest:
        if not self.sparse_emit_points:
            return self
        target = set(self.target_symbols)
        timestamps = [point.timestamp_ms for point in self.sparse_emit_points]
        if len(set(timestamps)) != len(timestamps):
            raise ValueError("sparse emit point timestamps must be unique")
        emit = self.emit_grid
        for point in self.sparse_emit_points:
            unknown = set(point.symbols) - target
            if unknown:
                raise ValueError("sparse emit point symbol not in target_symbols")
            if (
                point.timestamp_ms < emit.start_timestamp_ms
                or point.timestamp_ms > emit.end_timestamp_ms
            ):
                raise ValueError("sparse emit point outside emit_grid bounds")
            if point.timestamp_ms < self.partition.emit_start_ms:
                raise ValueError("sparse emit point outside partition emit bounds")
            if point.timestamp_ms > self.partition.emit_end_ms:
                raise ValueError("sparse emit point outside partition emit bounds")
        return self


class StructuralCanonicalizeMergeParams(Frozen):
    schema_version: Literal["pbe.replay.structural-canonicalize-merge.v1"]


class NullableJsonScalarProjection(Frozen):
    schema_version: Literal["pbe.replay.nullable-json-scalar-projection.v1"]
    source_column: str = Field(min_length=1)
    key_path: tuple[str, ...] = Field(min_length=1)
    scalar_type: Literal["integer", "string", "float", "boolean"]
    output_column: str = Field(min_length=1)


class AdministrativeRowHandling(Frozen):
    schema_version: Literal["pbe.replay.administrative-row-handling.v1"]
    predicate: SentinelPredicate


class PairedFillStateTransitionHandling(Frozen):
    schema_version: Literal["pbe.replay.paired-fill-state-transition.v1"]
    marker_exact_match_fields: dict[str, SentinelScalar] = Field(min_length=1)
    zero_numeric_fields: tuple[str, ...] = Field(min_length=1)
    shared_fields: tuple[str, ...] = Field(min_length=1)
    state_owner_role_value: SentinelScalar
    protocol_counterparty_role_value: SentinelScalar
    bypass_row_invariant_columns: tuple[str, ...] = Field(min_length=1)
    absolute_tolerance: float = Field(default=1e-12, ge=0.0)
    relative_tolerance: float = Field(default=1e-12, ge=0.0)

    @model_validator(mode="after")
    def _validate_transition_fields(self) -> PairedFillStateTransitionHandling:
        if self.state_owner_role_value == self.protocol_counterparty_role_value:
            raise ValueError("state transition role values must differ")
        for values, label in (
            (self.zero_numeric_fields, "zero_numeric_fields"),
            (self.shared_fields, "shared_fields"),
            (self.bypass_row_invariant_columns, "bypass_row_invariant_columns"),
        ):
            if len(values) != len(set(values)) or any(not value for value in values):
                raise ValueError(f"invalid {label}")
        if any(not key for key in self.marker_exact_match_fields):
            raise ValueError("state transition marker field name is empty")
        if not set(self.bypass_row_invariant_columns).issubset(
            set(self.zero_numeric_fields)
        ):
            raise ValueError(
                "state transition invariant bypass columns must be zero numeric fields"
            )
        return self


PAIRED_FILL_MAX_INPUT_FILES = 64
PAIRED_FILL_MAX_INPUT_BYTES = 1_610_612_736
PAIRED_FILL_MAX_OUTPUT_ROWS = 12_000_000
PAIRED_FILL_MAX_OUTPUT_BYTES = 1_073_741_824
PAIRED_FILL_MAX_EXCEPTION_ROWS = 100_000
PAIRED_FILL_MAX_CARRY_ROWS = 1
PAIRED_FILL_MAX_PAIR_SIZE = 2
_PARTICIPANT_FIELD_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_PARTICIPANT_RESERVED_OUTPUT_FIELDS = frozenset(
    {
        "role",
        "start_position",
        "signed_execution",
        "opening_quantity",
        "closing_quantity",
        "post_position",
        "source_input_index",
        "source_row_offset",
    }
)


class PairedFillIdentityMapping(Frozen):
    identity_source_column: str = Field(min_length=1)
    identity_normalized_column: str = Field(min_length=1)
    namespace_columns: tuple[str, ...] = ()


class SignedExecutionMapping(Frozen):
    schema_version: Literal["pbe.replay.signed-execution-mapping.v1"]
    quantity_column: str = Field(min_length=1)
    side_column: str = Field(min_length=1)
    positive_side_value: SentinelScalar
    negative_side_value: SentinelScalar


class SideSizeSignedExecutionSpec(Frozen):
    schema_version: Literal["pbe.replay.side-size-signed-execution.v1"]
    side_column: str = Field(min_length=1)
    size_column: str = Field(min_length=1)
    buy_side_value: SentinelScalar
    sell_side_value: SentinelScalar


class ParticipantFieldBinding(Frozen):
    output_field: str = Field(min_length=1)
    source_column: str = Field(min_length=1)


class PairedFillPairMapping(Frozen):
    pair_role_column: str = Field(min_length=1)
    aggressor_role_value: SentinelScalar
    passive_role_value: SentinelScalar
    measurement_core_fields: tuple[str, ...] = Field(min_length=1)
    start_position_column: str = Field(min_length=1)
    signed_execution_column: str | None = None
    signed_execution_mapping: SignedExecutionMapping | None = None
    side_size_signed_execution: SideSizeSignedExecutionSpec | None = None
    participant_passthrough_columns: tuple[str, ...] = ()
    participant_field_bindings: tuple[ParticipantFieldBinding, ...] = ()
    participant_identity_column: str | None = None

    @model_validator(mode="after")
    def _signed_execution_mode(self) -> PairedFillPairMapping:
        mode_count = sum(
            1
            for enabled in (
                self.signed_execution_column is not None,
                self.signed_execution_mapping is not None,
                self.side_size_signed_execution is not None,
            )
            if enabled
        )
        if mode_count != 1:
            raise ValueError("exactly one signed execution mapping mode is required")
        passthrough_columns: set[str] = set()
        for column in self.participant_passthrough_columns:
            if not column:
                raise ValueError("participant passthrough column name is empty")
            if column in passthrough_columns:
                raise ValueError("duplicate participant passthrough column")
            passthrough_columns.add(column)
        outputs: set[str] = set()
        for binding in self.participant_field_bindings:
            if binding.output_field in _PARTICIPANT_RESERVED_OUTPUT_FIELDS:
                raise ValueError("participant field output name is reserved")
            if not _PARTICIPANT_FIELD_PATTERN.fullmatch(binding.output_field):
                raise ValueError("participant field output name is unsafe")
            if binding.output_field in outputs:
                raise ValueError("duplicate participant field output name")
            outputs.add(binding.output_field)
        return self


class PairedFillCarryState(Frozen):
    schema_version: Literal["pbe.replay.paired-fill-reduce-carry.v1"]
    pending_row: dict[str, Any] | None = None
    pending_identity: int | None = None
    pending_group_key: tuple[Any, ...] | None = None


class PairedFillPartitionSpec(Frozen):
    terminal: bool = True
    incoming_carry: PairedFillCarryState | None = None


class PairedFillReduceJobParams(Frozen):
    schema_version: Literal["pbe.replay.paired-fill-reduce-job.v1"]


class PairedFillReduceRequest(Frozen):
    schema_version: Literal[
        "pbe.replay.paired-fill-reduce.v1",
        "pbe.replay.paired-fill-reduce.v2",
    ]
    request_id: str
    identity_mapping: PairedFillIdentityMapping
    pair_mapping: PairedFillPairMapping
    partition: PairedFillPartitionSpec = PairedFillPartitionSpec()
    nullable_json_scalar_projections: tuple[NullableJsonScalarProjection, ...] = ()
    row_invariants: tuple[RowInvariant, ...] = ()
    administrative_row_handling: AdministrativeRowHandling | None = None
    state_transition_handling: PairedFillStateTransitionHandling | None = None
    max_input_files: int = Field(
        default=PAIRED_FILL_MAX_INPUT_FILES,
        ge=1,
        le=PAIRED_FILL_MAX_INPUT_FILES,
    )
    max_input_bytes: int = Field(
        default=PAIRED_FILL_MAX_INPUT_BYTES,
        ge=1,
        le=PAIRED_FILL_MAX_INPUT_BYTES,
    )
    max_output_rows: int = Field(
        default=500_000,
        ge=1,
        le=PAIRED_FILL_MAX_OUTPUT_ROWS,
    )
    max_output_bytes: int = Field(
        default=PAIRED_FILL_MAX_OUTPUT_BYTES,
        ge=1,
        le=PAIRED_FILL_MAX_OUTPUT_BYTES,
    )
    max_exception_rows: int = Field(
        default=10_000,
        ge=0,
        le=PAIRED_FILL_MAX_EXCEPTION_ROWS,
    )

    @model_validator(mode="after")
    def _validate_nullable_projections(self) -> PairedFillReduceRequest:
        from .nullable_json_projection import (
            validate_nullable_json_scalar_projection_bundle,
        )

        validate_nullable_json_scalar_projection_bundle(
            self.nullable_json_scalar_projections
        )
        return self

    @model_validator(mode="after")
    def _validate_row_invariants(self) -> PairedFillReduceRequest:
        from .row_invariants import validate_row_invariant_bundle

        validate_row_invariant_bundle(self.row_invariants)
        return self

    @model_validator(mode="after")
    def _validate_state_transition_version(self) -> PairedFillReduceRequest:
        if self.state_transition_handling is None:
            if self.schema_version != "pbe.replay.paired-fill-reduce.v1":
                raise ValueError("paired-fill v2 requires state_transition_handling")
            return self
        if self.schema_version != "pbe.replay.paired-fill-reduce.v2":
            raise ValueError("state_transition_handling requires paired-fill v2")
        return self


PARAM_MODELS = {
    "replay.structural_canonicalize": StructuralCanonicalizeParams,
    "replay.event_window_extract": EventWindowExtractJobParams,
    "replay.causal_grid_extract": CausalGridExtractJobParams,
    "replay.structural_canonicalize_merge": StructuralCanonicalizeMergeParams,
    "replay.paired_fill_reduce": PairedFillReduceJobParams,
}
