import json

import polars as pl
import pytest
from pydantic import ValidationError

from portable_batch_execution.packs.replay_reduction.canonicalize import (
    StructuralCanonicalizeError,
    execute_structural_canonicalize,
)
from portable_batch_execution.packs.replay_reduction.causal_grid import (
    execute_causal_grid_extract,
)
from portable_batch_execution.packs.replay_reduction.event_window import (
    execute_event_window_extract,
)
from portable_batch_execution.packs.replay_reduction.models import (
    CanonicalTradeProfile,
    StructuralCanonicalizeParams,
)

_PROJECTION_V1 = "pbe.replay.json-scalar-projection.v1"


def _raw_json(*, event_id: int, direction: str, event_time_ms: int) -> str:
    return json.dumps(
        {
            "tick": {
                "event_id": event_id,
                "direction": direction,
                "event_time_ms": event_time_ms,
            }
        }
    )


def _chain_row(**fields):
    base = {
        "symbol": "AAA",
        "block": 5,
        "timestamp_ms": 9999,
        "price": 1.0,
        "seq": 0,
        "identity_norm": "1",
        "raw_json": _raw_json(event_id=1, direction="neutral", event_time_ms=3000),
    }
    base.update(fields)
    return base


def _identity_projection(**overrides):
    spec = {
        "schema_version": _PROJECTION_V1,
        "source_column": "raw_json",
        "key_path": ("tick", "event_id"),
        "scalar_type": "integer",
        "output_column": "identity",
    }
    spec.update(overrides)
    return spec


def _direction_projection(**overrides):
    spec = {
        "schema_version": _PROJECTION_V1,
        "source_column": "raw_json",
        "key_path": ("tick", "direction"),
        "scalar_type": "string",
        "output_column": "direction_tag",
    }
    spec.update(overrides)
    return spec


def _time_projection(**overrides):
    spec = {
        "schema_version": _PROJECTION_V1,
        "source_column": "raw_json",
        "key_path": ("tick", "event_time_ms"),
        "scalar_type": "integer",
        "output_column": "event_time_ms",
    }
    spec.update(overrides)
    return spec


def _projected_profile(**overrides):
    base = {
        "schema_version": "pbe.replay.canonical-trade-profile.v1",
        "identity_source_column": "identity",
        "identity_normalized_column": "identity_norm",
        "measurement_core_fields": ["price"],
        "json_scalar_projections": (
            _identity_projection(),
            _direction_projection(),
            _time_projection(),
        ),
    }
    base.update(overrides)
    return base


def _write(path, rows):
    pl.DataFrame(rows).write_parquet(path)
    return path


def test_structural_canonicalize_projects_positive_integer_identity(tmp_path):
    path = _write(
        tmp_path / "part.parquet",
        [
            _chain_row(
                identity_norm="42",
                raw_json=_raw_json(event_id=42, direction="neutral", event_time_ms=1000),
            ),
            _chain_row(
                identity_norm="43",
                raw_json=_raw_json(event_id=43, direction="neutral", event_time_ms=1001),
            ),
        ],
    )
    params = {
        "schema_version": "pbe.replay.structural-canonicalize.v1",
        "identity_source_column": "identity",
        "identity_normalized_column": "identity_norm",
        "measurement_core_fields": ["price"],
        "json_scalar_projections": (_identity_projection(),),
    }
    state = execute_structural_canonicalize([path], params)
    assert state.positive_group_count == 2
    assert state.positive_row_count == 2


def test_sentinel_exact_match_uses_projected_string(tmp_path):
    path = _write(
        tmp_path / "part.parquet",
        [
            _chain_row(
                identity_norm="0",
                raw_json=_raw_json(event_id=0, direction="witness", event_time_ms=100),
            ),
            _chain_row(
                identity_norm="1",
                raw_json=_raw_json(event_id=1, direction="neutral", event_time_ms=200),
            ),
        ],
    )
    params = {
        "schema_version": "pbe.replay.structural-canonicalize.v1",
        "identity_source_column": "identity",
        "identity_normalized_column": "identity_norm",
        "measurement_core_fields": ["price"],
        "json_scalar_projections": (
            _identity_projection(),
            _direction_projection(),
        ),
        "sentinel": {
            "identity_equals": 0,
            "exact_match_fields": {"direction_tag": "witness"},
        },
    }
    state = execute_structural_canonicalize([path], params)
    assert state.witness_row_count == 1
    assert state.positive_group_count == 1


def test_event_window_uses_projected_timestamp(tmp_path):
    path = _write(
        tmp_path / "part.parquet",
        [
            _chain_row(
                identity_norm="1",
                block=4,
                raw_json=_raw_json(event_id=1, direction="neutral", event_time_ms=2900),
                price=1.0,
            ),
            _chain_row(
                identity_norm="2",
                block=4,
                raw_json=_raw_json(event_id=2, direction="neutral", event_time_ms=3100),
                price=2.0,
            ),
        ],
    )
    request = {
        "schema_version": "pbe.replay.event-window-extract.v3",
        "request_id": "req-proj",
        "symbol": "AAA",
        "symbol_column": "symbol",
        "block_column": "block",
        "timestamp_column": "event_time_ms",
        "decision_timestamp_ms": 5000,
        "causal_cutoff_block": 5,
        "as_of_measurement_field": "price",
        "canonical_trade_profile": _projected_profile(),
        "trailing_windows": (
            {
                "fact_id": "trail",
                "measurement_field": "price",
                "trailing_width_ms": 2000,
            },
        ),
        "tie_break_columns": ("event_time_ms", "seq"),
    }
    facts = {
        item["fact_id"]: item["value"]
        for item in execute_event_window_extract([path], request)["facts"]
    }
    assert facts["trail.sum"] == 2.0
    assert facts["trail.count"] == 1


def test_causal_grid_with_projected_identity_and_timestamp(tmp_path):
    trades = _write(
        tmp_path / "trades.parquet",
        [
            _chain_row(
                identity_norm="1",
                block=90,
                raw_json=_raw_json(event_id=1, direction="neutral", event_time_ms=8_000),
                price=10.0,
                notional=50.0,
            ),
            _chain_row(
                identity_norm="2",
                block=100,
                raw_json=_raw_json(event_id=2, direction="neutral", event_time_ms=9_500),
                price=10.0,
                notional=25.0,
            ),
        ],
    )
    witness = _write(
        tmp_path / "witness.parquet",
        [
            {"block": 90, "timestamp_ms": 8_000},
            {"block": 100, "timestamp_ms": 9_500},
            {"block": 110, "timestamp_ms": 10_000},
        ],
    )
    mapping = {
        "canonical_trade_profile": _projected_profile(),
        "symbol_column": "symbol",
        "block_column": "block",
        "timestamp_column": "event_time_ms",
        "price_field": "price",
        "notional_field": "notional",
    }
    request = {
        "schema_version": "pbe.replay.causal-grid-extract.v1",
        "request_id": "grid-proj",
        "target_symbols": ("AAA",),
        "input_roles": (
            {"input_index": 0, "role": "canonical_trade"},
            {"input_index": 1, "role": "causal_witness"},
        ),
        "causal_witness_mapping": {
            "block_column": "block",
            "timestamp_column": "timestamp_ms",
        },
        "canonical_trade_mapping": mapping,
        "emit_grid": {
            "start_timestamp_ms": 10_000,
            "end_timestamp_ms": 10_000,
            "step_ms": 5_000,
        },
        "partition": {
            "emit_start_ms": 10_000,
            "emit_end_ms": 10_000,
            "overlap_ms": 60_000,
        },
        "as_of_measurement_field": "price",
        "trailing_windows": (
            {
                "fact_id": "trade_notional_60s",
                "measurement_field": "notional",
                "trailing_width_ms": 60_000,
            },
        ),
        "tie_break_columns": ("event_time_ms", "seq"),
    }
    row = execute_causal_grid_extract([trades, witness], request)["rows"][0]
    facts = {item["fact_id"]: item["value"] for item in row["facts"]}
    assert facts["trade_notional_60s.sum"] == 75.0


def test_cross_shard_projection_equivalent_to_single_file(tmp_path):
    rows_a = [
        _chain_row(
            identity_norm="1",
            raw_json=_raw_json(event_id=1, direction="neutral", event_time_ms=1000),
        )
    ]
    rows_b = [
        _chain_row(
            identity_norm="2",
            raw_json=_raw_json(event_id=2, direction="neutral", event_time_ms=1001),
        )
    ]
    path_a = _write(tmp_path / "a.parquet", rows_a)
    path_b = _write(tmp_path / "b.parquet", rows_b)
    params = {
        "schema_version": "pbe.replay.structural-canonicalize.v1",
        "identity_source_column": "identity",
        "identity_normalized_column": "identity_norm",
        "measurement_core_fields": ["price"],
        "json_scalar_projections": (_identity_projection(),),
    }
    split = execute_structural_canonicalize([path_a, path_b], params)
    merged = execute_structural_canonicalize(
        [_write(tmp_path / "merged.parquet", rows_a + rows_b)],
        params,
    )
    assert split.positive_group_count == merged.positive_group_count == 2


@pytest.mark.parametrize(
    ("raw_json", "message"),
    [
        ("{not json", "malformed"),
        (json.dumps({"tick": {"event_id": True}}), "type mismatch"),
        (json.dumps({"tick": {"event_id": 1.5}}), "type mismatch"),
        (json.dumps({"tick": {}}), "missing required key"),
        (json.dumps([1, 2, 3]), "traverses non-object"),
    ],
)
def test_projection_fail_closed_on_bad_json(tmp_path, raw_json, message):
    path = _write(
        tmp_path / "bad.parquet",
        [_chain_row(raw_json=raw_json, identity_norm="1")],
    )
    params = {
        "schema_version": "pbe.replay.structural-canonicalize.v1",
        "identity_source_column": "identity",
        "identity_normalized_column": "identity_norm",
        "measurement_core_fields": ["price"],
        "json_scalar_projections": (_identity_projection(),),
    }
    with pytest.raises(StructuralCanonicalizeError, match=message):
        execute_structural_canonicalize([path], params)


def test_projection_rejects_output_column_collision(tmp_path):
    path = _write(tmp_path / "part.parquet", [_chain_row(price=1.0)])
    params = {
        "schema_version": "pbe.replay.structural-canonicalize.v1",
        "identity_source_column": "identity",
        "identity_normalized_column": "identity_norm",
        "measurement_core_fields": ["price"],
        "json_scalar_projections": (
            _identity_projection(output_column="price"),
        ),
    }
    with pytest.raises(StructuralCanonicalizeError, match="collides"):
        execute_structural_canonicalize([path], params)


def test_projection_rejects_reserved_output_name():
    with pytest.raises(ValidationError, match="reserved"):
        StructuralCanonicalizeParams.model_validate(
            {
                "schema_version": "pbe.replay.structural-canonicalize.v1",
                "identity_source_column": "identity",
                "identity_normalized_column": "identity_norm",
                "measurement_core_fields": ["price"],
                "json_scalar_projections": (
                    _identity_projection(output_column="_identity_int"),
                ),
            }
        )


def test_projection_rejects_duplicate_output_columns():
    with pytest.raises(ValidationError, match="duplicate"):
        CanonicalTradeProfile.model_validate(
            _projected_profile(
                json_scalar_projections=(
                    _identity_projection(),
                    _identity_projection(output_column="identity"),
                )
            )
        )


def test_top_level_columns_backward_compatible_event_window(tmp_path):
    path = _write(
        tmp_path / "legacy.parquet",
        [
            {
                "identity": 1,
                "identity_norm": "1",
                "symbol": "AAA",
                "block": 5,
                "timestamp_ms": 3100,
                "price": 1.0,
                "seq": 0,
            }
        ],
    )
    request = {
        "schema_version": "pbe.replay.event-window-extract.v3",
        "request_id": "legacy",
        "symbol": "AAA",
        "symbol_column": "symbol",
        "block_column": "block",
        "timestamp_column": "timestamp_ms",
        "decision_timestamp_ms": 5000,
        "causal_cutoff_block": 5,
        "as_of_measurement_field": "price",
        "canonical_trade_profile": {
            "schema_version": "pbe.replay.canonical-trade-profile.v1",
            "identity_source_column": "identity",
            "identity_normalized_column": "identity_norm",
            "measurement_core_fields": ["price"],
        },
        "trailing_windows": (
            {
                "fact_id": "trail",
                "measurement_field": "price",
                "trailing_width_ms": 2000,
            },
        ),
        "tie_break_columns": ("timestamp_ms", "seq"),
    }
    facts = {
        item["fact_id"]: item["value"]
        for item in execute_event_window_extract([path], request)["facts"]
    }
    assert facts["trail.count"] == 1


def test_top_level_columns_backward_compatible_causal_grid(tmp_path):
    trades = _write(
        tmp_path / "trades.parquet",
        [
            {
                "identity": 1,
                "identity_norm": "1",
                "symbol": "AAA",
                "block": 100,
                "timestamp_ms": 9_000,
                "price": 10.0,
                "notional": 100.0,
                "seq": 0,
            }
        ],
    )
    witness = _write(tmp_path / "w.parquet", [{"block": 100, "timestamp_ms": 9_000}])
    request = {
        "schema_version": "pbe.replay.causal-grid-extract.v1",
        "request_id": "legacy-grid",
        "target_symbols": ("AAA",),
        "input_roles": (
            {"input_index": 0, "role": "canonical_trade"},
            {"input_index": 1, "role": "causal_witness"},
        ),
        "causal_witness_mapping": {
            "block_column": "block",
            "timestamp_column": "timestamp_ms",
        },
        "canonical_trade_mapping": {
            "canonical_trade_profile": {
                "schema_version": "pbe.replay.canonical-trade-profile.v1",
                "identity_source_column": "identity",
                "identity_normalized_column": "identity_norm",
                "measurement_core_fields": ["price"],
            },
            "symbol_column": "symbol",
            "block_column": "block",
            "timestamp_column": "timestamp_ms",
            "price_field": "price",
            "notional_field": "notional",
        },
        "emit_grid": {
            "start_timestamp_ms": 10_000,
            "end_timestamp_ms": 10_000,
            "step_ms": 5_000,
        },
        "partition": {
            "emit_start_ms": 10_000,
            "emit_end_ms": 10_000,
            "overlap_ms": 60_000,
        },
        "as_of_measurement_field": "price",
    }
    rows = execute_causal_grid_extract([trades, witness], request)["rows"]
    assert len(rows) == 1
