from __future__ import annotations

import polars as pl
import pytest

from portable_batch_execution.packs.replay_reduction.canonicalize import (
    StructuralCanonicalizeError,
)
from portable_batch_execution.packs.replay_reduction.paired_fill_reduce import (
    execute_paired_fill_reduce,
)


def _row(**overrides):
    row = {
        "identity": 7,
        "identity_norm": "7",
        "role": "owner",
        "event_type": "settlement",
        "price": 0.0,
        "size": 4.0,
        "raw_size": 4.0,
        "notional": 0.0,
        "start": 4.0,
        "signed": -4.0,
    }
    row.update(overrides)
    return row


def _request():
    return {
        "schema_version": "pbe.replay.paired-fill-reduce.v2",
        "request_id": "state-transition",
        "identity_mapping": {
            "identity_source_column": "identity",
            "identity_normalized_column": "identity_norm",
        },
        "pair_mapping": {
            "pair_role_column": "role",
            "aggressor_role_value": "owner",
            "passive_role_value": "system",
            "measurement_core_fields": ["identity", "price", "size", "notional"],
            "start_position_column": "start",
            "signed_execution_column": "signed",
        },
        "row_invariants": [
            {"schema_version": "pbe.replay.row-invariant.positive-finite.v1", "column": "price"},
            {"schema_version": "pbe.replay.row-invariant.positive-finite.v1", "column": "size"},
            {"schema_version": "pbe.replay.row-invariant.positive-finite.v1", "column": "notional"},
            {
                "schema_version": "pbe.replay.row-invariant.numeric.v1",
                "left_column": "size",
                "right_column": "raw_size",
            },
        ],
        "state_transition_handling": {
            "schema_version": "pbe.replay.paired-fill-state-transition.v1",
            "marker_exact_match_fields": {"event_type": "settlement"},
            "zero_numeric_fields": ["price", "notional"],
            "shared_fields": ["identity", "size"],
            "state_owner_role_value": "owner",
            "protocol_counterparty_role_value": "system",
            "bypass_row_invariant_columns": ["price", "notional"],
        },
        "partition": {"terminal": True},
        "max_output_rows": 100,
        "max_exception_rows": 10,
    }


def _write(path, rows):
    pl.DataFrame(rows).write_parquet(path)
    return path


def test_state_transition_emits_distinct_non_economic_record(tmp_path):
    rows = [
        _row(),
        _row(role="system", start=0.0, signed=4.0),
    ]
    result = execute_paired_fill_reduce([_write(tmp_path / "rows.parquet", rows)], _request())
    assert result["schema_version"] == "pbe.replay.paired-fill-reduce-result.v2"
    assert result["summary"]["state_transition_count"] == 1
    row = result["ledger_rows"][0]
    assert row["classification"] == "state_transition"
    assert row["state_owner"]["post_position"] == 0.0
    assert row["protocol_counterparty"]["inventory_effect"] == "none"
    assert row["economic_fill"] is False
    assert row["economic_notional"] == 0.0


def test_state_transition_is_sign_symmetric(tmp_path):
    rows = [
        _row(start=-4.0, signed=4.0),
        _row(role="system", start=0.0, signed=-4.0),
    ]
    result = execute_paired_fill_reduce([_write(tmp_path / "short.parquet", rows)], _request())
    assert result["ledger_rows"][0]["state_owner"]["post_position"] == 0.0


def test_malformed_marked_transition_fails_closed(tmp_path):
    rows = [
        _row(price=1.0, notional=4.0),
        _row(role="system", start=0.0, signed=4.0, price=1.0, notional=4.0),
    ]
    with pytest.raises(StructuralCanonicalizeError):
        execute_paired_fill_reduce([_write(tmp_path / "bad.parquet", rows)], _request())


def test_incomplete_marked_transition_fails_closed(tmp_path):
    with pytest.raises(StructuralCanonicalizeError):
        execute_paired_fill_reduce([_write(tmp_path / "one.parquet", [_row()])], _request())


def test_ordinary_pair_remains_economic_under_v2(tmp_path):
    rows = [
        _row(event_type="trade", price=10.0, notional=40.0, start=0.0, signed=4.0),
        _row(
            role="system",
            event_type="trade",
            price=10.0,
            notional=40.0,
            start=0.0,
            signed=-4.0,
        ),
    ]
    result = execute_paired_fill_reduce([_write(tmp_path / "trade.parquet", rows)], _request())
    assert result["ledger_rows"][0]["classification"] == "complete_pair"
    assert result["summary"]["state_transition_count"] == 0


def test_transition_pair_can_cross_input_boundary(tmp_path):
    first = _write(tmp_path / "a.parquet", [_row()])
    second = _write(
        tmp_path / "b.parquet",
        [_row(role="system", start=0.0, signed=4.0)],
    )
    result = execute_paired_fill_reduce([first, second], _request())
    assert result["ledger_rows"][0]["classification"] == "state_transition"
    assert result["summary"]["state_transition_count"] == 1


def test_transition_cannot_bypass_nonzero_semantic_invariant():
    request = _request()
    request["state_transition_handling"]["bypass_row_invariant_columns"].append(
        "size"
    )
    with pytest.raises(ValueError):
        from portable_batch_execution.packs.replay_reduction.models import (
            PairedFillReduceRequest,
        )

        PairedFillReduceRequest.model_validate(request)
