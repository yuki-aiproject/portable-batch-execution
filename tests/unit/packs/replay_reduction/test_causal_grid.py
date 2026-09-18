from itertools import permutations

import polars as pl
import pytest
from pydantic import ValidationError

from portable_batch_execution.packs.replay_reduction.canonicalize import (
    StructuralCanonicalizeError,
)
from portable_batch_execution.packs.replay_reduction.causal_grid import (
    execute_causal_grid_extract,
)
from portable_batch_execution.packs.replay_reduction.models import (
    CausalGridExtractRequest,
)

_PROFILE = {
    "schema_version": "pbe.replay.canonical-trade-profile.v1",
    "identity_source_column": "identity",
    "identity_normalized_column": "identity_norm",
    "measurement_core_fields": ["price"],
}

_SENTINEL = {
    "identity_equals": 0,
    "exact_match_fields": {"identity_norm": None},
}


def _write(path, rows):
    pl.DataFrame(rows).write_parquet(path)
    return path


def _request(**overrides):
    base = {
        "schema_version": "pbe.replay.causal-grid-extract.v1",
        "request_id": "grid-1",
        "target_symbols": ("AAA",),
        "input_roles": (
            {"input_index": 0, "role": "canonical_trade"},
        ),
        "causal_witness_mapping": {
            "block_column": "block",
            "timestamp_column": "timestamp_ms",
        },
        "canonical_trade_mapping": {
            "canonical_trade_profile": _PROFILE,
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
            "hard_gap_missing_dates": (),
        },
        "as_of_measurement_field": "price",
        "as_of_offsets_ms": (0, 5_000),
        "trailing_windows": (
            {
                "fact_id": "trade_notional_60s",
                "measurement_field": "notional",
                "trailing_width_ms": 60_000,
            },
        ),
        "tie_break_columns": ("timestamp_ms", "seq"),
    }
    base.update(overrides)
    return base


def _trade_row(**fields):
    base = {
        "identity": 1,
        "identity_norm": "1",
        "symbol": "AAA",
        "block": 100,
        "timestamp_ms": 9_000,
        "price": 10.0,
        "notional": 100.0,
        "seq": 0,
    }
    base.update(fields)
    return base


def _witness_row(**fields):
    base = {"block": 100, "timestamp_ms": 9_000}
    base.update(fields)
    return base


def _facts_by_id(row):
    return {item["fact_id"]: item["value"] for item in row["facts"]}


def test_positive_identity_duplicate_collapse(tmp_path):
    path = _write(
        tmp_path / "trades.parquet",
        [
            _trade_row(identity=1, identity_norm="1", block=90, timestamp_ms=8_000, price=10.0, notional=50.0),
            _trade_row(identity=1, identity_norm="1", block=91, timestamp_ms=8_100, price=10.0, notional=50.0),
            _trade_row(identity=2, identity_norm="2", block=100, timestamp_ms=9_500, notional=25.0),
        ],
    )
    witness = _write(
        tmp_path / "w.parquet",
        [
            _witness_row(block=90, timestamp_ms=8_000),
            _witness_row(block=100, timestamp_ms=9_500),
            _witness_row(block=110, timestamp_ms=10_000),
        ],
    )
    row = execute_causal_grid_extract(
        [path, witness],
        _request(
            input_roles=(
                {"input_index": 0, "role": "canonical_trade"},
                {"input_index": 1, "role": "causal_witness"},
            ),
        ),
    )["rows"][0]
    facts = _facts_by_id(row)
    assert facts["trade_notional_60s.sum"] == 75.0
    assert facts["trade_notional_60s.count"] == 2


def test_sentinel_exclusion(tmp_path):
    profile = {**_PROFILE, "sentinel": _SENTINEL}
    path = _write(
        tmp_path / "trades.parquet",
        [
            _trade_row(identity=0, identity_norm=None, price=1.0, notional=999.0),
            _trade_row(identity=1, identity_norm="1", block=90, timestamp_ms=8_000, notional=10.0),
            _trade_row(identity=2, identity_norm="2", block=110, timestamp_ms=9_000, notional=1.0),
        ],
    )
    request = _request()
    request["canonical_trade_mapping"]["canonical_trade_profile"] = profile
    witness = _write(
        tmp_path / "w.parquet",
        [
            _witness_row(block=90, timestamp_ms=8_000),
            _witness_row(block=100, timestamp_ms=9_000),
            _witness_row(block=110, timestamp_ms=10_000),
        ],
    )
    request["input_roles"] = (
        {"input_index": 0, "role": "canonical_trade"},
        {"input_index": 1, "role": "causal_witness"},
    )
    facts = _facts_by_id(
        execute_causal_grid_extract([path, witness], request)["rows"][0]
    )
    assert facts["trade_notional_60s.sum"] == 10.0


def test_malformed_identity_fail_closed(tmp_path):
    path = _write(tmp_path / "trades.parquet", [_trade_row(identity=-1, identity_norm="-1")])
    with pytest.raises(StructuralCanonicalizeError):
        execute_causal_grid_extract([path], _request())


def test_trade_and_liquidation_witness_cutoff(tmp_path):
    trades = _write(
        tmp_path / "trades.parquet",
        [
            _trade_row(block=100, timestamp_ms=8_000),
            _trade_row(identity=2, identity_norm="2", block=110, timestamp_ms=9_000),
        ],
    )
    liq = _write(
        tmp_path / "liq.parquet",
        [
            _witness_row(block=100, timestamp_ms=8_000),
            _witness_row(block=120, timestamp_ms=9_500),
        ],
    )
    request = _request(
        input_roles=(
            {"input_index": 0, "role": "canonical_trade"},
            {"input_index": 1, "role": "causal_witness"},
        ),
        emit_grid={
            "start_timestamp_ms": 10_000,
            "end_timestamp_ms": 10_000,
            "step_ms": 5_000,
        },
    )
    row = execute_causal_grid_extract([trades, liq], request)["rows"][0]
    assert row["causal_cutoff_block"] == 110


def test_witness_block_excluded_from_measurement(tmp_path):
    trades = _write(
        tmp_path / "trades.parquet",
        [
            _trade_row(block=100, timestamp_ms=8_000, price=1.0),
            _trade_row(
                identity=2,
                identity_norm="2",
                block=120,
                timestamp_ms=9_500,
                price=9.0,
                notional=1.0,
            ),
        ],
    )
    liq = _write(tmp_path / "liq.parquet", [_witness_row(block=120, timestamp_ms=9_500)])
    row = execute_causal_grid_extract(
        [trades, liq],
        _request(
            input_roles=(
                {"input_index": 0, "role": "canonical_trade"},
                {"input_index": 1, "role": "causal_witness"},
            ),
        ),
    )["rows"][0]
    facts = _facts_by_id(row)
    assert row["causal_cutoff_block"] == 100
    assert facts["as_of.0"] == 1.0


def test_no_predecessor_emits_unavailable(tmp_path):
    path = _write(tmp_path / "trades.parquet", [_trade_row(block=100, timestamp_ms=9_000)])
    row = execute_causal_grid_extract([path], _request())["rows"][0]
    assert row["causal_cutoff_block"] is None
    facts = _facts_by_id(row)
    assert facts["as_of.0"] is None
    assert facts["trade_notional_60s.sum"] is None


def test_hard_gap_boundary(tmp_path):
    path = _write(tmp_path / "trades.parquet", [_trade_row(timestamp_ms=86_400_000)])
    request = _request(
        partition={
            "emit_start_ms": 86_400_000,
            "emit_end_ms": 86_400_000,
            "overlap_ms": 60_000,
            "hard_gap_missing_dates": ("1970-01-02",),
        },
        emit_grid={
            "start_timestamp_ms": 86_400_000,
            "end_timestamp_ms": 86_400_000,
            "step_ms": 5_000,
        },
    )
    with pytest.raises(StructuralCanonicalizeError):
        execute_causal_grid_extract([path], request)


def test_as_of_tie_breaking(tmp_path):
    path = _write(
        tmp_path / "trades.parquet",
        [
            _trade_row(identity=1, identity_norm="1", block=100, timestamp_ms=10_000, price=1.0, seq=0),
            _trade_row(identity=2, identity_norm="2", block=101, timestamp_ms=10_000, price=2.0, seq=1),
            _trade_row(identity=3, identity_norm="3", block=91, timestamp_ms=5_000, price=3.0, seq=0),
            _trade_row(identity=4, identity_norm="4", block=92, timestamp_ms=5_000, price=4.0, seq=1),
        ],
    )
    request = _request(
        input_roles=({"input_index": 0, "role": "canonical_trade"},),
        emit_grid={"start_timestamp_ms": 10_000, "end_timestamp_ms": 10_000, "step_ms": 5_000},
    )
    request["canonical_trade_mapping"]["canonical_trade_profile"] = {
        **_PROFILE,
        "sentinel": None,
    }
    liq = _write(
        tmp_path / "witness.parquet",
        [
            _witness_row(block=90, timestamp_ms=4_000),
            _witness_row(block=100, timestamp_ms=8_000),
            _witness_row(block=104, timestamp_ms=10_000),
        ],
    )
    request["input_roles"] = (
        {"input_index": 0, "role": "canonical_trade"},
        {"input_index": 1, "role": "causal_witness"},
    )
    row = execute_causal_grid_extract([path, liq], request)["rows"][0]
    facts = _facts_by_id(row)
    assert row["causal_cutoff_block"] == 101
    assert facts["as_of.0"] == 2.0
    assert facts["as_of.5000"] == 4.0


def test_trailing_60s_notional(tmp_path):
    path = _write(
        tmp_path / "trades.parquet",
        [
            _trade_row(identity=1, identity_norm="1", block=95, timestamp_ms=9_000, notional=10.0),
            _trade_row(identity=2, identity_norm="2", block=102, timestamp_ms=9_500, notional=20.0),
            _trade_row(
                identity=3,
                identity_norm="3",
                block=115,
                timestamp_ms=10_000,
                notional=100.0,
            ),
        ],
    )
    liq = _write(
        tmp_path / "w.parquet",
        [
            _witness_row(block=90, timestamp_ms=8_000),
            _witness_row(block=101, timestamp_ms=9_001),
            _witness_row(block=110, timestamp_ms=10_000),
        ],
    )
    facts = _facts_by_id(
        execute_causal_grid_extract(
            [path, liq],
            _request(
                input_roles=(
                    {"input_index": 0, "role": "canonical_trade"},
                    {"input_index": 1, "role": "causal_witness"},
                ),
            ),
        )["rows"][0]
    )
    assert facts["trade_notional_60s.sum"] == 30.0


def test_zero_liquidation_control_grid_row_still_emitted(tmp_path):
    path = _write(tmp_path / "trades.parquet", [_trade_row(timestamp_ms=9_000)])
    rows = execute_causal_grid_extract([path], _request(target_symbols=("AAA", "BBB")))[
        "rows"
    ]
    assert len(rows) == 2
    assert {row["symbol"] for row in rows} == {"AAA", "BBB"}


def test_partition_carry_equivalence(tmp_path):
    trades = _write(
        tmp_path / "trades.parquet",
        [
            _trade_row(identity=1, identity_norm="1", timestamp_ms=40_000, notional=1.0),
            _trade_row(identity=2, identity_norm="2", timestamp_ms=80_000, notional=2.0),
            _trade_row(identity=3, identity_norm="3", timestamp_ms=120_000, notional=4.0),
        ],
    )
    witness = _write(
        tmp_path / "w.parquet",
        [
            _witness_row(block=10, timestamp_ms=30_000),
            _witness_row(block=20, timestamp_ms=50_000),
            _witness_row(block=30, timestamp_ms=90_000),
            _witness_row(block=40, timestamp_ms=130_000),
        ],
    )
    full = _request(
        input_roles=(
            {"input_index": 0, "role": "canonical_trade"},
            {"input_index": 1, "role": "causal_witness"},
        ),
        emit_grid={"start_timestamp_ms": 50_000, "end_timestamp_ms": 120_000, "step_ms": 10_000},
        partition={
            "emit_start_ms": 50_000,
            "emit_end_ms": 120_000,
            "overlap_ms": 70_000,
            "hard_gap_missing_dates": (),
        },
    )
    one_pass = execute_causal_grid_extract([trades, witness], full)["rows"]

    first = dict(full)
    first["partition"] = {
        "emit_start_ms": 50_000,
        "emit_end_ms": 80_000,
        "overlap_ms": 70_000,
        "hard_gap_missing_dates": (),
    }
    first_result = execute_causal_grid_extract([trades, witness], first)
    second = dict(full)
    second["partition"] = {
        "emit_start_ms": 90_000,
        "emit_end_ms": 120_000,
        "overlap_ms": 70_000,
        "hard_gap_missing_dates": (),
        "incoming_carry": first_result["outgoing_carry"],
    }
    merged = first_result["rows"] + execute_causal_grid_extract([trades, witness], second)["rows"]
    assert merged == one_pass


def test_outgoing_carry_excludes_post_emit_end_state(tmp_path):
    trades = _write(
        tmp_path / "trades.parquet",
        [
            _trade_row(identity=1, identity_norm="1", block=90, timestamp_ms=40_000, notional=1.0),
            _trade_row(identity=2, identity_norm="2", block=95, timestamp_ms=80_000, notional=2.0),
            _trade_row(identity=3, identity_norm="3", block=100, timestamp_ms=120_000, notional=4.0),
        ],
    )
    witness = _write(
        tmp_path / "w.parquet",
        [
            _witness_row(block=10, timestamp_ms=30_000),
            _witness_row(block=20, timestamp_ms=50_000),
            _witness_row(block=30, timestamp_ms=70_000),
            _witness_row(block=35, timestamp_ms=80_000),
            _witness_row(block=99, timestamp_ms=90_000),
            _witness_row(block=100, timestamp_ms=130_000),
        ],
    )
    request = _request(
        input_roles=(
            {"input_index": 0, "role": "canonical_trade"},
            {"input_index": 1, "role": "causal_witness"},
        ),
        emit_grid={"start_timestamp_ms": 50_000, "end_timestamp_ms": 80_000, "step_ms": 10_000},
        partition={
            "emit_start_ms": 50_000,
            "emit_end_ms": 80_000,
            "overlap_ms": 70_000,
            "hard_gap_missing_dates": (),
        },
    )
    carry = execute_causal_grid_extract([trades, witness], request)["outgoing_carry"]
    truncated_witness = _write(
        tmp_path / "w_trunc.parquet",
        [
            _witness_row(block=10, timestamp_ms=30_000),
            _witness_row(block=20, timestamp_ms=50_000),
            _witness_row(block=30, timestamp_ms=70_000),
            _witness_row(block=35, timestamp_ms=80_000),
        ],
    )
    reference_carry = execute_causal_grid_extract(
        [trades, truncated_witness],
        request,
    )["outgoing_carry"]
    assert carry == reference_carry
    assert all(row["exchange_time_ms"] <= 80_000 for row in carry["trade_rows"])
    assert not any(row["exchange_time_ms"] == 120_000 for row in carry["trade_rows"])


def test_insufficient_partition_overlap_fails_closed(tmp_path):
    path = _write(tmp_path / "trades.parquet", [_trade_row(timestamp_ms=9_000)])
    request = _request(
        partition={
            "emit_start_ms": 10_000,
            "emit_end_ms": 10_000,
            "overlap_ms": 1_000,
            "hard_gap_missing_dates": (),
        },
    )
    with pytest.raises(ValueError, match="overlap_ms shorter than required replay lookback"):
        execute_causal_grid_extract([path], request)


def test_multi_shard_deterministic_equivalence(tmp_path):
    rows_a = [
        _trade_row(identity=1, identity_norm="1", timestamp_ms=9_000, notional=1.0),
        _trade_row(identity=2, identity_norm="2", timestamp_ms=9_500, notional=2.0),
    ]
    rows_b = [
        _trade_row(identity=3, identity_norm="3", timestamp_ms=9_800, notional=3.0),
    ]
    path_a = _write(tmp_path / "a.parquet", rows_a)
    path_b = _write(tmp_path / "b.parquet", rows_b)
    witness = _write(
        tmp_path / "w.parquet",
        [
            _witness_row(block=10, timestamp_ms=8_000),
            _witness_row(block=20, timestamp_ms=9_000),
            _witness_row(block=30, timestamp_ms=10_000),
        ],
    )
    combined = execute_causal_grid_extract(
        [path_a, path_b, witness],
        _request(
            input_roles=(
                {"input_index": 0, "role": "canonical_trade"},
                {"input_index": 1, "role": "canonical_trade"},
                {"input_index": 2, "role": "causal_witness"},
            ),
        ),
    )["rows"]
    single = execute_causal_grid_extract(
        [_write(tmp_path / "merged.parquet", rows_a + rows_b), witness],
        _request(
            input_roles=(
                {"input_index": 0, "role": "canonical_trade"},
                {"input_index": 1, "role": "causal_witness"},
            ),
        ),
    )["rows"]
    assert combined == single


def test_non_contiguous_same_file_duplicate_identity_collapses(tmp_path):
    path = _write(
        tmp_path / "trades.parquet",
        [
            _trade_row(identity=1, identity_norm="1", block=90, timestamp_ms=8_000, notional=10.0),
            _trade_row(identity=2, identity_norm="2", block=91, timestamp_ms=8_100, notional=20.0),
            _trade_row(identity=1, identity_norm="1", block=92, timestamp_ms=8_200, notional=30.0),
        ],
    )
    witness = _write(
        tmp_path / "w.parquet",
        [
            _witness_row(block=90, timestamp_ms=8_000),
            _witness_row(block=100, timestamp_ms=9_000),
            _witness_row(block=110, timestamp_ms=10_000),
        ],
    )
    facts = _facts_by_id(
        execute_causal_grid_extract(
            [path, witness],
            _request(
                input_roles=(
                    {"input_index": 0, "role": "canonical_trade"},
                    {"input_index": 1, "role": "causal_witness"},
                ),
            ),
        )["rows"][0]
    )
    assert facts["trade_notional_60s.sum"] == 50.0
    assert facts["trade_notional_60s.count"] == 2


def test_cross_shard_duplicate_identity_collapses_once(tmp_path):
    path_a = _write(
        tmp_path / "a.parquet",
        [
            _trade_row(identity=1, identity_norm="1", block=90, timestamp_ms=8_000, notional=10.0),
            _trade_row(identity=2, identity_norm="2", block=91, timestamp_ms=8_050, notional=20.0),
        ],
    )
    path_b = _write(
        tmp_path / "b.parquet",
        [_trade_row(identity=1, identity_norm="1", block=92, timestamp_ms=8_100, notional=5.0)],
    )
    witness = _write(
        tmp_path / "w.parquet",
        [
            _witness_row(block=90, timestamp_ms=8_000),
            _witness_row(block=100, timestamp_ms=9_000),
            _witness_row(block=110, timestamp_ms=10_000),
        ],
    )
    facts = _facts_by_id(
        execute_causal_grid_extract(
            [path_a, path_b, witness],
            _request(
                input_roles=(
                    {"input_index": 0, "role": "canonical_trade"},
                    {"input_index": 1, "role": "canonical_trade"},
                    {"input_index": 2, "role": "causal_witness"},
                ),
            ),
        )["rows"][0]
    )
    assert facts["trade_notional_60s.sum"] == 25.0
    assert facts["trade_notional_60s.count"] == 2


def test_conflicting_duplicate_core_fail_closed(tmp_path):
    path = _write(
        tmp_path / "trades.parquet",
        [
            _trade_row(identity=1, identity_norm="1", block=90, timestamp_ms=8_000, price=10.0),
            _trade_row(identity=1, identity_norm="1", block=91, timestamp_ms=8_100, price=11.0),
        ],
    )
    with pytest.raises(StructuralCanonicalizeError):
        execute_causal_grid_extract([path], _request())


def test_cross_shard_conflicting_core_fail_closed(tmp_path):
    path_a = _write(
        tmp_path / "a.parquet",
        [
            _trade_row(identity=1, identity_norm="1", block=90, timestamp_ms=8_000, price=10.0),
            _trade_row(identity=2, identity_norm="2", block=91, timestamp_ms=8_050, price=20.0),
        ],
    )
    path_b = _write(
        tmp_path / "b.parquet",
        [_trade_row(identity=1, identity_norm="1", block=92, timestamp_ms=8_100, price=11.0)],
    )
    witness = _write(tmp_path / "w.parquet", [_witness_row(block=90, timestamp_ms=8_000)])
    with pytest.raises(StructuralCanonicalizeError):
        execute_causal_grid_extract(
            [path_a, path_b, witness],
            _request(
                input_roles=(
                    {"input_index": 0, "role": "canonical_trade"},
                    {"input_index": 1, "role": "canonical_trade"},
                    {"input_index": 2, "role": "causal_witness"},
                ),
            ),
        )


def test_output_row_bound_fail_closed(tmp_path):
    path = _write(tmp_path / "trades.parquet", [_trade_row(timestamp_ms=9_000)])
    request = _request(
        target_symbols=("AAA", "BBB", "CCC"),
        emit_grid={
            "start_timestamp_ms": 10_000,
            "end_timestamp_ms": 10_000,
            "step_ms": 5_000,
        },
        max_output_rows=2,
    )
    with pytest.raises(ValueError, match="output row count exceeds limit"):
        execute_causal_grid_extract([path], request)


def test_rolling_window_eviction_at_carry_boundary(tmp_path):
    trades = _write(
        tmp_path / "trades.parquet",
        [
            _trade_row(identity=1, identity_norm="1", block=90, timestamp_ms=40_000, notional=1.0),
            _trade_row(identity=2, identity_norm="2", block=95, timestamp_ms=80_000, notional=2.0),
            _trade_row(identity=3, identity_norm="3", block=100, timestamp_ms=120_000, notional=4.0),
        ],
    )
    witness = _write(
        tmp_path / "w.parquet",
        [
            _witness_row(block=10, timestamp_ms=30_000),
            _witness_row(block=20, timestamp_ms=50_000),
            _witness_row(block=30, timestamp_ms=90_000),
            _witness_row(block=40, timestamp_ms=130_000),
        ],
    )
    first = _request(
        input_roles=(
            {"input_index": 0, "role": "canonical_trade"},
            {"input_index": 1, "role": "causal_witness"},
        ),
        emit_grid={"start_timestamp_ms": 50_000, "end_timestamp_ms": 80_000, "step_ms": 10_000},
        partition={
            "emit_start_ms": 50_000,
            "emit_end_ms": 80_000,
            "overlap_ms": 70_000,
            "hard_gap_missing_dates": (),
        },
    )
    first_result = execute_causal_grid_extract([trades, witness], first)
    carry = first_result["outgoing_carry"]
    assert carry["schema_version"] == "pbe.replay.causal-grid-carry.v4"
    assert len(carry["causal_segment_frontiers"]) <= 2
    assert carry["causal_block_first_ms"] == ()
    assert all(row["exchange_time_ms"] >= 10_000 for row in carry["trade_rows"])


def test_unsorted_witness_matches_sorted_equivalent(tmp_path):
    trades = _write(
        tmp_path / "trades.parquet",
        [
            _trade_row(identity=1, identity_norm="1", block=100, timestamp_ms=8_000),
            _trade_row(identity=2, identity_norm="2", block=110, timestamp_ms=9_000),
        ],
    )
    sorted_witness = _write(
        tmp_path / "w_sorted.parquet",
        [
            _witness_row(block=100, timestamp_ms=8_000),
            _witness_row(block=120, timestamp_ms=9_500),
            _witness_row(block=130, timestamp_ms=10_000),
        ],
    )
    unsorted_witness = _write(
        tmp_path / "w_unsorted.parquet",
        [
            _witness_row(block=130, timestamp_ms=10_000),
            _witness_row(block=100, timestamp_ms=8_000),
            _witness_row(block=120, timestamp_ms=9_500),
        ],
    )
    request = _request(
        input_roles=(
            {"input_index": 0, "role": "canonical_trade"},
            {"input_index": 1, "role": "causal_witness"},
        ),
    )
    sorted_rows = execute_causal_grid_extract([trades, sorted_witness], request)["rows"]
    unsorted_rows = execute_causal_grid_extract([trades, unsorted_witness], request)["rows"]
    assert sorted_rows == unsorted_rows


def _frozen_causal_cutoff(
    observations: list[tuple[int, int]],
    *,
    decision_time_ms: int,
    monotone_ok: bool = True,
) -> int | None:
    if not monotone_ok:
        return None
    eligible = {block for time_ms, block in observations if time_ms <= decision_time_ms}
    if len(eligible) < 2:
        return None
    witness = max(eligible)
    lower = [block for block in eligible if block < witness]
    if not lower:
        return None
    return max(lower)


def test_large_observation_count_compact_carry_and_cutoff(tmp_path):
    witness_rows = [
        _witness_row(block=100 + index, timestamp_ms=8_000 + index)
        for index in range(80_000)
    ]
    trades = _write(
        tmp_path / "trades.parquet",
        [_trade_row(identity=1, identity_norm="1", block=200_000, timestamp_ms=100_000)],
    )
    witness = _write(tmp_path / "w.parquet", witness_rows)
    result = execute_causal_grid_extract(
        [trades, witness],
        _request(
            input_roles=(
                {"input_index": 0, "role": "canonical_trade"},
                {"input_index": 1, "role": "causal_witness"},
            ),
        ),
    )
    row = result["rows"][0]
    carry = result["outgoing_carry"]
    observations = [(8_000 + index, 100 + index) for index in range(80_000)]
    assert row["causal_cutoff_block"] == _frozen_causal_cutoff(
        observations,
        decision_time_ms=10_000,
    )
    assert carry["schema_version"] == "pbe.replay.causal-grid-carry.v4"
    assert len(carry["causal_segment_frontiers"]) == 1
    assert carry["causal_block_first_ms"] == ()


def test_causal_frontier_block_transitions_and_repeated_block(tmp_path):
    witness = _write(
        tmp_path / "w.parquet",
        [
            _witness_row(block=100, timestamp_ms=8_000),
            _witness_row(block=100, timestamp_ms=8_500),
            _witness_row(block=150, timestamp_ms=9_000),
            _witness_row(block=200, timestamp_ms=9_500),
        ],
    )
    trades = _write(
        tmp_path / "trades.parquet",
        [_trade_row(identity=1, identity_norm="1", block=200, timestamp_ms=100_000)],
    )
    row = execute_causal_grid_extract(
        [trades, witness],
        _request(
            input_roles=(
                {"input_index": 0, "role": "canonical_trade"},
                {"input_index": 1, "role": "causal_witness"},
            ),
        ),
    )["rows"][0]
    expected = _frozen_causal_cutoff(
        [(8_000, 100), (8_500, 100), (9_000, 150), (9_500, 200)],
        decision_time_ms=10_000,
    )
    assert row["causal_cutoff_block"] == expected == 150


def test_same_timestamp_witness_block_excluded_from_cutoff(tmp_path):
    trades = _write(
        tmp_path / "trades.parquet",
        [
            _trade_row(identity=1, identity_norm="1", block=100, timestamp_ms=10_000, price=1.0),
            _trade_row(
                identity=2,
                identity_norm="2",
                block=110,
                timestamp_ms=10_000,
                price=2.0,
                seq=1,
            ),
        ],
    )
    witness = _write(
        tmp_path / "w.parquet",
        [
            _witness_row(block=90, timestamp_ms=8_000),
            _witness_row(block=110, timestamp_ms=10_000),
        ],
    )
    row = execute_causal_grid_extract(
        [trades, witness],
        _request(
            input_roles=(
                {"input_index": 0, "role": "canonical_trade"},
                {"input_index": 1, "role": "causal_witness"},
            ),
            emit_grid={
                "start_timestamp_ms": 10_000,
                "end_timestamp_ms": 10_000,
                "step_ms": 5_000,
            },
        ),
    )["rows"][0]
    assert row["causal_cutoff_block"] == 100


def test_default_request_serializes_and_executes_unchanged(tmp_path):
    model = CausalGridExtractRequest.model_validate(_request())
    assert model.source_order_tie_break is False
    assert model.sparse_emit_points == ()
    assert model.causal_witness_mapping.timestamp_mode == "integer_ms"
    dumped = model.model_dump(mode="json")
    assert dumped["source_order_tie_break"] is False
    assert dumped["sparse_emit_points"] == []
    path = _write(tmp_path / "trades.parquet", [_trade_row(timestamp_ms=9_000)])
    rows = execute_causal_grid_extract([path], _request())["rows"]
    assert len(rows) == 1
    assert rows[0]["grid_timestamp_ms"] == 10_000


def test_same_index_bound_to_both_roles_accepted_and_exact_duplicate_rejected(tmp_path):
    path = _write(tmp_path / "trades.parquet", [_trade_row(block=100, timestamp_ms=9_000)])
    request = _request(
        input_roles=(
            {"input_index": 0, "role": "canonical_trade"},
            {"input_index": 0, "role": "causal_witness"},
        ),
    )
    CausalGridExtractRequest.model_validate(request)
    assert execute_causal_grid_extract([path], request)["rows"][0]["causal_cutoff_block"] is None

    duplicate = _request(
        input_roles=(
            {"input_index": 0, "role": "canonical_trade"},
            {"input_index": 0, "role": "canonical_trade"},
        ),
    )
    with pytest.raises(ValidationError):
        CausalGridExtractRequest.model_validate(duplicate)


def test_witness_iso8601_matches_integer_ms_and_rejects_naive_or_invalid(tmp_path):
    trades = _write(
        tmp_path / "trades.parquet",
        [
            _trade_row(block=100, timestamp_ms=8_000),
            _trade_row(identity=2, identity_norm="2", block=110, timestamp_ms=9_000),
        ],
    )
    integer_witness = _write(
        tmp_path / "w_ms.parquet",
        [
            _witness_row(block=100, timestamp_ms=8_000),
            _witness_row(block=120, timestamp_ms=9_500),
        ],
    )
    iso_witness = _write(
        tmp_path / "w_iso.parquet",
        [
            {"block": 100, "ts_iso": "1970-01-01T00:00:08Z"},
            {"block": 120, "ts_iso": "1970-01-01T00:00:09.500Z"},
        ],
    )
    integer_rows = execute_causal_grid_extract(
        [trades, integer_witness],
        _request(
            input_roles=(
                {"input_index": 0, "role": "canonical_trade"},
                {"input_index": 1, "role": "causal_witness"},
            ),
        ),
    )["rows"]
    iso_request = _request(
        input_roles=(
            {"input_index": 0, "role": "canonical_trade"},
            {"input_index": 1, "role": "causal_witness"},
        ),
        causal_witness_mapping={
            "block_column": "block",
            "timestamp_column": "ts_iso",
            "timestamp_mode": "iso8601",
        },
    )
    iso_rows = execute_causal_grid_extract([trades, iso_witness], iso_request)["rows"]
    assert integer_rows == iso_rows

    naive_witness = _write(
        tmp_path / "w_naive.parquet",
        [{"block": 100, "ts_iso": "1970-01-01T00:00:08"}],
    )
    with pytest.raises(StructuralCanonicalizeError):
        execute_causal_grid_extract([trades, naive_witness], iso_request)

    invalid_witness = _write(
        tmp_path / "w_bad.parquet",
        [{"block": 100, "ts_iso": "not-a-date"}],
    )
    with pytest.raises(StructuralCanonicalizeError):
        execute_causal_grid_extract([trades, invalid_witness], iso_request)


def _tie_witness(tmp_path):
    return _write(
        tmp_path / "w.parquet",
        [_witness_row(block=100, timestamp_ms=8_000), _witness_row(block=110, timestamp_ms=9_500)],
    )


def _tied_trade(identity, price):
    return _trade_row(
        identity=identity,
        identity_norm=str(identity),
        block=100,
        timestamp_ms=9_000,
        price=price,
    )


def test_source_order_tie_break_permutations_single_file(tmp_path):
    rows = [(1, 1.0), (2, 2.0), (3, 3.0)]
    for permutation in permutations(rows):
        trade_path = _write(
            tmp_path / "trades.parquet",
            [_tied_trade(identity, price) for identity, price in permutation],
        )
        witness = _tie_witness(tmp_path)
        row = execute_causal_grid_extract(
            [trade_path, witness],
            _request(
                input_roles=(
                    {"input_index": 0, "role": "canonical_trade"},
                    {"input_index": 1, "role": "causal_witness"},
                ),
                source_order_tie_break=True,
            ),
        )["rows"][0]
        assert row["causal_cutoff_block"] == 100
        assert _facts_by_id(row)["as_of.0"] == permutation[-1][1]


def test_source_order_tie_break_two_files_prefers_later_input_index(tmp_path):
    first = _write(tmp_path / "a.parquet", [_tied_trade(1, 1.0), _tied_trade(2, 2.0)])
    second = _write(tmp_path / "b.parquet", [_tied_trade(3, 3.0), _tied_trade(4, 4.0)])
    witness = _tie_witness(tmp_path)
    row = execute_causal_grid_extract(
        [first, second, witness],
        _request(
            input_roles=(
                {"input_index": 0, "role": "canonical_trade"},
                {"input_index": 1, "role": "canonical_trade"},
                {"input_index": 2, "role": "causal_witness"},
            ),
            source_order_tie_break=True,
        ),
    )["rows"][0]
    assert _facts_by_id(row)["as_of.0"] == 4.0


def test_sparse_emit_matches_dense_subset_and_is_deterministic(tmp_path):
    path = _write(tmp_path / "trades.parquet", [_trade_row(timestamp_ms=9_000)])
    dense_request = _request(
        target_symbols=("AAA", "BBB"),
        emit_grid={
            "start_timestamp_ms": 10_000,
            "end_timestamp_ms": 20_000,
            "step_ms": 5_000,
        },
        partition={
            "emit_start_ms": 10_000,
            "emit_end_ms": 20_000,
            "overlap_ms": 60_000,
            "hard_gap_missing_dates": (),
        },
    )
    dense_rows = execute_causal_grid_extract([path], dense_request)["rows"]
    sparse_request = {
        **dense_request,
        "sparse_emit_points": [
            {"timestamp_ms": 20_000, "symbols": ["BBB", "AAA"]},
            {"timestamp_ms": 15_000, "symbols": ["BBB"]},
        ],
    }
    sparse_rows = execute_causal_grid_extract([path], sparse_request)["rows"]
    assert sparse_rows == [
        row
        for row in dense_rows
        if (row["grid_timestamp_ms"], row["symbol"])
        in {(15_000, "BBB"), (20_000, "AAA"), (20_000, "BBB")}
    ]
    assert [(row["grid_timestamp_ms"], row["symbol"]) for row in sparse_rows] == [
        (15_000, "BBB"),
        (20_000, "AAA"),
        (20_000, "BBB"),
    ]
    assert {row["symbol"] for row in sparse_rows} == {"AAA", "BBB"}


def test_sparse_emit_enforces_max_output_rows(tmp_path):
    path = _write(tmp_path / "trades.parquet", [_trade_row(timestamp_ms=9_000)])
    sparse_request = _request(
        target_symbols=("AAA", "BBB"),
        emit_grid={
            "start_timestamp_ms": 10_000,
            "end_timestamp_ms": 15_000,
            "step_ms": 5_000,
        },
        partition={
            "emit_start_ms": 10_000,
            "emit_end_ms": 15_000,
            "overlap_ms": 60_000,
            "hard_gap_missing_dates": (),
        },
        sparse_emit_points=[
            {"timestamp_ms": 10_000, "symbols": ["AAA"]},
            {"timestamp_ms": 15_000, "symbols": ["BBB"]},
        ],
        max_output_rows=1,
    )
    with pytest.raises(ValueError, match="output row count exceeds limit"):
        execute_causal_grid_extract([path], sparse_request)


def test_sparse_emit_hard_gap_point_fails_closed(tmp_path):
    path = _write(tmp_path / "trades.parquet", [_trade_row(timestamp_ms=86_400_000)])
    sparse_request = _request(
        emit_grid={
            "start_timestamp_ms": 86_400_000,
            "end_timestamp_ms": 86_400_000,
            "step_ms": 5_000,
        },
        partition={
            "emit_start_ms": 86_400_000,
            "emit_end_ms": 86_400_000,
            "overlap_ms": 60_000,
            "hard_gap_missing_dates": ("1970-01-02",),
        },
        sparse_emit_points=[{"timestamp_ms": 86_400_000, "symbols": ["AAA"]}],
    )
    with pytest.raises(StructuralCanonicalizeError):
        execute_causal_grid_extract([path], sparse_request)


def test_sparse_emit_points_validate_against_request(tmp_path):
    base = _request()
    with pytest.raises(ValidationError):
        CausalGridExtractRequest.model_validate(
            {**base, "sparse_emit_points": [{"timestamp_ms": 10_000, "symbols": ["ZZZ"]}]}
        )
    with pytest.raises(ValidationError):
        CausalGridExtractRequest.model_validate(
            {
                **base,
                "sparse_emit_points": [
                    {"timestamp_ms": 10_000, "symbols": ["AAA"]},
                    {"timestamp_ms": 10_000, "symbols": ["AAA"]},
                ],
            }
        )
    with pytest.raises(ValidationError):
        CausalGridExtractRequest.model_validate(
            {**base, "sparse_emit_points": [{"timestamp_ms": 10_000, "symbols": ["AAA", "AAA"]}]}
        )
    with pytest.raises(ValidationError):
        CausalGridExtractRequest.model_validate(
            {**base, "sparse_emit_points": [{"timestamp_ms": 9_999, "symbols": ["AAA"]}]}
        )
