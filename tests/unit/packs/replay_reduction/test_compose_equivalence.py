import polars as pl
import pytest

from portable_batch_execution.packs.replay_reduction.canonicalize import (
    StructuralCanonicalizeError,
    execute_structural_canonicalize,
)
from portable_batch_execution.packs.replay_reduction.event_window import (
    execute_event_window_extract,
)

_PROFILE = {
    "schema_version": "pbe.replay.canonical-trade-profile.v1",
    "identity_source_column": "identity",
    "identity_normalized_column": "identity_norm",
    "measurement_core_fields": ["price"],
    "sentinel": {"identity_equals": -1, "exact_match_fields": {"identity_norm": "witness"}},
}

_REQUEST = {
    "schema_version": "pbe.replay.event-window-extract.v3",
    "request_id": "compose",
    "symbol": "AAA",
    "symbol_column": "symbol",
    "block_column": "block",
    "timestamp_column": "timestamp_ms",
    "decision_timestamp_ms": 4000,
    "causal_cutoff_block": 4,
    "as_of_measurement_field": "price",
    "as_of_offsets_ms": (0, 200),
    "trailing_windows": (
        {"fact_id": "trail", "measurement_field": "price", "trailing_width_ms": 2000},
    ),
    "future_windows": (
        {
            "fact_id": "future",
            "measurement_field": "price",
            "start_offset_ms": 1000,
            "end_offset_ms": 2000,
            "min_block_exclusive": 4,
        },
    ),
    "tie_break_columns": ("timestamp_ms", "seq"),
    "canonical_trade_profile": _PROFILE,
}

# Hand-computed from canonical economic rows only (not from execute_event_window_extract).
# Trailing (2000, 4000], block<=4: id1 price 10, id2 price 20, id3 price 5; sentinel excluded;
# id4 block 5 excluded by causal cutoff.
# as_of.0 at 4000: best tie-break among ts<=4000 -> id3 @ 3900 price 5.
# as_of.200 at 3800: id1 @ 3050, id2 @ 3350 -> best tie-break price 20.
# Future [5000,6000], block>4: id5 @ 5000 price 50.
_EXPECTED = {
    "trail.sum": 35.0,
    "trail.count": 3,
    "as_of.0": 5.0,
    "as_of.200": 20.0,
    "future": 50.0,
}


def _fixture_paths(tmp_path):
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    pl.DataFrame(
        [
            {
                "identity": 1,
                "identity_norm": "1",
                "price": 10.0,
                "symbol": "AAA",
                "block": 3,
                "timestamp_ms": 3000,
                "seq": 0,
            },
            {
                "identity": 1,
                "identity_norm": "1",
                "price": 10.0,
                "symbol": "AAA",
                "block": 3,
                "timestamp_ms": 3050,
                "seq": 1,
            },
            {
                "identity": -1,
                "identity_norm": "witness",
                "price": 9999.0,
                "symbol": "AAA",
                "block": 3,
                "timestamp_ms": 3100,
                "seq": 0,
            },
            {
                "identity": 2,
                "identity_norm": "2",
                "price": 20.0,
                "symbol": "AAA",
                "block": 3,
                "timestamp_ms": 3300,
                "seq": 0,
            },
        ]
    ).write_parquet(path_a)
    pl.DataFrame(
        [
            {
                "identity": 2,
                "identity_norm": "2",
                "price": 20.0,
                "symbol": "AAA",
                "block": 3,
                "timestamp_ms": 3350,
                "seq": 1,
            },
            {
                "identity": 3,
                "identity_norm": "3",
                "price": 5.0,
                "symbol": "AAA",
                "block": 4,
                "timestamp_ms": 3900,
                "seq": 0,
            },
            {
                "identity": 4,
                "identity_norm": "4",
                "price": 100.0,
                "symbol": "AAA",
                "block": 5,
                "timestamp_ms": 3900,
                "seq": 0,
            },
            {
                "identity": 5,
                "identity_norm": "5",
                "price": 50.0,
                "symbol": "AAA",
                "block": 6,
                "timestamp_ms": 5000,
                "seq": 0,
            },
        ]
    ).write_parquet(path_b)
    return path_a, path_b


def test_event_window_matches_hand_computed_reference(tmp_path):
    path_a, path_b = _fixture_paths(tmp_path)
    canonical = execute_structural_canonicalize(
        [path_a, path_b],
        {
            "schema_version": "pbe.replay.structural-canonicalize.v1",
            "identity_source_column": "identity",
            "identity_normalized_column": "identity_norm",
            "measurement_core_fields": ["price"],
            "sentinel": {"identity_equals": -1, "exact_match_fields": {"identity_norm": "witness"}},
        },
    )
    assert canonical.witness_row_count == 1
    assert canonical.positive_group_count == 5

    facts = {
        item["fact_id"]: item["value"]
        for item in execute_event_window_extract([path_a, path_b], _REQUEST)["facts"]
    }
    assert facts == _EXPECTED


def test_normalized_mismatch_fails_closed(tmp_path):
    path = tmp_path / "bad.parquet"
    pl.DataFrame(
        [
            {
                "identity": 1,
                "identity_norm": "2",
                "price": 1.0,
                "symbol": "AAA",
                "block": 1,
                "timestamp_ms": 1000,
                "seq": 0,
            }
        ]
    ).write_parquet(path)
    with pytest.raises(StructuralCanonicalizeError):
        execute_event_window_extract([path], _REQUEST)


def test_core_conflict_fails_closed(tmp_path):
    path = tmp_path / "conflict.parquet"
    pl.DataFrame(
        [
            {
                "identity": 1,
                "identity_norm": "1",
                "price": 1.0,
                "symbol": "AAA",
                "block": 1,
                "timestamp_ms": 1000,
                "seq": 0,
            },
            {
                "identity": 1,
                "identity_norm": "1",
                "price": 2.0,
                "symbol": "AAA",
                "block": 1,
                "timestamp_ms": 1001,
                "seq": 1,
            },
        ]
    ).write_parquet(path)
    with pytest.raises(StructuralCanonicalizeError):
        execute_event_window_extract([path], _REQUEST)


def test_invalid_non_positive_fails_closed(tmp_path):
    path = tmp_path / "zero.parquet"
    pl.DataFrame(
        [
            {
                "identity": 0,
                "identity_norm": "0",
                "price": 1.0,
                "symbol": "AAA",
                "block": 1,
                "timestamp_ms": 1000,
                "seq": 0,
            }
        ]
    ).write_parquet(path)
    with pytest.raises(StructuralCanonicalizeError):
        execute_event_window_extract([path], _REQUEST)


def test_non_contiguous_recurrence_fails_closed(tmp_path):
    path = tmp_path / "recur.parquet"
    pl.DataFrame(
        [
            {
                "identity": 1,
                "identity_norm": "1",
                "price": 1.0,
                "symbol": "AAA",
                "block": 1,
                "timestamp_ms": 1000,
                "seq": 0,
            },
            {
                "identity": 2,
                "identity_norm": "2",
                "price": 2.0,
                "symbol": "AAA",
                "block": 1,
                "timestamp_ms": 1001,
                "seq": 0,
            },
            {
                "identity": 1,
                "identity_norm": "1",
                "price": 1.0,
                "symbol": "AAA",
                "block": 1,
                "timestamp_ms": 1002,
                "seq": 0,
            },
        ]
    ).write_parquet(path)
    with pytest.raises(StructuralCanonicalizeError):
        execute_event_window_extract([path], _REQUEST)
