import polars as pl
import pytest
from pydantic import ValidationError

from portable_batch_execution.packs.replay_reduction.event_window import (
    execute_event_window_extract,
)

_DECISION_MS = 5000

_PROFILE = {
    "schema_version": "pbe.replay.canonical-trade-profile.v1",
    "identity_source_column": "identity",
    "identity_normalized_column": "identity_norm",
    "measurement_core_fields": ["price"],
}


def _write_rows(tmp_path, rows):
    path = tmp_path / "records.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


def _request(**overrides):
    base = {
        "schema_version": "pbe.replay.event-window-extract.v3",
        "request_id": "req-1",
        "symbol": "AAA",
        "symbol_column": "symbol",
        "block_column": "block",
        "timestamp_column": "timestamp_ms",
        "decision_timestamp_ms": _DECISION_MS,
        "causal_cutoff_block": 5,
        "as_of_measurement_field": "price",
        "canonical_trade_profile": _PROFILE,
        "as_of_offsets_ms": (0, 200),
        "trailing_windows": (
            {"fact_id": "trail", "measurement_field": "price", "trailing_width_ms": 2000},
        ),
        "future_windows": (
            {
                "fact_id": "future",
                "measurement_field": "price",
                "start_offset_ms": 1000,
                "end_offset_ms": 3000,
                "min_block_exclusive": 5,
            },
        ),
        "tie_break_columns": ("timestamp_ms", "seq"),
    }
    base.update(overrides)
    return base


def _row(**fields):
    base = {
        "identity": 1,
        "identity_norm": "1",
        "symbol": "AAA",
        "block": 5,
        "timestamp_ms": 3000,
        "price": 1.0,
        "seq": 0,
    }
    base.update(fields)
    return base


def test_trailing_interval_and_causal_cutoff_boundaries(tmp_path):
    path = _write_rows(
        tmp_path,
        [
            _row(identity=1, identity_norm="1", timestamp_ms=2900, price=1.0),
            _row(identity=2, identity_norm="2", timestamp_ms=3100, price=2.0),
            _row(identity=3, identity_norm="3", timestamp_ms=5000, price=3.0),
            _row(identity=4, identity_norm="4", block=6, timestamp_ms=4500, price=99.0),
        ],
    )
    facts = {
        item["fact_id"]: item["value"]
        for item in execute_event_window_extract([path], _request())["facts"]
    }
    assert facts["trail.sum"] == 5.0
    assert facts["trail.count"] == 2


def test_as_of_uses_at_or_before_timestamp_with_tie_break(tmp_path):
    path = _write_rows(
        tmp_path,
        [
            _row(identity=1, identity_norm="1", block=4, timestamp_ms=4800, price=1.0),
            _row(identity=2, identity_norm="2", timestamp_ms=5000, price=2.0),
            _row(identity=3, identity_norm="3", timestamp_ms=5000, price=9.0, seq=1),
        ],
    )
    facts = {
        item["fact_id"]: item["value"]
        for item in execute_event_window_extract([path], _request())["facts"]
    }
    assert facts["as_of.0"] == 9.0
    assert facts["as_of.200"] == 1.0


def test_as_of_missing_returns_none(tmp_path):
    path = _write_rows(
        tmp_path,
        [_row(identity=1, identity_norm="1", timestamp_ms=6000, price=1.0)],
    )
    facts = {
        item["fact_id"]: item["value"]
        for item in execute_event_window_extract(
            [path],
            _request(as_of_offsets_ms=(0,), trailing_windows=(), future_windows=()),
        )["facts"]
    }
    assert facts["as_of.0"] is None


def test_future_window_uses_timestamp_bounds_and_min_block_exclusive(tmp_path):
    path = _write_rows(
        tmp_path,
        [
            _row(identity=1, identity_norm="1", block=4, timestamp_ms=6200, price=1.0),
            _row(identity=2, identity_norm="2", block=5, timestamp_ms=6200, price=2.0),
            _row(identity=3, identity_norm="3", block=6, timestamp_ms=6200, price=4.0, seq=1),
            _row(identity=4, identity_norm="4", block=6, timestamp_ms=6200, price=8.0),
        ],
    )
    facts = {
        item["fact_id"]: item["value"]
        for item in execute_event_window_extract([path], _request())["facts"]
    }
    assert facts["future"] == 8.0


def test_future_respects_min_block_exclusive_independent_of_causal_cutoff(tmp_path):
    path = _write_rows(
        tmp_path,
        [
            _row(identity=1, identity_norm="1", block=5, timestamp_ms=6200, price=7.0),
            _row(identity=2, identity_norm="2", block=6, timestamp_ms=6100, price=3.0),
        ],
    )
    request = _request(
        causal_cutoff_block=10,
        future_windows=(
            {
                "fact_id": "future",
                "measurement_field": "price",
                "start_offset_ms": 1000,
                "end_offset_ms": 2000,
                "min_block_exclusive": 5,
            },
        ),
        trailing_windows=(),
        as_of_offsets_ms=(),
    )
    facts = {
        item["fact_id"]: item["value"]
        for item in execute_event_window_extract([path], request)["facts"]
    }
    assert facts["future"] == 3.0


def test_event_window_extract_requires_valid_request(tmp_path):
    path = _write_rows(tmp_path, [_row()])
    with pytest.raises(ValidationError):
        execute_event_window_extract([path], {"schema_version": "bad"})
