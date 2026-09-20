from __future__ import annotations

import json
import math
from pathlib import Path

import polars as pl
import pytest

from portable_batch_execution.packs.replay_reduction.two_leg_path_evaluate import (
    TwoLegPathEvaluateError,
    execute_two_leg_path_evaluate,
)


def _lifecycle() -> dict:
    return {
        "entry_z": 2.0,
        "exit_z": 0.5,
        "max_signal_holding_sessions": 20,
    }


def _request(**overrides) -> dict:
    base = {
        "schema_version": "pbe.replay.two-leg-path-evaluate.v1",
        "request_id": "req-test",
        "lifecycle": _lifecycle(),
        "censor_session_bound": None,
        "pair_gross_unit": 1.0,
        "max_output_events": 1000,
    }
    base.update(overrides)
    return base


def _write_path(tmp_path: Path, rows: list[dict]) -> str:
    path = tmp_path / "path.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return str(path)


def _base_row(
    *,
    event_id: str = "e1",
    session_idx: int,
    entry_exec_idx: int = 1,
    z: float = 3.0,
    leg1_observed: bool = True,
    leg0_open: float = 100.0,
    leg1_open: float = 50.0,
) -> dict:
    return {
        "event_id": event_id,
        "passthrough_json": json.dumps({"k": event_id}),
        "signal_idx": 0,
        "entry_exec_idx": entry_exec_idx,
        "session_idx": session_idx,
        "z": z,
        "leg0_side": -1,
        "leg0_weight": 0.5,
        "leg1_side": 1,
        "leg1_weight": 0.5,
        "leg0_raw_open": leg0_open,
        "leg1_raw_open": leg1_open,
        "leg0_raw_close": leg0_open,
        "leg1_raw_close": leg1_open,
        "leg0_observed": True,
        "leg1_observed": leg1_observed,
        "leg0_dividend": 0.0,
        "leg1_dividend": 0.0,
        "formation_median_tv_20d": 1e9,
        "formation_median_tv_60d": 1e9,
    }


def test_next_open_entry_cancellation(tmp_path: Path) -> None:
    rows = [
        _base_row(session_idx=1, entry_exec_idx=1, leg1_observed=False),
        _base_row(session_idx=2, z=3.0),
    ]
    path = _write_path(tmp_path, rows)
    result = execute_two_leg_path_evaluate([path], _request())
    fact = result["event_facts"][0]
    assert fact["status"] == "CANCELLED"
    assert fact["blocked_reason"] == "no_executable_open"


def test_close_signal_next_executable_open_exit(tmp_path: Path) -> None:
    rows = []
    for session in range(1, 6):
        z = 3.0 if session < 3 else 0.4
        rows.append(_base_row(session_idx=session, z=z))
    path = _write_path(tmp_path, rows)
    result = execute_two_leg_path_evaluate([path], _request())
    fact = result["event_facts"][0]
    assert fact["status"] == "CLOSED"
    assert fact["exit_exec_idx"] == 4


def test_latched_exit_not_unlatched_by_widening_z(tmp_path: Path) -> None:
    rows = []
    z_path = [3.0, 3.0, 0.4, 2.5, 0.3]
    for i, session in enumerate(range(1, 6), start=0):
        rows.append(_base_row(session_idx=session, z=z_path[i]))
    path = _write_path(tmp_path, rows)
    result = execute_two_leg_path_evaluate([path], _request())
    fact = result["event_facts"][0]
    assert fact["status"] == "CLOSED"
    assert fact["exit_exec_idx"] == 4


def test_twenty_session_horizon_rule(tmp_path: Path) -> None:
    rows = []
    for session in range(1, 25):
        rows.append(_base_row(session_idx=session, z=3.0))
    path = _write_path(tmp_path, rows)
    result = execute_two_leg_path_evaluate([path], _request())
    fact = result["event_facts"][0]
    assert fact["status"] == "CLOSED"
    assert fact["forced_by_horizon"] is True
    assert fact["exit_exec_idx"] == 21


def test_censoring_before_holdout(tmp_path: Path) -> None:
    rows = []
    for session in range(1, 8):
        rows.append(_base_row(session_idx=session, z=3.0))
    path = _write_path(tmp_path, rows)
    result = execute_two_leg_path_evaluate([path], _request(censor_session_bound=6))
    fact = result["event_facts"][0]
    assert fact["status"] == "CENSORED"


def test_dividends_separate_from_price_pnl(tmp_path: Path) -> None:
    rows = [
        _base_row(session_idx=1, z=3.0),
        _base_row(session_idx=2, z=0.2, leg0_open=110.0, leg1_open=55.0),
    ]
    rows[1]["leg0_dividend"] = 1.0
    path = _write_path(tmp_path, rows)
    result = execute_two_leg_path_evaluate([path], _request())
    fact = result["event_facts"][0]
    assert fact["status"] == "CLOSED"
    assert fact["dividend_cash_unit"] != 0.0
    assert fact["price_pnl_unit"] != fact["gross_pnl_unit"]


def test_mae_mfe_present_when_closed(tmp_path: Path) -> None:
    rows = [_base_row(session_idx=1, z=3.0), _base_row(session_idx=2, z=0.2)]
    path = _write_path(tmp_path, rows)
    result = execute_two_leg_path_evaluate([path], _request())
    fact = result["event_facts"][0]
    assert "mae_unit" in fact and "mfe_unit" in fact
    assert math.isfinite(fact["mae_unit"])
    assert math.isfinite(fact["mfe_unit"])


def test_deterministic_output_ordering(tmp_path: Path) -> None:
    rows = []
    for eid, sig in (("b", 2), ("a", 1)):
        rows.extend(
            [
                _base_row(event_id=eid, session_idx=1, z=3.0),
                _base_row(event_id=eid, session_idx=2, z=0.2),
            ]
        )
        rows[-2]["signal_idx"] = sig
        rows[-1]["signal_idx"] = sig
        rows[-2]["passthrough_json"] = json.dumps({"id": eid})
        rows[-1]["passthrough_json"] = json.dumps({"id": eid})
    path = _write_path(tmp_path, rows)
    first = execute_two_leg_path_evaluate([path], _request())
    second = execute_two_leg_path_evaluate([path], _request())
    assert first["event_facts"] == second["event_facts"]
    assert [f["event_id"] for f in first["event_facts"]] == ["a", "b"]


def test_malformed_input_fail_closed(tmp_path: Path) -> None:
    bad = _base_row(session_idx=1)
    bad["z"] = float("nan")
    path = _write_path(tmp_path, [bad])
    with pytest.raises(TwoLegPathEvaluateError):
        execute_two_leg_path_evaluate([path], _request())
