"""Generic bounded two-leg path evaluation (scenario-independent mechanical facts)."""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import polars as pl

from .canonicalize import StructuralCanonicalizeError, _require_columns
from .models import TwoLegPathEvaluateRequest

RESULT_SCHEMA_VERSION = "pbe.replay.two-leg-path-evaluate-result.v1"
METADATA_SCHEMA_VERSION = "pbe.replay.two-leg-path-evaluate-metadata.v1"
EVENTS_PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"
MARKS_PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"

_PATH_COLUMNS = (
    "event_id",
    "passthrough_json",
    "signal_idx",
    "entry_exec_idx",
    "session_idx",
    "z",
    "leg0_side",
    "leg0_weight",
    "leg1_side",
    "leg1_weight",
    "leg0_raw_open",
    "leg1_raw_open",
    "leg0_raw_close",
    "leg1_raw_close",
    "leg0_observed",
    "leg1_observed",
    "leg0_dividend",
    "leg1_dividend",
    "formation_median_tv_20d",
    "formation_median_tv_60d",
)


class TwoLegPathEvaluateError(StructuralCanonicalizeError):
    """Fail-closed path evaluation error."""


def _validated(
    request: dict[str, Any] | TwoLegPathEvaluateRequest,
) -> TwoLegPathEvaluateRequest:
    if isinstance(request, TwoLegPathEvaluateRequest):
        return request
    return TwoLegPathEvaluateRequest.model_validate(request)


def _finite(value: float, *, label: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise TwoLegPathEvaluateError(f"{label} invalid") from None
    if not math.isfinite(out):
        raise TwoLegPathEvaluateError(f"{label} non-finite")
    return out


def _executable(observed: bool, raw_open: float) -> bool:
    return bool(observed) and math.isfinite(raw_open) and raw_open > 0.0


def _sessions_between(entry_idx: int, upto_idx: int) -> int:
    return max(0, int(upto_idx) - int(entry_idx))


@dataclass
class _LegState:
    side: int
    weight: float
    shares: float = 0.0
    entry_price: float = 0.0
    entry_notional: float = 0.0


@dataclass
class _EventRows:
    event_id: str
    passthrough_json: str
    signal_idx: int
    entry_exec_idx: int
    by_session: dict[int, dict[str, Any]] = field(default_factory=dict)
    formation_tv_20: float | None = None
    formation_tv_60: float | None = None


def _load_path_rows(paths: list[str], model: TwoLegPathEvaluateRequest) -> list[_EventRows]:
    if not paths:
        raise TwoLegPathEvaluateError("path input missing")
    if len(paths) > model.max_input_files:
        raise TwoLegPathEvaluateError("too many input files")
    frames: list[pl.DataFrame] = []
    total_bytes = 0
    for path in paths:
        frame = pl.read_parquet(path)
        _require_columns(frame.collect_schema(), _PATH_COLUMNS)
        frames.append(frame)
        total_bytes += Path(path).stat().st_size
    if total_bytes > model.max_input_bytes:
        raise TwoLegPathEvaluateError("input byte limit exceeded")
    if not frames:
        raise TwoLegPathEvaluateError("empty path input")
    combined = pl.concat(frames, how="vertical_relaxed")
    if combined.height > model.max_input_rows:
        raise TwoLegPathEvaluateError("input row limit exceeded")
    events: dict[str, _EventRows] = {}
    for row in combined.iter_rows(named=True):
        event_id = str(row["event_id"])
        if not event_id:
            raise TwoLegPathEvaluateError("event_id empty")
        session_idx = int(row["session_idx"])
        if session_idx < 0:
            raise TwoLegPathEvaluateError("session_idx negative")
        entry_exec = int(row["entry_exec_idx"])
        signal_idx = int(row["signal_idx"])
        passthrough = row["passthrough_json"]
        if passthrough is None:
            raise TwoLegPathEvaluateError("passthrough_json missing")
        passthrough_str = str(passthrough)
        try:
            json.loads(passthrough_str)
        except (TypeError, ValueError):
            raise TwoLegPathEvaluateError("passthrough_json invalid") from None
        if event_id not in events:
            events[event_id] = _EventRows(
                event_id=event_id,
                passthrough_json=passthrough_str,
                signal_idx=signal_idx,
                entry_exec_idx=entry_exec,
            )
        ev = events[event_id]
        if ev.signal_idx != signal_idx or ev.entry_exec_idx != entry_exec:
            raise TwoLegPathEvaluateError("event metadata inconsistent")
        if passthrough_str != ev.passthrough_json:
            raise TwoLegPathEvaluateError("passthrough_json inconsistent")
        if session_idx in ev.by_session:
            raise TwoLegPathEvaluateError("duplicate session row")
        z = _finite(row["z"], label="z")
        leg0_side = int(row["leg0_side"])
        leg1_side = int(row["leg1_side"])
        if leg0_side not in (-1, 1) or leg1_side not in (-1, 1):
            raise TwoLegPathEvaluateError("leg side invalid")
        leg0_weight = _finite(row["leg0_weight"], label="leg0_weight")
        leg1_weight = _finite(row["leg1_weight"], label="leg1_weight")
        if leg0_weight <= 0.0 or leg1_weight <= 0.0:
            raise TwoLegPathEvaluateError("leg weight non-positive")
        ev.by_session[session_idx] = {
            "z": z,
            "leg0_side": leg0_side,
            "leg0_weight": leg0_weight,
            "leg1_side": leg1_side,
            "leg1_weight": leg1_weight,
            "leg0_raw_open": _finite(row["leg0_raw_open"], label="leg0_raw_open"),
            "leg1_raw_open": _finite(row["leg1_raw_open"], label="leg1_raw_open"),
            "leg0_raw_close": _finite(row["leg0_raw_close"], label="leg0_raw_close"),
            "leg1_raw_close": _finite(row["leg1_raw_close"], label="leg1_raw_close"),
            "leg0_observed": bool(row["leg0_observed"]),
            "leg1_observed": bool(row["leg1_observed"]),
            "leg0_dividend": _finite(row["leg0_dividend"], label="leg0_dividend"),
            "leg1_dividend": _finite(row["leg1_dividend"], label="leg1_dividend"),
        }
        tv20 = row["formation_median_tv_20d"]
        tv60 = row["formation_median_tv_60d"]
        if tv20 is not None and math.isfinite(float(tv20)):
            ev.formation_tv_20 = float(tv20)
        if tv60 is not None and math.isfinite(float(tv60)):
            ev.formation_tv_60 = float(tv60)
    if len(events) > model.max_output_events:
        raise TwoLegPathEvaluateError("event limit exceeded")
    return list(events.values())


def _exit_signal_fires(
    z: float, held: int, exit_z: float, max_hold: int
) -> tuple[bool, bool]:
    if held >= max_hold:
        return True, True
    if math.isfinite(z) and abs(z) <= exit_z:
        return True, False
    return False, False


def _row_executable(row: dict[str, Any]) -> bool:
    return _executable(row["leg0_observed"], row["leg0_raw_open"]) and _executable(
        row["leg1_observed"], row["leg1_raw_open"]
    )


def _open_legs(
    row: dict[str, Any], pair_gross_unit: float
) -> tuple[_LegState, _LegState]:
    if not _row_executable(row):
        raise TwoLegPathEvaluateError("entry not executable")
    legs = (
        _LegState(int(row["leg0_side"]), float(row["leg0_weight"])),
        _LegState(int(row["leg1_side"]), float(row["leg1_weight"])),
    )
    for leg, key in zip(legs, ("leg0_raw_open", "leg1_raw_open"), strict=True):
        px = float(row[key])
        leg.entry_price = px
        leg.entry_notional = pair_gross_unit * leg.weight
        leg.shares = leg.entry_notional / px
    return legs[0], legs[1]


def _accrued_dividends(
    legs: tuple[_LegState, _LegState],
    by_session: dict[int, dict[str, Any]],
    entry_idx: int,
    upto_idx: int,
) -> float:
    total = 0.0
    lo = entry_idx + 1
    hi = upto_idx + 1
    for session in range(lo, hi):
        row = by_session.get(session)
        if row is None:
            continue
        for i, leg in enumerate(legs):
            div = float(row[f"leg{i}_dividend"])
            if div != 0.0:
                total += div * leg.shares * leg.side
    return total


def _close_economics_fixed(
    legs: tuple[_LegState, _LegState],
    by_session: dict[int, dict[str, Any]],
    entry_idx: int,
    exit_idx: int,
) -> dict[str, float]:
    row = by_session.get(exit_idx)
    if row is None:
        raise TwoLegPathEvaluateError("exit session row missing")
    entry_notional = 0.0
    exit_notional = 0.0
    price_pnl = 0.0
    for i, leg in enumerate(legs):
        px_exit = float(row[f"leg{i}_raw_open"])
        if not _executable(row[f"leg{i}_observed"], px_exit):
            raise TwoLegPathEvaluateError("exit not executable")
        exit_n = abs(leg.shares * px_exit)
        entry_notional += abs(leg.entry_notional)
        exit_notional += exit_n
        price_pnl += leg.shares * (px_exit - leg.entry_price) * leg.side
    div_cash = _accrued_dividends(legs, by_session, entry_idx, exit_idx)
    short_notional = sum(abs(l.entry_notional) for l in legs if l.side < 0)
    holding_sessions = int(exit_idx - entry_idx)
    holding_days = _sessions_between(entry_idx, exit_idx)
    gross_pnl = price_pnl + div_cash
    return {
        "price_pnl_unit": price_pnl,
        "dividend_cash_unit": div_cash,
        "gross_pnl_unit": gross_pnl,
        "entry_notional_unit": entry_notional,
        "exit_notional_unit": exit_notional,
        "traded_notional_unit": entry_notional + exit_notional,
        "short_notional_unit": short_notional,
        "holding_sessions": float(holding_sessions),
        "holding_days": float(holding_days),
    }


def _raw_close_mark(
    leg: _LegState,
    by_session: dict[int, dict[str, Any]],
    entry_idx: int,
    idx: int,
    leg_index: int,
) -> tuple[float, bool, int]:
    for k in range(int(idx), int(entry_idx) - 1, -1):
        row = by_session.get(k)
        if row is None:
            continue
        px = float(row[f"leg{leg_index}_raw_close"])
        if math.isfinite(px) and px > 0.0:
            age = int(idx - k)
            return px, bool(age > 0), age
    return float("nan"), True, -1


def _mae_mfe(
    legs: tuple[_LegState, _LegState],
    by_session: dict[int, dict[str, Any]],
    entry_idx: int,
    exit_idx: int,
) -> tuple[float, float]:
    lo, hi = entry_idx + 1, exit_idx + 1
    if hi <= lo:
        return 0.0, 0.0
    path: list[float] = []
    for session in range(lo, hi):
        row = by_session.get(session)
        if row is None:
            return 0.0, 0.0
        total = 0.0
        ok = True
        for i, leg in enumerate(legs):
            px = float(row[f"leg{i}_raw_close"])
            if not math.isfinite(px):
                ok = False
                break
            total += leg.shares * (px - leg.entry_price) * leg.side
        if ok:
            path.append(total)
    if not path:
        return 0.0, 0.0
    return float(min(path)), float(max(path))


def _daily_marks(
    legs: tuple[_LegState, _LegState],
    by_session: dict[int, dict[str, Any]],
    entry_idx: int,
    last_idx: int,
    event_id: str,
) -> list[dict[str, Any]]:
    marks: list[dict[str, Any]] = []
    for idx in range(entry_idx, last_idx + 1):
        price = 0.0
        stale_legs = 0
        max_age = 0
        ok = True
        for i, leg in enumerate(legs):
            px, stale, age = _raw_close_mark(leg, by_session, entry_idx, idx, i)
            if not math.isfinite(px):
                ok = False
                break
            price += leg.shares * (px - leg.entry_price) * leg.side
            if stale:
                stale_legs += 1
                max_age = max(max_age, age)
        if not ok:
            price = float("nan")
            stale_legs = 2
            max_age = -1
        div = _accrued_dividends(legs, by_session, entry_idx, idx)
        marks.append(
            {
                "event_id": event_id,
                "session_idx": idx,
                "price_mtm_unit": price,
                "dividend_accrued_unit": div,
                "stale": bool(stale_legs > 0),
                "stale_legs": int(stale_legs),
                "stale_max_age": int(max_age),
            }
        )
    return marks


def _evaluate_event(ev: _EventRows, model: TwoLegPathEvaluateRequest) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lc = model.lifecycle
    entry_z = float(lc.entry_z)
    exit_z = float(lc.exit_z)
    max_hold = int(lc.max_signal_holding_sessions)
    pair_gross_unit = float(model.pair_gross_unit)
    hi_bound = (
        max(ev.by_session) + 1
        if model.censor_session_bound is None
        else min(max(ev.by_session) + 1, int(model.censor_session_bound))
    )
    entry_idx = ev.entry_exec_idx
    entry_row = ev.by_session.get(entry_idx)
    status: Literal["CLOSED", "CANCELLED", "CENSORED"] = "CANCELLED"
    blocked_reason: str | None = "no_executable_open"
    exit_idx: int | None = None
    forced_horizon = False
    delayed_exit = False
    marks: list[dict[str, Any]] = []
    econ: dict[str, float] = {}

    if entry_row is None or not _row_executable(entry_row):
        fact = {
            "event_id": ev.event_id,
            "passthrough_json": ev.passthrough_json,
            "signal_idx": ev.signal_idx,
            "entry_exec_idx": entry_idx,
            "status": status,
            "blocked_reason": blocked_reason,
            "exit_exec_idx": None,
            "forced_by_horizon": False,
            "delayed_exit": False,
            "formation_median_tv_20d": ev.formation_tv_20,
            "formation_median_tv_60d": ev.formation_tv_60,
        }
        return fact, marks

    try:
        legs = _open_legs(entry_row, pair_gross_unit)
    except TwoLegPathEvaluateError:
        fact = {
            "event_id": ev.event_id,
            "passthrough_json": ev.passthrough_json,
            "signal_idx": ev.signal_idx,
            "entry_exec_idx": entry_idx,
            "status": "CANCELLED",
            "blocked_reason": "no_executable_open",
            "exit_exec_idx": None,
            "forced_by_horizon": False,
            "delayed_exit": False,
            "formation_median_tv_20d": ev.formation_tv_20,
            "formation_median_tv_60d": ev.formation_tv_60,
        }
        return fact, marks

    latched = False
    pending_forced = False
    t = entry_idx
    while t < hi_bound:
        row = ev.by_session.get(t)
        if row is None:
            t += 1
            continue
        held = int(t - entry_idx + 1)
        if not latched:
            fire, forced = _exit_signal_fires(float(row["z"]), held, exit_z, max_hold)
            if fire:
                latched = True
                pending_forced = forced
        if latched:
            cand = t + 1
            if cand < hi_bound:
                cand_row = ev.by_session.get(cand)
                if cand_row is not None and _row_executable(cand_row):
                    exit_idx = cand
                    forced_horizon = pending_forced
                    delayed_exit = bool(cand - entry_idx > max_hold)
                    break
        t += 1

    if exit_idx is None:
        if model.censor_session_bound is not None and entry_idx < int(model.censor_session_bound):
            status = "CENSORED"
            blocked_reason = None
            marks = _daily_marks(legs, ev.by_session, entry_idx, hi_bound - 1, ev.event_id)
        else:
            status = "CANCELLED"
            blocked_reason = "unresolved_path"
        fact = {
            "event_id": ev.event_id,
            "passthrough_json": ev.passthrough_json,
            "signal_idx": ev.signal_idx,
            "entry_exec_idx": entry_idx,
            "status": status,
            "blocked_reason": blocked_reason,
            "exit_exec_idx": None,
            "forced_by_horizon": False,
            "delayed_exit": False,
            "formation_median_tv_20d": ev.formation_tv_20,
            "formation_median_tv_60d": ev.formation_tv_60,
        }
        return fact, marks

    econ = _close_economics_fixed(legs, ev.by_session, entry_idx, exit_idx)
    mae, mfe = _mae_mfe(legs, ev.by_session, entry_idx, exit_idx)
    marks = _daily_marks(legs, ev.by_session, entry_idx, exit_idx, ev.event_id)
    status = "CLOSED"
    fact = {
        "event_id": ev.event_id,
        "passthrough_json": ev.passthrough_json,
        "signal_idx": ev.signal_idx,
        "entry_exec_idx": entry_idx,
        "status": status,
        "blocked_reason": None,
        "exit_exec_idx": exit_idx,
        "forced_by_horizon": forced_horizon,
        "delayed_exit": delayed_exit,
        "formation_median_tv_20d": ev.formation_tv_20,
        "formation_median_tv_60d": ev.formation_tv_60,
        "mae_unit": mae,
        "mfe_unit": mfe,
        **econ,
    }
    return fact, marks


def execute_two_leg_path_evaluate(
    paths: list[str],
    request: dict[str, Any] | TwoLegPathEvaluateRequest,
) -> dict[str, Any]:
    model = _validated(request)
    events_in = _load_path_rows(paths, model)
    events_in.sort(key=lambda e: (e.signal_idx, e.event_id))
    event_facts: list[dict[str, Any]] = []
    mark_rows: list[dict[str, Any]] = []
    for ev in events_in:
        fact, marks = _evaluate_event(ev, model)
        event_facts.append(fact)
        mark_rows.extend(marks)
    event_facts.sort(key=lambda r: (int(r["signal_idx"]), str(r["event_id"])))
    events_bytes = b""
    marks_bytes = b""
    if event_facts:
        buf = io.BytesIO()
        pl.DataFrame(event_facts).write_parquet(buf)
        events_bytes = buf.getvalue()
    if mark_rows:
        buf = io.BytesIO()
        pl.DataFrame(mark_rows).write_parquet(buf)
        marks_bytes = buf.getvalue()
    if len(events_bytes) > model.max_output_bytes:
        raise TwoLegPathEvaluateError("events parquet exceeds byte limit")
    if len(marks_bytes) > model.max_output_bytes:
        raise TwoLegPathEvaluateError("marks parquet exceeds byte limit")
    identity = (
        f"sha256:{sha256(events_bytes + marks_bytes).hexdigest()}"
        if events_bytes or marks_bytes
        else None
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "summary": {
            "event_count": len(event_facts),
            "mark_row_count": len(mark_rows),
            "content_identity": identity,
        },
        "event_facts": event_facts,
        "events_parquet_bytes": events_bytes,
        "marks_parquet_bytes": marks_bytes,
        "max_output_bytes": model.max_output_bytes,
    }


def build_two_leg_path_metadata(
    result_payload: dict[str, Any],
    *,
    events_parquet_ref: Any | None,
    marks_parquet_ref: Any | None,
) -> dict[str, Any]:
    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "result_schema_version": result_payload["schema_version"],
        "summary": result_payload["summary"],
        "events_parquet_ref": (
            events_parquet_ref.model_dump(mode="json")
            if events_parquet_ref is not None
            else None
        ),
        "marks_parquet_ref": (
            marks_parquet_ref.model_dump(mode="json")
            if marks_parquet_ref is not None
            else None
        ),
    }


def publish_two_leg_path_evaluate_artifacts(
    plane,
    result_payload: dict[str, Any],
    *,
    artifact_ref_matches_bytes,
    shard_stage_failure,
) -> tuple[bytes, tuple[Any, ...]]:
    max_output_bytes = int(result_payload.get("max_output_bytes", 0))
    events_bytes = bytes(result_payload.get("events_parquet_bytes") or b"")
    marks_bytes = bytes(result_payload.get("marks_parquet_bytes") or b"")
    if max_output_bytes > 0 and len(events_bytes) + len(marks_bytes) > max_output_bytes:
        raise ValueError("two-leg path parquet output exceeds byte limit")
    output_refs: list[Any] = []
    events_ref = None
    marks_ref = None
    if events_bytes:
        events_ref = plane.write(events_bytes, EVENTS_PARQUET_MEDIA_TYPE)
        if not artifact_ref_matches_bytes(events_bytes, events_ref):
            raise shard_stage_failure("output_artifact_mismatch")
        output_refs.append(events_ref)
    if marks_bytes:
        marks_ref = plane.write(marks_bytes, MARKS_PARQUET_MEDIA_TYPE)
        if not artifact_ref_matches_bytes(marks_bytes, marks_ref):
            raise shard_stage_failure("output_artifact_mismatch")
        output_refs.append(marks_ref)
    metadata = build_two_leg_path_metadata(
        result_payload,
        events_parquet_ref=events_ref,
        marks_parquet_ref=marks_ref,
    )
    metadata_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
    metadata_ref = plane.write(metadata_bytes, "application/json")
    if not artifact_ref_matches_bytes(metadata_bytes, metadata_ref):
        raise shard_stage_failure("output_artifact_mismatch")
    return metadata_bytes, (metadata_ref, *output_refs)
