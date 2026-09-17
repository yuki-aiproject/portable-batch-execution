import polars as pl

from portable_batch_execution.packs.replay_reduction import spill_sort
from portable_batch_execution.packs.replay_reduction.causal_grid import (
    _GLOBAL_TRADE_BUCKET_BUFFER_ROWS,
    execute_causal_grid_extract,
)
from portable_batch_execution.packs.replay_reduction.spill_sort import (
    _merge_fan_in_group,
    _reduce_sorted_runs,
    iter_k_way_merge_dataframes,
)
from tests.unit.packs.replay_reduction.test_causal_grid import (
    _request,
    _trade_row,
    _witness_row,
    _write,
)


def test_hierarchical_merge_never_exceeds_fan_in(tmp_path, monkeypatch):
    monkeypatch.setattr(spill_sort, "_MERGE_FAN_IN", 3)
    max_group = 0
    original = _merge_fan_in_group

    def tracked(*args, **kwargs):
        nonlocal max_group
        runs = args[0] if args else kwargs.get("runs", ())
        max_group = max(max_group, len(runs))
        return original(*args, **kwargs)

    monkeypatch.setattr(spill_sort, "_merge_fan_in_group", tracked)
    sort_keys = ("k",)
    runs: list = []
    for index in range(20):
        frame = pl.DataFrame({"k": [index]})
        path = tmp_path / f"run_{index:02d}.parquet"
        frame.write_parquet(path)
        runs.append(path)
    _reduce_sorted_runs(
        runs,
        sort_keys=sort_keys,
        spill_dir=tmp_path / "reduce",
        run_prefix="t",
    )
    list(
        iter_k_way_merge_dataframes(
            runs,
            sort_keys=sort_keys,
            spill_dir=tmp_path / "final_merge",
        )
    )
    assert max_group <= 3


def test_hierarchical_merge_matches_naive_many_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(spill_sort, "_MERGE_FAN_IN", 2)
    sort_keys = ("k",)
    runs: list = []
    for index in range(7):
        frame = pl.DataFrame({"k": [index * 3, index * 3 + 1]})
        path = tmp_path / f"run_{index:02d}.parquet"
        frame.write_parquet(path)
        runs.append(path)
    naive = sorted(
        (row["k"] for path in runs for row in pl.read_parquet(path).iter_rows(named=True)),
    )
    reduced = _reduce_sorted_runs(
        runs,
        sort_keys=sort_keys,
        spill_dir=tmp_path / "reduce",
        run_prefix="t",
    )
    assert len(reduced) <= 2
    merged = [
        row["k"]
        for row in iter_k_way_merge_dataframes(
            runs,
            sort_keys=sort_keys,
            spill_dir=tmp_path / "final_merge",
        )
    ]
    assert merged == naive


def test_reduce_terminates_when_merge_emits_multiple_parts(tmp_path, monkeypatch):
    monkeypatch.setattr(spill_sort, "_MERGE_FAN_IN", 2)
    monkeypatch.setattr(spill_sort, "_SORT_RUN_ROWS", 2)
    sort_keys = ("k",)
    rows_per_run = 5
    num_runs = 32
    runs: list = []
    next_value = 0
    for run_index in range(num_runs):
        keys = list(range(next_value, next_value + rows_per_run))
        next_value += rows_per_run
        path = tmp_path / f"run_{run_index:02d}.parquet"
        pl.DataFrame({"k": keys}).write_parquet(path)
        runs.append(path)
    naive = sorted(
        int(value)
        for path in runs
        for value in pl.read_parquet(path)["k"].to_list()
    )

    max_logical_fan_in = 0
    original_merge = spill_sort._iter_k_way_merge_logical_runs

    def tracked_merge(logical_runs, **kwargs):
        nonlocal max_logical_fan_in
        max_logical_fan_in = max(max_logical_fan_in, len(logical_runs))
        yield from original_merge(logical_runs, **kwargs)

    monkeypatch.setattr(spill_sort, "_iter_k_way_merge_logical_runs", tracked_merge)

    max_parts_per_group = 0
    original_group = _merge_fan_in_group

    def tracked_group(*args, **kwargs):
        nonlocal max_parts_per_group
        merged = original_group(*args, **kwargs)
        max_parts_per_group = max(max_parts_per_group, len(merged.parts))
        return merged

    monkeypatch.setattr(spill_sort, "_merge_fan_in_group", tracked_group)

    reduced = _reduce_sorted_runs(
        runs,
        sort_keys=sort_keys,
        spill_dir=tmp_path / "reduce",
        run_prefix="t",
    )
    assert len(reduced) <= spill_sort._MERGE_FAN_IN
    merged = [
        int(row["k"])
        for row in iter_k_way_merge_dataframes(
            runs,
            sort_keys=sort_keys,
            spill_dir=tmp_path / "final_merge",
        )
    ]
    assert merged == naive
    assert max_parts_per_group >= 2
    assert max_logical_fan_in <= spill_sort._MERGE_FAN_IN


def test_many_witness_batches_equivalent_to_single_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "portable_batch_execution.packs.replay_reduction.causal_grid._IDENTITY_SCAN_BATCH",
        2,
    )
    trades = _write(
        tmp_path / "trades.parquet",
        [_trade_row(identity=1, identity_norm="1", block=100, timestamp_ms=8_000)],
    )
    witness_rows = [
        _witness_row(block=100 + index, timestamp_ms=8_000 + index) for index in range(12)
    ]
    witness = _write(tmp_path / "w.parquet", witness_rows)
    request = _request(
        input_roles=(
            {"input_index": 0, "role": "canonical_trade"},
            {"input_index": 1, "role": "causal_witness"},
        ),
    )
    many_batch = execute_causal_grid_extract([trades, witness], request)["rows"]
    monkeypatch.setattr(
        "portable_batch_execution.packs.replay_reduction.causal_grid._IDENTITY_SCAN_BATCH",
        65_536,
    )
    single_batch = execute_causal_grid_extract([trades, witness], request)["rows"]
    assert many_batch == single_batch


def test_global_trade_bucket_buffer_equivalence(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "portable_batch_execution.packs.replay_reduction.causal_grid._GLOBAL_TRADE_BUCKET_BUFFER_ROWS",
        8,
    )

    rows = [
        _trade_row(
            identity=index + 1,
            identity_norm=str(index + 1),
            block=100 + index,
            timestamp_ms=8_000 + index,
            notional=float(index + 1),
        )
        for index in range(32)
    ]
    trades = _write(tmp_path / "trades.parquet", rows)
    witness = _write(
        tmp_path / "w.parquet",
        [
            _witness_row(block=90, timestamp_ms=7_000),
            _witness_row(block=200, timestamp_ms=10_000),
        ],
    )
    small_buffer = execute_causal_grid_extract(
        [trades, witness],
        _request(
            input_roles=(
                {"input_index": 0, "role": "canonical_trade"},
                {"input_index": 1, "role": "causal_witness"},
            ),
        ),
    )["rows"]
    monkeypatch.setattr(
        "portable_batch_execution.packs.replay_reduction.causal_grid._GLOBAL_TRADE_BUCKET_BUFFER_ROWS",
        _GLOBAL_TRADE_BUCKET_BUFFER_ROWS,
    )
    default_buffer = execute_causal_grid_extract(
        [trades, witness],
        _request(
            input_roles=(
                {"input_index": 0, "role": "canonical_trade"},
                {"input_index": 1, "role": "causal_witness"},
            ),
        ),
    )["rows"]
    assert small_buffer == default_buffer
