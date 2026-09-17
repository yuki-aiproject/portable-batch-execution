import json

import polars as pl
import pytest

from portable_batch_execution.packs.replay_reduction import canonicalize, event_window

_PARAMS = {
    "schema_version": "pbe.replay.structural-canonicalize.v1",
    "identity_source_column": "identity",
    "identity_normalized_column": "identity_norm",
    "measurement_core_fields": ["price"],
}


def test_canonicalize_rejects_whole_corpus_read_and_concat(tmp_path, monkeypatch):
    path = tmp_path / "part.parquet"
    pl.DataFrame([{"identity": 1, "identity_norm": "1", "price": 1.0}]).write_parquet(path)

    def forbid_read(*args, **kwargs):
        raise AssertionError("read_parquet must not be used")

    def forbid_concat(*args, **kwargs):
        raise AssertionError("concat must not be used for whole-corpus materialization")

    monkeypatch.setattr(pl, "read_parquet", forbid_read)
    monkeypatch.setattr(pl, "concat", forbid_concat)

    result = canonicalize.execute_structural_canonicalize([path], _PARAMS)
    assert result.positive_group_count == 1
    assert len(json.dumps(canonicalize.state_summary(result))) < 4000


def test_event_window_rejects_whole_history_to_dicts(tmp_path, monkeypatch):
    path = tmp_path / "records.parquet"
    pl.DataFrame(
        [
            {
                "symbol": "AAA",
                "block": 1,
                "timestamp_ms": 1,
                "price": 1.0,
                "seq": 0,
                "identity": 1,
                "identity_norm": "1",
            }
        ]
    ).write_parquet(path)

    class _Frame:
        def to_dicts(self):
            raise AssertionError("to_dicts must not be used")

    def forbid_read(*args, **kwargs):
        return _Frame()

    monkeypatch.setattr(pl, "read_parquet", forbid_read)

    request = {
        "schema_version": "pbe.replay.event-window-extract.v3",
        "request_id": "req",
        "symbol": "AAA",
        "symbol_column": "symbol",
        "block_column": "block",
        "timestamp_column": "timestamp_ms",
        "decision_timestamp_ms": 1000,
        "causal_cutoff_block": 1,
        "as_of_measurement_field": "price",
        "canonical_trade_profile": {
            "schema_version": "pbe.replay.canonical-trade-profile.v1",
            "identity_source_column": "identity",
            "identity_normalized_column": "identity_norm",
            "measurement_core_fields": ["price"],
        },
    }
    result = event_window.execute_event_window_extract([path], request)
    assert result["facts"]


def test_canonicalize_does_not_materialize_all_bucket_lists(tmp_path):
    assert not hasattr(canonicalize, "_bucket_values")
    rows = [
        {"identity": i, "identity_norm": str(i), "price": float(i)} for i in range(1, 5001)
    ]
    path = tmp_path / "part.parquet"
    pl.DataFrame(rows).write_parquet(path)
    state = canonicalize.execute_structural_canonicalize([path], _PARAMS)
    assert state.positive_group_count == 5000
    assert len(json.dumps(canonicalize.state_summary(state))) < 4000


def test_canonicalize_materialization_guard_enforced(tmp_path, monkeypatch):
    monkeypatch.setattr(canonicalize, "_MAX_IDENTITY_MATERIALIZATION", 32)
    rows = [{"identity": i, "identity_norm": str(i), "price": 1.0} for i in range(1, 200)]
    path = tmp_path / "dense.parquet"
    pl.DataFrame(rows).write_parquet(path)
    with pytest.raises(canonicalize.StructuralCanonicalizeError):
        canonicalize.execute_structural_canonicalize([path], _PARAMS)


def test_canonicalize_avoids_global_group_and_nunique_aggregations(tmp_path, monkeypatch):
    def forbid_group_by(self, *args, **kwargs):
        raise AssertionError("group_by must not be used in canonicalize validation")

    def forbid_n_unique(self, *args, **kwargs):
        raise AssertionError("n_unique must not be used in canonicalize validation")

    monkeypatch.setattr(pl.DataFrame, "group_by", forbid_group_by)
    monkeypatch.setattr(pl.Expr, "n_unique", forbid_n_unique)
    path = tmp_path / "part.parquet"
    pl.DataFrame(
        [
            {"identity": 9, "identity_norm": "9", "price": 9.0},
            {"identity": 1, "identity_norm": "1", "price": 1.0},
            {"identity": 5, "identity_norm": "5", "price": 5.0},
        ]
    ).write_parquet(path)
    state = canonicalize.execute_structural_canonicalize([path], _PARAMS)
    assert state.positive_group_count == 3


def test_non_contiguous_recurrence_fails_across_scan_batches(tmp_path, monkeypatch):
    monkeypatch.setattr(canonicalize, "_IDENTITY_SCAN_BATCH", 2)
    path = tmp_path / "part.parquet"
    pl.DataFrame(
        [
            {"identity": 1, "identity_norm": "1", "price": 1.0},
            {"identity": 2, "identity_norm": "2", "price": 2.0},
            {"identity": 1, "identity_norm": "1", "price": 1.0},
        ]
    ).write_parquet(path)
    with pytest.raises(canonicalize.StructuralCanonicalizeError):
        canonicalize.execute_structural_canonicalize([path], _PARAMS)


def test_non_contiguous_recurrence_fails_across_sort_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(canonicalize, "_SORT_RUN_CAPACITY", 1)
    path = tmp_path / "part.parquet"
    pl.DataFrame(
        [
            {"identity": 1, "identity_norm": "1", "price": 1.0},
            {"identity": 2, "identity_norm": "2", "price": 2.0},
            {"identity": 1, "identity_norm": "1", "price": 1.0},
        ]
    ).write_parquet(path)
    with pytest.raises(canonicalize.StructuralCanonicalizeError):
        canonicalize.execute_structural_canonicalize([path], _PARAMS)


def test_production_publish_uses_incremental_bucket_payloads(tmp_path, monkeypatch):
    from hashlib import sha256

    from portable_batch_execution.contracts import ArtifactRef

    rows = [{"identity": 1, "identity_norm": "1", "price": 1.0}]
    path = tmp_path / "part.parquet"
    pl.DataFrame(rows).write_parquet(path)
    state = canonicalize.execute_structural_canonicalize([path], _PARAMS)

    def forbid_bulk(*args, **kwargs):
        raise AssertionError("encode_state_buckets must not be used in production publish")

    monkeypatch.setattr(canonicalize, "encode_state_buckets", forbid_bulk)

    class Plane:
        def __init__(self):
            self.max_live_payload = 0
            self.live = 0
            self.written = []

        def write(self, data, media_type):
            self.live += 1
            self.max_live_payload = max(self.max_live_payload, self.live)
            self.written.append((data, media_type))
            self.live -= 1
            return ArtifactRef(
                object_id=str(len(self.written)),
                uri=f"pbe://private/{len(self.written)}",
                sha256="sha256:" + sha256(data).hexdigest(),
                size_bytes=len(data),
                media_type=media_type,
            )

    from portable_batch_execution.worker import execute_wave as ew

    plane = Plane()
    ew._write_canonicalize_state(plane, state)
    assert plane.max_live_payload == 1
    assert len(plane.written) == 1 + state.bucket_count


def test_bucket_payload_bound_enforced(tmp_path, monkeypatch):
    monkeypatch.setattr(canonicalize, "_MAX_BUCKET_PAYLOAD_BYTES", 30)
    path = tmp_path / "part.parquet"
    pl.DataFrame([{"identity": 1, "identity_norm": "1", "price": 1.0}]).write_parquet(path)
    state = canonicalize.execute_structural_canonicalize([path], _PARAMS)
    with pytest.raises(canonicalize.StructuralCanonicalizeError):
        for _ in canonicalize.iter_state_bucket_payloads(state):
            pass


def test_read_canonicalize_state_consumes_one_bucket_payload_at_a_time(tmp_path, monkeypatch):
    from hashlib import sha256

    from portable_batch_execution.contracts import ArtifactRef
    from portable_batch_execution.worker import execute_wave as ew

    path = tmp_path / "part.parquet"
    pl.DataFrame([{"identity": 1, "identity_norm": "1", "price": 1.0}]).write_parquet(path)
    state = canonicalize.execute_structural_canonicalize([path], _PARAMS)

    payloads = {}
    bucket_refs = []
    for index, payload in enumerate(canonicalize.iter_state_bucket_payloads(state)):
        object_id = f"bucket-{index}"
        payloads[object_id] = payload
        bucket_refs.append(
            ArtifactRef(
                object_id=object_id,
                uri=f"pbe://private/{object_id}",
                sha256="sha256:" + sha256(payload).hexdigest(),
                size_bytes=len(payload),
                media_type=canonicalize.BUCKET_MEDIA_TYPE,
            )
        )
    summary = canonicalize.attach_bucket_refs(
        canonicalize.state_summary(state), tuple(bucket_refs)
    )
    summary_bytes = json.dumps(summary, sort_keys=True).encode("utf-8")
    summary_id = "summary"
    payloads[summary_id] = summary_bytes
    summary_ref = ArtifactRef(
        object_id=summary_id,
        uri="pbe://private/summary",
        sha256="sha256:" + sha256(summary_bytes).hexdigest(),
        size_bytes=len(summary_bytes),
        media_type="application/json",
    )

    class Plane:
        def read(self, ref):
            return payloads[ref.object_id]

    plane = Plane()
    live = 0
    max_live = 0
    original_read = ew._read_verified_artifact_bytes

    def tracked_read(data_plane, ref):
        nonlocal live, max_live
        live += 1
        max_live = max(max_live, live)
        try:
            return original_read(data_plane, ref)
        finally:
            live -= 1

    monkeypatch.setattr(ew, "_read_verified_artifact_bytes", tracked_read)
    decoded = ew._read_canonicalize_state(plane, summary_ref)
    assert decoded.positive_group_count == state.positive_group_count
    assert max_live == 1
