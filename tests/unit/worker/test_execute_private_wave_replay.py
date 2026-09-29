import io
import itertools
import json
from datetime import UTC, datetime
from hashlib import sha256

import polars as pl

from portable_batch_execution.contracts import ArtifactRef
from portable_batch_execution.packs.replay_reduction.canonicalize import (
    BUCKET_MEDIA_TYPE,
    STATE_SCHEMA_VERSION,
    decode_state,
)
from portable_batch_execution.worker.execute_wave import execute_private_wave

_OBJECT_SEQUENCE = itertools.count()


def _parquet_payload(rows):
    buffer = io.BytesIO()
    pl.DataFrame(rows).write_parquet(buffer)
    return buffer.getvalue()


def _ref(name: str, payload: bytes) -> ArtifactRef:
    return ArtifactRef(
        object_id=name,
        uri=f"pbe://private/{name}",
        sha256="sha256:" + sha256(payload).hexdigest(),
        size_bytes=len(payload),
        media_type="application/vnd.apache.parquet",
    )


def _plane_for_replay(*, operation, input_refs, operation_params):
    payloads = {ref.object_id: None for ref in input_refs}

    class Plane:
        def __init__(self):
            self.appended = []
            self.written = []
            self._payloads = dict(payloads)

        def resolve_wave(self, run_id, wave_id):
            now = datetime.now(UTC).isoformat()
            return {
                "job": {
                    "job_id": "job",
                    "logical_run_id": "opaque-run",
                    "pack": "replay-batch",
                    "operation": operation,
                    "input_manifest_ref": input_refs[0].model_dump(mode="json"),
                    "sharding": {},
                    "execution": {"max_parallel": 1, "max_attempts_per_shard": 4},
                    "security_profile": "offline",
                    "provenance": {"producer": "test", "revision": "1", "created_at": now},
                    "operation_params": operation_params,
                },
                "wave": {
                    "logical_run_id": "opaque-run",
                    "wave_id": "opaque-wave",
                    "ordinal": 0,
                    "shard_ids": ["opaque-shard"],
                    "max_parallel": 1,
                },
                "shards": [
                    {
                        "logical_run_id": "opaque-run",
                        "shard_id": "opaque-shard",
                        "ordinal": 0,
                        "correctness": {},
                        "input_refs": [ref.model_dump(mode="json") for ref in input_refs],
                        "input_digest": "current",
                        "execution_fingerprint": "fixed",
                    }
                ],
            }

        def read_attempts(self, run_id):
            return tuple(self.appended)

        def read(self, ref):
            return self._payloads[ref.object_id]

        def open_content(self, ref):
            from portable_batch_execution.data_plane.base import ArtifactContentStream

            payload = self._payloads[ref.object_id]

            def chunks():
                for offset in range(0, len(payload), 13):
                    yield payload[offset : offset + 13]

            return ArtifactContentStream(size_bytes=len(payload), chunks=chunks())

        def write(self, data, media_type):
            object_id = f"out-{next(_OBJECT_SEQUENCE)}"
            self._payloads[object_id] = data
            self.written.append((object_id, data, media_type))
            return ArtifactRef(
                object_id=object_id,
                uri=f"pbe://private/{object_id}",
                sha256="sha256:" + sha256(data).hexdigest(),
                size_bytes=len(data),
                media_type=media_type,
            )

        def append_attempt(self, record):
            self.appended.append(record)

    plane = Plane()
    for ref in input_refs:
        plane._payloads[ref.object_id] = None
    return plane


def test_private_wave_materializes_multiple_parquet_inputs_for_canonicalize():
    payload_a = _parquet_payload(
        [{"identity": 1, "identity_norm": "1", "price": 1.0, "symbol": "AAA", "block": 1}]
    )
    payload_b = _parquet_payload(
        [{"identity": 2, "identity_norm": "2", "price": 2.0, "symbol": "AAA", "block": 2}]
    )
    refs = [_ref("a", payload_a), _ref("b", payload_b)]
    plane = _plane_for_replay(
        operation="replay.structural_canonicalize",
        input_refs=refs,
        operation_params={
            "schema_version": "pbe.replay.structural-canonicalize.v1",
            "identity_source_column": "identity",
            "identity_normalized_column": "identity_norm",
            "measurement_core_fields": ["price"],
        },
    )
    plane._payloads["a"] = payload_a
    plane._payloads["b"] = payload_b

    attempts = execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert attempts[0].status == "succeeded"
    assert attempts[0].counts["input_rows"] == 2

    record = attempts[0]
    assert len(record.output_refs) == 1 + 256
    summary_ref = record.output_refs[0]
    bucket_refs = record.output_refs[1:]
    assert summary_ref.media_type == "application/json"
    assert all(ref.media_type == BUCKET_MEDIA_TYPE for ref in bucket_refs)
    summary = json.loads(plane._payloads[summary_ref.object_id].decode())
    assert summary["schema_version"] == STATE_SCHEMA_VERSION
    assert summary["bucket_count"] == 256
    assert [ref.object_id for ref in bucket_refs] == [
        item["object_id"] for item in summary["bucket_refs"]
    ]
    state = decode_state(
        summary, tuple(plane._payloads[ref.object_id] for ref in bucket_refs)
    )
    assert state.positive_group_count == 2


def test_private_wave_merge_reads_and_republishes_multi_artifact_state():
    payload = _parquet_payload(
        [{"identity": 1, "identity_norm": "1", "price": 5.0, "symbol": "AAA", "block": 1}]
    )
    refs = [_ref("seed", payload)]
    params = {
        "schema_version": "pbe.replay.structural-canonicalize.v1",
        "identity_source_column": "identity",
        "identity_normalized_column": "identity_norm",
        "measurement_core_fields": ["price"],
    }
    seed_plane = _plane_for_replay(
        operation="replay.structural_canonicalize",
        input_refs=refs,
        operation_params=params,
    )
    seed_plane._payloads["seed"] = payload
    seed_attempts = execute_private_wave("opaque-run", "opaque-wave", plane=seed_plane)
    left_refs = seed_attempts[0].output_refs

    payload_right = _parquet_payload(
        [
            {"identity": 1, "identity_norm": "1", "price": 5.0, "symbol": "AAA", "block": 1},
            {"identity": 2, "identity_norm": "2", "price": 6.0, "symbol": "AAA", "block": 2},
        ]
    )
    right_plane = _plane_for_replay(
        operation="replay.structural_canonicalize",
        input_refs=[_ref("seed2", payload_right)],
        operation_params=params,
    )
    right_plane._payloads["seed2"] = payload_right
    right_attempts = execute_private_wave("opaque-run", "opaque-wave", plane=right_plane)
    right_refs = right_attempts[0].output_refs

    merge_plane = _plane_for_replay(
        operation="replay.structural_canonicalize_merge",
        input_refs=[left_refs[0], right_refs[0]],
        operation_params={
            "schema_version": "pbe.replay.structural-canonicalize-merge.v1"
        },
    )
    for ref in left_refs:
        merge_plane._payloads[ref.object_id] = seed_plane._payloads[ref.object_id]
    for ref in right_refs:
        merge_plane._payloads[ref.object_id] = right_plane._payloads[ref.object_id]

    merged = execute_private_wave("opaque-run", "opaque-wave", plane=merge_plane)
    assert merged[0].status == "succeeded"
    assert merged[0].counts["input_rows"] == 3
    summary_ref = merged[0].output_refs[0]
    summary = json.loads(merge_plane._payloads[summary_ref.object_id].decode())
    state = decode_state(
        summary,
        tuple(
            merge_plane._payloads[ref.object_id] for ref in merged[0].output_refs[1:]
        ),
    )
    assert state.positive_group_count == 2
    assert state.positive_row_count == 2


def test_private_wave_merge_fails_closed_on_non_contiguous_recurrence():
    def seed(rows, name):
        payload = _parquet_payload(rows)
        plane = _plane_for_replay(
            operation="replay.structural_canonicalize",
            input_refs=[_ref(name, payload)],
            operation_params={
                "schema_version": "pbe.replay.structural-canonicalize.v1",
                "identity_source_column": "identity",
                "identity_normalized_column": "identity_norm",
                "measurement_core_fields": ["price"],
            },
        )
        plane._payloads[name] = payload
        return plane, execute_private_wave("opaque-run", "opaque-wave", plane=plane)[0]

    left_plane, left = seed(
        [{"identity": 1, "identity_norm": "1", "price": 1.0}], "l"
    )
    right_plane, right = seed(
        [
            {"identity": 2, "identity_norm": "2", "price": 2.0},
            {"identity": 1, "identity_norm": "1", "price": 1.0},
        ],
        "r",
    )
    merge_plane = _plane_for_replay(
        operation="replay.structural_canonicalize_merge",
        input_refs=[left.output_refs[0], right.output_refs[0]],
        operation_params={
            "schema_version": "pbe.replay.structural-canonicalize-merge.v1"
        },
    )
    for ref in left.output_refs:
        merge_plane._payloads[ref.object_id] = left_plane._payloads[ref.object_id]
    for ref in right.output_refs:
        merge_plane._payloads[ref.object_id] = right_plane._payloads[ref.object_id]

    from portable_batch_execution.worker.execute_wave import PrivateWaveExecutionError

    try:
        execute_private_wave("opaque-run", "opaque-wave", plane=merge_plane)
    except PrivateWaveExecutionError as error:
        assert error.attempts[0].failure == "shard_pack_execution_failed"
    else:
        raise AssertionError("merge must fail closed on non-contiguous recurrence")


def test_private_wave_merge_fails_closed_on_mismatched_bucket_state():
    payload = _parquet_payload(
        [{"identity": 1, "identity_norm": "1", "price": 1.0}]
    )
    plane = _plane_for_replay(
        operation="replay.structural_canonicalize",
        input_refs=[_ref("seed", payload)],
        operation_params={
            "schema_version": "pbe.replay.structural-canonicalize.v1",
            "identity_source_column": "identity",
            "identity_normalized_column": "identity_norm",
            "measurement_core_fields": ["price"],
        },
    )
    plane._payloads["seed"] = payload
    refs = execute_private_wave("opaque-run", "opaque-wave", plane=plane)[0].output_refs

    merge_plane = _plane_for_replay(
        operation="replay.structural_canonicalize_merge",
        input_refs=[refs[0], refs[0]],
        operation_params={
            "schema_version": "pbe.replay.structural-canonicalize-merge.v1"
        },
    )
    for ref in refs:
        merge_plane._payloads[ref.object_id] = plane._payloads[ref.object_id]
    del merge_plane._payloads[refs[-1].object_id]

    from portable_batch_execution.worker.execute_wave import PrivateWaveExecutionError

    try:
        execute_private_wave("opaque-run", "opaque-wave", plane=merge_plane)
    except PrivateWaveExecutionError as error:
        assert error.attempts[0].failure in {
            "input_artifact_invalid",
            "input_artifact_read_failed",
        }
    else:
        raise AssertionError("merge must fail closed on malformed bucket state")


def test_private_wave_causal_grid_extract_dispatches_typed_request():
    trade_payload = _parquet_payload(
        [
            {
                "identity": 1,
                "identity_norm": "1",
                "symbol": "AAA",
                "block": 90,
                "timestamp_ms": 8_000,
                "price": 1.0,
                "notional": 10.0,
                "seq": 0,
            },
            {
                "identity": 2,
                "identity_norm": "2",
                "symbol": "AAA",
                "block": 100,
                "timestamp_ms": 9_500,
                "price": 2.0,
                "notional": 20.0,
                "seq": 0,
            },
        ]
    )
    witness_payload = _parquet_payload(
        [
            {"block": 90, "timestamp_ms": 8_000},
            {"block": 110, "timestamp_ms": 10_000},
        ]
    )
    request = {
        "schema_version": "pbe.replay.causal-grid-extract.v1",
        "request_id": "worker-grid",
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
    request_payload = json.dumps(request).encode("utf-8")
    refs = [
        _ref("trades", trade_payload),
        _ref("witness", witness_payload),
        ArtifactRef(
            object_id="request",
            uri="pbe://private/request",
            sha256="sha256:" + sha256(request_payload).hexdigest(),
            size_bytes=len(request_payload),
            media_type="application/json",
        ),
    ]
    plane = _plane_for_replay(
        operation="replay.causal_grid_extract",
        input_refs=refs,
        operation_params={"schema_version": "pbe.replay.causal-grid-extract-job.v1"},
    )
    plane._payloads["trades"] = trade_payload
    plane._payloads["witness"] = witness_payload
    plane._payloads["request"] = request_payload
    attempts = execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert attempts[0].status == "succeeded"
    result = json.loads(plane._payloads[attempts[0].output_refs[0].object_id].decode())
    assert result["schema_version"] == "pbe.replay.causal-grid-extract-result.v1"
    assert len(result["rows"]) == 1

