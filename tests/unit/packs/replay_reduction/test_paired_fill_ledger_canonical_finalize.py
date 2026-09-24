import io
import json
from hashlib import sha256

import polars as pl
import pytest
from pydantic import ValidationError

from portable_batch_execution.packs.replay_reduction.models import (
    PairedFillLedgerCanonicalFinalizeRequest,
)
from portable_batch_execution.packs.replay_reduction.paired_fill_ledger_canonical_finalize import (
    CANONICAL_METADATA_SCHEMA_VERSION,
    PairedFillLedgerCanonicalFinalizeError,
    PairedFillLedgerCanonicalizeCrossShardState,
    assert_verified_payload_bytes,
    build_canonical_finalize_metadata,
    canonicalize_verified_shard_ledger,
    execute_paired_fill_ledger_canonical_finalize,
    publish_paired_fill_ledger_canonical_finalize_artifacts,
)
from portable_batch_execution.packs.replay_reduction.paired_fill_reduce import (
    LEDGER_PARQUET_MEDIA_TYPE,
    build_paired_fill_metadata,
    execute_paired_fill_reduce,
    publish_paired_fill_reduce_artifacts,
)

_ROLE_A = "side_a"
_ROLE_B = "side_b"


def _row(**fields):
    base = {
        "identity": 1,
        "identity_norm": "1",
        "pair_role": _ROLE_A,
        "core_price": 100.0,
        "core_size": 2.0,
        "start_pos": 0.0,
        "signed_qty": 2.0,
    }
    base.update(fields)
    return base


def _reduce_request(**overrides):
    base = {
        "schema_version": "pbe.replay.paired-fill-reduce.v1",
        "request_id": "pf-reduce",
        "identity_mapping": {
            "identity_source_column": "identity",
            "identity_normalized_column": "identity_norm",
        },
        "pair_mapping": {
            "pair_role_column": "pair_role",
            "aggressor_role_value": _ROLE_A,
            "passive_role_value": _ROLE_B,
            "measurement_core_fields": ["core_price", "core_size"],
            "start_position_column": "start_pos",
            "signed_execution_column": "signed_qty",
        },
        "partition": {"terminal": True},
        "max_output_rows": 100,
        "max_exception_rows": 10,
    }
    base.update(overrides)
    return base


def _write(path, rows):
    pl.DataFrame(rows).write_parquet(path)
    return path


def _digest_binding(data: bytes, media_type: str) -> dict:
    return {
        "schema_version": "pbe.replay.verified-payload-digest.v1",
        "sha256": sha256(data).hexdigest(),
        "size_bytes": len(data),
        "media_type": media_type,
    }


def _carry_binding(incoming_carry, global_ordinal: int) -> dict:
    row = incoming_carry["pending_row"]
    local = row.get("_source_input_index", row.get("source_input_index"))
    offset = row.get("_source_row_offset", row.get("source_row_offset"))
    return {
        "schema_version": "pbe.replay.carry-source-ordinal-binding.v1",
        "source_input_index": int(local),
        "source_row_offset": int(offset),
        "global_source_ordinal": global_ordinal,
    }


def _canonical_request(
    *,
    shard_ordinal: int,
    ledger_bytes: bytes,
    metadata_bytes: bytes,
    global_ordinals: tuple[int, ...],
    incoming_carry=None,
    incoming_carry_global_ordinal: int | None = None,
    terminal: bool = True,
    request_id: str = "pf-canonical",
) -> dict:
    carry_binding = None
    if incoming_carry is not None and incoming_carry.get("pending_row") is not None:
        if incoming_carry_global_ordinal is None:
            raise ValueError("incoming_carry_global_ordinal required")
        carry_binding = _carry_binding(incoming_carry, incoming_carry_global_ordinal)
    return {
        "schema_version": "pbe.replay.paired-fill-ledger-canonical-finalize.v1",
        "request_id": request_id,
        "partition_id": "partition-0",
        "terminal": terminal,
        "shard": {
            "shard_ordinal": shard_ordinal,
            "ledger": _digest_binding(ledger_bytes, LEDGER_PARQUET_MEDIA_TYPE),
            "reducer_metadata": _digest_binding(metadata_bytes, "application/json"),
            "local_index_to_global_ordinal": list(global_ordinals),
            "incoming_carry": incoming_carry,
            "incoming_carry_source_binding": carry_binding,
        },
    }


def _participant(row, role):
    return next(item for item in row["participants"] if item["role"] == role)


def _reduce_shard(tmp_path, rows, *, partition, paths=None):
    if paths is None:
        path = _write(tmp_path / f"part-{partition.get('terminal')}.parquet", rows)
        paths = [path]
    else:
        paths = list(paths)
    return execute_paired_fill_reduce(paths, _reduce_request(partition=partition))


def _metadata_bytes(result) -> bytes:
    return json.dumps(
        build_paired_fill_metadata(result, ledger_parquet_ref=None),
        sort_keys=True,
    ).encode("utf-8")


def test_distinct_global_ordinals_for_same_local_index_zero(tmp_path):
    first = _reduce_shard(
        tmp_path,
        [
            _row(identity=7, identity_norm="7"),
            _row(identity=7, identity_norm="7", pair_role=_ROLE_B, signed_qty=-2.0),
        ],
        partition={"terminal": True},
    )
    second = _reduce_shard(
        tmp_path,
        [_row(identity=8, identity_norm="8")],
        partition={"terminal": True},
    )
    ledger0 = bytes(first["ledger_parquet_bytes"])
    meta0 = _metadata_bytes(first)
    ledger1 = bytes(second["ledger_parquet_bytes"])
    meta1 = _metadata_bytes(second)

    rows0, _, state0 = canonicalize_verified_shard_ledger(
        ledger0,
        meta0,
        _canonical_request(
            shard_ordinal=0,
            ledger_bytes=ledger0,
            metadata_bytes=meta0,
            global_ordinals=(100,),
            terminal=True,
        ),
    )
    row0 = rows0[0]
    assert _participant(row0, "aggressor")["source_input_ordinal"] == 100

    rows1, _, _ = canonicalize_verified_shard_ledger(
        ledger1,
        meta1,
        _canonical_request(
            shard_ordinal=0,
            ledger_bytes=ledger1,
            metadata_bytes=meta1,
            global_ordinals=(200,),
            terminal=True,
        ),
    )
    assert rows1[0]["participants"][0]["source_input_ordinal"] == 200


def test_cross_shard_carry_pair_preserves_both_global_ordinals(tmp_path):
    path_a = _write(
        tmp_path / "a.parquet",
        [_row(identity=30, identity_norm="30", start_pos=1.0, signed_qty=1.0)],
    )
    first = execute_paired_fill_reduce(
        [path_a],
        _reduce_request(partition={"terminal": False}),
    )
    incoming = first["outgoing_carry"]
    ledger_row = {
        "identity_namespace": None,
        "ledger_identity": 30,
        "classification": "complete_pair",
        "measurement_core": {"core_price": 100.0, "core_size": 2.0},
        "first_source_input_index": 0,
        "first_source_row_offset": 0,
        "last_source_input_index": 1,
        "last_source_row_offset": 0,
        "source_cursor_first": {"source_input_index": 0, "source_row_offset": 0},
        "source_cursor_last": {"source_input_index": 1, "source_row_offset": 0},
        "participants": [
            {
                "role": "aggressor",
                "start_position": 1.0,
                "signed_execution": 1.0,
                "opening_quantity": 1.0,
                "closing_quantity": 0.0,
                "post_position": 2.0,
                "source_input_index": 0,
                "source_row_offset": 0,
            },
            {
                "role": "passive",
                "start_position": 0.0,
                "signed_execution": -1.0,
                "opening_quantity": 1.0,
                "closing_quantity": 0.0,
                "post_position": -1.0,
                "source_input_index": 1,
                "source_row_offset": 0,
            },
        ],
    }
    buffer = io.BytesIO()
    pl.DataFrame([ledger_row]).write_parquet(buffer)
    ledger1 = buffer.getvalue()
    meta1 = _metadata_bytes(
        {
            "schema_version": "pbe.replay.paired-fill-reduce-result.v1",
            "summary": {"ledger_row_count": 1},
            "exceptions": [],
            "outgoing_carry": {
                "schema_version": "pbe.replay.paired-fill-reduce-carry.v1",
                "pending_row": None,
                "pending_identity": None,
                "pending_group_key": None,
            },
        }
    )
    ledger0 = bytes(first["ledger_parquet_bytes"])
    meta0 = _metadata_bytes(first)

    _, result0, state0 = canonicalize_verified_shard_ledger(
        ledger0,
        meta0,
        _canonical_request(
            shard_ordinal=0,
            ledger_bytes=ledger0,
            metadata_bytes=meta0,
            global_ordinals=(10,),
            terminal=False,
        ),
    )
    assert len(result0["canonical_ledger_rows"]) == 0

    _, result1, _ = canonicalize_verified_shard_ledger(
        ledger1,
        meta1,
        _canonical_request(
            shard_ordinal=1,
            ledger_bytes=ledger1,
            metadata_bytes=meta1,
            global_ordinals=(500, 20),
            incoming_carry=incoming,
            incoming_carry_global_ordinal=10,
            terminal=True,
        ),
        cross_shard_state=state0,
    )
    row = result1["canonical_ledger_rows"][0]
    aggressor = _participant(row, "aggressor")
    passive = _participant(row, "passive")
    assert aggressor["source_input_ordinal"] == 10
    assert passive["source_input_ordinal"] == 20


def test_fail_closed_invalid_local_index_mapping(tmp_path):
    bad_row = {
        "identity_namespace": None,
        "ledger_identity": 1,
        "classification": "singleton",
        "measurement_core": {"core_price": 100.0, "core_size": 2.0},
        "participants": [
            {
                "role": "aggressor",
                "start_position": 0.0,
                "signed_execution": 2.0,
                "opening_quantity": 2.0,
                "closing_quantity": 0.0,
                "post_position": 2.0,
                "source_input_index": 3,
                "source_row_offset": 0,
            }
        ],
    }
    buffer = io.BytesIO()
    pl.DataFrame([bad_row]).write_parquet(buffer)
    ledger = buffer.getvalue()
    meta = _metadata_bytes(
        {
            "schema_version": "pbe.replay.paired-fill-reduce-result.v1",
            "summary": {"ledger_row_count": 1},
            "exceptions": [],
            "outgoing_carry": {
                "schema_version": "pbe.replay.paired-fill-reduce-carry.v1",
                "pending_row": None,
                "pending_identity": None,
                "pending_group_key": None,
            },
        }
    )
    with pytest.raises(PairedFillLedgerCanonicalFinalizeError, match="invalid shard-local"):
        canonicalize_verified_shard_ledger(
            ledger,
            meta,
            _canonical_request(
                shard_ordinal=0,
                ledger_bytes=ledger,
                metadata_bytes=meta,
                global_ordinals=(0,),
            ),
        )


def test_fail_closed_carry_without_prior_mapping(tmp_path):
    path_a = _write(tmp_path / "a.parquet", [_row(identity=30, identity_norm="30")])
    path_b = _write(
        tmp_path / "b.parquet",
        [_row(identity=30, identity_norm="30", pair_role=_ROLE_B, signed_qty=-1.0)],
    )
    first = execute_paired_fill_reduce([path_a], _reduce_request(partition={"terminal": False}))
    incoming = first["outgoing_carry"]
    second = execute_paired_fill_reduce(
        [path_b],
        _reduce_request(partition={"terminal": True, "incoming_carry": incoming}),
    )
    ledger1 = bytes(second["ledger_parquet_bytes"])
    meta1 = _metadata_bytes(second)
    with pytest.raises(ValidationError, match="incoming_carry_source_binding"):
        PairedFillLedgerCanonicalFinalizeRequest.model_validate(
            {
                "schema_version": "pbe.replay.paired-fill-ledger-canonical-finalize.v1",
                "request_id": "pf-canonical",
                "partition_id": "partition-0",
                "terminal": True,
                "shard": {
                    "shard_ordinal": 0,
                    "ledger": _digest_binding(ledger1, LEDGER_PARQUET_MEDIA_TYPE),
                    "reducer_metadata": _digest_binding(meta1, "application/json"),
                    "local_index_to_global_ordinal": [1],
                    "incoming_carry": incoming,
                },
            }
        )


def test_fail_closed_payload_identity_mismatch(tmp_path):
    result = _reduce_shard(tmp_path, [_row(identity=2, identity_norm="2")], partition={"terminal": True})
    ledger = bytes(result["ledger_parquet_bytes"])
    meta = _metadata_bytes(result)
    tampered = bytearray(ledger)
    tampered[-1] ^= 0xFF
    with pytest.raises(PairedFillLedgerCanonicalFinalizeError, match="sha256 mismatch"):
        canonicalize_verified_shard_ledger(
            bytes(tampered),
            meta,
            _canonical_request(
                shard_ordinal=0,
                ledger_bytes=ledger,
                metadata_bytes=meta,
                global_ordinals=(0,),
            ),
        )


def test_bounded_processing_does_not_accumulate_all_shards(tmp_path):
    first = _reduce_shard(
        tmp_path,
        [_row(identity=41, identity_norm="41")],
        partition={"terminal": False},
    )
    second = _reduce_shard(
        tmp_path,
        [_row(identity=42, identity_norm="42")],
        partition={"terminal": True, "incoming_carry": first["outgoing_carry"]},
    )
    shards = [first, second]
    state = PairedFillLedgerCanonicalizeCrossShardState()
    peak_rows = 0
    incoming = None
    for ordinal, (global_ord, result) in enumerate(zip((501, 502), shards, strict=True)):
        ledger = bytes(result["ledger_parquet_bytes"])
        meta = _metadata_bytes(result)
        rows, _, state = canonicalize_verified_shard_ledger(
            ledger,
            meta,
            _canonical_request(
                shard_ordinal=ordinal,
                ledger_bytes=ledger,
                metadata_bytes=meta,
                global_ordinals=(global_ord,),
                incoming_carry=incoming,
                incoming_carry_global_ordinal=501 if ordinal == 1 else None,
                terminal=ordinal == 1,
            ),
            cross_shard_state=state,
        )
        incoming = result["outgoing_carry"]
        peak_rows = max(peak_rows, len(rows))
    assert peak_rows <= 2
    assert state.shards_processed == 2


def test_output_identity_differs_from_intermediate_and_rerun_stable(tmp_path):
    result = _reduce_shard(tmp_path, [_row(identity=3, identity_norm="3")], partition={"terminal": True})
    ledger = bytes(result["ledger_parquet_bytes"])
    meta = _metadata_bytes(result)
    request = _canonical_request(
        shard_ordinal=0,
        ledger_bytes=ledger,
        metadata_bytes=meta,
        global_ordinals=(7,),
    )
    first = execute_paired_fill_ledger_canonical_finalize(ledger, meta, request)
    second = execute_paired_fill_ledger_canonical_finalize(ledger, meta, request)
    assert first["canonical_ledger_identity"] != first["source_ledger_identity"]
    assert first["canonical_ledger_identity"] == second["canonical_ledger_identity"]


def test_publish_and_readback_identity(tmp_path):
    class _Ref:
        def __init__(self, object_id, payload, media_type):
            self.object_id = object_id
            self.uri = f"pbe://private/{object_id}"
            self.sha256 = "sha256:" + sha256(payload).hexdigest()
            self.size_bytes = len(payload)
            self.media_type = media_type

        def model_dump(self, mode="json"):
            return {
                "object_id": self.object_id,
                "uri": self.uri,
                "sha256": self.sha256,
                "size_bytes": self.size_bytes,
                "media_type": self.media_type,
            }

    class Plane:
        def __init__(self):
            self.store = {}
            self.counter = 0

        def write(self, data, media_type):
            object_id = f"obj-{self.counter}"
            self.counter += 1
            self.store[object_id] = data
            return _Ref(object_id, data, media_type)

    result = _reduce_shard(tmp_path, [_row(identity=4, identity_norm="4")], partition={"terminal": True})
    ledger = bytes(result["ledger_parquet_bytes"])
    meta = _metadata_bytes(result)
    payload = execute_paired_fill_ledger_canonical_finalize(
        ledger,
        meta,
        _canonical_request(
            shard_ordinal=0,
            ledger_bytes=ledger,
            metadata_bytes=meta,
            global_ordinals=(9,),
        ),
    )
    plane = Plane()

    def _matches(data, ref):
        return ref.sha256 == "sha256:" + sha256(data).hexdigest()

    metadata_bytes, refs = publish_paired_fill_ledger_canonical_finalize_artifacts(
        plane,
        payload,
        artifact_ref_matches_bytes=_matches,
        shard_stage_failure=RuntimeError,
    )
    metadata = json.loads(metadata_bytes.decode())
    assert metadata["schema_version"] == CANONICAL_METADATA_SCHEMA_VERSION
    assert metadata["canonical_ledger_ref"]["object_id"] == refs[1].object_id
    assert metadata["canonical_ledger_identity"] == payload["canonical_ledger_identity"]
    read_back = plane.store[refs[1].object_id]
    from portable_batch_execution.packs.replay_reduction.models import (
        VerifiedReplayPayloadDigest,
    )

    assert_verified_payload_bytes(
        read_back,
        VerifiedReplayPayloadDigest(
            schema_version="pbe.replay.verified-payload-digest.v1",
            sha256=refs[1].sha256.removeprefix("sha256:"),
            size_bytes=refs[1].size_bytes,
            media_type=refs[1].media_type,
        ),
        label="readback",
    )
    rebuilt = build_canonical_finalize_metadata(payload, canonical_ledger_ref=refs[0])
    assert rebuilt["canonical_ledger_identity"] == metadata["canonical_ledger_identity"]
