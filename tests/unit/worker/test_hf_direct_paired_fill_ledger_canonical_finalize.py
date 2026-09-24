from __future__ import annotations

import json
import os
import unittest
from hashlib import sha256
from unittest import mock

import polars as pl

from portable_batch_execution.contracts import (
    ArtifactRef,
    ExecutionPolicy,
    JobSpec,
    Provenance,
    ShardCorrectnessSpec,
    ShardSpec,
    WaveSpec,
)
from portable_batch_execution.packs.replay_reduction.paired_fill_reduce import (
    LEDGER_PARQUET_MEDIA_TYPE,
    build_paired_fill_metadata,
    execute_paired_fill_reduce,
)
from portable_batch_execution.transport.hf_bucket import (
    HfBucketTransport,
    HfBucketTransportError,
    InMemoryHfBucketStorage,
    artifact_ref_from_hf_object,
)
from portable_batch_execution.worker.hf_direct import (
    PAIRED_FILL_LEDGER_CANONICAL_FINALIZE_OPERATION,
)
from portable_batch_execution.worker.hf_direct_wave import (
    HfDirectWaveError,
    SerializedWaveDescriptor,
    execute_hf_direct_wave,
    validate_wave_descriptor_payload,
    wave_result_manifest_object_path,
)


def _hf_direct_env(public_revision: str) -> dict[str, str]:
    return {
        "PBE_MODE": "hf-direct",
        "PBE_JPX_HF_DIRECT": "1",
        "HF_TOKEN": "hf_test",
        "PBE_EXPECTED_PUBLIC_REVISION": public_revision,
    }


def _reduce_fixture_rows():
    return [
        {
            "identity": 3,
            "identity_norm": "3",
            "pair_role": "A",
            "core_price": 1.0,
            "core_size": 2.0,
            "start_pos": 0.0,
            "signed_qty": 2.0,
        },
        {
            "identity": 3,
            "identity_norm": "3",
            "pair_role": "B",
            "core_price": 1.0,
            "core_size": 2.0,
            "start_pos": 0.0,
            "signed_qty": -2.0,
        },
    ]


def _reduce_request():
    return {
        "schema_version": "pbe.replay.paired-fill-reduce.v1",
        "request_id": "hf-reduce",
        "identity_mapping": {
            "identity_source_column": "identity",
            "identity_normalized_column": "identity_norm",
        },
        "pair_mapping": {
            "pair_role_column": "pair_role",
            "aggressor_role_value": "A",
            "passive_role_value": "B",
            "measurement_core_fields": ["core_price", "core_size"],
            "start_position_column": "start_pos",
            "signed_execution_column": "signed_qty",
        },
        "partition": {"terminal": True},
        "max_output_rows": 10,
        "max_exception_rows": 10,
    }


def _digest_binding(data: bytes, media_type: str) -> dict:
    return {
        "schema_version": "pbe.replay.verified-payload-digest.v1",
        "sha256": sha256(data).hexdigest(),
        "size_bytes": len(data),
        "media_type": media_type,
    }


def _store_hf_object(storage: InMemoryHfBucketStorage, object_path: str, data: bytes, media_type: str):
    storage.objects[object_path] = data
    return artifact_ref_from_hf_object(
        object_path=object_path,
        sha256_hex=sha256(data).hexdigest(),
        size_bytes=len(data),
        media_type=media_type,
    )


def _build_canonical_finalize_descriptor(storage: InMemoryHfBucketStorage, public_revision: str):
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as temporary:
        parquet_path = Path(temporary) / "reduce-input.parquet"
        pl.DataFrame(_reduce_fixture_rows()).write_parquet(parquet_path)
        reduce_result = execute_paired_fill_reduce(
            [parquet_path],
            _reduce_request(),
        )
    ledger_bytes = bytes(reduce_result["ledger_parquet_bytes"])
    reducer_metadata_bytes = json.dumps(
        build_paired_fill_metadata(reduce_result, ledger_parquet_ref=None),
        sort_keys=True,
    ).encode("utf-8")

    manifest_digest = "paired-ledger-canonical-manifest"
    bucket_prefix = "replay/public/paired-ledger-canonical/"
    logical_run_id = "run-canonical"
    wave_id = "wave-canonical-0001"
    base = f"{bucket_prefix.rstrip('/')}/inputs/{public_revision}/{manifest_digest}/{wave_id}"

    ledger_ref = _store_hf_object(
        storage,
        f"{base}/intermediate-ledger.parquet",
        ledger_bytes,
        LEDGER_PARQUET_MEDIA_TYPE,
    )
    reducer_metadata_ref = _store_hf_object(
        storage,
        f"{base}/reducer-metadata.json",
        reducer_metadata_bytes,
        "application/json",
    )

    canonical_request = {
        "schema_version": "pbe.replay.paired-fill-ledger-canonical-finalize.v1",
        "request_id": "hf-canonical",
        "partition_id": "partition-0",
        "terminal": True,
        "shard": {
            "shard_ordinal": 0,
            "ledger": _digest_binding(ledger_bytes, LEDGER_PARQUET_MEDIA_TYPE),
            "reducer_metadata": _digest_binding(reducer_metadata_bytes, "application/json"),
            "local_index_to_global_ordinal": [42],
        },
    }
    request_bytes = json.dumps(canonical_request, sort_keys=True).encode("utf-8")
    request_ref = _store_hf_object(
        storage,
        f"{base}/canonical-request.json",
        request_bytes,
        "application/json",
    )

    shard_id = f"{wave_id}-shard-0000"
    shards = [
        ShardSpec(
            logical_run_id=logical_run_id,
            shard_id=shard_id,
            ordinal=0,
            correctness=ShardCorrectnessSpec(mode="independent"),
            input_refs=(ledger_ref, reducer_metadata_ref, request_ref),
            input_digest=sha256(ledger_bytes).hexdigest(),
            execution_fingerprint=sha256(
                f"{public_revision}|{shard_id}|canonical".encode()
            ).hexdigest(),
        )
    ]
    job = JobSpec(
        job_id="job-canonical",
        logical_run_id=logical_run_id,
        pack="replay-batch",
        operation=PAIRED_FILL_LEDGER_CANONICAL_FINALIZE_OPERATION,
        input_manifest_ref=request_ref,
        sharding=ShardCorrectnessSpec(mode="independent"),
        execution=ExecutionPolicy(
            max_parallel=1,
            max_attempts_per_shard=4,
            resume_enabled=True,
        ),
        security_profile="offline",
        provenance=Provenance(
            producer="test",
            revision=public_revision,
            created_at="1970-01-01T00:00:00+00:00",
        ),
        operation_params={
            "schema_version": "pbe.replay.paired-fill-ledger-canonical-finalize-job.v1",
            "transport_profile": "hf_direct",
            "bucket_prefix": bucket_prefix,
        },
    )
    wave = WaveSpec(
        logical_run_id=logical_run_id,
        wave_id=wave_id,
        ordinal=0,
        shard_ids=(shard_id,),
        max_parallel=1,
    )
    serialized = SerializedWaveDescriptor.build(
        logical_run_id=logical_run_id,
        wave_id=wave_id,
        manifest_digest=manifest_digest,
        public_revision=public_revision,
        bucket_prefix=bucket_prefix,
        job=job,
        wave=wave,
        shards=tuple(shards),
    )
    storage.objects[serialized.hf_ref["object_path"]] = serialized.bytes
    manifest_path = wave_result_manifest_object_path(
        bucket_prefix=bucket_prefix,
        public_revision=public_revision,
        manifest_digest=manifest_digest,
        wave_id=wave_id,
    )
    return serialized.hf_ref, manifest_path, public_revision, ledger_ref, request_ref


class TestHfDirectPairedFillLedgerCanonicalFinalize(unittest.TestCase):
    def test_descriptor_accepts_operation(self):
        storage = InMemoryHfBucketStorage()
        public_revision = "f" * 40
        ref, _manifest_path, _rev, _ledger, _request = _build_canonical_finalize_descriptor(
            storage, public_revision
        )
        descriptor = json.loads(storage.objects[ref["object_path"]].decode())
        validated = validate_wave_descriptor_payload(descriptor)
        self.assertEqual(
            validated["job"].operation,
            PAIRED_FILL_LEDGER_CANONICAL_FINALIZE_OPERATION,
        )

    def test_hf_direct_wave_succeeds_with_verified_refs(self):
        storage = InMemoryHfBucketStorage()
        public_revision = "a" * 40
        ref, manifest_path, revision, _ledger, _request = _build_canonical_finalize_descriptor(
            storage, public_revision
        )
        transport = HfBucketTransport(storage)
        env = _hf_direct_env(revision)
        with mock.patch.dict(os.environ, env, clear=False):
            summary = execute_hf_direct_wave(
                wave_descriptor_ref=ref,
                expected_manifest_path=manifest_path,
                transport=transport,
            )
        row = summary["manifest"]["shards"][0]
        self.assertEqual(row["status"], "succeeded")
        self.assertIn("output_hf_refs", row)
        self.assertGreaterEqual(len(row["output_hf_refs"]), 1)
        for output_ref in row["output_hf_refs"]:
            self.assertTrue(storage.remote_exists(output_ref["object_path"]))
        metadata_path = row["output_hf_ref"]["object_path"]
        metadata = json.loads(storage.objects[metadata_path].decode())
        self.assertNotEqual(
            metadata.get("source_ledger_identity"),
            metadata.get("canonical_ledger_identity"),
        )

    def test_idempotent_rerun_reuses_verified_outputs(self):
        storage = InMemoryHfBucketStorage()
        public_revision = "b" * 40
        ref, manifest_path, revision, _ledger, _request = _build_canonical_finalize_descriptor(
            storage, public_revision
        )
        transport = HfBucketTransport(storage)
        env = _hf_direct_env(revision)
        with mock.patch.dict(os.environ, env, clear=False):
            first = execute_hf_direct_wave(
                wave_descriptor_ref=ref,
                expected_manifest_path=manifest_path,
                transport=transport,
            )
            second = execute_hf_direct_wave(
                wave_descriptor_ref=ref,
                expected_manifest_path=manifest_path,
                transport=transport,
            )
        self.assertEqual(
            first["manifest"]["shards"][0]["output_hf_ref"],
            second["manifest"]["shards"][0]["output_hf_ref"],
        )

    def test_sha_mismatch_fails_closed(self):
        storage = InMemoryHfBucketStorage()
        public_revision = "c" * 40
        ref, manifest_path, revision, ledger_ref, _request = _build_canonical_finalize_descriptor(
            storage, public_revision
        )
        transport = HfBucketTransport(storage)
        env = _hf_direct_env(revision)
        from portable_batch_execution.transport.hf_bucket import parse_hf_object_uri

        ledger_path = parse_hf_object_uri(ledger_ref.uri)
        tampered = bytearray(storage.objects[ledger_path])
        tampered[-1] ^= 0xFF
        storage.objects[ledger_path] = bytes(tampered)
        with (
            mock.patch.dict(os.environ, env, clear=False),
            self.assertRaises((HfDirectWaveError, HfBucketTransportError)),
        ):
            execute_hf_direct_wave(
                wave_descriptor_ref=ref,
                expected_manifest_path=manifest_path,
                transport=transport,
            )

    def test_append_only_collision_mismatch_fails(self):
        storage = InMemoryHfBucketStorage()
        public_revision = "d" * 40
        ref, manifest_path, revision, _ledger, _request = _build_canonical_finalize_descriptor(
            storage, public_revision
        )
        transport = HfBucketTransport(storage)
        env = _hf_direct_env(revision)
        with mock.patch.dict(os.environ, env, clear=False):
            execute_hf_direct_wave(
                wave_descriptor_ref=ref,
                expected_manifest_path=manifest_path,
                transport=transport,
            )
        storage.objects[manifest_path] = b"{}"
        with (
            mock.patch.dict(os.environ, env, clear=False),
            self.assertRaises(HfBucketTransportError),
        ):
            execute_hf_direct_wave(
                wave_descriptor_ref=ref,
                expected_manifest_path=manifest_path,
                transport=transport,
            )


if __name__ == "__main__":
    unittest.main()
