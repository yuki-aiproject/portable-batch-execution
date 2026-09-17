import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import pytest

from portable_batch_execution.contracts import ArtifactRef, ShardAttemptRecord
from portable_batch_execution.data_plane.base import ArtifactContentStream
from portable_batch_execution.kernel import exhausted_shards
from portable_batch_execution.worker.execute_wave import (
    PrivateWaveExecutionError,
    _private_attempt_id,
    execute_private_wave,
)

_SENTINEL = "SENTINEL_PRIVATE_LEAK_DO_NOT_LOG"


def _plane(
    *,
    rows=None,
    shards=None,
    max_attempts=4,
    appended=None,
    read_raises=None,
    write_raises=False,
    operation="tabular.rolling",
    pack="tabular-batch",
    ml_payload=None,
    binary_payload=None,
    tabular_payload=None,
    operation_params=None,
    input_ref_overrides=None,
    write_ref_overrides=None,
):
    if binary_payload is not None:
        payload = binary_payload
    elif pack == "media-batch":
        payload = binary_payload if binary_payload is not None else b"\x00\x01private-audio"
    elif pack == "ml-batch":
        document = ml_payload if ml_payload is not None else {
            "schema_version": "pbe.ml.char-wb-tfidf-logistic-score.v1",
            "model": {
                "features": [{"feature": "ab", "idf": 1.0, "coefficient": 0.0}],
                "intercept": 0.0,
            },
            "rows": [{"row_id": "row-1", "text": "ab"}],
        }
        payload = json.dumps(document).encode()
    else:
        if tabular_payload is not None:
            document = tabular_payload
        else:
            rows = rows if rows is not None else [{"group": "a", "value": 1}]
            document = rows
        payload = json.dumps(document).encode()
    input_ref_fields = {
        "object_id": "input",
        "uri": "pbe://private/input",
        "sha256": "sha256:" + sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    input_ref_fields.update(input_ref_overrides or {})
    input_ref = ArtifactRef(**input_ref_fields)
    now = datetime.now(UTC).isoformat()
    shard_specs = shards or [
        {
            "logical_run_id": "opaque-run",
            "shard_id": "opaque-shard",
            "ordinal": 0,
            "correctness": {},
            "input_refs": [input_ref.model_dump(mode="json")],
            "input_digest": "current",
            "execution_fingerprint": "fixed",
        }
    ]
    job = {
        "job_id": "job",
        "logical_run_id": "opaque-run",
        "pack": pack,
        "operation": operation,
        "input_manifest_ref": input_ref.model_dump(mode="json"),
        "sharding": {},
        "execution": {"max_parallel": 1, "max_attempts_per_shard": max_attempts},
        "security_profile": "offline",
        "provenance": {"producer": "test", "revision": "1", "created_at": now},
        "operation_params": (
            operation_params
            if operation_params is not None
            else (
                {}
                if pack in {"ml-batch", "media-batch"}
                else {
                    "column": "value",
                    "window_size": 2,
                    "output_column": "rolling",
                }
            )
        ),
    }
    wave = {
        "logical_run_id": "opaque-run",
        "wave_id": "opaque-wave",
        "ordinal": 0,
        "shard_ids": [item["shard_id"] for item in shard_specs],
        "max_parallel": 1,
    }

    class Plane:
        def __init__(self):
            self.appended = list(appended or [])

        def resolve_wave(self, run_id, wave_id):
            return {"job": job, "wave": wave, "shards": shard_specs}

        def read_attempts(self, run_id):
            return tuple(self.appended)

        def read(self, ref):
            if read_raises:
                raise read_raises
            return payload

        def open_content(self, ref):
            if read_raises:
                raise read_raises

            def chunks():
                for offset in range(0, len(payload), 17):
                    yield payload[offset : offset + 17]

            return ArtifactContentStream(size_bytes=len(payload), chunks=chunks())

        def write(self, data, media_type):
            self.last_written = data
            if write_raises:
                raise RuntimeError(_SENTINEL)
            output_ref_fields = {
                "object_id": "output",
                "uri": "pbe://private/output",
                "sha256": "sha256:" + sha256(data).hexdigest(),
                "size_bytes": len(data),
            }
            output_ref_fields.update(write_ref_overrides or {})
            return ArtifactRef(**output_ref_fields)

        def append_attempt(self, record):
            self.appended.append(record)

    return Plane()


def test_failed_execution_appends_one_failed_record():
    plane = _plane()
    with patch(
        "portable_batch_execution.worker.execute_wave.TabularPack.execute",
        side_effect=RuntimeError(_SENTINEL),
    ), pytest.raises(PrivateWaveExecutionError) as error:
        execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert len(plane.appended) == 1
    record = plane.appended[0]
    assert record.status == "failed"
    assert record.failure == "shard_pack_execution_failed"
    assert record.output_refs == ()
    assert record.output_digest is None
    assert _SENTINEL not in json.dumps(record.model_dump(mode="json"))
    assert _SENTINEL not in str(error.value)
    assert error.value.attempts == (record,)


def test_four_matching_failures_exhaust_attempt_budget():
    plane = _plane()
    with patch(
        "portable_batch_execution.worker.execute_wave.TabularPack.execute",
        side_effect=RuntimeError(_SENTINEL),
    ):
        for _ in range(4):
            with pytest.raises(PrivateWaveExecutionError):
                execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert len(plane.appended) == 4
    assert execute_private_wave("opaque-run", "opaque-wave", plane=plane) == ()
    shard = plane.resolve_wave("opaque-run", "opaque-wave")["shards"][0]
    from portable_batch_execution.contracts import ExecutionPolicy, ShardSpec

    policy = ExecutionPolicy(max_parallel=1, max_attempts_per_shard=4)
    assert (
        exhausted_shards(
            (ShardSpec.model_validate(shard),),
            tuple(plane.appended),
            policy,
        )
        != ()
    )


def test_stale_attempts_do_not_consume_ordinal_or_budget():
    now = datetime.now(UTC)
    stale = [
        ShardAttemptRecord(
            logical_run_id="opaque-run",
            shard_id="opaque-shard",
            attempt_id=f"stale-{index}",
            status="failed",
            input_digest="stale",
            execution_fingerprint="stale",
            started_at=now,
            finished_at=now,
            failure="shard_execution_failed",
        )
        for index in range(9)
    ]
    plane = _plane(appended=stale)
    with patch(
        "portable_batch_execution.worker.execute_wave.TabularPack.execute",
        side_effect=RuntimeError(_SENTINEL),
    ), pytest.raises(PrivateWaveExecutionError):
        execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert plane.appended[-1].attempt_id == _private_attempt_id(
        "opaque-wave",
        "opaque-shard",
        input_digest="current",
        execution_fingerprint="fixed",
        current_attempt_count=0,
    )
    assert len([item for item in plane.appended if item.input_digest == "current"]) == 1


def test_new_generation_reuses_ordinal_without_colliding_with_stale_generation():
    now = datetime.now(UTC)
    stale_attempt_id = _private_attempt_id(
        "opaque-wave",
        "opaque-shard",
        input_digest="stale",
        execution_fingerprint="stale",
        current_attempt_count=0,
    )
    stale = ShardAttemptRecord(
        logical_run_id="opaque-run",
        shard_id="opaque-shard",
        attempt_id=stale_attempt_id,
        status="failed",
        input_digest="stale",
        execution_fingerprint="stale",
        started_at=now,
        finished_at=now,
        failure="shard_execution_failed",
    )
    plane = _plane(appended=[stale])
    with patch(
        "portable_batch_execution.worker.execute_wave.TabularPack.execute",
        side_effect=RuntimeError(_SENTINEL),
    ), pytest.raises(PrivateWaveExecutionError):
        execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    current_attempt_id = _private_attempt_id(
        "opaque-wave",
        "opaque-shard",
        input_digest="current",
        execution_fingerprint="fixed",
        current_attempt_count=0,
    )
    assert current_attempt_id.endswith("-1")
    assert current_attempt_id != stale_attempt_id
    assert plane.appended[-1].attempt_id == current_attempt_id
    assert len([item for item in plane.appended if item.input_digest == "current"]) == 1


def test_successful_shards_continue_when_another_shard_fails():
    rows = [{"group": "a", "value": 1}, {"group": "a", "value": 2}]
    input_ref = ArtifactRef(
        object_id="input",
        uri="pbe://private/input",
        sha256="sha256:" + sha256(json.dumps(rows).encode()).hexdigest(),
    )
    input_payload = input_ref.model_dump(mode="json")
    plane = _plane(
        rows=rows,
        shards=[
            {
                "logical_run_id": "opaque-run",
                "shard_id": "fail-shard",
                "ordinal": 0,
                "correctness": {},
                "input_refs": [input_payload],
                "input_digest": "current",
                "execution_fingerprint": "fixed",
            },
            {
                "logical_run_id": "opaque-run",
                "shard_id": "ok-shard",
                "ordinal": 1,
                "correctness": {},
                "input_refs": [input_payload],
                "input_digest": "current",
                "execution_fingerprint": "fixed",
            },
        ],
    )

    from portable_batch_execution.packs import TabularPack

    original_execute = TabularPack.execute

    def execute_side_effect(pack, job, shard, params, inputs):
        if shard.shard_id == "fail-shard":
            raise RuntimeError(_SENTINEL)
        return original_execute(pack, job, shard, params, inputs)

    with (
        patch.object(TabularPack, "execute", execute_side_effect),
        pytest.raises(PrivateWaveExecutionError) as error,
    ):
        execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert len(error.value.attempts) == 2
    by_shard = {item.shard_id: item for item in plane.appended[-2:]}
    assert by_shard["fail-shard"].status == "failed"
    assert by_shard["ok-shard"].status == "succeeded"
    assert _SENTINEL not in str(error.value)


def test_rejects_format_migration_on_private_runner():
    plane = _plane(operation="tabular.format_migration")
    with pytest.raises(ValueError, match="typed multi-input contract"):
        execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert plane.appended == []


def test_tabular_join_success_records_row_counts():
    left = [{"id": "a", "v": 1}, {"id": "b", "v": 2}]
    right = [{"id": "a", "label": "A"}, {"id": "c", "label": "C"}]
    plane = _plane(
        operation="tabular.join",
        tabular_payload={"left": left, "right": right},
        operation_params={"on": ["id"], "how": "left"},
    )
    attempts = execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert len(attempts) == 1
    record = attempts[0]
    assert record.status == "succeeded"
    assert record.counts == {"input_rows": 4, "output_rows": 2}
    output = json.loads(plane.last_written.decode())
    assert {row["id"] for row in output} == {"a", "b"}
    assert next(row for row in output if row["id"] == "a")["label"] == "A"


def test_tabular_pit_join_success_includes_exact_boundary_match():
    left = [
        {"id": "a", "at": 10, "v": 1},
        {"id": "a", "at": 20, "v": 2},
        {"id": "b", "at": 10, "v": 3},
    ]
    right = [
        {"id": "a", "seen": 10, "label": "exact"},
        {"id": "a", "seen": 15, "label": "past"},
        {"id": "b", "seen": 9, "label": "early"},
    ]
    plane = _plane(
        operation="tabular.pit_join",
        tabular_payload={"left": left, "right": right},
        operation_params={
            "on": ["id"],
            "left_time": "at",
            "right_time": "seen",
        },
    )
    attempts = execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert attempts[0].status == "succeeded"
    assert attempts[0].counts == {"input_rows": 6, "output_rows": 3}
    output = json.loads(plane.last_written.decode())
    exact = next(row for row in output if row["id"] == "a" and row["at"] == 10)
    assert exact["label"] == "exact"


@pytest.mark.parametrize(
    "tabular_payload",
    (
        [{"id": "a"}],
        {"left": [{"id": "a"}], "right": "bad"},
        {"left": [{"id": "a"}], "right": [{"id": "b"}], "extra": True},
        {"left": "bad", "right": [{"id": "b"}]},
    ),
)
def test_tabular_two_table_invalid_envelope_records_failed_attempt(
    tabular_payload,
):
    plane = _plane(
        operation="tabular.join",
        tabular_payload=tabular_payload,
        operation_params={"on": ["id"], "how": "left"},
    )
    with pytest.raises(PrivateWaveExecutionError):
        execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    record = plane.appended[0]
    assert record.status == "failed"
    assert record.failure == "input_artifact_invalid"


def test_tabular_join_invalid_envelope_does_not_consume_success_skip():
    left = [{"id": "a", "v": 1}]
    right = [{"id": "a", "label": "A"}]
    valid_payload = {"left": left, "right": right}
    plane = _plane(
        operation="tabular.join",
        tabular_payload=valid_payload,
        operation_params={"on": ["id"], "how": "left"},
    )
    attempts = execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert attempts[0].status == "succeeded"
    assert execute_private_wave("opaque-run", "opaque-wave", plane=plane) == ()


def test_input_digest_mismatch_records_sanitized_failure():
    plane = _plane(
        input_ref_overrides={
            "sha256": "sha256:" + ("0" * 64),
        }
    )
    with pytest.raises(PrivateWaveExecutionError) as error:
        execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    record = plane.appended[0]
    assert record.status == "failed"
    assert record.failure == "input_artifact_mismatch"
    assert _SENTINEL not in json.dumps(record.model_dump(mode="json"))
    assert _SENTINEL not in str(error.value)


def test_input_size_mismatch_records_sanitized_failure():
    plane = _plane(input_ref_overrides={"size_bytes": 0})
    with pytest.raises(PrivateWaveExecutionError):
        execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert plane.appended[0].failure == "input_artifact_mismatch"


def test_ml_char_wb_success_records_row_counts():
    plane = _plane(
        pack="ml-batch",
        operation="ml.char_wb_tfidf_logistic_score",
        ml_payload={
            "schema_version": "pbe.ml.char-wb-tfidf-logistic-score.v1",
            "model": {
                "features": [{"feature": "ab", "idf": 1.0, "coefficient": 0.0}],
                "intercept": 0.0,
            },
            "rows": [
                {"row_id": "a", "text": "ab"},
                {"row_id": "b", "text": "xy"},
            ],
        },
    )
    attempts = execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert len(attempts) == 1
    record = attempts[0]
    assert record.status == "succeeded"
    assert record.counts == {"input_rows": 2, "output_rows": 2}
    output = json.loads(plane.last_written.decode())
    assert (
        output["schema_version"]
        == "pbe.ml.char-wb-tfidf-logistic-score-result.v1"
    )
    assert [item["row_id"] for item in output["rows"]] == ["a", "b"]


def test_ml_cosine_similarity_matrix_success_records_counts():
    plane = _plane(
        pack="ml-batch",
        operation="ml.cosine_similarity_matrix",
        ml_payload={
            "left": [
                {"row_id": "a", "vector": [1.0, 0.0]},
                {"row_id": "b", "vector": [0.0, 1.0]},
            ],
            "right": [
                {"row_id": "x", "vector": [1.0, 1.0]},
                {"row_id": "y", "vector": [1.0, -1.0]},
            ],
        },
    )
    attempts = execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert len(attempts) == 1
    record = attempts[0]
    assert record.status == "succeeded"
    assert record.counts == {"input_rows": 4, "output_rows": 4}
    output = json.loads(plane.last_written.decode())
    assert output["left_ids"] == ["a", "b"]
    assert output["right_ids"] == ["x", "y"]
    assert len(output["scores"]) == 2
    assert len(output["scores"][0]) == 2


def test_ml_cosine_similarity_matrix_serializes_null_for_zero_norm_pairs():
    plane = _plane(
        pack="ml-batch",
        operation="ml.cosine_similarity_matrix",
        ml_payload={
            "left": [
                {"row_id": "a", "vector": [1.0, 0.0]},
                {"row_id": "z", "vector": [0.0, 0.0]},
            ],
            "right": [
                {"row_id": "x", "vector": [0.0, 1.0]},
                {"row_id": "y", "vector": [0.0, 0.0]},
            ],
        },
    )
    attempts = execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert attempts[0].status == "succeeded"
    output = json.loads(plane.last_written.decode())
    assert output["scores"][0] == [0.0, None]
    assert output["scores"][1] == [None, None]


def test_ml_cosine_malformed_contract_records_sanitized_failure():
    plane = _plane(
        pack="ml-batch",
        operation="ml.cosine_similarity_matrix",
        ml_payload={
            "left": [
                {"row_id": "a", "vector": [1.0]},
                {"row_id": "a", "vector": [_SENTINEL]},
            ],
            "right": [{"row_id": "x", "vector": [1.0]}],
        },
    )
    with pytest.raises(PrivateWaveExecutionError) as error:
        execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    record = plane.appended[0]
    assert record.status == "failed"
    assert record.failure == "shard_pack_execution_failed"
    assert _SENTINEL not in json.dumps(record.model_dump(mode="json"))
    assert _SENTINEL not in str(error.value)


def test_ml_malformed_contract_records_sanitized_failure():
    plane = _plane(
        pack="ml-batch",
        operation="ml.char_wb_tfidf_logistic_score",
        ml_payload={
            "schema_version": "pbe.ml.char-wb-tfidf-logistic-score.v1",
            "model": {
                "features": [{"feature": "ab", "idf": 1.0, "coefficient": 0.0}],
                "intercept": 0.0,
            },
            "rows": [
                {"row_id": "a", "text": "ok"},
                {"row_id": "a", "text": _SENTINEL},
            ],
        },
    )
    with pytest.raises(PrivateWaveExecutionError) as error:
        execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    record = plane.appended[0]
    assert record.status == "failed"
    assert record.failure == "shard_pack_execution_failed"
    assert _SENTINEL not in json.dumps(record.model_dump(mode="json"))
    assert _SENTINEL not in str(error.value)


def test_rejects_unapproved_ml_operation():
    plane = _plane(pack="ml-batch", operation="ml.tfidf")
    with pytest.raises(ValueError, match="not available on the public runner"):
        execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert plane.appended == []


def test_media_asr_normalize_flac_success_writes_flac_bytes():
    flac_bytes = b"fLaC\x00private-output"
    plane = _plane(
        pack="media-batch",
        operation="media.asr_normalize_flac",
        binary_payload=b"private-input-bytes",
    )

    def fake_execute(pack, job, shard, params, context):
        Path(context["output"]).write_bytes(flac_bytes)
        return Path(context["output"])

    with patch(
        "portable_batch_execution.worker.execute_wave.MediaPack.execute",
        fake_execute,
    ):
        attempts = execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert len(attempts) == 1
    record = attempts[0]
    assert record.status == "succeeded"
    assert record.counts == {
        "input_bytes": len(b"private-input-bytes"),
        "output_bytes": len(flac_bytes),
    }
    assert plane.last_written == flac_bytes
    assert _SENTINEL not in json.dumps(record.model_dump(mode="json"))


def test_media_pack_failure_records_sanitized_failure():
    plane = _plane(pack="media-batch", operation="media.asr_normalize_flac")
    with patch(
        "portable_batch_execution.worker.execute_wave.MediaPack.execute",
        side_effect=RuntimeError(_SENTINEL),
    ), pytest.raises(PrivateWaveExecutionError) as error:
        execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    record = plane.appended[0]
    assert record.failure == "shard_pack_execution_failed"
    assert _SENTINEL not in json.dumps(record.model_dump(mode="json"))
    assert _SENTINEL not in str(error.value)


def test_media_successful_shard_is_skipped_on_rerun():
    now = datetime.now(UTC)
    prior = ShardAttemptRecord(
        logical_run_id="opaque-run",
        shard_id="opaque-shard",
        attempt_id="prior-success",
        status="succeeded",
        input_digest="current",
        execution_fingerprint="fixed",
        started_at=now,
        finished_at=now,
        wave_id="opaque-wave",
        output_refs=(
            ArtifactRef(
                object_id="output",
                uri="pbe://private/output",
                sha256="sha256:" + ("a" * 64),
            ),
        ),
        output_digest="a" * 64,
    )
    plane = _plane(
        pack="media-batch",
        operation="media.asr_normalize_flac",
        appended=[prior],
    )
    with patch(
        "portable_batch_execution.worker.execute_wave.MediaPack.execute",
        side_effect=RuntimeError(_SENTINEL),
    ):
        assert execute_private_wave("opaque-run", "opaque-wave", plane=plane) == ()
    assert len(plane.appended) == 1


def test_rejects_non_empty_media_operation_params():
    plane = _plane(pack="media-batch", operation="media.asr_normalize_flac")
    plane.resolve_wave("opaque-run", "opaque-wave")["job"]["operation_params"] = {
        "codec": "mp3"
    }
    with pytest.raises(ValueError, match="closed operation parameters"):
        execute_private_wave("opaque-run", "opaque-wave", plane=plane)


def test_output_reference_mismatch_records_sanitized_failure():
    plane = _plane(
        write_ref_overrides={
            "sha256": "sha256:" + ("f" * 64),
        }
    )
    with pytest.raises(PrivateWaveExecutionError) as error:
        execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    record = plane.appended[0]
    assert record.status == "failed"
    assert record.failure == "output_artifact_mismatch"
    assert record.output_refs == ()
    assert _SENTINEL not in str(error.value)


def test_private_wave_executes_parquet_single_input_tabular_operation():
    import io

    import polars as pl

    buffer = io.BytesIO()
    pl.DataFrame({"group": ["a", "a"], "value": [1, 2]}).write_parquet(buffer)
    payload = buffer.getvalue()
    plane = _plane(
        binary_payload=payload,
        input_ref_overrides={
            "media_type": "application/vnd.apache.parquet",
        },
    )
    attempts = execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert len(attempts) == 1
    assert attempts[0].status == "succeeded"
    assert attempts[0].counts == {"input_rows": 2, "output_rows": 2}


def test_private_wave_parquet_digest_mismatch_fails_before_pack_execution():
    import io

    import polars as pl

    buffer = io.BytesIO()
    pl.DataFrame({"group": ["a"], "value": [1]}).write_parquet(buffer)
    payload = buffer.getvalue()
    plane = _plane(
        binary_payload=payload,
        input_ref_overrides={
            "media_type": "application/vnd.apache.parquet",
            "sha256": "sha256:" + ("0" * 64),
        },
    )
    with patch(
        "portable_batch_execution.worker.execute_wave.TabularPack.execute"
    ) as execute:
        with pytest.raises(PrivateWaveExecutionError):
            execute_private_wave("opaque-run", "opaque-wave", plane=plane)
        execute.assert_not_called()
    assert plane.appended[0].failure == "input_artifact_mismatch"


def test_private_wave_parquet_size_mismatch_fails_before_pack_execution():
    import io

    import polars as pl

    buffer = io.BytesIO()
    pl.DataFrame({"group": ["a"], "value": [1]}).write_parquet(buffer)
    payload = buffer.getvalue()
    plane = _plane(
        binary_payload=payload,
        input_ref_overrides={
            "media_type": "application/vnd.apache.parquet",
            "size_bytes": len(payload) + 1,
        },
    )
    with patch(
        "portable_batch_execution.worker.execute_wave.TabularPack.execute"
    ) as execute:
        with pytest.raises(PrivateWaveExecutionError):
            execute_private_wave("opaque-run", "opaque-wave", plane=plane)
        execute.assert_not_called()
    assert plane.appended[0].failure == "input_artifact_mismatch"


def test_private_wave_json_record_array_input_remains_compatible():
    plane = _plane(rows=[{"group": "a", "value": 1}, {"group": "a", "value": 2}])
    attempts = execute_private_wave("opaque-run", "opaque-wave", plane=plane)
    assert len(attempts) == 1
    assert attempts[0].status == "succeeded"
    assert attempts[0].counts == {"input_rows": 2, "output_rows": 2}

def test_private_wave_parquet_oversize_stream_fails_before_pack_execution():
    import io

    import polars as pl

    buffer = io.BytesIO()
    pl.DataFrame({"group": ["a"], "value": [1]}).write_parquet(buffer)
    payload = buffer.getvalue()
    plane = _plane(
        binary_payload=payload,
        input_ref_overrides={
            "media_type": "application/vnd.apache.parquet",
            "size_bytes": 10,
        },
    )

    def open_content(ref):
        return ArtifactContentStream(size_bytes=10, chunks=iter([payload[:11]]))

    plane.open_content = open_content
    with patch(
        "portable_batch_execution.worker.execute_wave.TabularPack.execute"
    ) as execute:
        with pytest.raises(PrivateWaveExecutionError):
            execute_private_wave("opaque-run", "opaque-wave", plane=plane)
        execute.assert_not_called()
    assert plane.appended[0].failure == "input_artifact_mismatch"

def test_parquet_row_count_uses_scan_not_eager_read_parquet(tmp_path):
    import io

    import polars as pl

    from portable_batch_execution.worker.execute_wave import _parquet_row_count

    buffer = io.BytesIO()
    pl.DataFrame({"group": ["a", "a"], "value": [1, 2]}).write_parquet(buffer)
    path = tmp_path / "input.parquet"
    path.write_bytes(buffer.getvalue())

    with patch.object(
        pl, "read_parquet", side_effect=AssertionError("read_parquet must not be used")
    ), patch.object(pl, "scan_parquet", wraps=pl.scan_parquet) as scan:
        assert _parquet_row_count(path) == 2
        scan.assert_called_once()
