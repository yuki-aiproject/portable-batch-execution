import json
from datetime import UTC, datetime

from portable_batch_execution.contracts import (
    RunManifest,
    ShardAttemptRecord,
)
from portable_batch_execution.data_plane.service import PrivateDataPlaneService


def _service(tmp_path, token: str = "plane-token") -> PrivateDataPlaneService:
    return PrivateDataPlaneService(tmp_path, token)


def test_auth_fails_closed_without_leaking_token(tmp_path):
    service = _service(tmp_path, "SENTINEL_TOKEN")
    status, _, body = service.dispatch(
        "GET",
        "/v1/runs/run/waves/wave",
        authorization="Bearer wrong",
    )
    assert status == 401
    assert b"SENTINEL" not in (body or b"")


def test_resolve_wave_requires_auth_and_exact_pair(tmp_path):
    from portable_batch_execution.controller.a1_controller import A1Controller

    controller = A1Controller(tmp_path)
    controller.prepare_private_synthetic_run(
        logical_run_id="opaque-run",
        wave_id="opaque-wave",
    )
    service = _service(tmp_path)
    status, _, body = service.dispatch(
        "GET",
        "/v1/runs/opaque-run/waves/opaque-wave",
        authorization="Bearer plane-token",
    )
    assert status == 200
    payload = json.loads(body or b"{}")
    assert payload["wave"]["wave_id"] == "opaque-wave"
    missing, _, _ = service.dispatch(
        "GET",
        "/v1/runs/opaque-run/waves/missing",
        authorization="Bearer plane-token",
    )
    assert missing == 404


def test_artifact_write_read_and_head(tmp_path):
    service = _service(tmp_path)
    status, _, body = service.dispatch(
        "POST",
        "/v1/artifacts",
        authorization="Bearer plane-token",
        headers={"content-type": "application/json"},
        body=b"[{\"value\":1}]",
    )
    assert status == 200
    ref = json.loads(body or b"{}")
    object_id = ref["object_id"]
    get_status, get_headers, payload = service.dispatch(
        "GET",
        f"/v1/artifacts/{object_id}/content",
        authorization="Bearer plane-token",
    )
    assert get_status == 200
    assert get_headers["Content-Type"] == "application/octet-stream"
    assert get_headers["Content-Length"] == "13"
    assert b"".join(payload.chunks) == b"[{\"value\":1}]"
    head_status, head_headers, _ = service.dispatch(
        "HEAD",
        f"/v1/artifacts/{object_id}/content",
        authorization="Bearer plane-token",
    )
    assert head_status == 200
    assert head_headers["Content-Length"] == "13"


def test_large_artifact_get_streams_without_whole_object_read(tmp_path, monkeypatch):
    service = _service(tmp_path)
    data = bytes(range(256)) * 4096
    status, _, body = service.dispatch(
        "POST",
        "/v1/artifacts",
        authorization="Bearer plane-token",
        body=data,
    )
    assert status == 200
    object_id = json.loads(body or b"{}")["object_id"]

    def reject_read(ref):
        raise AssertionError("whole-object read path must not be used for artifact GET")

    monkeypatch.setattr(service.store, "read", reject_read)
    get_status, get_headers, payload = service.dispatch(
        "GET",
        f"/v1/artifacts/{object_id}/content",
        authorization="Bearer plane-token",
    )
    assert get_status == 200
    assert get_headers["Content-Length"] == str(len(data))
    assert get_headers["Content-Type"] == "application/octet-stream"
    assert payload.size_bytes == len(data)
    chunks = list(payload.chunks)
    assert len(chunks) > 1
    assert b"".join(chunks) == data


def test_attempts_are_immutable(tmp_path):
    service = _service(tmp_path)
    now = datetime.now(UTC)
    record = ShardAttemptRecord(
        logical_run_id="opaque-run",
        shard_id="shard",
        attempt_id="attempt-1",
        status="failed",
        input_digest="d",
        execution_fingerprint="f",
        started_at=now,
        finished_at=now,
        failure="shard_execution_failed",
    )
    first, _, _ = service.dispatch(
        "POST",
        "/v1/runs/opaque-run/attempts",
        authorization="Bearer plane-token",
        body=record.model_dump_json().encode("utf-8"),
    )
    assert first == 204
    conflict, _, _ = service.dispatch(
        "POST",
        "/v1/runs/opaque-run/attempts",
        authorization="Bearer plane-token",
        body=record.model_copy(update={"failure": "other"}).model_dump_json().encode(
            "utf-8"
        ),
    )
    assert conflict == 409


def test_manifest_put_is_controller_only(tmp_path):
    from portable_batch_execution.controller.a1_controller import A1Controller

    controller = A1Controller(tmp_path)
    controller.prepare_private_synthetic_run(
        logical_run_id="opaque-run",
        wave_id="opaque-wave",
    )
    service = _service(tmp_path)
    get_status, _, body = service.dispatch(
        "GET",
        "/v1/runs/opaque-run/manifest",
        authorization="Bearer plane-token",
    )
    assert get_status == 200
    manifest = RunManifest.model_validate_json(body or b"{}")
    assert manifest.revision == 0

    put_status, _, put_body = service.dispatch(
        "PUT",
        "/v1/runs/opaque-run/manifest",
        authorization="Bearer plane-token",
        headers={"if-match": "0"},
        body=manifest.model_copy(update={"revision": 1, "status": "running"}).model_dump_json().encode(
            "utf-8"
        ),
    )
    assert put_status == 403
    assert json.loads(put_body or b"{}") == {"error": "controller-only"}

    again_status, _, again_body = service.dispatch(
        "GET",
        "/v1/runs/opaque-run/manifest",
        authorization="Bearer plane-token",
    )
    assert again_status == 200
    stored = RunManifest.model_validate_json(again_body or b"{}")
    assert stored.revision == 0
