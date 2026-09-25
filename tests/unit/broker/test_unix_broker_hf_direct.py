import json
from hashlib import sha256
from unittest.mock import patch

import httpx

from portable_batch_execution.backends.github_actions import GitHubActionsBackend
from portable_batch_execution.broker.config import BrokerConfig
from portable_batch_execution.broker.hf_direct import (
    BrokerHfDirectExecuteResponse,
    parse_hf_direct_request,
)
from portable_batch_execution.broker.service import UnixBrokerService
from portable_batch_execution.broker.state import BrokerRequestStore
from portable_batch_execution.controller.a1_controller import A1Controller
from portable_batch_execution.transport.hf_bucket import HF_BUCKET_REF_SCHEMA

_PUBLIC_SHA = "ac3a69d2c818526b87f38c848d324221e2dc2775"
_UID = 1000


def _mock_backend(handler):
    return GitHubActionsBackend(
        "owner",
        "repo",
        "execute-wave.yml",
        private_data_plane=True,
        client=httpx.Client(
            base_url="https://api.github.com", transport=httpx.MockTransport(handler)
        ),
    )


def _service(tmp_path, config, handler):
    controller = A1Controller(tmp_path, backend=_mock_backend(handler))
    return UnixBrokerService(
        state_root=tmp_path,
        config=config,
        controller=controller,
        poll_interval_seconds=0.0,
        sleeper=lambda _seconds: None,
    )


_FIXED_SET_OPERATION = "replay.trade_path_scenario_evaluate_fixed_set"
_FIXED_SET_JOB_SCHEMA = "pbe.replay.trade-path-scenario-evaluate-fixed-set-job.v1"
_PUBLIC_REVISION = _PUBLIC_SHA
_REQUEST_ID = sha256(b"hf-direct-broker-test").hexdigest()


def _hf_config(tmp_path):
    path = tmp_path / "broker-config-hf.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "pbe.a1-unix-broker.config.v1",
                "public_sha": _PUBLIC_SHA,
                "max_input_bytes": 1_048_576,
                "allowed_operations_by_uid": {
                    str(_UID): [
                        [
                            "replay-batch",
                            _FIXED_SET_OPERATION,
                        ],
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    return BrokerConfig.load(path)


def _wave_descriptor_hf_ref() -> dict:
    payload = b'{"schema_version":"pbe.hf-direct.wave-descriptor.v1"}'
    digest = sha256(payload).hexdigest()
    return {
        "schema_version": HF_BUCKET_REF_SCHEMA,
        "bucket_id": "yamauchiJP/system-trading-data",
        "object_path": "pair-trading/v1/public-eval/waves/"
        f"{_PUBLIC_REVISION}/digest-1/wave-0001-descriptor-0123456789abcdef.json",
        "sha256": digest,
        "size_bytes": len(payload),
        "media_type": "application/vnd.pbe.hf-direct-wave-descriptor.v1",
    }


def _hf_request(
    *,
    request_id: str = _REQUEST_ID,
    result_manifest_object_path: str = (
        f"pair-trading/v1/public-eval/wave-results/{_PUBLIC_REVISION}/"
        "digest-1/wave-0001-manifest-0123456789abcdef.json"
    ),
) -> dict:
    return {
        "schema_version": "pbe.a1-unix-broker.hf-direct-request.v1",
        "request_id": request_id,
        "pack": "replay-batch",
        "operation": _FIXED_SET_OPERATION,
        "operation_params": {
            "schema_version": _FIXED_SET_JOB_SCHEMA,
            "transport_profile": "hf_bucket_direct",
        },
        "logical_run_id": "run-1",
        "wave_id": "wave-0001",
        "public_revision": _PUBLIC_REVISION,
        "wave_descriptor_hf_ref": _wave_descriptor_hf_ref(),
        "result_manifest_object_path": result_manifest_object_path,
        "shard_count": 8,
        "max_parallel": 8,
    }


def _github_handler():
    def handler(request):
        if request.method == "POST":
            body = json.loads(request.content.decode("utf-8"))
            inputs = body.get("inputs") or {}
            assert inputs.get("jpx_hf_direct") is True
            assert inputs.get("private") is False
            assert "hf_wave_descriptor_ref" in inputs
            assert "input_b64" not in inputs
            return httpx.Response(
                201, json={"workflow_run_id": 1, "html_url": "https://run"}
            )
        return httpx.Response(
            200,
            json={"status": "completed", "conclusion": "success", "updated_at": "t"},
        )

    return handler


def test_hf_direct_success_metadata_only_without_artifact_reader(tmp_path):
    config = _hf_config(tmp_path)
    service = _service(tmp_path, config, _github_handler())
    request = _hf_request()
    manifest_path = request["result_manifest_object_path"]

    def fail_read(ref):
        raise AssertionError(
            "HF-direct broker must not read Private Data Plane artifacts"
        )

    service._artifact_reader.read = fail_read  # type: ignore[method-assign]

    with patch.object(
        service.controller,
        "dispatch_hf_direct_wave",
        wraps=service.controller.dispatch_hf_direct_wave,
    ) as dispatch:
        response = service.handle_payload(_UID, request)

    assert dispatch.call_count == 1
    assert isinstance(response, BrokerHfDirectExecuteResponse)
    assert response.status == "succeeded"
    assert response.result_manifest_object_path == manifest_path
    dumped = response.model_dump(exclude_none=True)
    assert "output_b64" not in dumped
    assert "input_hf_ref" not in dumped
    assert dumped["schema_version"] == "pbe.a1-unix-broker.hf-direct-response.v1"


def test_hf_direct_rejects_public_revision_mismatch(tmp_path):
    config = _hf_config(tmp_path)
    service = _service(tmp_path, config, _github_handler())
    request = _hf_request()
    request["public_revision"] = "b" * 40
    response = service.handle_payload(_UID, request)
    assert response.status == "failed"
    assert response.error_code == "public_revision_mismatch"


def test_hf_direct_rejects_byte_fields(tmp_path):
    config = _hf_config(tmp_path)
    service = _service(tmp_path, config, _github_handler())
    bad = _hf_request()
    bad["input_b64"] = "abcd"
    response = service.handle_payload(_UID, bad)
    assert response.status == "failed"
    assert response.error_code == "request_invalid"


def test_hf_direct_recovery_after_state_loss(tmp_path):
    config = _hf_config(tmp_path)
    service = _service(tmp_path, config, _github_handler())
    request = _hf_request(request_id=sha256(b"recover-hf").hexdigest())
    first = service.handle_payload(_UID, request)
    assert first.status == "succeeded"
    store = BrokerRequestStore(service.state_root / "controller")
    store._path(request["request_id"]).unlink()
    second = service.handle_payload(_UID, request)
    assert second.status == "succeeded"
    assert second.result_manifest_object_path == first.result_manifest_object_path


def test_legacy_fixed_set_with_input_b64_rejected(tmp_path):
    config = _hf_config(tmp_path)
    service = _service(tmp_path, config, _github_handler())
    legacy = {
        "schema_version": "pbe.a1-unix-broker.request.v1",
        "request_id": sha256(b"legacy-fixed").hexdigest(),
        "pack": "replay-batch",
        "operation": _FIXED_SET_OPERATION,
        "operation_params": {
            "schema_version": _FIXED_SET_JOB_SCHEMA,
            "transport_profile": "hf_bucket_direct",
        },
        "input_media_type": "application/json",
        "input_b64": "e30=",
    }
    response = service.handle_payload(_UID, legacy)
    assert response.status == "failed"
    assert response.error_code == "request_invalid"


def test_hf_direct_request_accepts_paired_ledger_canonical_finalize():
    request = _hf_request()
    request["operation"] = "replay.paired_fill_ledger_canonical_finalize"
    request["operation_params"] = {
        "schema_version": "pbe.replay.paired-fill-ledger-canonical-finalize-job.v1",
        "transport_profile": "hf_direct",
    }
    parsed = parse_hf_direct_request(request)
    assert parsed.operation == "replay.paired_fill_ledger_canonical_finalize"


def test_hf_direct_request_existing_operations_regression():
    assert parse_hf_direct_request(_hf_request()).operation == _FIXED_SET_OPERATION
    paired = _hf_request()
    paired["operation"] = "replay.paired_fill_reduce"
    paired["operation_params"] = {
        "schema_version": "pbe.replay.paired-fill-reduce-job.v1",
        "transport_profile": "hf_direct",
    }
    assert parse_hf_direct_request(paired).operation == "replay.paired_fill_reduce"
