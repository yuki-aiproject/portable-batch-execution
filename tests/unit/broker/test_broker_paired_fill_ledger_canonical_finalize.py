import json
from hashlib import sha256

import httpx

from portable_batch_execution.backends.github_actions import GitHubActionsBackend
from portable_batch_execution.broker.config import BrokerConfig
from portable_batch_execution.broker.service import UnixBrokerService
from portable_batch_execution.controller.a1_controller import A1Controller
from portable_batch_execution.transport.hf_bucket import HF_BUCKET_REF_SCHEMA

_OPERATION = "replay.paired_fill_ledger_canonical_finalize"
_JOB_SCHEMA = "pbe.replay.paired-fill-ledger-canonical-finalize-job.v1"
_PUBLIC_SHA = "ac3a69d2c818526b87f38c848d324221e2dc2775"
_UID = 1000
_REQUEST_ID = sha256(b"broker-canonical-finalize-acceptance").hexdigest()


def _mock_backend():
    def handler(request):
        if request.method == "POST":
            return httpx.Response(
                201, json={"workflow_run_id": 1, "html_url": "https://run"}
            )
        return httpx.Response(
            200,
            json={"status": "completed", "conclusion": "success", "updated_at": "t"},
        )

    return GitHubActionsBackend(
        "owner",
        "repo",
        "execute-wave.yml",
        private_data_plane=True,
        client=httpx.Client(
            base_url="https://api.github.com", transport=httpx.MockTransport(handler)
        ),
    )


def _service(tmp_path, config):
    controller = A1Controller(tmp_path, backend=_mock_backend())
    return UnixBrokerService(
        state_root=tmp_path,
        config=config,
        controller=controller,
        poll_interval_seconds=0.0,
        sleeper=lambda _seconds: None,
    )


def _hf_config(tmp_path, *, allowed_operations):
    path = tmp_path / "broker-config-hf.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "pbe.a1-unix-broker.config.v1",
                "public_sha": _PUBLIC_SHA,
                "max_input_bytes": 1_048_576,
                "allowed_operations_by_uid": {
                    str(_UID): allowed_operations,
                },
            }
        ),
        encoding="utf-8",
    )
    return BrokerConfig.load(path)


def _wave_descriptor_hf_ref(public_revision: str) -> dict:
    payload = b'{"schema_version":"pbe.hf-direct.wave-descriptor.v1"}'
    digest = sha256(payload).hexdigest()
    return {
        "schema_version": HF_BUCKET_REF_SCHEMA,
        "bucket_id": "yamauchiJP/system-trading-data",
        "object_path": "pair-trading/v1/public-eval/waves/"
        f"{public_revision}/digest-1/wave-0001-descriptor-0123456789abcdef.json",
        "sha256": digest,
        "size_bytes": len(payload),
        "media_type": "application/vnd.pbe.hf-direct-wave-descriptor.v1",
    }


def _canonical_finalize_hf_request(
    *,
    public_revision: str = _PUBLIC_SHA,
    request_id: str = _REQUEST_ID,
) -> dict:
    return {
        "schema_version": "pbe.a1-unix-broker.hf-direct-request.v1",
        "request_id": request_id,
        "pack": "replay-batch",
        "operation": _OPERATION,
        "operation_params": {
            "schema_version": _JOB_SCHEMA,
            "transport_profile": "hf_direct",
        },
        "logical_run_id": "run-1",
        "wave_id": "wave-0001",
        "public_revision": public_revision,
        "wave_descriptor_hf_ref": _wave_descriptor_hf_ref(public_revision),
        "result_manifest_object_path": (
            f"pair-trading/v1/public-eval/wave-results/{public_revision}/"
            "digest-1/wave-0001-manifest-0123456789abcdef.json"
        ),
        "shard_count": 4,
        "max_parallel": 4,
    }


def test_hf_direct_canonical_finalize_rejects_peer_not_authorized(tmp_path):
    config = _hf_config(
        tmp_path,
        allowed_operations=[
            ["replay-batch", "replay.trade_path_scenario_evaluate_fixed_set"],
        ],
    )
    service = _service(tmp_path, config)
    response = service.handle_payload(_UID, _canonical_finalize_hf_request())
    assert response.status == "failed"
    assert response.error_code == "peer_not_authorized"


def test_hf_direct_canonical_finalize_rejects_public_revision_mismatch(tmp_path):
    config = _hf_config(
        tmp_path,
        allowed_operations=[["replay-batch", _OPERATION]],
    )
    service = _service(tmp_path, config)
    request = _canonical_finalize_hf_request()
    request["public_revision"] = "b" * 40
    response = service.handle_payload(_UID, request)
    assert response.status == "failed"
    assert response.error_code == "public_revision_mismatch"


def test_hf_direct_canonical_finalize_rejects_inline_byte_payload(tmp_path):
    config = _hf_config(
        tmp_path,
        allowed_operations=[["replay-batch", _OPERATION]],
    )
    service = _service(tmp_path, config)
    bad = _canonical_finalize_hf_request()
    bad["input_b64"] = "e30="
    response = service.handle_payload(_UID, bad)
    assert response.status == "failed"
    assert response.error_code == "request_invalid"
