import base64
import json
from unittest.mock import patch

import httpx
import pytest

from portable_batch_execution.backends.github_actions import GitHubActionsBackend
from portable_batch_execution.broker.config import BrokerConfig
from portable_batch_execution.broker.planning import (
    broker_allows_operation,
    canonical_operation_params,
    register_broker_private_run,
)
from portable_batch_execution.broker.protocol import (
    BrokerExecuteResponse,
    parse_request,
)
from portable_batch_execution.broker.service import UnixBrokerService
from portable_batch_execution.controller.a1_controller import A1Controller
from portable_batch_execution.packs.ml import MLPack

_PUBLIC_SHA = "ac3a69d2c818526b87f38c848d324221e2dc2775"
_UID = 1000

_CLOSED_ML_OPS = (
    "ml.char_wb_tfidf_logistic_score",
    "ml.cosine_similarity_matrix",
)


@pytest.mark.parametrize("operation", _CLOSED_ML_OPS)
def test_closed_ml_operations_accept_empty_params(operation: str):
    assert canonical_operation_params("ml-batch", operation, {}) == {}


@pytest.mark.parametrize("operation", _CLOSED_ML_OPS)
def test_closed_ml_operations_reject_non_empty_params(operation: str):
    with pytest.raises(ValueError, match="closed operation parameters"):
        canonical_operation_params("ml-batch", operation, {"extra": 1})


_TRADE_PATH_JOB_PARAMS = {
    "schema_version": "pbe.replay.trade-path-scenario-evaluate-job.v1",
}


def test_replay_batch_trade_path_scenario_evaluate_accepts_closed_job_params():
    validated = canonical_operation_params(
        "replay-batch",
        "replay.trade_path_scenario_evaluate",
        _TRADE_PATH_JOB_PARAMS,
    )
    assert validated == _TRADE_PATH_JOB_PARAMS


def test_replay_batch_rejects_unknown_replay_operation():
    assert not broker_allows_operation("replay-batch", "replay.structural_canonicalize")
    with pytest.raises(ValueError, match="unsupported broker operation"):
        canonical_operation_params(
            "replay-batch",
            "replay.structural_canonicalize",
            _TRADE_PATH_JOB_PARAMS,
        )


def test_replay_batch_trade_path_rejects_extra_operation_params():
    with pytest.raises(ValueError):
        canonical_operation_params(
            "replay-batch",
            "replay.trade_path_scenario_evaluate",
            {**_TRADE_PATH_JOB_PARAMS, "extra": 1},
        )


def test_replay_batch_request_rejects_reserved_operation_param_keys():
    with pytest.raises(ValueError, match="reserved request field"):
        parse_request(
            {
                "schema_version": "pbe.a1-unix-broker.request.v1",
                "request_id": "req-replay",
                "pack": "replay-batch",
                "operation": "replay.trade_path_scenario_evaluate",
                "operation_params": {"executable": "evil"},
                "input_media_type": "application/json",
                "input_b64": base64.b64encode(b"{}").decode("ascii"),
            }
        )


def test_register_broker_private_run_replay_batch_trade_path(tmp_path):
    job, _wave, shard, manifest = register_broker_private_run(
        state_root=tmp_path,
        request_id="req-replay-plan",
        pack="replay-batch",
        operation="replay.trade_path_scenario_evaluate",
        operation_params=_TRADE_PATH_JOB_PARAMS,
        input_bytes=b"{}",
        input_media_type="application/json",
        public_sha=_PUBLIC_SHA,
    )
    assert job.pack == "replay-batch"
    assert job.operation == "replay.trade_path_scenario_evaluate"
    assert job.operation_params == _TRADE_PATH_JOB_PARAMS
    assert job.security_profile == "offline"
    assert shard.shard_id == "shard-000000"
    assert manifest.status == "planned"


def test_legacy_ml_operation_still_uses_mlpack_validation():
    validated = canonical_operation_params("ml-batch", "ml.tfidf", {"max_features": 8})
    assert validated == MLPack().validate_params("ml.tfidf", {"max_features": 8})


def test_char_wb_service_request_passes_param_validation(tmp_path):
    config_path = tmp_path / "broker-config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "pbe.a1-unix-broker.config.v1",
                "public_sha": _PUBLIC_SHA,
                "max_input_bytes": 1_048_576,
                "allowed_operations_by_uid": {
                    str(_UID): [
                        ["ml-batch", "ml.char_wb_tfidf_logistic_score"],
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    config = BrokerConfig.load(config_path)
    backend = GitHubActionsBackend(
        "owner",
        "repo",
        "execute-wave.yml",
        private_data_plane=True,
        client=httpx.Client(
            base_url="https://api.github.com",
            transport=httpx.MockTransport(lambda request: httpx.Response(500)),
        ),
    )
    controller = A1Controller(tmp_path, backend=backend)
    service = UnixBrokerService(
        state_root=tmp_path,
        config=config,
        controller=controller,
        poll_interval_seconds=0.0,
    )
    ml_input = json.dumps(
        {
            "schema_version": "pbe.ml.char-wb-tfidf-logistic-score.v1",
            "model": {
                "features": [{"feature": "ab", "idf": 1.0, "coefficient": 0.0}],
                "intercept": 0.0,
            },
            "rows": [{"row_id": "a", "text": "ab"}],
        }
    ).encode()
    request = {
        "schema_version": "pbe.a1-unix-broker.request.v1",
        "request_id": "req-char-wb",
        "pack": "ml-batch",
        "operation": "ml.char_wb_tfidf_logistic_score",
        "operation_params": {},
        "input_media_type": "application/json",
        "input_b64": base64.b64encode(ml_input).decode("ascii"),
    }
    stub = BrokerExecuteResponse(request_id="req-char-wb", status="succeeded")
    with patch.object(UnixBrokerService, "_drive_to_terminal", return_value=stub):
        response = service.handle_payload(_UID, request)
    assert response.error_code != "operation_params_invalid"
    assert response.status == "succeeded"


_PAIRED_LEDGER_CANONICAL_FINALIZE_JOB_PARAMS = {
    "schema_version": "pbe.replay.paired-fill-ledger-canonical-finalize-job.v1",
    "transport_profile": "hf_direct",
}


def test_broker_allows_paired_ledger_canonical_finalize():
    assert broker_allows_operation(
        "replay-batch", "replay.paired_fill_ledger_canonical_finalize"
    )


def test_paired_ledger_canonical_finalize_uses_typed_job_params():
    validated = canonical_operation_params(
        "replay-batch",
        "replay.paired_fill_ledger_canonical_finalize",
        _PAIRED_LEDGER_CANONICAL_FINALIZE_JOB_PARAMS,
    )
    assert validated["schema_version"] == (
        "pbe.replay.paired-fill-ledger-canonical-finalize-job.v1"
    )
    assert validated["transport_profile"] == "hf_direct"


def test_paired_ledger_canonical_finalize_rejects_extra_job_params():
    with pytest.raises(ValueError):
        canonical_operation_params(
            "replay-batch",
            "replay.paired_fill_ledger_canonical_finalize",
            {**_PAIRED_LEDGER_CANONICAL_FINALIZE_JOB_PARAMS, "extra": 1},
        )


def test_existing_fixed_set_remains_allowed_regression():
    assert broker_allows_operation(
        "replay-batch", "replay.trade_path_scenario_evaluate_fixed_set"
    )


def test_existing_paired_fill_reduce_remains_allowed_regression():
    assert broker_allows_operation("replay-batch", "replay.paired_fill_reduce")
