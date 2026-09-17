import json
from datetime import UTC, datetime
from hashlib import sha256

import pytest

from portable_batch_execution.contracts import ArtifactRef
from portable_batch_execution.worker.runtime_profile import (
    RuntimeProfileError,
    classify_operation,
    resolve_private_profile,
    resolve_profile,
    resolve_public_profile,
)

_TABULAR = "tabular"
_ML = "ml"
_DISTILBERT = "distilbert"
_MEDIA = "media"


@pytest.mark.parametrize(
    ("pack", "operation", "expected"),
    (
        ("tabular-batch", "tabular.normalize", _TABULAR),
        ("tabular-batch", "tabular.cast", _TABULAR),
        ("tabular-batch", "tabular.sort", _TABULAR),
        ("tabular-batch", "tabular.dedup", _TABULAR),
        ("tabular-batch", "tabular.window", _TABULAR),
        ("tabular-batch", "tabular.rolling", _TABULAR),
        ("tabular-batch", "tabular.statistics", _TABULAR),
        ("tabular-batch", "tabular.text_event_features.v1", _TABULAR),
        ("tabular-batch", "tabular.join", _TABULAR),
        ("tabular-batch", "tabular.pit_join", _TABULAR),
        ("tabular-batch", "tabular.trailing_sparse_window_aggregate.v1", _TABULAR),
        ("ml-batch", "ml.char_wb_tfidf_logistic_score", _ML),
        ("ml-batch", "ml.cosine_similarity_matrix", _ML),
        ("ml-batch", "ml.distilbert_pair_binary_scores", _DISTILBERT),
        ("media-batch", "media.asr_normalize_flac", _MEDIA),
    ),
)
def test_supported_execute_wave_operations_classify_to_one_profile(pack, operation, expected):
    assert classify_operation(pack, operation) == expected


@pytest.mark.parametrize(
    ("pack", "operation"),
    (
        ("tabular-batch", "tabular.format_migration"),
        ("ml-batch", "ml.tfidf"),
        ("ml-batch", "ml.embedding"),
        ("media-batch", "media.decode"),
        ("media-batch", "media.metadata"),
        ("acquisition-batch", "acquisition.rest"),
        ("replay-eval-batch", "replay_eval.replay"),
        ("tabular-batch", "tabular.unknown"),
    ),
)
def test_unsupported_operations_fail_closed(pack, operation):
    with pytest.raises(RuntimeProfileError):
        classify_operation(pack, operation)


def test_public_synthetic_mode_resolves_deterministically():
    first = resolve_public_profile("wave-0000")
    second = resolve_public_profile("wave-0000")
    assert first == second
    assert first.mode == "public"
    assert first.profile == _TABULAR
    assert first.pack == "tabular-batch"
    assert first.operation == "tabular.rolling"
    assert first.run_id is None


@pytest.mark.parametrize(
    "wave_id",
    ("", "wave-0001", "../wave-0000", "wave-0000; echo x", "$(whoami)"),
)
def test_public_mode_rejects_non_allowlisted_wave_ids(wave_id):
    with pytest.raises(RuntimeProfileError):
        resolve_public_profile(wave_id)


def _plane(*, pack="tabular-batch", operation="tabular.rolling", run_id="opaque-run", wave_id="opaque-wave"):
    payload = b"payload"
    input_ref = ArtifactRef(
        object_id="input",
        uri="pbe://private/input",
        sha256="sha256:" + sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )
    now = datetime.now(UTC).isoformat()
    job = {
        "job_id": "job",
        "logical_run_id": run_id,
        "pack": pack,
        "operation": operation,
        "input_manifest_ref": input_ref.model_dump(mode="json"),
        "sharding": {},
        "execution": {"max_parallel": 1, "max_attempts_per_shard": 1},
        "security_profile": "offline",
        "provenance": {"producer": "test", "revision": "1", "created_at": now},
        "operation_params": {},
    }
    shard = {
        "logical_run_id": run_id,
        "shard_id": "opaque-shard",
        "ordinal": 0,
        "correctness": {},
        "input_refs": [input_ref.model_dump(mode="json")],
        "input_digest": "current",
        "execution_fingerprint": "fixed",
    }
    wave = {
        "logical_run_id": run_id,
        "wave_id": wave_id,
        "ordinal": 0,
        "shard_ids": ["opaque-shard"],
        "max_parallel": 1,
    }

    class Plane:
        def __init__(self):
            self.payload = {"job": job, "wave": wave, "shards": [shard]}
            self.resolved: list[tuple[str, str]] = []

        def resolve_wave(self, requested_run, requested_wave):
            self.resolved.append((requested_run, requested_wave))
            return json.loads(json.dumps(self.payload))

    return Plane()


@pytest.mark.parametrize(
    ("pack", "operation", "expected"),
    (
        ("tabular-batch", "tabular.join", _TABULAR),
        ("ml-batch", "ml.cosine_similarity_matrix", _ML),
        ("ml-batch", "ml.distilbert_pair_binary_scores", _DISTILBERT),
        ("media-batch", "media.asr_normalize_flac", _MEDIA),
    ),
)
def test_private_mode_resolves_supported_profile_without_domain_imports(pack, operation, expected):
    plane = _plane(pack=pack, operation=operation)
    resolution = resolve_private_profile("opaque-run", "opaque-wave", plane=plane)
    assert resolution.profile == expected
    assert resolution.mode == "private"
    assert resolution.run_id == "opaque-run"
    assert plane.resolved == [("opaque-run", "opaque-wave")]


def test_private_mode_rejects_unavailable_operation():
    plane = _plane(operation="tabular.format_migration")
    with pytest.raises(RuntimeProfileError):
        resolve_private_profile("opaque-run", "opaque-wave", plane=plane)


def test_private_mode_rejects_mismatched_run_or_wave():
    plane = _plane()
    with pytest.raises(RuntimeProfileError):
        resolve_private_profile("different-run", "opaque-wave", plane=plane)
    with pytest.raises(RuntimeProfileError):
        resolve_private_profile("opaque-run", "different-wave", plane=plane)


def test_private_mode_rejects_shard_plan_mismatch():
    plane = _plane()
    plane.payload["wave"]["shard_ids"] = ["other-shard"]
    with pytest.raises(RuntimeProfileError):
        resolve_private_profile("opaque-run", "opaque-wave", plane=plane)


def test_resolve_profile_dispatches_on_mode():
    assert resolve_profile("wave-0000").profile == _TABULAR
    plane = _plane()
    assert (
        resolve_profile("opaque-wave", mode="private", run_id="opaque-run", plane=plane).profile
        == _TABULAR
    )
    with pytest.raises(RuntimeProfileError):
        resolve_profile("opaque-wave", mode="private")
    with pytest.raises(RuntimeProfileError):
        resolve_profile("wave-0000", mode="bogus")
