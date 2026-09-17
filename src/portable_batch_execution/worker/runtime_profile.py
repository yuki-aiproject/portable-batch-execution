"""Lightweight runtime-profile resolver for one already closed execute-wave job.

This module is importable from a base-only install: it must never import a
domain pack (polars, scikit-learn, torch, transformers, or the FFmpeg media
pack).  It reads the closed contracts, validates the wave against its job, and
maps the operation onto exactly one bootstrap profile:

* ``tabular``   -- the Polars-backed tabular pack operations
* ``ml``        -- the scikit-learn / NumPy backed closed ML operations
* ``distilbert``-- the torch / transformers pair scoring operation
* ``media``     -- the FFmpeg-backed normalization operation

Unsupported, unavailable, or internally mismatched jobs fail closed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from portable_batch_execution.contracts import JobSpec, ShardSpec, WaveSpec

PROFILES = ("tabular", "ml", "distilbert", "media")

TABULAR_SINGLE_INPUT_OPERATIONS = frozenset(
    {
        "tabular.normalize",
        "tabular.cast",
        "tabular.sort",
        "tabular.dedup",
        "tabular.window",
        "tabular.rolling",
        "tabular.statistics",
        "tabular.text_event_features.v1",
    }
)
TABULAR_TWO_TABLE_OPERATIONS = frozenset(
    {
        "tabular.join",
        "tabular.pit_join",
    }
)
TABULAR_FEATURE_REQUEST_OPERATIONS = frozenset(
    {"tabular.trailing_sparse_window_aggregate.v1"}
)
TABULAR_UNAVAILABLE_OPERATIONS = frozenset({"tabular.format_migration"})
ML_SINGLE_INPUT_OPERATIONS = frozenset(
    {
        "ml.char_wb_tfidf_logistic_score",
        "ml.cosine_similarity_matrix",
    }
)
ML_FIVE_INPUT_OPERATIONS = frozenset({"ml.distilbert_pair_binary_scores"})
MEDIA_SINGLE_INPUT_OPERATIONS = frozenset({"media.asr_normalize_flac"})

_WAVE_ID_PATTERN = r"wave-[0-9]{4}"
_PUBLIC_WAVES = frozenset({"wave-0000"})


class RuntimeProfileError(RuntimeError):
    """The closed wave cannot be mapped onto a supported bootstrap profile."""


@dataclass(frozen=True)
class RuntimeProfileResolution:
    mode: str
    wave_id: str
    profile: str
    pack: str
    operation: str
    run_id: str | None = None


def classify_operation(pack: str, operation: str) -> str:
    """Return the bootstrap profile for one closed pack operation, or fail closed."""
    if pack == "tabular-batch":
        if operation in TABULAR_UNAVAILABLE_OPERATIONS:
            raise RuntimeProfileError("operation requires a typed multi-input contract")
        if (
            operation in TABULAR_SINGLE_INPUT_OPERATIONS
            or operation in TABULAR_TWO_TABLE_OPERATIONS
            or operation in TABULAR_FEATURE_REQUEST_OPERATIONS
        ):
            return "tabular"
    elif pack == "ml-batch":
        if operation in ML_SINGLE_INPUT_OPERATIONS:
            return "ml"
        if operation in ML_FIVE_INPUT_OPERATIONS:
            return "distilbert"
    elif pack == "media-batch":
        if operation in MEDIA_SINGLE_INPUT_OPERATIONS:
            return "media"
    raise RuntimeProfileError("operation is not supported by the public runner")


def _validate_closed_contracts(payload: Any, run_id: str, wave_id: str) -> tuple[JobSpec, WaveSpec, tuple[ShardSpec, ...]]:
    try:
        job = JobSpec.model_validate(payload["job"])
        wave = WaveSpec.model_validate(payload["wave"])
        shards = tuple(ShardSpec.model_validate(item) for item in payload["shards"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeProfileError("closed wave contracts are invalid") from exc
    if job.logical_run_id != run_id or wave.logical_run_id != run_id or wave.wave_id != wave_id:
        raise RuntimeProfileError("closed wave does not match the requested run or wave")
    if tuple(shard.shard_id for shard in shards) != wave.shard_ids or any(
        shard.logical_run_id != run_id for shard in shards
    ):
        raise RuntimeProfileError("closed wave shards do not match the wave plan")
    return job, wave, shards


def resolve_private_profile(
    run_id: str, wave_id: str, *, plane=None
) -> RuntimeProfileResolution:
    """Resolve a private closed wave through the data plane without domain imports."""
    from portable_batch_execution.data_plane import HttpPrivateDataPlane

    plane = plane or HttpPrivateDataPlane.from_environment()
    payload = plane.resolve_wave(run_id, wave_id)
    job, _wave, _shards = _validate_closed_contracts(payload, run_id, wave_id)
    profile = classify_operation(job.pack, job.operation)
    return RuntimeProfileResolution(
        mode="private",
        wave_id=wave_id,
        run_id=run_id,
        profile=profile,
        pack=job.pack,
        operation=job.operation,
    )


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _plan_path(wave_id: str, repository_root: Path) -> Path:
    import re

    if not isinstance(wave_id, str) or not re.fullmatch(_WAVE_ID_PATTERN, wave_id):
        raise RuntimeProfileError("wave_id must be a closed planned wave identifier")
    if wave_id not in _PUBLIC_WAVES:
        raise RuntimeProfileError("wave_id is not an approved public synthetic wave")
    return repository_root / "fixtures" / "public" / "synthetic" / f"{wave_id}.json"


def resolve_public_profile(
    wave_id: str, *, repository_root: Path | None = None
) -> RuntimeProfileResolution:
    """Resolve a committed public synthetic plan deterministically without domain imports."""
    from hashlib import sha256

    from portable_batch_execution.contracts import ArtifactRef

    root = (repository_root or _repository_root()).resolve()
    path = _plan_path(wave_id, root)
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeProfileError("public synthetic wave plan is invalid") from exc
    if not isinstance(plan, dict) or "job" not in plan or "wave" not in plan:
        raise RuntimeProfileError("public synthetic wave plan must be an object")
    fixture_root = (root / "fixtures" / "public").resolve()
    fixture = plan.get("input_fixture")
    if not isinstance(fixture, str) or Path(fixture).name != fixture:
        raise RuntimeProfileError("public synthetic input fixture is invalid")
    data_path = (fixture_root / fixture).resolve()
    if fixture_root not in data_path.parents or not data_path.is_file():
        raise RuntimeProfileError("public synthetic input fixture is unavailable")
    try:
        payload = data_path.read_bytes()
    except OSError as exc:
        raise RuntimeProfileError("public synthetic input fixture is unavailable") from exc
    digest = sha256(payload).hexdigest()
    input_ref = ArtifactRef(
        object_id=digest,
        uri=data_path.as_uri(),
        sha256=f"sha256:{digest}",
        size_bytes=len(payload),
    ).model_dump(mode="json")
    try:
        job = JobSpec.model_validate({**plan["job"], "input_manifest_ref": input_ref})
        wave = WaveSpec.model_validate(plan["wave"])
        shards = tuple(
            ShardSpec.model_validate(
                {
                    **item,
                    "logical_run_id": job.logical_run_id,
                    "correctness": job.sharding.model_dump(mode="json"),
                    "input_refs": [input_ref],
                }
            )
            for item in plan["shards"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeProfileError("public synthetic plan is invalid") from exc
    if wave.wave_id != wave_id or wave.logical_run_id != job.logical_run_id:
        raise RuntimeProfileError("public synthetic wave does not match its job")
    if tuple(shard.shard_id for shard in shards) != wave.shard_ids:
        raise RuntimeProfileError("public synthetic wave shard plan is invalid")
    profile = classify_operation(job.pack, job.operation)
    return RuntimeProfileResolution(
        mode="public",
        wave_id=wave_id,
        profile=profile,
        pack=job.pack,
        operation=job.operation,
    )


def resolve_profile(
    wave_id: str,
    *,
    mode: str = "public",
    run_id: str | None = None,
    repository_root: Path | None = None,
    plane=None,
) -> RuntimeProfileResolution:
    if mode == "private":
        if not run_id:
            raise RuntimeProfileError("private mode requires a run identifier")
        return resolve_private_profile(run_id, wave_id, plane=plane)
    if mode == "public":
        return resolve_public_profile(wave_id, repository_root=repository_root)
    raise RuntimeProfileError("mode must be public or private")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve one closed wave's runtime profile.")
    parser.add_argument("--wave-id", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--mode", choices=("public", "private"), default="public")
    parser.add_argument("--json", action="store_true", help="Emit a JSON resolution object")
    args = parser.parse_args(argv)
    if args.mode == "private" and not args.run_id:
        parser.error("--mode private requires --run-id")
    try:
        resolution = resolve_profile(
            args.wave_id,
            mode=args.mode,
            run_id=args.run_id,
        )
    except RuntimeProfileError as exc:
        print(f"runtime-profile error: {exc}", file=__import__("sys").stderr)
        return 1
    if args.json:
        print(
            json.dumps(
                {
                    "mode": resolution.mode,
                    "wave_id": resolution.wave_id,
                    "run_id": resolution.run_id,
                    "pack": resolution.pack,
                    "operation": resolution.operation,
                    "profile": resolution.profile,
                }
            )
        )
    else:
        print(resolution.profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
