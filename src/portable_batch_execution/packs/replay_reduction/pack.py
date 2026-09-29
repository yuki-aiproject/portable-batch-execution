"""Replay reduction pack: structural canonicalize, merge, and event-window extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonicalize import (
    CanonicalizeState,
    execute_structural_canonicalize,
    merge_structural_canonicalize_states,
)
from .causal_grid import execute_causal_grid_extract
from .event_window import execute_event_window_extract
from .models import PARAM_MODELS


class ReplayReductionPack:
    pack_id = "replay-batch"
    supported_operations = tuple(PARAM_MODELS)

    def validate_params(self, operation: str, params: dict) -> dict:
        try:
            model = PARAM_MODELS[operation]
        except KeyError as exc:
            raise ValueError(f"unsupported replay operation: {operation}") from exc
        return model.model_validate(params).model_dump(mode="json")

    def execute(self, job, shard, params, context):
        operation = getattr(job, "operation", None) or context["operation"]
        if operation == "replay.structural_canonicalize":
            paths = context.get("parquet_paths") or context.get("paths")
            if paths is None:
                raise TypeError("structural canonicalize requires parquet_paths")
            return execute_structural_canonicalize(paths, params)
        if operation == "replay.structural_canonicalize_merge":
            return merge_structural_canonicalize_states(
                context["left_state"], context["right_state"]
            )
        if operation == "replay.event_window_extract":
            paths = context.get("parquet_paths")
            request = context["request"]
            if paths is None:
                raise TypeError("event window extract requires parquet_paths")
            return execute_event_window_extract(paths, request)
        if operation == "replay.causal_grid_extract":
            paths = context.get("parquet_paths")
            request = context["request"]
            if paths is None:
                raise TypeError("causal grid extract requires parquet_paths")
            return execute_causal_grid_extract(paths, request)
        raise ValueError(f"unsupported replay operation: {operation}")

    def finalize(self, job, canonical_attempts, context):
        return canonical_attempts

    def run(
        self,
        operation: str,
        *,
        paths: list[str | Path] | None = None,
        left_state: CanonicalizeState | None = None,
        right_state: CanonicalizeState | None = None,
        request: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ):
        if operation == "replay.structural_canonicalize":
            return execute_structural_canonicalize(paths or [], params or {})
        if operation == "replay.structural_canonicalize_merge":
            return merge_structural_canonicalize_states(left_state, right_state)
        if operation == "replay.event_window_extract":
            return execute_event_window_extract(paths or [], request or {})
        if operation == "replay.causal_grid_extract":
            return execute_causal_grid_extract(paths or [], request or {})
        raise ValueError(f"unsupported replay operation: {operation}")

