"""Bounded JSON string scalar projection for replay ingestion (ST-0093)."""

from __future__ import annotations

import json
import math
import re
from typing import Any

import polars as pl

from .models import JsonScalarProjection

JSON_SCALAR_PROJECTION_MAX = 16
JSON_SCALAR_KEY_PATH_MAX_DEPTH = 8

_OUTPUT_COLUMN_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

_RESERVED_OUTPUT_COLUMNS = frozenset(
    {
        "_identity_int",
        "_block_int",
        "_timestamp_ms",
        "_row_index",
        "schema_version",
    }
)


class JsonScalarProjectionError(Exception):
    """JSON scalar projection contract violation."""


def validate_json_scalar_projection_bundle(
    projections: tuple[JsonScalarProjection, ...],
) -> None:
    if len(projections) > JSON_SCALAR_PROJECTION_MAX:
        raise ValueError("json scalar projection count exceeds bounded limit")
    outputs: set[str] = set()
    paths: set[tuple[str, tuple[str, ...]]] = set()
    projection_outputs: set[str] = set()
    for projection in projections:
        if len(projection.key_path) > JSON_SCALAR_KEY_PATH_MAX_DEPTH:
            raise ValueError("json scalar projection key_path exceeds bounded depth")
        if projection.output_column in _RESERVED_OUTPUT_COLUMNS:
            raise ValueError("json scalar projection output column is reserved")
        if not _OUTPUT_COLUMN_PATTERN.fullmatch(projection.output_column):
            raise ValueError("json scalar projection output column is unsafe")
        if projection.output_column in outputs:
            raise ValueError("duplicate json scalar projection output column")
        outputs.add(projection.output_column)
        projection_outputs.add(projection.output_column)
        path_key = (projection.source_column, projection.key_path)
        if path_key in paths:
            raise ValueError("duplicate json scalar projection")
        paths.add(path_key)
        if projection.source_column in projection_outputs:
            raise ValueError("ambiguous json scalar projection source column")


def _require_string_source_column(schema: pl.Schema, source_column: str) -> None:
    if source_column not in schema:
        raise JsonScalarProjectionError(
            f"json scalar projection source column missing: {source_column}"
        )
    dtype = schema[source_column]
    if dtype != pl.Utf8 and dtype != pl.String:
        raise JsonScalarProjectionError(
            f"json scalar projection source column is not string: {source_column}"
        )


def validate_json_scalar_projections_against_schema(
    projections: tuple[JsonScalarProjection, ...],
    schema: pl.Schema,
) -> None:
    if not projections:
        return
    validate_json_scalar_projection_bundle(projections)
    column_names = set(schema.names())
    for projection in projections:
        if projection.output_column in column_names:
            raise JsonScalarProjectionError(
                "json scalar projection output column collides with input column"
            )
        _require_string_source_column(schema, projection.source_column)


def _schema_with_projections(
    schema: pl.Schema, projections: tuple[JsonScalarProjection, ...]
) -> pl.Schema:
    if not projections:
        return schema
    fields = dict(schema)
    for projection in projections:
        if projection.scalar_type == "integer":
            fields[projection.output_column] = pl.Int64
        else:
            fields[projection.output_column] = pl.Utf8
    return pl.Schema(fields)


def _extract_scalar(
    raw: Any,
    projection: JsonScalarProjection,
) -> int | str:
    if raw is None:
        raise JsonScalarProjectionError("json scalar projection source is null")
    if not isinstance(raw, str):
        raise JsonScalarProjectionError("json scalar projection source is not a string")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise JsonScalarProjectionError("json scalar projection malformed") from exc

    current: Any = parsed
    for key in projection.key_path:
        if not isinstance(current, dict):
            raise JsonScalarProjectionError(
                "json scalar projection path traverses non-object"
            )
        if key not in current:
            raise JsonScalarProjectionError(
                "json scalar projection missing required key"
            )
        current = current[key]

    if projection.scalar_type == "integer":
        if isinstance(current, bool):
            raise JsonScalarProjectionError("json scalar projection type mismatch")
        if isinstance(current, int):
            return int(current)
        if isinstance(current, float):
            if not math.isfinite(current) or current != math.trunc(current):
                raise JsonScalarProjectionError("json scalar projection type mismatch")
            return int(current)
        raise JsonScalarProjectionError("json scalar projection type mismatch")

    if not isinstance(current, str):
        raise JsonScalarProjectionError("json scalar projection type mismatch")
    return current


def project_json_scalar_batch(
    batch: pl.DataFrame,
    projections: tuple[JsonScalarProjection, ...],
) -> pl.DataFrame:
    if not projections:
        return batch
    if batch.is_empty():
        extra = {
            projection.output_column: pl.Series(
                projection.output_column,
                [],
                dtype=pl.Int64 if projection.scalar_type == "integer" else pl.Utf8,
            )
            for projection in projections
        }
        return batch.with_columns(**extra)
    validate_json_scalar_projections_against_schema(projections, batch.schema)
    new_columns: dict[str, list[Any]] = {
        projection.output_column: [] for projection in projections
    }
    unique_sources = list(dict.fromkeys(projection.source_column for projection in projections))
    for row in batch.select(unique_sources).iter_rows():
        source_by_column = dict(zip(unique_sources, row, strict=True))
        for projection in projections:
            new_columns[projection.output_column].append(
                _extract_scalar(source_by_column[projection.source_column], projection)
            )
    series = [
        pl.Series(
            projection.output_column,
            new_columns[projection.output_column],
            dtype=pl.Int64 if projection.scalar_type == "integer" else pl.Utf8,
        )
        for projection in projections
    ]
    return batch.with_columns(series)


def apply_json_scalar_projections_to_lazy(
    lazy: pl.LazyFrame,
    projections: tuple[JsonScalarProjection, ...],
) -> pl.LazyFrame:
    if not projections:
        return lazy
    input_schema = lazy.collect_schema()
    validate_json_scalar_projections_against_schema(projections, input_schema)
    output_schema = _schema_with_projections(input_schema, projections)

    def _map_batch(batch: pl.DataFrame) -> pl.DataFrame:
        try:
            return project_json_scalar_batch(batch, projections)
        except JsonScalarProjectionError as exc:
            from .canonicalize import StructuralCanonicalizeError

            raise StructuralCanonicalizeError(str(exc)) from exc

    return lazy.map_batches(_map_batch, schema=output_schema)
