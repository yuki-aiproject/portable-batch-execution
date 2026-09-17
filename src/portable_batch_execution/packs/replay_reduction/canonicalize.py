"""Exact structural canonicalization via partitioned exact-ID spill state.

Identity membership lives in per-bucket sorted spill files. Merge and
canonicalization stream one bucket at a time so peak memory does not scale
with total distinct identities across a wave.
"""

from __future__ import annotations

import heapq
import shutil
import struct
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from .models import (
    BUCKET_COUNT_MAX,
    BUCKET_COUNT_MIN,
    SentinelPredicate,
    SentinelScalar,
    StructuralCanonicalizeParams,
)

STATE_SCHEMA_VERSION = "pbe.replay.structural-canonicalize-state.v4"
BUCKET_FORMAT = "pbe.replay.exact-id-bucket.v1"
BUCKET_FORMAT_VERSION = 1
BUCKET_MEDIA_TYPE = "application/vnd.pbe.exact-id-bucket.v1"
BUCKET_MAGIC = b"PBEBKT01"
_BUCKET_HEADER = struct.Struct("<8sHIIQ")
_UINT64_PACK = struct.Struct("<Q")
_UINT64_MASK = (1 << 64) - 1
_SPLITMIX_GOLDEN = 0x9E3779B97F4A7C15
_SPLITMIX_MUL_A = 0xBF58476D1CE4E5B9
_SPLITMIX_MUL_B = 0x94D049BB133111EB
_IDENTITY_SCAN_BATCH = 65_536
_SORT_RUN_CAPACITY = 8_192
_MAX_IDENTITY_MATERIALIZATION = 1 << 20
_MAX_BUCKET_PAYLOAD_BYTES = _BUCKET_HEADER.size + (_MAX_IDENTITY_MATERIALIZATION * 8)


class StructuralCanonicalizeError(Exception):
    """Input rows violate structural canonicalization invariants."""


@dataclass(frozen=True)
class BucketSet:
    """One deterministic sorted unique uint64 identity bucket (artifact view)."""

    bucket_index: int
    values: tuple[int, ...]


@dataclass(frozen=True)
class CanonicalizeState:
    """Mergeable canonicalization state backed by on-disk bucket spill files."""

    bucket_count: int
    spill_dir: Path
    bucket_counts: tuple[int, ...]
    positive_row_count: int
    positive_group_count: int
    witness_row_count: int
    first_boundary: dict[str, Any] | None
    last_boundary: dict[str, Any] | None


def splitmix64(value: int) -> int:
    z = (value + _SPLITMIX_GOLDEN) & _UINT64_MASK
    z = ((z ^ (z >> 30)) * _SPLITMIX_MUL_A) & _UINT64_MASK
    z = ((z ^ (z >> 27)) * _SPLITMIX_MUL_B) & _UINT64_MASK
    return (z ^ (z >> 31)) & _UINT64_MASK


def bucket_index(value: int, bucket_count: int) -> int:
    return splitmix64(value) & (bucket_count - 1)


def _assert_bounded_materialization(count: int) -> None:
    if count > _MAX_IDENTITY_MATERIALIZATION:
        raise StructuralCanonicalizeError("identity materialization exceeds bounded limit")


def _assert_bucket_payload_bounded(payload_size: int) -> None:
    if payload_size > _MAX_BUCKET_PAYLOAD_BYTES:
        raise StructuralCanonicalizeError("bucket payload materialization exceeds bounded limit")


def _validated(
    params: dict[str, Any] | StructuralCanonicalizeParams,
) -> StructuralCanonicalizeParams:
    if isinstance(params, StructuralCanonicalizeParams):
        return params
    return StructuralCanonicalizeParams.model_validate(params)


def _require_columns(schema: pl.Schema, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in schema]
    if missing:
        raise StructuralCanonicalizeError(
            f"missing required columns: {', '.join(missing)}"
        )


def _exact_match_expr(field: str, expected: SentinelScalar) -> pl.Expr:
    if expected is None:
        return pl.col(field).is_null()
    return (pl.col(field) == expected).fill_null(False)


def _sentinel_witness_expr(sentinel: SentinelPredicate) -> pl.Expr:
    expr = (pl.col("_identity_int") == sentinel.identity_equals).fill_null(False)
    for field, expected in sentinel.exact_match_fields.items():
        expr = expr & _exact_match_expr(field, expected)
    return expr


def _positive_normalized_mismatch_expr(normalized_col: str) -> pl.Expr:
    normalized_utf8 = pl.col(normalized_col).cast(pl.Utf8, strict=False)
    expected = pl.col("_identity_int").cast(pl.Utf8)
    return normalized_utf8.is_null() | (normalized_utf8 != expected)


def _positive_null_core_expr(core_fields: list[str]) -> pl.Expr:
    if len(core_fields) == 1:
        return pl.col(core_fields[0]).is_null()
    return pl.any_horizontal([pl.col(field).is_null() for field in core_fields])


def _source_identity_malformed_expr(identity_col: str, identity_dtype: pl.DataType) -> pl.Expr:
    if identity_dtype == pl.Boolean:
        return pl.lit(True)
    source = pl.col(identity_col)
    cast_int = source.cast(pl.Int64, strict=False)
    cast_float = source.cast(pl.Float64, strict=False)
    return (
        source.is_null()
        | cast_int.is_null()
        | cast_float.is_null()
        | ~cast_float.is_finite()
        | (cast_float != cast_float.floor())
    )


def _boundary_profile(row: dict[str, Any], core_fields: list[str]) -> dict[str, Any]:
    return {
        "identity": int(row["_identity_int"]),
        "core": {field: row[field] for field in core_fields},
    }


def _bucket_file(spill_dir: Path, index: int) -> Path:
    return spill_dir / f"bucket_{index:05d}.bin"


def _make_spill_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="pbe-canonicalize-"))


def _init_empty_spill(bucket_count: int, spill_dir: Path) -> tuple[int, ...]:
    counts: list[int] = []
    for index in range(bucket_count):
        path = _bucket_file(spill_dir, index)
        path.touch()
        counts.append(0)
    return tuple(counts)


def _iter_uint64_path(path: Path) -> Iterator[int]:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb") as handle:
        while True:
            payload = handle.read(8)
            if not payload:
                return
            if len(payload) != 8:
                raise StructuralCanonicalizeError("bucket spill file is corrupt")
            yield int(_UINT64_PACK.unpack(payload)[0])


def _k_way_merge_sorted_fail_on_duplicate(sources: list[Path], dest: Path) -> None:
    heap: list[tuple[int, int, Iterator[int]]] = []
    for source_index, source in enumerate(sources):
        iterator = _iter_uint64_path(source)
        value = next(iterator, None)
        if value is not None:
            heapq.heappush(heap, (value, source_index, iterator))
    last_written: int | None = None
    with dest.open("wb") as output:
        while heap:
            value, source_index, iterator = heapq.heappop(heap)
            if last_written == value:
                raise StructuralCanonicalizeError(
                    "positive identity recurs non-contiguously within one input"
                )
            output.write(_UINT64_PACK.pack(value & _UINT64_MASK))
            last_written = value
            next_value = next(iterator, None)
            if next_value is not None:
                heapq.heappush(heap, (next_value, source_index, iterator))


def _finalize_sorted_bucket_file(path: Path) -> int:
    previous = 0
    count = 0
    for value in _iter_uint64_path(path):
        if count and value <= previous:
            if value == previous:
                raise StructuralCanonicalizeError(
                    "positive identity recurs non-contiguously within one input"
                )
            raise StructuralCanonicalizeError("bucket values must be sorted unique")
        previous = value
        count += 1
    return count


def _sort_unique_bucket_file_in_place(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        path.touch()
        return 0
    run_paths: list[Path] = []
    with path.open("rb") as source:
        while True:
            chunk = source.read(_SORT_RUN_CAPACITY * 8)
            if not chunk:
                break
            if len(chunk) % 8:
                raise StructuralCanonicalizeError("bucket spill file is corrupt")
            values = sorted(
                _UINT64_PACK.unpack_from(chunk, offset)[0]
                for offset in range(0, len(chunk), 8)
            )
            _assert_bounded_materialization(len(values))
            run_path = path.with_name(f"{path.name}.run{len(run_paths)}")
            with run_path.open("wb") as run:
                for value in values:
                    run.write(_UINT64_PACK.pack(value & _UINT64_MASK))
            run_paths.append(run_path)
    path.unlink(missing_ok=True)
    if not run_paths:
        path.touch()
        return _finalize_sorted_bucket_file(path)
    if len(run_paths) == 1:
        run_paths[0].replace(path)
        return _finalize_sorted_bucket_file(path)
    _k_way_merge_sorted_fail_on_duplicate(run_paths, path)
    for run_path in run_paths:
        run_path.unlink(missing_ok=True)
    return _finalize_sorted_bucket_file(path)


def _stream_validate_and_spill_positive_rows(
    ordered: pl.LazyFrame,
    positive_row_count: int,
    spill_dir: Path,
    bucket_count: int,
    core_fields: list[str],
) -> tuple[int, tuple[int, ...], dict[str, Any], dict[str, Any]]:
    _init_empty_spill(bucket_count, spill_dir)
    handles = [_bucket_file(spill_dir, index).open("ab") for index in range(bucket_count)]
    previous_identity: int | None = None
    group_core: dict[str, Any] | None = None
    group_count = 0
    first_boundary: dict[str, Any] | None = None
    last_boundary: dict[str, Any] | None = None
    select_columns = ["_identity_int", *core_fields]
    try:
        offset = 0
        while offset < positive_row_count:
            batch_size = min(_IDENTITY_SCAN_BATCH, positive_row_count - offset)
            batch = ordered.slice(offset, batch_size).select(select_columns).collect()
            _assert_bounded_materialization(batch.height)
            for row in batch.iter_rows(named=True):
                for field in core_fields:
                    if row[field] is None:
                        raise StructuralCanonicalizeError("measurement core fields disagree")
                identity = int(row["_identity_int"])
                if previous_identity is None or identity != previous_identity:
                    index = bucket_index(identity, bucket_count)
                    handles[index].write(_UINT64_PACK.pack(identity & _UINT64_MASK))
                    group_core = {field: row[field] for field in core_fields}
                    group_count += 1
                    profile = _boundary_profile(row, core_fields)
                    if first_boundary is None:
                        first_boundary = profile
                    last_boundary = profile
                else:
                    if group_core is None:
                        raise StructuralCanonicalizeError("measurement core fields disagree")
                    for field in core_fields:
                        if row[field] != group_core[field]:
                            raise StructuralCanonicalizeError(
                                "measurement core fields disagree"
                            )
                    last_boundary = _boundary_profile(row, core_fields)
                previous_identity = identity
            offset += batch_size
    finally:
        for handle in handles:
            handle.close()
    if first_boundary is None or last_boundary is None:
        raise StructuralCanonicalizeError("positive rows missing boundary profiles")
    bucket_counts: list[int] = []
    for index in range(bucket_count):
        bucket_counts.append(
            _sort_unique_bucket_file_in_place(_bucket_file(spill_dir, index))
        )
    return group_count, tuple(bucket_counts), first_boundary, last_boundary


def _stream_merge_sorted_unique_files(
    left_path: Path,
    right_path: Path,
    dest_path: Path,
    *,
    allowed_identity: int | None,
) -> int:
    left_iter = _iter_uint64_path(left_path)
    right_iter = _iter_uint64_path(right_path)
    left_value = next(left_iter, None)
    right_value = next(right_iter, None)
    last_written: int | None = None
    count = 0
    with dest_path.open("wb") as output:
        while left_value is not None or right_value is not None:
            if right_value is None or (
                left_value is not None and left_value < right_value
            ):
                chosen = left_value
                left_value = next(left_iter, None)
            elif left_value is None or right_value < left_value:
                chosen = right_value
                right_value = next(right_iter, None)
            else:
                if allowed_identity is None or left_value != allowed_identity:
                    raise StructuralCanonicalizeError(
                        "non-contiguous recurrence of a positive identity"
                    )
                chosen = left_value
                left_value = next(left_iter, None)
                right_value = next(right_iter, None)
            if last_written == chosen:
                continue
            output.write(_UINT64_PACK.pack(chosen & _UINT64_MASK))
            last_written = chosen
            count += 1
    return count


def _bucket_contains_identity(path: Path, bucket_count: int, identity: int) -> bool:
    target = identity & _UINT64_MASK
    for value in _iter_uint64_path(path):
        if value == target:
            return True
        if value > target:
            return False
    return False


def _validate_bucket_file(path: Path, bucket_count: int, expected_index: int) -> int:
    previous = 0
    count = 0
    for value in _iter_uint64_path(path):
        if value <= previous:
            raise StructuralCanonicalizeError("bucket values must be sorted unique")
        if bucket_index(value, bucket_count) != expected_index:
            raise StructuralCanonicalizeError("bucket value is not in its bucket")
        previous = value
        count += 1
    return count


def _empty_state(bucket_count: int, witness_row_count: int) -> CanonicalizeState:
    spill_dir = _make_spill_dir()
    bucket_counts = _init_empty_spill(bucket_count, spill_dir)
    return CanonicalizeState(
        bucket_count=bucket_count,
        spill_dir=spill_dir,
        bucket_counts=bucket_counts,
        positive_row_count=0,
        positive_group_count=0,
        witness_row_count=witness_row_count,
        first_boundary=None,
        last_boundary=None,
    )


def _single_input_state(
    path: str | Path,
    model: StructuralCanonicalizeParams,
) -> CanonicalizeState:
    identity_col = model.identity_source_column
    normalized_col = model.identity_normalized_column
    core_fields = list(model.measurement_core_fields)
    sentinel = model.sentinel
    bucket_count = model.bucket_count

    lazy = pl.scan_parquet(str(path))
    required = (identity_col, normalized_col, *core_fields)
    if sentinel is not None:
        required = required + tuple(sentinel.exact_match_fields)
    _require_columns(lazy.collect_schema(), required)
    identity_dtype = lazy.collect_schema()[identity_col]
    lazy = lazy.with_row_index("_row_index")
    if (
        int(
            lazy.filter(_source_identity_malformed_expr(identity_col, identity_dtype))
            .select(pl.len())
            .collect()
            .item()
        )
        > 0
    ):
        raise StructuralCanonicalizeError("identity is missing or not positive")
    lazy = lazy.with_columns(
        pl.col(identity_col).cast(pl.Int64, strict=False).alias("_identity_int"),
    )

    if sentinel is not None:
        witness_expr = _sentinel_witness_expr(sentinel)
        witness_count = int(lazy.filter(witness_expr).select(pl.len()).collect().item())
    else:
        witness_expr = None
        witness_count = 0

    nonpositive = pl.col("_identity_int") <= 0
    if witness_expr is None:
        invalid_expr = pl.col("_identity_int").is_null() | nonpositive
    else:
        invalid_expr = pl.col("_identity_int").is_null() | (
            nonpositive & witness_expr.not_()
        )
    if int(lazy.filter(invalid_expr).select(pl.len()).collect().item()) > 0:
        raise StructuralCanonicalizeError("identity is missing or not positive")

    positive = lazy.filter(pl.col("_identity_int") > 0)
    mismatch = positive.filter(_positive_normalized_mismatch_expr(normalized_col))
    if int(mismatch.select(pl.len()).collect().item()) > 0:
        raise StructuralCanonicalizeError("identity normalized column mismatch")

    null_core = positive.filter(_positive_null_core_expr(core_fields))
    if int(null_core.select(pl.len()).collect().item()) > 0:
        raise StructuralCanonicalizeError("measurement core fields disagree")

    positive_row_count = int(positive.select(pl.len()).collect().item())
    if positive_row_count == 0:
        return _empty_state(bucket_count, witness_count)

    spill_dir = _make_spill_dir()
    group_count, bucket_counts, first_boundary, last_boundary = (
        _stream_validate_and_spill_positive_rows(
            positive,
            positive_row_count,
            spill_dir,
            bucket_count,
            core_fields,
        )
    )

    return CanonicalizeState(
        bucket_count=bucket_count,
        spill_dir=spill_dir,
        bucket_counts=bucket_counts,
        positive_row_count=positive_row_count,
        positive_group_count=group_count,
        witness_row_count=witness_count,
        first_boundary=first_boundary,
        last_boundary=last_boundary,
    )


def _validate_state(state: CanonicalizeState) -> None:
    if (
        not isinstance(state.bucket_count, int)
        or isinstance(state.bucket_count, bool)
        or state.bucket_count < BUCKET_COUNT_MIN
        or state.bucket_count > BUCKET_COUNT_MAX
        or state.bucket_count & (state.bucket_count - 1)
    ):
        raise StructuralCanonicalizeError("invalid bucket_count")
    if len(state.bucket_counts) != state.bucket_count:
        raise StructuralCanonicalizeError("bucket count metadata mismatch")
    total_identities = 0
    for index in range(state.bucket_count):
        count = _validate_bucket_file(
            _bucket_file(state.spill_dir, index), state.bucket_count, index
        )
        if count != state.bucket_counts[index]:
            raise StructuralCanonicalizeError("bucket count metadata mismatch")
        total_identities += count
    if total_identities != state.positive_group_count:
        raise StructuralCanonicalizeError("bucket identity total does not match group count")
    if state.positive_row_count < state.positive_group_count or state.witness_row_count < 0:
        raise StructuralCanonicalizeError("invalid positive or witness counts")
    if state.positive_group_count == 0:
        if state.positive_row_count != 0:
            raise StructuralCanonicalizeError("empty state cannot carry positive rows")
        if state.first_boundary is not None or state.last_boundary is not None:
            raise StructuralCanonicalizeError("empty state cannot carry boundaries")
        return
    for name, boundary in (
        ("first", state.first_boundary),
        ("last", state.last_boundary),
    ):
        if not isinstance(boundary, dict) or "identity" not in boundary or "core" not in boundary:
            raise StructuralCanonicalizeError(f"{name} boundary is missing")
        identity = boundary["identity"]
        if not isinstance(identity, int) or isinstance(identity, bool) or identity <= 0:
            raise StructuralCanonicalizeError(f"{name} boundary identity is not positive")
        bucket_path = _bucket_file(state.spill_dir, bucket_index(identity, state.bucket_count))
        if not _bucket_contains_identity(bucket_path, state.bucket_count, identity):
            raise StructuralCanonicalizeError(f"{name} boundary identity is not in its bucket")


def _merge_states(
    left: CanonicalizeState,
    right: CanonicalizeState,
    *,
    subtract_boundary_row: bool,
) -> CanonicalizeState:
    _validate_state(left)
    _validate_state(right)
    if left.bucket_count != right.bucket_count:
        raise StructuralCanonicalizeError(
            "bucket_count mismatch between canonicalization states"
        )

    left_positive = left.positive_group_count > 0
    right_positive = right.positive_group_count > 0
    shared_boundary = (
        left_positive
        and right_positive
        and left.last_boundary["identity"] == right.first_boundary["identity"]
    )
    if shared_boundary and left.last_boundary["core"] != right.first_boundary["core"]:
        raise StructuralCanonicalizeError(
            "measurement core fields disagree at boundary identity"
        )
    allowed_identity = left.last_boundary["identity"] if shared_boundary else None

    if not left_positive:
        shutil.rmtree(left.spill_dir, ignore_errors=True)
        return right
    if not right_positive:
        shutil.rmtree(right.spill_dir, ignore_errors=True)
        return left

    merged_dir = _make_spill_dir()
    merged_counts: list[int] = []
    for index in range(left.bucket_count):
        count = _stream_merge_sorted_unique_files(
            _bucket_file(left.spill_dir, index),
            _bucket_file(right.spill_dir, index),
            _bucket_file(merged_dir, index),
            allowed_identity=allowed_identity,
        )
        merged_counts.append(count)

    if shared_boundary:
        group_count = left.positive_group_count + right.positive_group_count - 1
        row_count = left.positive_row_count + right.positive_row_count - (
            1 if subtract_boundary_row else 0
        )
    else:
        group_count = left.positive_group_count + right.positive_group_count
        row_count = left.positive_row_count + right.positive_row_count

    merged_state = CanonicalizeState(
        bucket_count=left.bucket_count,
        spill_dir=merged_dir,
        bucket_counts=tuple(merged_counts),
        positive_row_count=row_count,
        positive_group_count=group_count,
        witness_row_count=left.witness_row_count + right.witness_row_count,
        first_boundary=left.first_boundary,
        last_boundary=right.last_boundary,
    )
    shutil.rmtree(left.spill_dir, ignore_errors=True)
    shutil.rmtree(right.spill_dir, ignore_errors=True)
    _validate_state(merged_state)
    return merged_state


def execute_structural_canonicalize(
    paths: list[str | Path],
    params: dict[str, Any] | StructuralCanonicalizeParams,
) -> CanonicalizeState:
    if not paths:
        raise StructuralCanonicalizeError("at least one parquet input is required")
    model = _validated(params)
    state: CanonicalizeState | None = None
    for path in paths:
        part = _single_input_state(path, model)
        state = part if state is None else _merge_states(state, part, subtract_boundary_row=False)
    assert state is not None
    return state


def merge_structural_canonicalize_states(
    left: CanonicalizeState,
    right: CanonicalizeState,
) -> CanonicalizeState:
    return _merge_states(left, right, subtract_boundary_row=True)


def read_bucket_values(state: CanonicalizeState, index: int) -> tuple[int, ...]:
    """Read one bucket into memory (tests and small artifacts only)."""
    values = tuple(_iter_uint64_path(_bucket_file(state.spill_dir, index)))
    _assert_bounded_materialization(len(values))
    return values


def states_equal(left: CanonicalizeState, right: CanonicalizeState) -> bool:
    try:
        _validate_state(left)
        _validate_state(right)
    except StructuralCanonicalizeError:
        return False
    fields = (
        left.bucket_count,
        left.bucket_counts,
        left.positive_row_count,
        left.positive_group_count,
        left.witness_row_count,
        left.first_boundary,
        left.last_boundary,
    )
    other_fields = (
        right.bucket_count,
        right.bucket_counts,
        right.positive_row_count,
        right.positive_group_count,
        right.witness_row_count,
        right.first_boundary,
        right.last_boundary,
    )
    if fields != other_fields:
        return False
    for index in range(left.bucket_count):
        if _bucket_file(left.spill_dir, index).read_bytes() != _bucket_file(
            right.spill_dir, index
        ).read_bytes():
            return False
    return True


def encode_bucket(bucket_count: int, bucket: BucketSet) -> bytes:
    header = _BUCKET_HEADER.pack(
        BUCKET_MAGIC,
        BUCKET_FORMAT_VERSION,
        bucket_count,
        bucket.bucket_index,
        len(bucket.values),
    )
    if not bucket.values:
        return header
    _assert_bounded_materialization(len(bucket.values))
    return header + struct.pack(f"<{len(bucket.values)}Q", *bucket.values)


def encode_bucket_from_path(
    bucket_count: int, bucket_index_value: int, path: Path, value_count: int
) -> bytes:
    header = _BUCKET_HEADER.pack(
        BUCKET_MAGIC,
        BUCKET_FORMAT_VERSION,
        bucket_count,
        bucket_index_value,
        value_count,
    )
    body_size = value_count * 8
    payload_size = _BUCKET_HEADER.size + body_size
    _assert_bucket_payload_bounded(payload_size)
    if value_count == 0:
        return header
    buffer = bytearray(header)
    with path.open("rb") as handle:
        remaining = body_size
        while remaining > 0:
            chunk = handle.read(min(8192 * 8, remaining))
            if not chunk:
                raise StructuralCanonicalizeError("bucket spill file is corrupt")
            buffer.extend(chunk)
            remaining -= len(chunk)
    return bytes(buffer)


def iter_state_bucket_payloads(state: CanonicalizeState):
    _validate_state(state)
    for index in range(state.bucket_count):
        yield encode_bucket_from_path(
            state.bucket_count,
            index,
            _bucket_file(state.spill_dir, index),
            state.bucket_counts[index],
        )


def encode_state_buckets(state: CanonicalizeState) -> tuple[bytes, ...]:
    """Test helper: materializes every bucket payload (not for production)."""
    return tuple(iter_state_bucket_payloads(state))


def decode_bucket(payload: bytes, bucket_count: int, expected_index: int) -> tuple[int, ...]:
    if len(payload) < _BUCKET_HEADER.size:
        raise StructuralCanonicalizeError("bucket artifact is truncated")
    magic, version, encoded_count, encoded_index, value_count = _BUCKET_HEADER.unpack_from(
        payload, 0
    )
    if magic != BUCKET_MAGIC:
        raise StructuralCanonicalizeError("bucket artifact magic mismatch")
    if version != BUCKET_FORMAT_VERSION:
        raise StructuralCanonicalizeError("bucket artifact version mismatch")
    if encoded_count != bucket_count:
        raise StructuralCanonicalizeError("bucket artifact bucket_count mismatch")
    if encoded_index != expected_index:
        raise StructuralCanonicalizeError("bucket artifact order mismatch")
    if len(payload) != _BUCKET_HEADER.size + value_count * 8:
        raise StructuralCanonicalizeError("bucket artifact length mismatch")
    values = (
        struct.unpack_from(f"<{value_count}Q", payload, _BUCKET_HEADER.size)
        if value_count
        else ()
    )
    previous = 0
    for value in values:
        if value <= previous:
            raise StructuralCanonicalizeError("bucket artifact values are not sorted unique")
        previous = value
    _assert_bounded_materialization(len(values))
    return values


def _write_bucket_payload(path: Path, payload: bytes, bucket_count: int, index: int) -> int:
    _assert_bucket_payload_bounded(len(payload))
    if len(payload) < _BUCKET_HEADER.size:
        raise StructuralCanonicalizeError("bucket artifact is truncated")
    magic, version, encoded_count, encoded_index, value_count = _BUCKET_HEADER.unpack_from(
        payload, 0
    )
    if magic != BUCKET_MAGIC:
        raise StructuralCanonicalizeError("bucket artifact magic mismatch")
    if version != BUCKET_FORMAT_VERSION:
        raise StructuralCanonicalizeError("bucket artifact version mismatch")
    if encoded_count != bucket_count or encoded_index != index:
        raise StructuralCanonicalizeError("bucket artifact order mismatch")
    if len(payload) != _BUCKET_HEADER.size + value_count * 8:
        raise StructuralCanonicalizeError("bucket artifact length mismatch")
    previous = 0
    with path.open("wb") as handle:
        for offset in range(_BUCKET_HEADER.size, len(payload), 8):
            value = _UINT64_PACK.unpack_from(payload, offset)[0]
            if value <= previous:
                raise StructuralCanonicalizeError("bucket artifact values are not sorted unique")
            if bucket_index(value, bucket_count) != index:
                raise StructuralCanonicalizeError("bucket value is not in its bucket")
            handle.write(_UINT64_PACK.pack(value & _UINT64_MASK))
            previous = value
    return value_count


def state_summary(state: CanonicalizeState) -> dict[str, Any]:
    _validate_state(state)
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "bucket_count": state.bucket_count,
        "bucket_format": BUCKET_FORMAT,
        "bucket_format_version": BUCKET_FORMAT_VERSION,
        "positive_row_count": state.positive_row_count,
        "positive_group_count": state.positive_group_count,
        "witness_row_count": state.witness_row_count,
        "first_boundary": state.first_boundary,
        "last_boundary": state.last_boundary,
        "bucket_counts": list(state.bucket_counts),
    }


def attach_bucket_refs(
    summary: dict[str, Any], bucket_refs: tuple[Any, ...]
) -> dict[str, Any]:
    if len(bucket_refs) != int(summary["bucket_count"]):
        raise StructuralCanonicalizeError("bucket reference count mismatch")
    enriched = dict(summary)
    enriched["bucket_refs"] = [
        dict(ref) if isinstance(ref, dict) else ref.model_dump(mode="json")
        for ref in bucket_refs
    ]
    return enriched


def _validated_summary_for_decode(summary: dict[str, Any]) -> tuple[int, tuple[int, ...]]:
    if not isinstance(summary, dict):
        raise StructuralCanonicalizeError("canonicalization summary must be an object")
    if summary.get("schema_version") != STATE_SCHEMA_VERSION:
        raise StructuralCanonicalizeError("canonicalization state schema mismatch")
    if summary.get("bucket_format") != BUCKET_FORMAT:
        raise StructuralCanonicalizeError("canonicalization bucket format mismatch")
    if summary.get("bucket_format_version") != BUCKET_FORMAT_VERSION:
        raise StructuralCanonicalizeError("canonicalization bucket format version mismatch")
    bucket_count = summary.get("bucket_count")
    if (
        not isinstance(bucket_count, int)
        or isinstance(bucket_count, bool)
        or bucket_count < BUCKET_COUNT_MIN
        or bucket_count > BUCKET_COUNT_MAX
        or bucket_count & (bucket_count - 1)
    ):
        raise StructuralCanonicalizeError("canonicalization bucket_count is invalid")
    bucket_counts = summary.get("bucket_counts")
    if not isinstance(bucket_counts, list) or len(bucket_counts) != bucket_count:
        raise StructuralCanonicalizeError("canonicalization bucket_counts are invalid")
    if not all(
        isinstance(count, int) and not isinstance(count, bool) and count >= 0
        for count in bucket_counts
    ):
        raise StructuralCanonicalizeError("canonicalization bucket_counts are invalid")
    for name in ("positive_row_count", "positive_group_count", "witness_row_count"):
        value = summary.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise StructuralCanonicalizeError(f"canonicalization {name} is invalid")
    return bucket_count, tuple(int(count) for count in bucket_counts)


def decode_state_from_bucket_payloads(
    summary: dict[str, Any],
    bucket_payloads: Iterator[bytes],
) -> CanonicalizeState:
    """Decode state from bucket payloads consumed one at a time (production path)."""
    bucket_count, expected_counts = _validated_summary_for_decode(summary)
    spill_dir = _make_spill_dir()
    written_counts: list[int] = []
    for index in range(bucket_count):
        try:
            payload = next(bucket_payloads)
        except StopIteration as exc:
            raise StructuralCanonicalizeError(
                "canonicalization bucket artifacts are incomplete"
            ) from exc
        written = _write_bucket_payload(
            _bucket_file(spill_dir, index), payload, bucket_count, index
        )
        if written != expected_counts[index]:
            raise StructuralCanonicalizeError("canonicalization bucket count mismatch")
        written_counts.append(written)
    if next(bucket_payloads, None) is not None:
        raise StructuralCanonicalizeError("canonicalization bucket artifacts are incomplete")
    state = CanonicalizeState(
        bucket_count=bucket_count,
        spill_dir=spill_dir,
        bucket_counts=tuple(written_counts),
        positive_row_count=int(summary["positive_row_count"]),
        positive_group_count=int(summary["positive_group_count"]),
        witness_row_count=int(summary["witness_row_count"]),
        first_boundary=summary.get("first_boundary"),
        last_boundary=summary.get("last_boundary"),
    )
    _validate_state(state)
    return state


def decode_state(
    summary: dict[str, Any], bucket_payloads: tuple[bytes, ...]
) -> CanonicalizeState:
    """Test helper: decode from a fully materialized bucket payload tuple."""
    if len(bucket_payloads) != int(summary.get("bucket_count", -1)):
        raise StructuralCanonicalizeError("canonicalization bucket artifacts are incomplete")
    return decode_state_from_bucket_payloads(summary, iter(bucket_payloads))
