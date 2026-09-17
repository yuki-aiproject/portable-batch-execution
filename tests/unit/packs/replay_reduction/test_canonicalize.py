import itertools
import json

import polars as pl
import pytest

from portable_batch_execution.packs.replay_reduction.canonicalize import (
    BUCKET_FORMAT,
    BUCKET_FORMAT_VERSION,
    STATE_SCHEMA_VERSION,
    StructuralCanonicalizeError,
    bucket_index,
    decode_bucket,
    decode_state,
    encode_state_buckets,
    execute_structural_canonicalize,
    merge_structural_canonicalize_states,
    read_bucket_values,
    state_summary,
    states_equal,
)

_PARAMS = {
    "schema_version": "pbe.replay.structural-canonicalize.v1",
    "identity_source_column": "identity",
    "identity_normalized_column": "identity_norm",
    "measurement_core_fields": ["price"],
    "sentinel": {"identity_equals": -1},
}


def _paths(tmp_path, *row_groups):
    paths = []
    for index, rows in enumerate(row_groups):
        path = tmp_path / f"part-{index}.parquet"
        pl.DataFrame(rows).write_parquet(path)
        paths.append(path)
    return paths


_STATE_SEQUENCE = itertools.count()


def _state(tmp_path, rows, params=_PARAMS):
    path = tmp_path / f"state-{next(_STATE_SEQUENCE)}.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return execute_structural_canonicalize([path], params)


def _roundtrip(state):
    decoded = decode_state(state_summary(state), encode_state_buckets(state))
    assert states_equal(decoded, state)
    return decoded


def test_many_distinct_identities_keep_constant_summary_cardinality(tmp_path):
    rows = [
        {"identity": i, "identity_norm": str(i), "price": float(i)}
        for i in range(1, 5001)
    ]
    state = _state(tmp_path, rows)
    summary = state_summary(state)
    assert summary["schema_version"] == STATE_SCHEMA_VERSION
    assert summary["bucket_count"] == 256
    assert summary["positive_group_count"] == 5000
    assert len(json.dumps(summary)) < 4000
    assert len(summary["bucket_counts"]) == summary["bucket_count"]


def test_summary_cardinality_independent_of_identity_count(tmp_path):
    small = state_summary(
        _state(
            tmp_path,
            [{"identity": i, "identity_norm": str(i), "price": float(i)} for i in range(1, 11)],
        )
    )
    large = state_summary(
        _state(
            tmp_path,
            [{"identity": i, "identity_norm": str(i), "price": float(i)} for i in range(1, 10001)],
        )
    )
    assert sorted(small) == sorted(large)
    assert len(small["bucket_counts"]) == len(large["bucket_counts"]) == 256
    assert len(json.dumps(small)) < 4000
    assert len(json.dumps(large)) < 4000


def test_bucket_artifacts_carry_exact_identities(tmp_path):
    rows = [{"identity": value, "identity_norm": str(value), "price": 1.0} for value in (5, 1, 9, 300)]
    state = _state(tmp_path, rows)
    carried = {
        value
        for index in range(state.bucket_count)
        for value in read_bucket_values(state, index)
    }
    assert carried == {5, 1, 9, 300}
    for index in range(state.bucket_count):
        values = read_bucket_values(state, index)
        assert list(values) == sorted(set(values))
    for identity in carried:
        assert identity in read_bucket_values(state, bucket_index(identity, state.bucket_count))


def test_non_monotonic_unique_identities_pass(tmp_path):
    state = _state(
        tmp_path,
        [
            {"identity": 9, "identity_norm": "9", "price": 9.0},
            {"identity": 1, "identity_norm": "1", "price": 1.0},
            {"identity": 5, "identity_norm": "5", "price": 5.0},
        ],
    )
    assert state.positive_row_count == 3
    assert state.positive_group_count == 3
    assert state.first_boundary["identity"] == 9
    assert state.last_boundary["identity"] == 5


def test_adjacent_repeated_identity_with_matching_core_is_one_group(tmp_path):
    state = _state(
        tmp_path,
        [
            {"identity": 3, "identity_norm": "3", "price": 3.0},
            {"identity": 3, "identity_norm": "3", "price": 3.0},
            {"identity": 4, "identity_norm": "4", "price": 4.0},
        ],
    )
    assert state.positive_row_count == 3
    assert state.positive_group_count == 2


def test_ordered_disjoint_shards_pass(tmp_path):
    state = execute_structural_canonicalize(
        _paths(
            tmp_path,
            [
                {"identity": 1, "identity_norm": "1", "price": 1.0},
                {"identity": 2, "identity_norm": "2", "price": 2.0},
            ],
            [{"identity": 3, "identity_norm": "3", "price": 3.0}],
        ),
        _PARAMS,
    )
    assert state.positive_group_count == 3
    assert state.positive_row_count == 3


def test_shared_boundary_split_across_adjacent_shards_passes(tmp_path):
    state = execute_structural_canonicalize(
        _paths(
            tmp_path,
            [{"identity": 1, "identity_norm": "1", "price": 10.0}],
            [
                {"identity": 1, "identity_norm": "1", "price": 10.0},
                {"identity": 2, "identity_norm": "2", "price": 20.0},
            ],
        ),
        _PARAMS,
    )
    assert state.positive_group_count == 2
    assert state.positive_row_count == 3
    assert state.last_boundary["identity"] == 2


def test_shared_boundary_core_conflict_fails(tmp_path):
    with pytest.raises(StructuralCanonicalizeError):
        execute_structural_canonicalize(
            _paths(
                tmp_path,
                [{"identity": 1, "identity_norm": "1", "price": 1.0}],
                [{"identity": 1, "identity_norm": "1", "price": 2.0}],
            ),
            _PARAMS,
        )


def test_non_contiguous_recurrence_within_one_input_fails(tmp_path):
    with pytest.raises(StructuralCanonicalizeError):
        execute_structural_canonicalize(
            _paths(
                tmp_path,
                [
                    {"identity": 1, "identity_norm": "1", "price": 1.0},
                    {"identity": 2, "identity_norm": "2", "price": 2.0},
                    {"identity": 1, "identity_norm": "1", "price": 1.0},
                ],
            ),
            _PARAMS,
        )


def test_non_contiguous_recurrence_across_source_files_fails(tmp_path):
    with pytest.raises(StructuralCanonicalizeError):
        execute_structural_canonicalize(
            _paths(
                tmp_path,
                [{"identity": 1, "identity_norm": "1", "price": 1.0}],
                [{"identity": 2, "identity_norm": "2", "price": 2.0}],
                [{"identity": 1, "identity_norm": "1", "price": 1.0}],
            ),
            _PARAMS,
        )


def test_non_contiguous_recurrence_across_merge_waves_fails(tmp_path):
    left = _state(tmp_path, [{"identity": 1, "identity_norm": "1", "price": 1.0}])
    right = _state(
        tmp_path,
        [
            {"identity": 2, "identity_norm": "2", "price": 2.0},
            {"identity": 1, "identity_norm": "1", "price": 1.0},
        ],
    )
    with pytest.raises(StructuralCanonicalizeError):
        merge_structural_canonicalize_states(left, right)


def test_group_split_exactly_across_adjacent_wave_boundary_dedups_once(tmp_path):
    wave_a = _state(tmp_path, [{"identity": 1, "identity_norm": "1", "price": 5.0}])
    wave_b = _state(
        tmp_path,
        [
            {"identity": 1, "identity_norm": "1", "price": 5.0},
            {"identity": 2, "identity_norm": "2", "price": 6.0},
        ],
    )
    merged = merge_structural_canonicalize_states(wave_a, wave_b)
    assert merged.positive_group_count == 2
    assert merged.positive_row_count == 2
    assert merged.last_boundary["identity"] == 2
    assert _roundtrip(merged).positive_group_count == 2


def test_mismatching_boundary_core_fails(tmp_path):
    wave_a = _state(tmp_path, [{"identity": 1, "identity_norm": "1", "price": 1.0}])
    wave_b = _state(tmp_path, [{"identity": 1, "identity_norm": "1", "price": 2.0}])
    with pytest.raises(StructuralCanonicalizeError):
        merge_structural_canonicalize_states(wave_a, wave_b)


def test_merge_rejects_bucket_count_mismatch(tmp_path):
    left = _state(tmp_path, [{"identity": 1, "identity_norm": "1", "price": 1.0}])
    right = _state(
        tmp_path,
        [{"identity": 2, "identity_norm": "2", "price": 2.0}],
        {**_PARAMS, "bucket_count": 64},
    )
    with pytest.raises(StructuralCanonicalizeError):
        merge_structural_canonicalize_states(left, right)


def test_bucket_count_must_be_power_of_two_in_range():
    from portable_batch_execution.packs.replay_reduction.models import (
        StructuralCanonicalizeParams,
    )

    with pytest.raises(ValueError):
        StructuralCanonicalizeParams.model_validate({**_PARAMS, "bucket_count": 3})
    with pytest.raises(ValueError):
        StructuralCanonicalizeParams.model_validate({**_PARAMS, "bucket_count": 0})
    assert (
        StructuralCanonicalizeParams.model_validate({**_PARAMS, "bucket_count": 512}).bucket_count
        == 512
    )


def test_structural_canonicalize_counts_sentinel_witness_rows(tmp_path):
    paths = _paths(tmp_path, [{"identity": -1, "identity_norm": "x", "price": 0.0}])
    result = execute_structural_canonicalize(paths, _PARAMS)
    assert result.witness_row_count == 1


_EXACT_PARAMS = {
    "schema_version": "pbe.replay.structural-canonicalize.v1",
    "identity_source_column": "identity",
    "identity_normalized_column": "identity_norm",
    "measurement_core_fields": ["price"],
    "sentinel": {
        "identity_equals": -1,
        "exact_match_fields": {"marker_kind": "boundary", "marker_rank": 7},
    },
}


def test_sentinel_exact_match_accepts_matching_witness(tmp_path):
    paths = _paths(
        tmp_path,
        [
            {
                "identity": -1,
                "identity_norm": "x",
                "price": 0.0,
                "marker_kind": "boundary",
                "marker_rank": 7,
            },
            {"identity": 1, "identity_norm": "1", "price": 1.0, "marker_kind": "a", "marker_rank": 1},
        ],
    )
    result = execute_structural_canonicalize(paths, _EXACT_PARAMS)
    assert result.witness_row_count == 1


def test_sentinel_exact_match_wrong_field_value_fails_closed(tmp_path):
    paths = _paths(
        tmp_path,
        [
            {
                "identity": -1,
                "identity_norm": "x",
                "price": 0.0,
                "marker_kind": "boundary",
                "marker_rank": 8,
            }
        ],
    )
    with pytest.raises(StructuralCanonicalizeError):
        execute_structural_canonicalize(paths, _EXACT_PARAMS)


def test_sentinel_exact_match_missing_predicate_field_fails_closed(tmp_path):
    paths = _paths(
        tmp_path,
        [{"identity": -1, "identity_norm": "x", "price": 0.0, "marker_kind": "boundary"}],
    )
    with pytest.raises(StructuralCanonicalizeError):
        execute_structural_canonicalize(paths, _EXACT_PARAMS)


def test_sentinel_exact_match_null_predicate_value_fails_closed(tmp_path):
    paths = _paths(
        tmp_path,
        [
            {
                "identity": -1,
                "identity_norm": "x",
                "price": 0.0,
                "marker_kind": "boundary",
                "marker_rank": None,
            }
        ],
    )
    with pytest.raises(StructuralCanonicalizeError):
        execute_structural_canonicalize(paths, _EXACT_PARAMS)


@pytest.mark.parametrize(
    "rows",
    [
        [{"identity": None, "identity_norm": "1", "price": 1.0}],
        [{"identity": 0, "identity_norm": "0", "price": 1.0}],
        [{"identity": -2, "identity_norm": "-2", "price": 1.0}],
    ],
)
def test_structural_canonicalize_rejects_missing_or_nonpositive(tmp_path, rows):
    with pytest.raises(StructuralCanonicalizeError):
        execute_structural_canonicalize(_paths(tmp_path, rows), _PARAMS)


def test_structural_canonicalize_rejects_core_field_disagreement_within_group(tmp_path):
    with pytest.raises(StructuralCanonicalizeError):
        execute_structural_canonicalize(
            _paths(
                tmp_path,
                [
                    {"identity": 1, "identity_norm": "1", "price": 1.0},
                    {"identity": 1, "identity_norm": "1", "price": 2.0},
                ],
            ),
            _PARAMS,
        )


def test_singleton_positive_rejects_null_normalized_identity(tmp_path):
    with pytest.raises(StructuralCanonicalizeError, match="identity normalized column mismatch"):
        execute_structural_canonicalize(
            _paths(tmp_path, [{"identity": 1, "identity_norm": None, "price": 1.0}]),
            _PARAMS,
        )


def test_singleton_positive_rejects_null_measurement_core_field(tmp_path):
    with pytest.raises(StructuralCanonicalizeError, match="measurement core fields disagree"):
        execute_structural_canonicalize(
            _paths(tmp_path, [{"identity": 1, "identity_norm": "1", "price": None}]),
            _PARAMS,
        )


def test_multi_shard_path_rejects_null_normalized_identity(tmp_path):
    with pytest.raises(StructuralCanonicalizeError, match="identity normalized column mismatch"):
        execute_structural_canonicalize(
            _paths(
                tmp_path,
                [{"identity": 1, "identity_norm": "1", "price": 1.0}],
                [{"identity": 2, "identity_norm": None, "price": 2.0}],
            ),
            _PARAMS,
        )


def test_multi_shard_path_rejects_null_measurement_core_field(tmp_path):
    with pytest.raises(StructuralCanonicalizeError, match="measurement core fields disagree"):
        execute_structural_canonicalize(
            _paths(
                tmp_path,
                [{"identity": 1, "identity_norm": "1", "price": 1.0}],
                [{"identity": 2, "identity_norm": "2", "price": None}],
            ),
            _PARAMS,
        )


def test_sentinel_and_positive_rows_still_pass_after_identity_core_guards(tmp_path):
    result = execute_structural_canonicalize(
        _paths(
            tmp_path,
            [
                {"identity": -1, "identity_norm": "x", "price": 0.0},
                {"identity": 1, "identity_norm": "1", "price": 1.0},
            ],
        ),
        _PARAMS,
    )
    assert result.witness_row_count == 1
    assert result.positive_group_count == 1


@pytest.mark.parametrize(
    "identity_value,schema",
    [
        (1.5, {"identity": pl.Float64}),
        (True, {"identity": pl.Boolean}),
        (float("nan"), {"identity": pl.Float64}),
        (float("inf"), {"identity": pl.Float64}),
    ],
)
def test_rejects_lossy_or_non_integer_source_identity_singleton(
    tmp_path, identity_value, schema
):
    path = tmp_path / "part.parquet"
    pl.DataFrame(
        [{"identity": identity_value, "identity_norm": "1", "price": 1.0}],
        schema={
            **schema,
            "identity_norm": pl.Utf8,
            "price": pl.Float64,
        },
    ).write_parquet(path)
    with pytest.raises(StructuralCanonicalizeError, match="identity is missing or not positive"):
        execute_structural_canonicalize([path], _PARAMS)


def test_multi_shard_rejects_lossy_source_identity(tmp_path):
    good = tmp_path / "good.parquet"
    pl.DataFrame([{"identity": 1, "identity_norm": "1", "price": 1.0}]).write_parquet(good)
    bad = tmp_path / "bad.parquet"
    pl.DataFrame(
        [{"identity": 2.5, "identity_norm": "2", "price": 2.0}],
        schema={"identity": pl.Float64, "identity_norm": pl.Utf8, "price": pl.Float64},
    ).write_parquet(bad)
    with pytest.raises(StructuralCanonicalizeError, match="identity is missing or not positive"):
        execute_structural_canonicalize([good, bad], _PARAMS)


def test_roundtrip_preserves_state_across_binary_buckets(tmp_path):
    state = _state(
        tmp_path,
        [{"identity": value, "identity_norm": str(value), "price": float(value)} for value in (7, 3, 11, 300)],
    )
    decoded = decode_state(state_summary(state), encode_state_buckets(state))
    assert states_equal(decoded, state)


def test_malformed_bucket_state_fails_closed(tmp_path):
    state = _state(tmp_path, [{"identity": 1, "identity_norm": "1", "price": 1.0}])
    summary = state_summary(state)
    buckets = list(encode_state_buckets(state))

    truncated = list(buckets)
    truncated[0] = truncated[0][:6]
    with pytest.raises(StructuralCanonicalizeError):
        decode_state(summary, tuple(truncated))

    reordered = list(reversed(buckets))
    with pytest.raises(StructuralCanonicalizeError):
        decode_state(summary, tuple(reordered))

    with pytest.raises(StructuralCanonicalizeError):
        decode_state({**summary, "bucket_format": "pbe.replay.exact-id-bucket.v0"}, tuple(buckets))

    with pytest.raises(StructuralCanonicalizeError):
        decode_state({**summary, "positive_group_count": 999}, tuple(buckets))

    inflated = list(summary["bucket_counts"])
    inflated[0] += 1
    with pytest.raises(StructuralCanonicalizeError):
        decode_state({**summary, "bucket_counts": inflated}, tuple(buckets))

    with pytest.raises(StructuralCanonicalizeError):
        decode_state(
            {**summary, "first_boundary": {"identity": 123456, "core": {"price": 1.0}}},
            tuple(buckets),
        )

    with pytest.raises(StructuralCanonicalizeError):
        decode_state(summary, tuple(buckets[:-1]))


def test_decode_bucket_rejects_unsorted_values():
    import struct

    header = struct.Struct("<8sHIIQ")
    payload = header.pack(b"PBEBKT01", BUCKET_FORMAT_VERSION, 256, 0, 2) + struct.pack("<2Q", 5, 3)
    with pytest.raises(StructuralCanonicalizeError):
        decode_bucket(payload, 256, 0)
    assert BUCKET_FORMAT == "pbe.replay.exact-id-bucket.v1"

