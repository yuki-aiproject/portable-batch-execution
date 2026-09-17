"""Generic replay reduction primitives (structural canonicalize, event windows)."""

from .canonicalize import (
    BUCKET_FORMAT,
    BUCKET_FORMAT_VERSION,
    BUCKET_MEDIA_TYPE,
    STATE_SCHEMA_VERSION,
    BucketSet,
    CanonicalizeState,
    StructuralCanonicalizeError,
    attach_bucket_refs,
    bucket_index,
    decode_bucket,
    decode_state,
    decode_state_from_bucket_payloads,
    encode_bucket,
    encode_state_buckets,
    execute_structural_canonicalize,
    merge_structural_canonicalize_states,
    read_bucket_values,
    splitmix64,
    state_summary,
    states_equal,
)
from .causal_grid import execute_causal_grid_extract
from .event_window import execute_event_window_extract
from .pack import ReplayReductionPack

__all__ = [
    "BUCKET_FORMAT",
    "BUCKET_FORMAT_VERSION",
    "BUCKET_MEDIA_TYPE",
    "STATE_SCHEMA_VERSION",
    "BucketSet",
    "CanonicalizeState",
    "ReplayReductionPack",
    "StructuralCanonicalizeError",
    "attach_bucket_refs",
    "bucket_index",
    "decode_bucket",
    "decode_state",
    "decode_state_from_bucket_payloads",
    "encode_bucket",
    "encode_state_buckets",
    "execute_causal_grid_extract",
    "execute_event_window_extract",
    "execute_structural_canonicalize",
    "merge_structural_canonicalize_states",
    "read_bucket_values",
    "splitmix64",
    "state_summary",
    "states_equal",
]

