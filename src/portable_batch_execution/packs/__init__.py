"""Public exports for the v1 closed domain pack implementations.

Every export resolves lazily (PEP 562) so importing this package never forces an
unrelated domain dependency -- scikit-learn, torch, transformers, or the
FFmpeg-backed media pack -- to be importable for a tabular-only job.  The public
names, including ``from portable_batch_execution.packs import TabularPack``,
keep working unchanged.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "AcquisitionPack",
    "FakeEncoder",
    "MLPack",
    "MediaPack",
    "ReplayEvalPack",
    "TabularPack",
    "rolling_halo",
    "rolling_halo_rows",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "AcquisitionPack": (".acquisition", "AcquisitionPack"),
    "FakeEncoder": (".ml", "FakeEncoder"),
    "MLPack": (".ml", "MLPack"),
    "MediaPack": (".media", "MediaPack"),
    "ReplayEvalPack": (".replay_eval", "ReplayEvalPack"),
    "TabularPack": (".tabular", "TabularPack"),
    "rolling_halo": (".tabular", "rolling_halo"),
    "rolling_halo_rows": (".tabular", "rolling_halo_rows"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
