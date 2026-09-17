"""Closed worker entry points for public synthetic wave execution.

``execute_public_wave`` resolves lazily (PEP 562) so that importing this package
-- including running ``python -m portable_batch_execution.worker.runtime_profile``
-- does not import the worker entry point or any domain pack.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["execute_public_wave"]


def __getattr__(name: str) -> Any:
    if name != "execute_public_wave":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = importlib.import_module(".execute_wave", __name__).execute_public_wave
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
