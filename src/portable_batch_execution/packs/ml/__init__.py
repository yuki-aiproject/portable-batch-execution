"""Small, offline scikit-learn workloads for the ML batch pack.

Exports resolve lazily (PEP 562) so importing this package -- or the
dependency-free char-wb scoring module inside it -- never requires
scikit-learn to be installed.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["FakeEncoder", "MLPack", "execute_char_wb_tfidf_logistic_score"]

_EXPORTS: dict[str, tuple[str, str]] = {
    "FakeEncoder": (".pack", "FakeEncoder"),
    "MLPack": (".pack", "MLPack"),
    "execute_char_wb_tfidf_logistic_score": (
        ".char_wb_tfidf_logistic_score",
        "execute_char_wb_tfidf_logistic_score",
    ),
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
