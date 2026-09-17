"""Clean-process assertions that a tabular bootstrap loads no heavy domain stack.

Each check runs in a fresh interpreter so that unrelated test imports cannot
mask an eager domain import.  The worker and resolver must be importable -- and
must resolve a tabular profile -- without loading scikit-learn, torch,
transformers, or the FFmpeg-backed media pack.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

_HEAVY_MODULES = (
    "sklearn",
    "torch",
    "transformers",
    "portable_batch_execution.packs.ml.pack",
    "portable_batch_execution.packs.media",
)

_TABULAR_PROBE = """
import sys

{imports}

loaded = [name for name in {heavy!r} if name in sys.modules]
assert loaded == [], f"unexpected heavy modules loaded: {{loaded}}"
print("clean")
"""


def _run_clean_process(source: str) -> str:
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


@pytest.mark.parametrize(
    "imports",
    (
        "import portable_batch_execution.worker.execute_wave",
        "from portable_batch_execution.worker import runtime_profile",
        (
            "from portable_batch_execution.worker import runtime_profile\n"
            "assert runtime_profile.resolve_public_profile('wave-0000').profile == 'tabular'"
        ),
        "import portable_batch_execution.packs",
        (
            "import portable_batch_execution.packs as packs\n"
            "assert 'TabularPack' in dir(packs) and 'MLPack' in dir(packs)"
        ),
    ),
)
def test_tabular_bootstrap_path_imports_no_heavy_domain_stack(imports):
    source = _TABULAR_PROBE.format(imports=imports, heavy=_HEAVY_MODULES)
    assert _run_clean_process(source) == "clean"


def test_worker_module_does_not_eagerly_import_domain_packs():
    source = """
import sys
import portable_batch_execution.worker.execute_wave  # noqa: F401

for name in (
    "portable_batch_execution.packs.tabular",
    "portable_batch_execution.packs.ml.pack",
    "portable_batch_execution.packs.media",
    "portable_batch_execution.packs.acquisition",
    "portable_batch_execution.packs.replay_eval",
):
    assert name not in sys.modules, name
print("clean")
"""
    assert _run_clean_process(source) == "clean"


def test_lazy_pack_exports_preserve_public_surface():
    source = """
from portable_batch_execution.packs import (
    AcquisitionPack,
    FakeEncoder,
    MLPack,
    MediaPack,
    ReplayEvalPack,
    TabularPack,
    rolling_halo,
    rolling_halo_rows,
)

assert TabularPack().pack_id == "tabular-batch"
assert MLPack().pack_id == "ml-batch"
assert MediaPack().pack_id == "media-batch"
assert AcquisitionPack().pack_id == "acquisition-batch"
assert rolling_halo(3) == 2
assert rolling_halo_rows({"column": "value", "window_size": 4, "output_column": "out"}) == 3
assert callable(FakeEncoder)
assert ReplayEvalPack is not None
print("clean")
"""
    assert _run_clean_process(source) == "clean"
