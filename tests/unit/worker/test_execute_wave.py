from pathlib import Path

import pytest

from portable_batch_execution.data_plane import LocalFilesystemDataPlane
from portable_batch_execution.worker import execute_public_wave


def test_public_worker_executes_committed_wave_and_only_appends_attempts(tmp_path):
    attempts = execute_public_wave("wave-0000", state_root=tmp_path)

    assert len(attempts) == 1
    assert attempts[0].status == "succeeded"
    assert attempts[0].output_refs
    state = LocalFilesystemDataPlane(tmp_path)
    assert state.read_attempts("public-synthetic-rolling-v1") == attempts
    assert state.read_manifest("public-synthetic-rolling-v1") is None


@pytest.mark.parametrize("wave_id", ["", "wave-0001", "../wave-0000", "wave-0000; echo x", "$(whoami)"])
def test_public_worker_rejects_non_allowlisted_or_executable_wave_input(wave_id):
    with pytest.raises(ValueError):
        execute_public_wave(wave_id)


def test_execute_wave_workflow_invokes_worker_with_environment_boundary():
    workflow = (Path(__file__).parents[3] / ".github" / "workflows" / "execute-wave.yml").read_text()

    assert "python -m portable_batch_execution.worker.execute_wave" in workflow
    assert '"$PBE_WAVE_ID"' in workflow
    assert "PBE_WAVE_ID: ${{ inputs.wave_id }}" in workflow
    assert "pytest" not in workflow


def test_execute_wave_workflow_bootstraps_base_then_conditional_extras():
    workflow = (Path(__file__).parents[3] / ".github" / "workflows" / "execute-wave.yml").read_text()

    assert "uv sync --no-dev" in workflow
    assert "--dev" not in workflow.replace("--no-dev", "")
    assert "python -m portable_batch_execution.worker.runtime_profile" in workflow
    assert "uv sync --no-dev --extra tabular" in workflow
    assert "uv sync --no-dev --extra ml" in workflow
    assert "uv sync --no-dev --extra distilbert" in workflow
    assert workflow.index("uv sync --no-dev\n") < workflow.index(
        "python -m portable_batch_execution.worker.runtime_profile"
    )
    assert workflow.index(
        "python -m portable_batch_execution.worker.runtime_profile"
    ) < workflow.index("uv sync --no-dev --extra tabular")
    assert "sudo apt-get install -y --no-install-recommends ffmpeg" in workflow
    assert workflow.count("sudo apt-get install") == 1
