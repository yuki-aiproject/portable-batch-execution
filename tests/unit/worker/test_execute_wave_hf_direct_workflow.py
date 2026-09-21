from pathlib import Path

_WORKFLOW = (
    Path(__file__).parents[3] / ".github" / "workflows" / "execute-wave.yml"
)


def _text() -> str:
    return _WORKFLOW.read_text(encoding="utf-8")


def test_hf_direct_mode_is_a_closed_dispatch_selection():
    workflow = _text()
    assert "hf_direct:" in workflow
    assert "type: boolean" in workflow
    assert "inputs.hf_direct && 'hf-direct'" in workflow
    assert "hf_bucket:" in workflow
    assert "hf_prefix:" in workflow
    assert "hf_object_layout:" in workflow
    assert "default: sha256-flat.v1" in workflow


def test_hf_direct_jobs_inject_fixed_secret_as_process_env_only():
    workflow = _text()
    assert (
        "HF_SYSTEM_TRADING_DATA_RW_TOKEN: ${{ inputs.hf_direct && "
        "secrets.HF_SYSTEM_TRADING_DATA_RW_TOKEN || '' }}" in workflow
    )
    assert (
        "HF_TOKEN: ${{ inputs.hf_direct && secrets.HF_SYSTEM_TRADING_DATA_RW_TOKEN || '' }}"
        in workflow
    )
    assert "env.HF_SYSTEM_TRADING_DATA_RW_TOKEN" not in workflow
    for forbidden in ("gh api", "curl", "inputs.token", "hf_token:", "token:"):
        assert forbidden not in workflow


def test_hf_direct_dispatch_carries_only_bounded_metadata_not_payloads():
    workflow = _text()
    assert "PBE_MODE: ${{ inputs.hf_direct && 'hf-direct'" in workflow
    assert "PBE_HF_BUCKET: ${{ inputs.hf_direct && inputs.hf_bucket || '' }}" in workflow
    assert "PBE_HF_PREFIX: ${{ inputs.hf_direct && inputs.hf_prefix || '' }}" in workflow
    assert "PBE_RUN_ID: ${{ inputs.run_id }}" in workflow
    assert "PBE_WAVE_ID: ${{ inputs.wave_id }}" in workflow


def test_hf_direct_installs_pinned_huggingface_hub_before_resolution():
    workflow = _text()
    install = "uv sync --no-dev --extra hf-direct"
    assert install in workflow
    assert workflow.index(install) < workflow.index(
        "python -m portable_batch_execution.worker.runtime_profile"
    )
    assert "uv tool install" not in workflow
    assert "huggingface_hub[cli]" not in workflow
    assert "--mode hf-direct" in workflow
    assert workflow.count("--mode hf-direct") == 1


def test_hf_direct_workflow_keeps_existing_modes_and_no_test_runner():
    workflow = _text()
    assert "--mode private" in workflow
    assert "--mode public" in workflow
    assert "pytest" not in workflow


def test_hf_direct_profile_sync_retains_hf_extra_and_mode_guard():
    workflow = _text()
    guard = 'if [ "$PBE_MODE" = "hf-direct" ]; then'
    assert workflow.count(guard) == 3
    assert "uv sync --no-dev --extra tabular --extra hf-direct" in workflow
    assert "uv sync --no-dev --extra ml --extra hf-direct" in workflow
    assert "uv sync --no-dev --extra distilbert --extra hf-direct" in workflow
    assert 'if [ "" = "hf-direct" ]; then' not in workflow
