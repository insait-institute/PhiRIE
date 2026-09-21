from pathlib import Path

import pytest

from run.sim_recon_sim.build_isolated import allocated_devices, CODE_DIRECTORIES, CODE_FILES


def test_physical_scheduler_index_not_visible_zero():
    paths = allocated_devices({"SLURM_JOB_GPUS": "3", "CUDA_VISIBLE_DEVICES": "3"})
    assert "/dev/nvidia3" in paths
    assert "/dev/nvidia0" not in paths


@pytest.mark.parametrize("value", ["GPU-uuid", "", "0,1", "0-7", "3-2", "../3"])
def test_no_unallocated_gpu_fallback(value):
    with pytest.raises(ValueError):
        allocated_devices({"SLURM_JOB_GPUS": value})


def test_constructor_code_allowlist_excludes_native_adapter_and_data():
    for member in (*CODE_DIRECTORIES, *CODE_FILES):
        assert "adapters" not in Path(member).parts
        assert "outputs" not in Path(member).parts
        assert "third_party" not in Path(member).parts
        assert member not in {".", "robo", "agents", "run"}


def test_container_slurm_nvml_and_minor_indices_differ():
    from robo.roundtrip.capture import slurm_device_minors
    uuid = "GPU-9d07d567-97b1-7ded-dafc-9e49be4decbf"
    def query(command, text):
        assert command == ["nvidia-smi", "-q", "-x", "--id=" + uuid]
        return f"<nvidia_smi_log><gpu><uuid>{uuid}</uuid><minor_number>3</minor_number></gpu></nvidia_smi_log>"
    assert slurm_device_minors({"SLURM_STEP_GPUS": "1", "CUDA_VISIBLE_DEVICES": uuid}, query=query) == {3}
    with pytest.raises(ValueError, match="UUID count"):
        slurm_device_minors({"SLURM_STEP_GPUS": "1,2", "CUDA_VISIBLE_DEVICES": uuid}, query=query)
    with pytest.raises(ValueError, match="unique physical"):
        slurm_device_minors({"SLURM_STEP_GPUS": "1", "CUDA_VISIBLE_DEVICES": uuid},
                            query=lambda *a, **kw: "<nvidia_smi_log/>")


def test_explicit_shared_cpu_stage_never_gpu_fallback():
    from run.sim_recon_sim.build_isolated import cpu_stage
    assert cpu_stage({'shared_candidates':{'phase':'select'}})
    assert cpu_stage({'observed_surface':{}})
    assert not cpu_stage({'shared_candidates':{'phase':'rvg'}})
    assert not cpu_stage({'shared_candidates':{}})
    with pytest.raises(ValueError,match='unsupported'):cpu_stage({'shared_candidates':{'phase':'retry-silently'}})
