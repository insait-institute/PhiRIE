import copy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from run.icra2027.policy_service_smoke import (
    LoggedTransport, observation_manifest, stop_child, synthetic_observation,
)


def test_synthetic_input_is_fixed_readonly_and_rejects_room_image_mode():
    config = yaml.safe_load((Path(__file__).resolve().parents[1]/
        "configs/experiments/icra2027/policy_service_smoke.yaml").read_text())
    obs,prompt = synthetic_observation(config)
    assert obs["observation/joint_position"].shape == (7,)
    assert all(not a.flags.writeable for a in obs.values())
    assert not obs["observation/exterior_image_1_left"].any()
    assert observation_manifest(obs,prompt) == observation_manifest(*synthetic_observation(config))
    altered = copy.deepcopy(config)
    altered["synthetic_observation"]["image_fill"] = 1
    with pytest.raises(ValueError,match="synthetic-only"):
        synthetic_observation(altered)


@pytest.mark.parametrize("bad", [np.zeros((14,8)),np.full((15,8),np.nan)])
def test_transport_rejects_invalid_raw_warmup_chunk_and_preserves_failure(bad):
    class Transport:
        def infer(self,request): return {"actions":bad}
    events=[]
    with pytest.raises(ValueError):
        LoggedTransport(Transport(),events).infer({})
    assert events[0]["status"] == "failed"
    assert events[0]["elapsed_s"] >= 0


def test_cleanup_terminates_only_its_owned_child_group():
    child = subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"],
                             start_new_session=True)
    try:
        result = stop_child(child,1)
        assert result["pid"] == child.pid
        assert result["terminated"] is True
        assert child.poll() is not None
    finally:
        if child.poll() is None:
            child.kill();child.wait()
