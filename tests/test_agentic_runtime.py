from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh

from agents.orchestrator.runtime import _apply_contact_dynamics, align_and_probe


def test_open3d_registration_backend_isolated_from_user_and_visualization():
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    code = """
import sys
from agents.assets.s5_align import open3d_registration_backend
backend = open3d_registration_backend()
assert hasattr(backend.pipelines.registration, "registration_icp")
assert "open3d.visualization" not in sys.modules
assert "dash" not in sys.modules
assert "flask" not in sys.modules
print(backend.__file__)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "/open3d/cpu/pybind" in result.stdout


def test_align_and_probe_emits_construction_only_evidence(tmp_path: Path):
    mesh = trimesh.creation.box(extents=[0.12, 0.08, 0.05])
    mesh_path = tmp_path / "proposal.obj"
    mesh.export(mesh_path)
    rng = np.random.RandomState(4)
    points, _ = trimesh.sample.sample_surface(mesh, 800, seed=9)
    observation = np.asarray(points)[rng.choice(len(points), 500, replace=False)]
    observation = observation + np.array([0.3, -0.1, 0.2])
    out = tmp_path / "candidate"
    result = align_and_probe(
        mesh_path,
        observation,
        label="box",
        visible_fraction=0.8,
        out_dir=out,
        signed_source_up=False,
        producer_commit="a" * 40,
        input_hashes={
            "raw_mesh": "b" * 64,
            "raw_gaussian": "c" * 64,
            "visible_observation": "d" * 64,
        },
    )
    evidence = result["evidence"]
    assert evidence["schema_frame_unit_valid"] is True
    assert evidence["observation_point_count"] == 500
    assert evidence["usable_convex_parts"] == 1
    assert evidence["settle_drift_m"] is not None
    assert evidence["missing_evidence"] == []
    evidence_manifest = json.loads((out / "evidence.json").read_text())
    assert evidence_manifest["producer_commit"] == "a" * 40
    assert evidence_manifest["raw_values"] == evidence
    assert evidence_manifest["missing_flags"] == []
    assert evidence_manifest["input_hashes"] == {
        "raw_gaussian": "c" * 64,
        "raw_mesh": "b" * 64,
        "visible_observation": "d" * 64,
    }
    assert not any("gt" in key.lower() for key in evidence)
    physics = json.loads((out / "physical" / "physics.json").read_text())
    assert physics == {
        "friction": 0.5,
        "mass_kg": 0.3,
        "restitution": 0.1,
        "source": "e3_category_independent_constant_v1",
    }
    probe = json.loads((out / "physical" / "probe.json").read_text())
    assert probe["contact_dynamics"] == {
        "object": {
            "application": "pybullet.changeDynamics_getDynamicsInfo_v1",
            "lateral_friction": 0.5,
            "link_index": -1,
            "restitution": 0.1,
            "role": "object",
        },
        "plane": {
            "application": "pybullet.changeDynamics_getDynamicsInfo_v1",
            "lateral_friction": 0.5,
            "link_index": -1,
            "restitution": 0.1,
            "role": "plane",
        },
    }


def test_contact_dynamics_are_applied_to_the_declared_body_and_link():
    class FakeBullet:
        def __init__(self):
            self.calls = []

        def changeDynamics(self, body_id, link_index, **kwargs):
            self.calls.append((body_id, link_index, kwargs))

        def getDynamicsInfo(self, body_id, link_index):
            assert (body_id, link_index) in {(17, -1), (23, -1)}
            return (0.3, 0.5, (0.0, 0.0, 0.0), (), (), 0.1)

    bullet = FakeBullet()
    object_recorded = _apply_contact_dynamics(
        bullet,
        17,
        role="object",
        physics={"friction": 0.5, "restitution": 0.1},
    )
    plane_recorded = _apply_contact_dynamics(
        bullet,
        23,
        role="plane",
        physics={"friction": 0.5, "restitution": 0.1},
    )
    assert bullet.calls == [
        (17, -1, {"lateralFriction": 0.5, "restitution": 0.1}),
        (23, -1, {"lateralFriction": 0.5, "restitution": 0.1}),
    ]
    assert object_recorded == {
        "application": "pybullet.changeDynamics_getDynamicsInfo_v1",
        "lateral_friction": 0.5,
        "link_index": -1,
        "restitution": 0.1,
        "role": "object",
    }
    assert plane_recorded == {
        "application": "pybullet.changeDynamics_getDynamicsInfo_v1",
        "lateral_friction": 0.5,
        "link_index": -1,
        "restitution": 0.1,
        "role": "plane",
    }


def test_signed_retry_records_nonlegacy_hypothesis_when_needed(tmp_path: Path):
    source = trimesh.creation.box(extents=[0.24, 0.04, 0.07])
    mesh_path = tmp_path / "proposal.obj"
    source.export(mesh_path)
    # Map source +x to world +z, then keep only a partial visible half.
    rotation = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    points, _ = trimesh.sample.sample_surface(source, 1600, seed=11)
    target = np.asarray(points) @ rotation.T + np.array([0.2, 0.4, 0.6])
    target = target[target[:, 0] >= np.median(target[:, 0])]
    result = align_and_probe(
        mesh_path,
        target,
        label="box",
        visible_fraction=0.75,
        out_dir=tmp_path / "retry",
        signed_source_up=True,
        producer_commit="a" * 40,
        input_hashes={
            "raw_mesh": "b" * 64,
            "raw_gaussian": "c" * 64,
            "visible_observation": "d" * 64,
        },
    )
    assert result["alignment"]["source_up_hypothesis"] in {
        "-z", "+x", "-x", "+y", "-y"
    }
    assert result["alignment"]["mesh_sample_seed"] == 42


def test_existing_public_open3d_does_not_reload_incompatible_pybind_library():
    env = dict(os.environ, PYTHONNOUSERSITE='1')
    code = '''
import open3d, sys
from agents.assets.s5_align import open3d_registration_backend
backend = open3d_registration_backend()
assert backend.geometry.PointCloud is open3d.geometry.PointCloud
assert backend.pipelines.registration.registration_icp is open3d.pipelines.registration.registration_icp
assert open3d_registration_backend() is backend
'''
    result = subprocess.run([sys.executable, '-c', code], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
