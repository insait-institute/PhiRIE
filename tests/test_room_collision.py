"""Tests for robo/sim/room_collision.py (plan Task 06).

Exercises the full room-collision export against the synthetic fixture in
tests/data/scenes/collision_fixture/ (floor + table + wall + two obstacle
boxes + three dynamic objects + a scripted robot end-effector sweep + two
scripted pushes -- see that dir's scene.json for exact numbers).

The core regression this file guards is the negative-check from
plan/06_FULL_ROOM_COLLISION_EXPORT.md's FAST_VALIDATION_MATRIX: an object
pushed off its original support location must continue to contact the
actual room in `--collision-mode room`, and must FAIL to (i.e. fall through
to the floor) under the legacy `--collision-mode shim` ablation. Every test
here is a real MuJoCo rollout on real (if synthetic) mesh geometry -- no
hand-computed expected XML, no mocking.

Run: .venv/bin/python -m pytest -q tests/test_room_collision.py
"""
from pathlib import Path

import numpy as np
import pytest
import torch  # noqa: F401  see room_collision.coacd_decompose: MUST precede
              # `import coacd` anywhere in this process, or it segfaults.

from robo.sim import room_collision as rc

FIXTURE_DIR = Path(__file__).parent / "data" / "scenes" / "collision_fixture"
STEPS = 1000


# --------------------------------------------------------------- fixtures --

@pytest.fixture(scope="module")
def spec():
    return rc.load_scene_spec(FIXTURE_DIR)


@pytest.fixture(scope="module")
def room_report(tmp_path_factory):
    out = tmp_path_factory.mktemp("room_mode")
    return rc.run_fixture(FIXTURE_DIR, collision_mode="room",
                          smoke_steps=STEPS, out_dir=out, seed=0)


@pytest.fixture(scope="module")
def shim_report(tmp_path_factory):
    out = tmp_path_factory.mktemp("shim_mode")
    return rc.run_fixture(FIXTURE_DIR, collision_mode="shim",
                          smoke_steps=STEPS, out_dir=out, seed=0)


def _touches(contact_pairs, geom_a_substr, geom_b_prefix):
    """True if some contact pair has one geom containing `geom_a_substr` and
    the other starting with `geom_b_prefix` (order-agnostic)."""
    for pair in contact_pairs:
        a, b = pair.split("|")
        if (geom_a_substr in a and b.startswith(geom_b_prefix)) or \
           (geom_a_substr in b and a.startswith(geom_b_prefix)):
            return True
    return False


# ------------------------------------------------------- geometry unit tests

def test_extract_room_features_on_simple_mesh():
    """Floor + table + wall + 2 obstacle boxes, built straight from
    primitives (not via the fixture loader) -> sane classification: one
    floor, exactly one big table-sized support, at least one validated
    wall, and a non-trivial residual left for CoACD (the obstacles + debris
    ribbons carved out by classification)."""
    import trimesh

    floor = trimesh.creation.box(extents=(4.0, 4.0, 0.05))
    floor.apply_translation((0, 0, -0.025))
    table = trimesh.creation.box(extents=(0.8, 0.6, 0.02))
    table.apply_translation((0.6, 0.0, 0.39))
    wall = trimesh.creation.box(extents=(0.05, 2.0, 1.2))
    wall.apply_translation((1.4, 0.0, 0.6))
    mesh = trimesh.util.concatenate([floor, table, wall])

    feats = rc.extract_room_features(mesh)
    assert feats.floor is not None
    assert feats.floor.area == pytest.approx(16.0, rel=1e-3)
    assert len(feats.supports) >= 1
    table_support = max(feats.supports, key=lambda p: p.area)
    assert table_support.area == pytest.approx(0.48, rel=1e-3)
    assert table_support.z == pytest.approx(0.40, abs=1e-6)
    # the wall's two large faces must be found as separate, correctly
    # positioned single-plane walls (regression: an earlier version merged
    # opposite-facing faces into one "ring" and rejected the whole cluster)
    big_walls = [w for w in feats.walls if w.area > 2.0]
    assert len(big_walls) == 2
    xs = sorted(w.lo[0] for w in big_walls)
    assert xs[0] == pytest.approx(1.375, abs=1e-3)
    assert xs[1] == pytest.approx(1.425, abs=1e-3)
    assert feats.classified_tris < feats.source_tris  # something is residual


def test_decimate_mesh_noop_below_budget():
    import trimesh
    mesh = trimesh.creation.box(extents=(1, 1, 1))
    dec, stats = rc.decimate_mesh(mesh, max_tris=10_000)
    assert stats["tris_before"] == stats["tris_after"] == len(mesh.faces)
    assert stats["hausdorff_p95_m"] == 0.0
    assert stats["hausdorff_max_m"] == 0.0


def test_greedy_channel_color_keeps_close_footprints_apart():
    # two footprints closer than `safe_margin` must get DIFFERENT channels
    # (a shared channel would let one's shim reach into the other's object)
    close = [(np.array([0.0, 0.0]), np.array([0.1, 0.1])),
            (np.array([0.12, 0.0]), np.array([0.22, 0.1]))]
    chan = rc.greedy_channel_color(close, safe_margin=0.05)
    assert chan[0] != chan[1]
    # far-apart footprints MAY share a channel (greedy colouring is free to
    # reuse one), so only assert the close case is forced apart above.


def test_collision_coverage_metrics_self_agreement():
    """A collision representation identical to the source mesh must score
    (near-)perfect ray/point agreement -- sanity check on the metric itself
    before trusting it to flag real discrepancies."""
    import trimesh
    mesh = trimesh.creation.box(extents=(0.5, 0.4, 0.3))
    out = rc.collision_coverage_metrics(mesh, mesh, n_rays=500, n_points=1000, seed=1)
    assert out["occupancy_agreement"] > 0.98
    if out["ray_hit_agreement"] is not None:  # None only if rtree/pyembree missing
        assert out["ray_hit_agreement"] > 0.98


def test_collision_coverage_surface_sampling_is_seeded(monkeypatch):
    import trimesh

    observed_seeds = []
    original = trimesh.sample.sample_surface

    def recorded_sample_surface(*args, **kwargs):
        observed_seeds.append(kwargs.get("seed"))
        return original(*args, **kwargs)

    monkeypatch.setattr(trimesh.sample, "sample_surface", recorded_sample_surface)
    mesh = trimesh.creation.box(extents=(0.5, 0.4, 0.3))
    first = rc.collision_coverage_metrics(
        mesh, mesh, n_rays=50, n_points=100, seed=1729
    )
    second = rc.collision_coverage_metrics(
        mesh, mesh, n_rays=50, n_points=100, seed=1729
    )
    assert observed_seeds == [1729, 1729]
    assert first == second


# --------------------------------------------------------- fixture-level tests

def test_room_mode_has_no_private_shims(room_report):
    """Acceptance criterion: 'main policy runs use no private support
    shims.' room mode's manifest must show zero shim channels."""
    assert "shim" not in room_report["manifest"]
    assert "room" in room_report["manifest"]
    assert room_report["manifest"]["room"]["n_supports"] >= 1  # the table
    with open(room_report["scene_xml"]) as f:
        xml = f.read()
    assert 'name="support_0"' not in xml  # legacy per-object shim geom name
    assert "room_support_0" in xml


def test_shim_mode_has_no_room_geometry(shim_report):
    """The inverse: the legacy ablation must not accidentally pick up the
    new room geometry (that would defeat the point of the ablation)."""
    assert "room" not in shim_report["manifest"]
    assert "shim" in shim_report["manifest"]
    with open(shim_report["scene_xml"]) as f:
        xml = f.read()
    assert "room_support" not in xml
    assert "room_obstacle" not in xml
    assert "room_wall" not in xml
    assert 'name="support_0"' in xml  # legacy per-object shim IS present


def test_room_mode_reports_coverage_and_penetration(room_report):
    cov = room_report["manifest"]["room"]["coverage"]
    assert cov["occupancy_agreement"] is not None
    assert cov["occupancy_agreement"] > 0.95
    assert cov["penetration_frac"] is not None
    assert cov["penetration_frac"] < 0.05
    assert room_report["manifest"]["room"]["hausdorff_p95_m"] < 0.05


def test_robot_sweep_hits_table_and_obstacle(room_report):
    """Required test 1: robot sweep collides with the table AND a nearby
    obstacle. Only meaningful in room mode -- shim mode has no table/
    obstacle collision at all by construction."""
    contacts = room_report["push_off_support"]["contact_pairs"]
    assert _touches(contacts, "robot_probe_geom", "room_support")
    assert _touches(contacts, "robot_probe_geom", "room_obstacle")


def test_object_pushed_off_support_room_vs_shim(room_report, shim_report):
    """Required test 2 (the core negative-check): obj_mug is shoved -Y,
    clearing its own PAD-extended shim footprint but staying well within
    the table's real extent (see scene.json's push_off_support._note).

    - room mode: the room's real table collision must still be there, so
      obj_mug should still be resting at ~table height and still in contact
      with a room_support/room_obstacle geom at the end of the rollout.
    - shim mode: nothing represents the table except the (now-missed)
      private shim, so obj_mug must fall all the way to the floor.
    """
    table_top = 0.40
    floor_top = 0.0

    room_z = room_report["push_off_support"]["final_body_pos"]["obj_mug"][2]
    shim_z = shim_report["push_off_support"]["final_body_pos"]["obj_mug"][2]

    assert room_z > table_top - 0.05, (
        f"room mode: obj_mug ended at z={room_z:.3f}, expected ~table height "
        f"({table_top}); the room collision failed to catch it")
    assert shim_z < floor_top + 0.10, (
        f"shim mode: obj_mug ended at z={shim_z:.3f}, expected it to have "
        f"fallen to ~floor height ({floor_top}) -- if it did NOT fall, the "
        f"ablation is not exercising the private-shim limitation")
    assert room_z - shim_z > 0.2  # unambiguous separation between the two modes

    room_contacts = room_report["push_off_support"]["contact_pairs"]
    assert _touches(room_contacts, "obj_mug_geom", "room_support") or \
        _touches(room_contacts, "obj_mug_geom", "room_obstacle"), (
        "room mode: obj_mug never contacted any room_* geom after the push")

    shim_contacts = shim_report["push_off_support"]["contact_pairs"]
    assert _touches(shim_contacts, "obj_mug_geom", "floor"), (
        "shim mode: obj_mug never contacted the (real, universal) floor "
        "plane -- expected it to fall through to the floor")


@pytest.mark.parametrize("mode_report", ["room_report", "shim_report"])
def test_cross_object_collision_active(mode_report, room_report, shim_report):
    """Required test 3: cross-object collision is active, regardless of
    collision mode (this was already true before Task 06 -- object-object
    collision never depended on the room; this is a regression guard)."""
    report = room_report if mode_report == "room_report" else shim_report
    contacts = report["cross_object_push"]["contact_pairs"]
    assert _touches(contacts, "obj_bottle_a_geom", "obj_bottle_b_geom") or \
        _touches(contacts, "obj_bottle_b_geom", "obj_bottle_a_geom")


@pytest.mark.parametrize("mode", ["room", "shim"])
def test_deterministic_smoke_1000_steps(mode, tmp_path_factory):
    """Required test 4: same seed/inputs -> same final state hash, twice."""
    out1 = tmp_path_factory.mktemp(f"det_{mode}_1")
    out2 = tmp_path_factory.mktemp(f"det_{mode}_2")
    r1 = rc.run_fixture(FIXTURE_DIR, collision_mode=mode, smoke_steps=STEPS,
                        out_dir=out1, seed=0)
    r2 = rc.run_fixture(FIXTURE_DIR, collision_mode=mode, smoke_steps=STEPS,
                        out_dir=out2, seed=0)
    assert r1["push_off_support"]["state_hash"] == r2["push_off_support"]["state_hash"]
    assert r1["cross_object_push"]["state_hash"] == r2["cross_object_push"]["state_hash"]
    assert r1["finite"] and r2["finite"]


def test_benchmark_report_has_required_fields(room_report):
    """Acceptance criterion: compile time, physics speed, memory, settle
    drift, and contact trajectories are all reported per scene."""
    r = room_report["push_off_support"]
    for key in ("compile_time_s", "steps_per_sec", "mem_mb", "settle_drift_m",
               "contact_pairs", "state_hash", "finite"):
        assert key in r
    assert r["compile_time_s"] >= 0
    assert r["steps_per_sec"] > 0
    assert r["mem_mb"] > 0


def test_shim_geoms_isolated_from_room_style_universal_channel():
    """Unit-level check on the legacy ablation's own invariant: a private
    shim's mask must never include bit 0 (the shared/universal channel),
    since export_mjcf.py's floor/robot/other-objects all key off bit 0."""
    supports = [
        (0.40, np.array([0.5, 0.05]), np.array([0.6, 0.15])),
        (0.40, np.array([0.62, -0.09]), np.array([0.68, -0.03])),
    ]
    masks, slabs = rc.build_shim_geoms(supports)
    assert len(slabs) == 2
    for m in masks:
        assert m & 1 == 0  # channel-only, bit 0 excluded
