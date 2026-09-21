from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run/icra2027/e4_collision_diagnostic_sweep.py"


def _load():
    spec = importlib.util.spec_from_file_location("e4_collision_diagnostic_sweep", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_sweep_has_exact_4_plus_12_cells() -> None:
    sweep = _load()
    from robo.sim.room_collision import validate_room_diagnostic_spec

    assert sweep.VARIANT_IDS == (
        "3db-hraw-pfail",
        "3db-hraw-pdemote",
        "3db-hclip-pfail",
        "3db-hclip-pdemote",
        "d755-r20-zextent-hraw",
        "d755-r20-zextent-hclip",
        "d755-r20-zmean-hraw",
        "d755-r20-zmean-hclip",
        "d755-r25-zextent-hraw",
        "d755-r25-zextent-hclip",
        "d755-r25-zmean-hraw",
        "d755-r25-zmean-hclip",
        "d755-r30-zextent-hraw",
        "d755-r30-zextent-hclip",
        "d755-r30-zmean-hraw",
        "d755-r30-zmean-hclip",
    )
    expected_keys = {
        "schema_version",
        "variant_id",
        "scene_id",
        "hull_bottom",
        "intrusive_primitive",
        "plane_residual_tol_m",
        "support_z",
        "min_support_area_m2",
        "carve_side_top_margin_m",
        "lower_carve_margin_m",
        "support_clip_offset_m",
    }
    assert all(set(row) == expected_keys for row in sweep.VARIANTS.values())

    three_db = [row for row in sweep.VARIANTS.values() if row["scene_id"] == "3db0a1c8f3"]
    d755 = [row for row in sweep.VARIANTS.values() if row["scene_id"] == "d755b3d9d8"]
    assert len(three_db) == 4
    assert {
        (row["hull_bottom"], row["intrusive_primitive"]) for row in three_db
    } == {
        ("raw", "fail"),
        ("raw", "demote_to_residual"),
        ("clip_scan_aabb_bottom", "fail"),
        ("clip_scan_aabb_bottom", "demote_to_residual"),
    }
    assert all(row["plane_residual_tol_m"] == 0.02 for row in three_db)
    assert all(row["support_z"] == "extent" for row in three_db)

    assert len(d755) == 12
    assert all(row["intrusive_primitive"] == "fail" for row in d755)
    assert {
        (row["plane_residual_tol_m"], row["support_z"], row["hull_bottom"])
        for row in d755
    } == {
        (residual, support, hull)
        for residual in (0.02, 0.025, 0.03)
        for support in ("extent", "fitted_mean_20mm")
        for hull in ("raw", "clip_scan_aabb_bottom")
    }
    assert all(
        row["min_support_area_m2"] == 0.05
        and row["carve_side_top_margin_m"] == 0.02
        and row["lower_carve_margin_m"] == 0.0
        and row["support_clip_offset_m"] == 0.005
        for row in sweep.VARIANTS.values()
    )
    assert all(
        validate_room_diagnostic_spec(row, expected_scene_id=row["scene_id"]) == row
        for row in sweep.VARIANTS.values()
    )


def test_exporter_commands_bind_two_factories_spec_and_one_package_direction() -> None:
    sweep = _load()
    factories = [Path("/e/A0"), Path("/e/A4")]
    build = sweep._exporter_argv(
        factories=factories,
        spec_path=Path("/e/spec.json"),
        package_out=Path("/e/common_static"),
    )
    assert build.count("--background-carve-factory") == 2
    assert "--room-diagnostic-spec" in build
    assert "--room-static-package-out" in build
    assert "--room-static-package" not in build
    assert "--test" not in build

    evaluate = sweep._exporter_argv(
        factories=factories,
        spec_path=Path("/e/spec.json"),
        package_in=Path("/e/common_static"),
        test=True,
    )
    assert evaluate.count("--background-carve-factory") == 2
    assert "--room-static-package" in evaluate
    assert "--room-static-package-out" not in evaluate
    assert "--test" in evaluate
    with pytest.raises(sweep.DiagnosticSweepError):
        sweep._exporter_argv(
            factories=factories,
            spec_path=Path("/e/spec.json"),
        )


def test_static_package_identity_is_exact_and_hash_only() -> None:
    sweep = _load()
    identity = {
        "manifest_sha256": "a" * 64,
        "content_sha256": "b" * 64,
        "static_xml_sha256": "c" * 64,
        "background_sha256": "d" * 64,
    }
    assert sweep._static_identity(identity) == identity
    with pytest.raises(sweep.DiagnosticSweepError):
        sweep._static_identity({**identity, "extension": "e" * 64})
    with pytest.raises(sweep.DiagnosticSweepError):
        sweep._static_identity({**identity, "manifest_sha256": "A" * 64})


def test_collision_coverage_schema_is_exact_finite_and_bounded() -> None:
    sweep = _load()
    valid = {
        "n_rays": 1500,
        "ray_hit_agreement": 0.8,
        "ray_dist_p95_m": 0.4,
        "n_points": 3000,
        "occupancy_agreement": 0.75,
        "penetration_frac": 0.05,
    }
    assert sweep._collision_coverage(valid) == valid
    assert sweep._collision_coverage({**valid, "extra": 1}) is None
    assert sweep._collision_coverage({**valid, "n_rays": True}) is None
    assert sweep._collision_coverage({**valid, "occupancy_agreement": 1.01}) is None
    assert sweep._collision_coverage({**valid, "ray_dist_p95_m": float("nan")}) is None


def test_preregistered_ranking_prefers_stability_then_intervention_then_coverage() -> None:
    sweep = _load()

    def row(
        variant_id: str,
        *,
        stable: int = 2,
        occupancy: float = 0.7,
    ) -> dict:
        spec = sweep.VARIANTS[variant_id]
        return {
            "comparison_pass": True,
            "paired_stable_count": stable,
            "variant_id": variant_id,
            "intervention_rank": {
                "uses_clipped_hulls": spec["hull_bottom"] != "raw",
                "uses_primitive_demotion": spec["intrusive_primitive"] != "fail",
                "plane_residual_tol_m": spec["plane_residual_tol_m"],
                "uses_fitted_mean_support": spec["support_z"] != "extent",
            },
            "conservative_collision_coverage": {
                "min_occupancy_agreement": occupancy,
                "min_ray_hit_agreement": 0.8,
                "max_penetration_frac": 0.05,
                "max_ray_dist_p95_m": 0.4,
            },
        }

    # More paired-stable slots dominate every intervention preference.
    assert sweep._ranking_key(
        row("d755-r30-zmean-hclip", stable=3)
    ) < sweep._ranking_key(row("d755-r20-zextent-hraw", stable=2))
    # With stability tied: raw, smaller residual, and extent are preferred.
    assert sweep._ranking_key(
        row("d755-r20-zextent-hraw")
    ) < sweep._ranking_key(row("d755-r20-zextent-hclip"))
    assert sweep._ranking_key(
        row("d755-r20-zextent-hraw")
    ) < sweep._ranking_key(row("d755-r25-zextent-hraw"))
    assert sweep._ranking_key(
        row("d755-r20-zextent-hraw")
    ) < sweep._ranking_key(row("d755-r20-zmean-hraw"))
    # Coverage breaks a tie only after intervention is identical.
    lower = row("d755-r20-zextent-hraw", occupancy=0.6)
    higher = row("d755-r20-zextent-hraw", occupancy=0.9)
    assert sweep._ranking_key(higher) < sweep._ranking_key(lower)


def test_package_loader_temporarily_binds_variant_scene(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sweep = _load()
    from robo.sim import export_mjcf

    spec = sweep.VARIANTS["d755-r20-zextent-hraw"]
    original_scene = export_mjcf.C.SCENE_ID
    identity = {
        "manifest_sha256": "1" * 64,
        "content_sha256": "2" * 64,
        "static_xml_sha256": "3" * 64,
        "background_sha256": "4" * 64,
    }

    def loader(package_dir, observed_spec, observed_hash):
        assert package_dir == tmp_path / "common_static"
        assert observed_spec == spec
        assert observed_hash == "a" * 64
        assert export_mjcf.C.SCENE_ID == "d755b3d9d8"
        return {
            "asset_xml_lines": [],
            "geom_xml_lines": [],
            "room_report": {},
            "source_manifest_sha256": {"A0": "0" * 64, "A4": "4" * 64},
            "static_package_identity": identity,
            "package_dir": str(tmp_path / "common_static"),
        }

    monkeypatch.setattr(export_mjcf, "load_common_room_static_package", loader)
    _package, observed_identity = sweep._load_package(
        tmp_path / "common_static", spec=spec, spec_sha256="a" * 64
    )
    assert observed_identity == identity
    assert export_mjcf.C.SCENE_ID == original_scene


def test_afterany_comparison_seals_missing_policy_as_diagnostic_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sweep = _load()
    variant_id = "3db-hraw-pfail"
    monkeypatch.setattr(
        sweep,
        "_code_snapshot",
        lambda _commit: {"code_root": str(sweep.CODE_ROOT), "commit": "a" * 40, "dirty": False},
    )
    monkeypatch.setattr(sweep, "_evidence_root", lambda: tmp_path)
    monkeypatch.setattr(
        sweep,
        "_spec_and_hash",
        lambda *_args, **_kwargs: (sweep.VARIANTS[variant_id], "b" * 64),
    )
    monkeypatch.setattr(
        sweep,
        "_build_policy_gate",
        lambda **kwargs: (_ for _ in ()).throw(FileNotFoundError(kwargs["policy_id"])),
    )
    gate = sweep._build_comparison_gate(
        sweep_id="diag-test",
        variant_id=variant_id,
        expected_commit="a" * 40,
    )
    assert set(gate["errors"]) == {"A0", "A4"}
    assert gate["comparison_pass"] is False
    assert gate["static_package_identity_equal"] is False
    assert gate["status"] == "diagnostic_fail"
    assert gate["gpu_launch_allowed"] is False
    assert gate["large_rollout_launch_allowed"] is False
    assert gate["paper_ready"] is False


def test_pair_comparison_requires_two_shared_stable_objects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sweep = _load()
    variant_id = "3db-hclip-pdemote"
    identity = {
        "manifest_sha256": "1" * 64,
        "content_sha256": "2" * 64,
        "static_xml_sha256": "3" * 64,
        "background_sha256": "4" * 64,
    }
    monkeypatch.setattr(
        sweep,
        "_code_snapshot",
        lambda _commit: {"code_root": str(sweep.CODE_ROOT), "commit": "a" * 40, "dirty": False},
    )
    monkeypatch.setattr(sweep, "_evidence_root", lambda: tmp_path)
    monkeypatch.setattr(
        sweep,
        "_spec_and_hash",
        lambda *_args, **_kwargs: (sweep.VARIANTS[variant_id], "b" * 64),
    )

    def policy_gate(**kwargs):
        return {
            "discovered_slots": ["obj_00", "obj_01"],
            "policy_id": kwargs["policy_id"],
            "scientific_pass": True,
            "stable_slots": ["obj_00"],
            "static_package_identity": identity,
            "sweep_id": kwargs["sweep_id"],
            "variant_id": kwargs["variant_id"],
        }

    monkeypatch.setattr(sweep, "_build_policy_gate", policy_gate)
    monkeypatch.setattr(
        sweep,
        "_validated_bundle_gate",
        lambda *_args, **kwargs: (
            policy_gate(
                sweep_id=kwargs["sweep_id"],
                variant_id=kwargs["variant_id"],
                policy_id=kwargs["policy_id"],
                expected_commit="a" * 40,
            ),
            {"manifest_sha256": ("0" if kwargs["policy_id"] == "A0" else "5") * 64},
        ),
    )
    monkeypatch.setattr(
        sweep,
        "_candidate",
        lambda: SimpleNamespace(_same_replay_structure=lambda left, right: left == right),
    )
    gate = sweep._build_comparison_gate(
        sweep_id="diag-test",
        variant_id=variant_id,
        expected_commit="a" * 40,
    )
    assert gate["static_package_identity_equal"] is True
    assert gate["discovered_roster_equal"] is True
    assert gate["paired_stable_count"] == 1
    assert gate["paired_stable_at_least_two"] is False
    assert gate["comparison_pass"] is False


def test_aggregate_is_permanently_diagnostics_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sweep = _load()
    commit = "a" * 40
    monkeypatch.setattr(
        sweep,
        "_code_snapshot",
        lambda _commit: {"code_root": str(sweep.CODE_ROOT), "commit": commit, "dirty": False},
    )
    monkeypatch.setattr(sweep, "_evidence_root", lambda: tmp_path)

    def comparison(**kwargs):
        variant_id = kwargs["variant_id"]
        spec = sweep.VARIANTS[variant_id]
        return {
            "comparison_pass": True,
            "conservative_collision_coverage": {
                "min_n_points": 3000,
                "min_n_rays": 1500,
                "min_occupancy_agreement": 0.75,
                "max_penetration_frac": 0.05,
                "max_ray_dist_p95_m": 0.4,
                "min_ray_hit_agreement": 0.8,
            },
            "scene_id": spec["scene_id"],
            "paired_stable_count": 2,
            "policies": {
                policy: {
                    "initial_contacts": {"max_static_penetration_m": 0.001}
                }
                for policy in sweep.POLICIES
            },
            "policy_scientific_pass": {"A0": True, "A4": True},
            "sweep_id": kwargs["sweep_id"],
            "variant_id": variant_id,
        }

    monkeypatch.setattr(sweep, "_build_comparison_gate", comparison)
    monkeypatch.setattr(
        sweep,
        "_validated_bundle_gate",
        lambda *_args, **kwargs: (
            comparison(
                sweep_id=kwargs["sweep_id"],
                variant_id=kwargs["variant_id"],
                expected_commit=commit,
            ),
            {"manifest_sha256": "b" * 64},
        ),
    )
    monkeypatch.setattr(
        sweep,
        "_candidate",
        lambda: SimpleNamespace(_same_replay_structure=lambda left, right: left == right),
    )
    monkeypatch.setattr(
        sweep,
        "_publish",
        lambda *_args, **_kwargs: {"manifest_sha256": "c" * 64},
    )
    report = sweep.aggregate(sweep_id="diag-test", expected_commit=commit)
    assert report["complete_variant_count"] == 16
    assert report["variant_count"] == 16
    assert report["errors"] == {}
    assert report["scene_winners"] == {
        "3db0a1c8f3": "3db-hraw-pfail",
        "d755b3d9d8": "d755-r20-zextent-hraw",
    }
    assert report["selection_rule"] == list(sweep.SELECTION_RULE)
    assert report["gpu_launch_allowed"] is False
    assert report["large_rollout_launch_allowed"] is False
    assert report["paper_ready"] is False


def test_incomplete_aggregate_never_names_a_scene_winner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sweep = _load()
    commit = "a" * 40
    missing = sweep.VARIANT_IDS[0]
    monkeypatch.setattr(
        sweep,
        "_code_snapshot",
        lambda _commit: {"code_root": str(sweep.CODE_ROOT), "commit": commit, "dirty": False},
    )
    monkeypatch.setattr(sweep, "_evidence_root", lambda: tmp_path)

    def comparison(**kwargs):
        variant_id = kwargs["variant_id"]
        if variant_id == missing:
            raise FileNotFoundError(variant_id)
        spec = sweep.VARIANTS[variant_id]
        return {
            "comparison_pass": True,
            "conservative_collision_coverage": {
                "min_n_points": 3000,
                "min_n_rays": 1500,
                "min_occupancy_agreement": 0.75,
                "max_penetration_frac": 0.05,
                "max_ray_dist_p95_m": 0.4,
                "min_ray_hit_agreement": 0.8,
            },
            "scene_id": spec["scene_id"],
            "paired_stable_count": 2,
            "policies": {
                policy: {"initial_contacts": {"max_static_penetration_m": 0.001}}
                for policy in sweep.POLICIES
            },
            "policy_scientific_pass": {"A0": True, "A4": True},
            "sweep_id": kwargs["sweep_id"],
            "variant_id": variant_id,
        }

    monkeypatch.setattr(sweep, "_build_comparison_gate", comparison)
    monkeypatch.setattr(
        sweep,
        "_validated_bundle_gate",
        lambda *_args, **kwargs: (
            comparison(
                sweep_id=kwargs["sweep_id"],
                variant_id=kwargs["variant_id"],
                expected_commit=commit,
            ),
            {"manifest_sha256": "b" * 64},
        ),
    )
    monkeypatch.setattr(
        sweep,
        "_candidate",
        lambda: SimpleNamespace(_same_replay_structure=lambda left, right: left == right),
    )
    monkeypatch.setattr(
        sweep,
        "_publish",
        lambda *_args, **_kwargs: {"manifest_sha256": "c" * 64},
    )
    report = sweep.aggregate(sweep_id="diag-test", expected_commit=commit)
    assert report["status"] == "incomplete"
    assert report["complete_variant_count"] == 15
    assert missing in report["errors"]
    assert report["scene_winners"] == {
        "3db0a1c8f3": None,
        "d755b3d9d8": None,
    }
