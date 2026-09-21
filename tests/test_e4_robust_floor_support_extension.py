from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run/icra2027/e4_robust_floor_support_extension.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "e4_robust_floor_support_extension_test", RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _comparison(passed: bool, paired: int, *, errors: dict | None = None) -> dict:
    return {
        "comparison_pass": passed,
        "errors": {} if errors is None else errors,
        "paired_stable_count": paired,
    }


def _external_registry() -> dict:
    return {
        "schema_version": 1,
        "registry_status": "validated",
        "source_sweep_id": "sealed-robust-source",
        "source_code_commit": "d" * 40,
        "winners": {
            "d755b3d9d8": {
                "scene_id": "d755b3d9d8",
                "winner": "d755-fs",
            }
        },
        "authorization": {
            "gpu_launch_allowed": False,
            "large_rollout_launch_allowed": False,
            "paper_ready": False,
        },
    }


def _fake_aggregate(module, monkeypatch, comparisons: dict[str, dict]) -> dict:
    code = {
        "code_root": str(module.CODE_ROOT),
        "commit": "a" * 40,
        "dirty": False,
    }
    monkeypatch.setattr(module._ROBUST._BASE, "_validated_id", lambda value: value)
    monkeypatch.setattr(module._ROBUST._BASE, "_code_snapshot", lambda _commit: code)
    monkeypatch.setattr(
        module._ROBUST._BASE, "_evidence_root", lambda: Path("/evidence")
    )
    monkeypatch.setattr(
        module._ROBUST._BASE,
        "_comparison_bundle",
        lambda _root, _sweep_id, variant_id: Path("/comparison") / variant_id,
    )
    monkeypatch.setattr(
        module,
        "_build_comparison_gate",
        lambda *, variant_id, **_kwargs: deepcopy(comparisons[variant_id]),
    )
    monkeypatch.setattr(
        module._ROBUST._BASE,
        "_validated_bundle_gate",
        lambda _directory, *, variant_id, **_kwargs: (
            deepcopy(comparisons[variant_id]),
            {"manifest_sha256": "b" * 64},
        ),
    )
    monkeypatch.setattr(
        module._ROBUST._BASE,
        "_candidate",
        lambda: SimpleNamespace(
            _same_replay_structure=lambda left, right: left == right
        ),
    )
    monkeypatch.setattr(
        module,
        "_validated_external_winners",
        lambda _root: _external_registry(),
    )
    monkeypatch.setattr(
        module,
        "_publish",
        lambda _destination, **_kwargs: {"manifest_sha256": "c" * 64},
    )
    return module.aggregate(sweep_id="extension-test", expected_commit="a" * 40)


def test_extension_matrix_is_four_explicit_scene_pairs() -> None:
    module = _load_runner()
    assert tuple(module.SCENE_VARIANTS) == (
        "27dd4da69e",
        "acd95847c5",
        "1ada7a0617",
        "25f3b7a318",
    )
    assert len(module.VARIANT_IDS) == 8
    assert tuple(module.VARIANTS) == tuple(
        variant
        for pair in module.SCENE_VARIANTS.values()
        for variant in pair
    )
    for scene_id, (floor_id, support_id) in module.SCENE_VARIANTS.items():
        assert module.VARIANTS[floor_id]["scene_id"] == scene_id
        assert module.VARIANTS[floor_id]["room_surface_policy"] == "robust_floor_only"
        assert module.VARIANTS[support_id]["scene_id"] == scene_id
        assert (
            module.VARIANTS[support_id]["room_surface_policy"]
            == "robust_floor_plus_support"
        )
    assert set(module.SCENE_VARIANTS).isdisjoint({"3db0a1c8f3", "d755b3d9d8"})


def test_extension_rebinds_both_reused_runner_layers() -> None:
    module = _load_runner()
    assert module._ROBUST.VARIANTS is module.VARIANTS
    assert module._ROBUST.VARIANT_IDS == module.VARIANT_IDS
    assert module._ROBUST._BASE.VARIANTS is module.VARIANTS
    assert module._ROBUST._BASE.VARIANT_IDS == module.VARIANT_IDS


def test_selection_rule_is_applied_independently_per_explicit_scene() -> None:
    module = _load_runner()
    for scene_id, (floor_id, support_id) in module.SCENE_VARIANTS.items():
        variants = {
            floor_id: _comparison(True, 2),
            support_id: _comparison(True, 2),
        }
        assert module._choose_scene_winner(scene_id, variants)[0] == floor_id
        variants[support_id] = _comparison(True, 3)
        assert module._choose_scene_winner(scene_id, variants)[0] == support_id
        variants[floor_id] = _comparison(False, 20)
        assert module._choose_scene_winner(scene_id, variants)[0] == support_id
    with pytest.raises(module.DiagnosticSweepError, match="outside the extension"):
        module._choose_scene_winner("d755b3d9d8", {})


def test_aggregate_is_complete_for_eight_scientific_failures(monkeypatch) -> None:
    module = _load_runner()
    comparisons = {
        variant_id: _comparison(False, 0) for variant_id in module.VARIANT_IDS
    }
    result = _fake_aggregate(module, monkeypatch, comparisons)
    assert result["status"] == "complete"
    assert result["errors"] == {}
    assert result["comparison_error_variants"] == []
    assert result["complete_variant_count"] == 8
    assert result["scene_winners"] == {
        scene_id: None for scene_id in module.SCENE_IDS
    }
    assert result["combined_scene_winners"] == {
        "d755b3d9d8": "d755-fs",
        **{scene_id: None for scene_id in module.SCENE_IDS},
    }


def test_aggregate_fails_closed_on_any_nested_comparison_error(monkeypatch) -> None:
    module = _load_runner()
    comparisons = {
        variant_id: _comparison(True, 2) for variant_id in module.VARIANT_IDS
    }
    broken = module.VARIANT_IDS[-1]
    comparisons[broken] = _comparison(
        False, 0, errors={"A4": "missing sealed policy bundle"}
    )
    result = _fake_aggregate(module, monkeypatch, comparisons)
    assert result["status"] == "incomplete"
    assert result["comparison_error_variants"] == [broken]
    assert set(result["scene_winners"].values()) == {None}
    assert set(result["combined_scene_winners"].values()) == {None}
    assert {
        row["reason"] for row in result["scene_decisions"].values()
    } == {"comparison_artifact_or_runtime_error"}


def test_aggregate_fails_closed_when_external_registry_is_invalid(
    monkeypatch,
) -> None:
    module = _load_runner()
    comparisons = {
        variant_id: _comparison(True, 2) for variant_id in module.VARIANT_IDS
    }
    monkeypatch.setattr(
        module,
        "_validated_external_winners",
        lambda _root: (_ for _ in ()).throw(ValueError("tampered external seal")),
    )
    result = _fake_aggregate(module, monkeypatch, comparisons)
    # _fake_aggregate installs the valid stub, so replace it once more and rerun.
    monkeypatch.setattr(
        module,
        "_validated_external_winners",
        lambda _root: (_ for _ in ()).throw(ValueError("tampered external seal")),
    )
    result = module.aggregate(
        sweep_id="extension-test", expected_commit="a" * 40
    )
    assert result["status"] == "incomplete"
    assert "external:d755b3d9d8" in result["errors"]
    assert set(result["scene_winners"].values()) == {None}
    assert set(result["combined_scene_winners"].values()) == {None}


def test_external_reference_is_aggregate_only_and_authorization_is_permanent_false() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    aggregate_source = source.split("def aggregate(", 1)[1]
    before_aggregate = source.split("def aggregate(", 1)[0]
    assert "_validated_external_winners(root)" in aggregate_source
    assert "_validated_external_winners(root)" not in before_aggregate
    assert "external_factories_are_never_reused" in source
    for key in ("gpu_launch_allowed", "large_rollout_launch_allowed", "paper_ready"):
        assert f'"{key}": False' in source
