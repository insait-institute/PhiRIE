import copy
from pathlib import Path
import subprocess

import pytest

from run.icra2027 import e4_compact_canonical as compact


def scene(name, n, labels):
    rows = [dict(automatic_instance_id=1000+i, label=label, prepared=True)
            for i, label in enumerate(labels)]
    return dict(scene_id=name, planned_objects=n, pairs=compact.semantic_pairs(name, rows))


def test_input_population_count_then_scene_tie_without_quality():
    rows = [scene("cccccccccc", 20, ["bottle", "bowl"]),
            scene("bbbbbbbbbb", 10, ["box", "bottle", "box"]),
            scene("aaaaaaaaaa", 10, ["bottle", "bowl"]),
            scene("0000000000", 1, ["chair"])]
    selected, tasks = compact.select_compact(rows)
    assert [s["scene_id"] for s in selected] == ["aaaaaaaaaa", "bbbbbbbbbb"]
    assert len(tasks) == 4
    assert [t["task_id"] for t in tasks] == ["aaaaaaaaaa__obj_1000_to_region",
        "aaaaaaaaaa__obj_1000_to_obj_1001", "bbbbbbbbbb__obj_1000_to_region",
        "bbbbbbbbbb__obj_1000_to_obj_1002"]


def test_unprepared_and_unavailable_targets_remain_in_task_declaration():
    rows = [dict(automatic_instance_id=1000, label="bottle", prepared=False),
            dict(automatic_instance_id=1001, label="bowl", prepared=False)]
    result = compact.semantic_pairs("aaaaaaaaaa", rows)
    assert len(result) == 2 and result[1]["receptacle"] == "obj_1001"


def test_quality_or_later_success_cannot_change_input_selection():
    rows = [scene("aaaaaaaaaa", 7, ["box", "bottle", "bowl"]),
            scene("bbbbbbbbbb", 10, ["bottle", "bowl"])]
    original = compact.select_compact(rows)
    changed = copy.deepcopy(rows)
    for row in changed:
        row.update(policy_success=False, qualification_pass=False, construction_accepted=False)
        for pair in row["pairs"]:
            pair.update(policy_success=False, qualification_pass=False)
    _, tasks = compact.select_compact(changed)
    assert [t["task_id"] for t in tasks] == [t["task_id"] for t in original[1]]


def test_requires_both_task_families_and_two_scenes():
    with pytest.raises(ValueError, match="two"):
        compact.select_compact([scene("aaaaaaaaaa", 1, ["bottle"])])


def test_duplicate_automatic_ids_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        compact.semantic_pairs("aaaaaaaaaa", [dict(automatic_instance_id=1000, label="box")] * 2)


def test_protocol_mutation_rejected_before_pool_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(compact, "CODE", tmp_path)
    path = tmp_path / "protocol.yaml"
    path.write_text("no_substitution_after_qualification: false\n")
    monkeypatch.setattr(compact, "make_protocol", lambda: {"no_substitution_after_qualification": True})
    with pytest.raises(ValueError, match="protocol changed"):
        compact.prepare(path)


@pytest.mark.parametrize("extra", [dict(SLURM_ARRAY_JOB_ID="123"), dict(E4_COMPACT_SCENE="not-declared"),
                                    dict(E4_COMPACT_PHASE="rollout"), dict(E4_COMPACT_CODE="relative")])
def test_wrapper_refuses_array_or_undeclared_launch(extra):
    import os
    script = compact.CODE / "run/icra2027/e4_compact_canonical.sh"
    env = dict(os.environ, E4_COMPACT_CODE="/missing/code", E4_COMPACT_FREEZE="/missing/freeze",
               E4_COMPACT_PHASE="control", E4_COMPACT_SCENE="27dd4da69e")
    env.update(extra)
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert result.returncode == 2


def test_wrapper_passes_only_existing_canonical_phase(tmp_path):
    import os
    script = compact.CODE / "run/icra2027/e4_compact_canonical.sh"
    old = tmp_path / "run/icra2027/e3_fresh_canonical.sbatch"
    old.parent.mkdir(parents=True)
    old.write_text('printf "%s %s %s" "$E3_CANONICAL_PHASE" "$E3_CANONICAL_SCENE" "$E3_CANONICAL_CODE"\n')
    env = dict(os.environ, E4_COMPACT_CODE=str(tmp_path), E4_COMPACT_FREEZE=str(tmp_path / "freeze"),
               E4_COMPACT_PHASE="control", E4_COMPACT_SCENE="27dd4da69e")
    env.pop("SLURM_ARRAY_JOB_ID", None)
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout == f"control 27dd4da69e {tmp_path}"
