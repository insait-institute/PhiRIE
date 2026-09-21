"""Freeze a small, input-selected E4 prerequisite through existing E3 producers.

No construction, physics, metric, or rollout implementation lives here. Scene
and future manipulation task selection use the complete discovery population
before any canonical-controller or robot-policy outcomes are inspected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from run.icra2027 import e3_fresh_canonical_config as canonical
from run.icra2027.e3_auto_discovery_pilot import identity, sha
from robo.tasks.pi05_tasks import GRASP_LABELS, RECEPTACLE_LABELS

CODE = Path(__file__).resolve().parents[2]
EVIDENCE = Path("/group/worldcept/PhiRIE/code/SimAny/outputs/icra2027")
DISCOVERY = EVIDENCE / "20260905-76c15d5-v1/auto_discovery_pilot"
TRELLIS = EVIDENCE / "20260905-02b54da-v2"
RVG = EVIDENCE / "20260905-d1e8e21-v1"
EXPECTED_SCENES = ["27dd4da69e", "40aec5fffa"]
COHORT = "configs/experiments/icra2027/construction_regimes.yaml"
OVERRIDES = "configs/experiments/icra2027/canonical_rvg_pool_overrides.yaml"
DIRECTORY = "configs/experiments/icra2027/e4_compact_canonical"


def semantic_pairs(scene, rows):
    """Use existing task-role vocabulary; no geometry/quality filtering."""
    objects = {f"obj_{r['automatic_instance_id']}": r for r in rows}
    if len(objects) != len(rows):
        raise ValueError("duplicate discovery automatic instance")
    targets = sorted(slot for slot, row in objects.items() if row["label"] in GRASP_LABELS)
    destinations = sorted(slot for slot, row in objects.items() if row["label"] in RECEPTACLE_LABELS)
    result = []
    for target in targets:
        for destination in [None, *(slot for slot in destinations if slot != target)]:
            result.append(dict(scene_id=scene, target=target, receptacle=destination,
                               task_family="object_to_region" if destination is None else "object_to_receptacle",
                               task_id=f"{scene}__{target}_to_{destination or 'region'}"))
    return result


def select_compact(scenes):
    """Deterministic input-only selection; no substitution after qualification."""
    eligible = [scene for scene in scenes if any(p["receptacle"] is not None for p in scene["pairs"])]
    chosen = sorted(eligible, key=lambda r: (r["planned_objects"], r["scene_id"]))[:2]
    if len(chosen) != 2:
        raise ValueError("the predeclared pilot requires two semantic-family scenes")
    tasks = []
    for scene in chosen:
        for family in ("object_to_region", "object_to_receptacle"):
            candidates = [p for p in scene["pairs"] if p["task_family"] == family]
            tasks.append(min(candidates, key=lambda p: p["task_id"]))
    return chosen, tasks


def make_protocol():
    cohort_path = CODE / COHORT
    scene_ids = [canonical.e3._require_scene_id(value) for value in
                 yaml.safe_load(cohort_path.read_text())["population"]["scene_ids"]]
    if len(scene_ids) != 50 or len(set(scene_ids)) != 50:
        raise ValueError("selection must inspect the complete frozen discovery roster")
    scenes = []
    for scene in scene_ids:
        path = DISCOVERY / scene / "all_jobs_manifest.json"
        report = json.loads(path.read_text())
        audit_path = path.parent / "postrun_audit.json"
        audit = json.loads(audit_path.read_text())
        if (sha(path) != audit["all_jobs_manifest_sha256"] or report["scene_id"] != scene
                or report["source_gaussian_training_provenance"] != canonical.FRESH
                or report["planned_jobs"] != len(report["rows"])):
            raise ValueError("complete input population is not authenticated")
        scenes.append(dict(scene_id=scene, planned_objects=report["planned_jobs"],
                           all_jobs_manifest=identity(path), discovery_audit=identity(audit_path),
                           pairs=semantic_pairs(scene, report["rows"])))
    chosen, tasks = select_compact(scenes)
    if [s["scene_id"] for s in chosen] != EXPECTED_SCENES:
        raise ValueError("input-only selection no longer yields the declared pilot scenes")
    return dict(schema_version=1, study_scope="e4_compact_canonical_engineering", paper_ready=False,
                selection_rule="two smallest complete discovered-object populations with both semantic task families; tie by scene_id",
                task_selection_rule="lexicographically first task_id per family per selected scene before any qualification",
                no_substitution_after_qualification=True, no_policy_outcome_selection=True,
                no_full_cohort_replacement=True, source_cohort=identity(cohort_path),
                role_definition=identity(CODE / "robo/tasks/pi05_tasks.py"),
                grasp_labels=sorted(GRASP_LABELS), receptacle_labels=sorted(RECEPTACLE_LABELS),
                population=dict(scene_ids=EXPECTED_SCENES, planned_objects=17, planned_policy_object_rows=85),
                source_populations=[{**{k: v for k, v in s.items() if k != "pairs"},
                    "semantic_region_queries": sum(p["receptacle"] is None for p in s["pairs"]),
                    "semantic_receptacle_queries": sum(p["receptacle"] is not None for p in s["pairs"])} for s in scenes],
                selected_semantic_queries=[p for s in chosen for p in s["pairs"]],
                planned_qualification_queries=28, planned_qualification_cells=280,
                fixed_manipulation_tasks=tasks, episodes_per_task_arm=5, reset_base_seed=0,
                jitter_xy_m=.01, planned_manipulation_episodes=40,
                construction_arms=["A0", "A4"], first_real_policy="pi05_droid_jointpos",
                raw_pool_override_manifest=identity(CODE / OVERRIDES),
                manipulation_launch_gate="canonical source, rig, fixed-task qualification, camera, scorer, scripted smoke and real-policy identity must pass",
                absent_or_failed_selected_tasks="retain all five reset rows for that arm; never replace a task",
                full_manipulation_claim="NOT_RUN")


def write_protocol(path):
    payload = make_protocol()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)
    return payload


def prepare(protocol_path, *, freeze_id=None, config_directory=None):
    protocol_path = Path(protocol_path).resolve(strict=True)
    protocol_path.relative_to(CODE)
    protocol = yaml.safe_load(protocol_path.read_text())
    if protocol != make_protocol():
        raise ValueError("predeclared input-only protocol changed")
    roster = yaml.safe_load((CODE / COHORT).read_text())["population"]["scene_ids"]
    overrides, override_anchor = canonical.load_pool_overrides(CODE / OVERRIDES, roster)
    bundles, histories = [], []
    for scene in EXPECTED_SCENES:
        replacement = overrides.get(scene)
        rvg_root = Path(replacement["freeze_root"]) if replacement else RVG
        rvg_dir = Path(replacement["path"]).parent if replacement else RVG / "rvg_initial" / scene
        terminal = (TRELLIS / "terminal_audit/20260905T161447Z" / f"{scene}.json")
        kwargs = dict(trellis_freeze_root=TRELLIS, rvg_freeze_root=rvg_root)
        if terminal.is_file():
            kwargs["trellis_terminal_audit"] = terminal
        missing = canonical._missing_inputs(DISCOVERY / scene, TRELLIS / "trellis_initial" / scene, rvg_dir,
                    **({"trellis_terminal_audit": terminal} if terminal.is_file() else {}))
        if missing:
            return dict(status="WAITING_REAL_INITIAL_POOLS", config_written=False, paper_ready=False, missing_inputs=missing)
        source, jobs = canonical._source_payload(DISCOVERY / scene, TRELLIS / "trellis_initial" / scene, rvg_dir, **kwargs)
        bundles.append((source, jobs))
        if replacement:
            histories.append(canonical.override_runtime_history(replacement, source, jobs))
    population = dict(scene_roster_config=str(protocol_path),
                      scene_roster_config_file_sha256=sha(protocol_path),
                      scene_roster_sha256=hashlib.sha256(("\n".join(EXPECTED_SCENES) + "\n").encode()).hexdigest(),
                      planned_scenes=2, planned_jobs_per_policy=17, planned_policy_object_rows=85)
    payload = dict(schema_version=1, study_scope="automatic_training_only_engineering", paper_ready=False,
                   output_dir="outputs/icra2027/{freeze_id}/agentic", population=population,
                   automatic_sources=[s for s, _ in bundles],
                   runtime_accounting=dict(schema_version=1, process_histories=histories),
                   pool_override_manifest=override_anchor)
    from agents.orchestrator.automatic_inventory import build_payloads
    _, proposals, audit, _ = build_payloads(payload, {"contract_sha256": "0" * 64,
        "freeze_id": "config-validation", "code": {"commit": "0" * 40}}, "config-validation")
    if not audit["initial_pool_complete"] or sum(len(jobs) for _, jobs in bundles) != 17:
        raise ValueError("canonical initial population is incomplete or differs")
    result = dict(status="READY_FOR_RESERVED_FREEZE", config_written=False, paper_ready=False,
                  scope="e4_compact_canonical_engineering", planned_scenes=2, planned_jobs=17,
                  planned_policy_object_rows=85, proposal_counts=proposals["counts"], population=population,
                  protocol=identity(protocol_path), fixed_manipulation_tasks=protocol["fixed_manipulation_tasks"],
                  planned_manipulation_episodes=40, runtime_accounting=payload["runtime_accounting"])
    if config_directory is None:
        return result
    return canonical._write_configs(payload, bundles, result, freeze_id, config_directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(CODE / DIRECTORY / "protocol.yaml"))
    parser.add_argument("--write-protocol", action="store_true")
    parser.add_argument("--freeze-id")
    parser.add_argument("--config-directory")
    args = parser.parse_args()
    result = write_protocol(args.protocol) if args.write_protocol else prepare(
        args.protocol, freeze_id=args.freeze_id, config_directory=args.config_directory)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
