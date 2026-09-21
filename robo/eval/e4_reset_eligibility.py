"""Consume sealed CPU reset evidence without dropping failed construction arms."""
from pathlib import Path
import json

from robo.eval import e4_candidate_screen as screen


def load_reset_eligibility(config, spec, states, *, root):
    declaration = config.get("cpu_reset_eligibility")
    if declaration is None:
        return {}
    if declaration.get("kind") in {"automatic_compact", "automatic_full"}:
        return _automatic_eligibility(config, spec, states, root=root)["indexed"]
    if config.get("paper_mode") or config.get("policy") != "scripted_sinusoid":
        raise ValueError("CPU reset eligibility adapter is engineering smoke only")
    if config.get("jitter_first_episode") is not True \
            or config.get("jitter") != screen.JITTER_XY_M:
        raise ValueError("harness jitter must match every CPU reset")
    bundle = screen._validate_bundle(
        Path(declaration["bundle"]), root=root,
        expected_kind="e4_one_target_winner_smoke_artifacts")
    directory = bundle["directory"]
    if screen._sha256(directory / "gate.json") != declaration["gate_sha256"]:
        raise ValueError("CPU eligibility gate hash differs")
    gate = json.loads((directory / "gate.json").read_text())
    code = screen._code_snapshot(declaration["source_commit"])
    if gate["code"] != code or gate["study_scope"] != "one_target_engineering_smoke":
        raise ValueError("CPU eligibility source or scope differs")
    task_bundle = gate["task_bundle"]
    from robo.eval.e4_task_freeze import validate_task_bundle
    from robo.eval import e4_camera_scorer_gate as camera_gate
    scene = config["scenes"][0]
    fresh_task_bundle = validate_task_bundle(
        scene["task_freeze_manifest"], expected_scene_id=scene["id"],
        repository_root=root)
    if fresh_task_bundle["bundle_manifest_sha256"] != task_bundle["bundle_manifest_sha256"]:
        raise ValueError("CPU eligibility task-bundle digest differs")
    fresh_menagerie = camera_gate._menagerie_snapshot(
        screen.EXPECTED_MENAGERIE_ROOT, screen.EXPECTED_MENAGERIE_COMMIT)
    if {k: v for k, v in fresh_menagerie.items() if k != "files"} != gate["menagerie"]:
        raise ValueError("CPU eligibility menagerie digest differs")
    if Path(scene.get("menagerie_root", "")).resolve() != screen.EXPECTED_MENAGERIE_ROOT:
        raise ValueError("CPU eligibility configured menagerie root differs")
    suites = {p: json.loads(Path(task_bundle["variant_tasks"][p]).read_text())
              for p in screen.POLICIES}
    factories = {p: Path(task_bundle["factories"][p]) for p in screen.POLICIES}
    contracts = screen._prepared_task_contracts({
        "gate": {**gate, "footprint_replays": json.loads(
            (directory / "footprint_replays.json").read_text())},
        "suites": suites, "factories": factories, "task_bundle": task_bundle})
    rows = [json.loads(line) for line in
            (directory / "metrics.jsonl").read_text().splitlines()]
    scene_id = suites["A0"]["scene"]
    # The existing strict replay checks geometry, reasons, thresholds, reset
    # jitter, all cells and their denominators. This adapter defines no metric.
    screen._replay_qualifier_metrics(
        rows, scene_id=scene_id, expected_task_contracts=contracts)
    treatments = {t.scene: t.id for t in spec.treatments.values()}
    if set(treatments) != {"fixed_single_path", "agentic"} or len(spec.treatments) != 2:
        raise ValueError("CPU eligibility needs both declared construction arms")
    if any(s.base_seed != screen.BASE_SEED for s in states):
        raise ValueError("CPU eligibility base seed differs")
    if len(config["scenes"]) != 1:
        raise ValueError("winner smoke requires its single declared scene")
    scene = config["scenes"][0]
    for policy, variant in (("A0", "fixed_single_path"), ("A4", "agentic")):
        configured = scene["construction_variants"][variant]
        if Path(configured.get("menagerie_root", scene["menagerie_root"])).resolve() != screen.EXPECTED_MENAGERIE_ROOT:
            raise ValueError("CPU eligibility variant menagerie root differs")
        for field, value in (("tasks_json", task_bundle["variant_tasks"][policy]),
                             ("factory_dir", task_bundle["factories"][policy]),
                             ("scene_xml", task_bundle["scene_xml"][policy])):
            if Path(configured[field]).resolve() != Path(value).resolve():
                raise ValueError("CPU eligibility artifact binding differs")
    return index_reset_eligibility(rows, states, treatments, declaration)


def index_reset_eligibility(rows, states, treatments, declaration):
    expected = {(s.scene_id, s.task_id, s.ep, s.reset_seed): s for s in states}
    if len(expected) != len(states):
        raise ValueError("duplicate planned reset")
    result = {}
    for policy, variant in (("A0", "fixed_single_path"), ("A4", "agentic")):
        observed = set()
        for row in (r for r in rows if r["policy_id"] == policy):
            identity = (row["scene_id"], row["task_id"], row["episode"], row["reset_seed"])
            if identity not in expected or identity in observed:
                raise ValueError("CPU eligibility unknown or duplicate reset")
            observed.add(identity)
            state = expected[identity]
            result[(treatments[variant], state.reset_state_id)] = {
                "passed": row["passed"],
                "failure_type": None if row["passed"] else "cpu_reset_validity_failure",
                "failed_checks": sorted(k for k, v in row["checks"].items() if not v),
                "cpu_cell_id": row["cell_id"],
                "gate_sha256": declaration["gate_sha256"],
                "source_commit": declaration["source_commit"],
                "bundle": declaration["bundle"],
            }
        if observed != set(expected):
            raise ValueError("CPU eligibility missing planned reset")
    if len(rows) != len(result):
        raise ValueError("CPU eligibility unknown policy rows")
    return result


def certify_prebuild_failure(record, spec, *, root):
    """Authenticate an unexecuted reset against its original sealed CPU cell.

    This reads historical producer evidence; its commit is kept distinct from
    the executing validator. It never supplies a fabricated simulator state.
    """
    evidence = record.get("construction_validity_evidence")
    if record.get("outcome") != "build_failure" or not isinstance(evidence, dict):
        return False
    if record.get("reset_provenance") is not None:
        raise ValueError("prebuild failure must not contain measured reset provenance")
    if spec.raw.get("cpu_reset_eligibility", {}).get("kind") in {"automatic_compact", "automatic_full"}:
        return _certify_automatic_failure(record, spec, root=root)
    if spec.raw.get("paper_mode") or spec.raw.get("policy") != "scripted_sinusoid":
        raise ValueError("prebuild CPU eligibility is engineering smoke only")
    declared_treatment = spec.treatments.get(record.get("treatment_id"))
    if declared_treatment is None or record.get("treatment") != declared_treatment.to_dict():
        raise ValueError("prebuild failure treatment differs from declared arm")
    declaration = spec.raw.get("cpu_reset_eligibility")
    if not isinstance(declaration, dict) or evidence.get("failure_type") != "cpu_reset_validity_failure":
        raise ValueError("prebuild failure has no declared CPU eligibility evidence")
    if any(evidence.get(key) != declaration.get(key)
           for key in ("bundle", "gate_sha256", "source_commit")):
        raise ValueError("prebuild failure CPU evidence identity differs")
    bundle = screen._validate_bundle(Path(declaration["bundle"]), root=root,
        expected_kind="e4_one_target_winner_smoke_artifacts")
    directory = bundle["directory"]
    if screen._sha256(directory / "gate.json") != declaration["gate_sha256"]:
        raise ValueError("prebuild failure CPU gate digest differs")
    gate = json.loads((directory / "gate.json").read_text())
    if gate["code"]["commit"] != declaration["source_commit"] or gate["code"]["dirty"]:
        raise ValueError("prebuild failure CPU producer differs")
    rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    matches = [row for row in rows if row["cell_id"] == evidence.get("cpu_cell_id")]
    if len(matches) != 1:
        raise ValueError("prebuild failure CPU cell is unknown")
    row = matches[0]
    policy = {"fixed_single_path": "A0", "agentic": "A4"}.get(declared_treatment.scene)
    contract = record["contract"]
    definition = contract.get("reset_definition", {})
    expected_id = f"{row['task_id']}__seed{screen.BASE_SEED}__ep{row['episode']}"
    if row["passed"] is not False or evidence.get("passed") is not False \
            or all(row["checks"].values()) \
            or evidence.get("failed_checks") != sorted(k for k,v in row["checks"].items() if not v) \
            or row["policy_id"] != policy \
            or any(row[k] != record.get(k) for k in ("scene_id", "task_id", "reset_seed")) \
            or record.get("reset_state_id") != expected_id \
            or definition.get("reset_state_id") != expected_id \
            or definition.get("reset_seed") != row["reset_seed"] \
            or definition.get("episode_index") != row["episode"] \
            or definition.get("base_seed") != screen.BASE_SEED \
            or definition.get("jitter") != row["reset_jitter"] \
            or contract.get("construction_artifacts", {}).get("task_freeze_manifest_sha256") \
                != gate["task_bundle"]["bundle_manifest_sha256"]:
        raise ValueError("prebuild failure differs from its measured CPU cell or planned reset")
    return True



def full_qualified_policy_selection(camera_config, chain, checked):
    """Replay the predeclared qualification-only selector, never policy outcomes."""
    import yaml
    from robo.eval import e4_camera_scorer_gate as camera
    from robo.eval import episode_log
    from run.icra2027 import e4_full_protocol
    if camera_config.get("study_scope")!=camera.FULL_AUTOMATIC_SCOPE:
        raise ValueError("full policy requires explicit full-cohort camera scope")
    if checked["gate"]["upstream"]!=chain["summary"]:
        raise ValueError("full policy selection source differs")
    protocol=yaml.safe_load(Path(camera_config["protocol"]["path"]).read_text())
    if (len(protocol['qualification_tasks'])!=269 or chain['summary'].get('planned')!=2690
            or len(chain['rows'])+len(chain.get('planned_unavailable',[]))!=2690):
        raise ValueError('full policy selection requires the complete 269-query source')
    rows={r["cell_id"]:r for r in chain["rows"]}
    cells={r["cell_id"]:r for r in checked["cells"]}
    if len(rows)!=len(chain["rows"]) or len(cells)!=len(checked["cells"]) or set(rows)!=set(cells):
        raise ValueError("full policy selection CPU/camera cell identities differ")
    qualifications=[]
    for q in protocol["qualification_tasks"]:
        ids=[f"{arm.lower()}__{q['task_id']}__seed0__ep{ep}" for arm in screen.POLICIES for ep in range(5)]
        qualifications.append(dict(task_id=q["task_id"],all_frozen_qualification_gates_pass=all(
            identity in rows and rows[identity]["passed"] is True
            and cells[identity]["passed"] is True for identity in ids)))
    matrix=e4_full_protocol.select_matrix(protocol,qualifications)
    if matrix["status"]!="READY_FOR_FROZEN_PILOT":
        return matrix,[],[],chain,checked["cells"]
    states=[episode_log.ResetState(reset_state_id=f"{q['task_id']}__seed0__ep{ep}",scene_id=q["scene_id"],
        task_id=q["task_id"],ep=ep,base_seed=0,reset_seed=episode_log.derive_reset_seed(0,q["task_id"],ep))
        for q in matrix["tasks"] for ep in range(5)]
    pilot_tasks=[min((q['task_id'] for q in matrix['tasks'] if q['scene_id']==scene and q['task_family']==family))
        for scene in matrix['scene_ids'][:2] for family in e4_full_protocol.FAMILIES]
    pilot_ids=[state.reset_state_id for state in states if state.task_id in pilot_tasks]
    selected_ids={q['task_id'] for q in matrix['tasks']}
    selected={**chain,'rows':[r for r in chain['rows'] if r['task_id'] in selected_ids],
        'scenes':{scene:chain['scenes'][scene] for scene in matrix['scene_ids']}}
    selected_cells=[r for r in checked['cells'] if r['task_id'] in selected_ids]
    if len(states)!=80 or len(pilot_ids)!=20 or len(selected['rows'])!=160 or len(selected_cells)!=160:
        raise ValueError("full qualified policy matrix/reset denominator differs")
    return matrix,states,pilot_ids,selected,selected_cells


def _automatic_eligibility(config, spec, states, *, root):
    """Normalize only the two producer validation errors at the adapter boundary."""
    from robo.eval.e4_camera_scorer_gate import CameraScorerGateError
    try:
        return _validated_automatic_eligibility(config, spec, states, root=root)
    except (CameraScorerGateError, screen.CandidateScreenError) as exc:
        raise ValueError(f"automatic eligibility {type(exc).__name__}: {exc}") from exc


def _validated_automatic_eligibility(config, spec, states, *, root):
    """Read the complete canonical CPU/camera evidence chain; never execute it."""
    import yaml
    from robo.eval import e4_camera_scorer_gate as camera
    from run.icra2027 import e4_compact_harness as compact

    declaration = config["cpu_reset_eligibility"]
    if (config.get("paper_mode") or config.get("jitter_first_episode") is not True
            or config.get("jitter") != screen.JITTER_XY_M
            or not (config.get("policy") == "pi05_droid_jointpos"
                or (config.get("policy") == "scripted_sinusoid"
                    and config.get("study_scope") in {"e4_compact_scripted_smoke","e4_full_scripted_smoke"}
                    and config.get("contract", {}).get("policy", {}).get("kind") == "scripted_smoke"))):
        raise ValueError("automatic eligibility requires the frozen engineering policy/reset protocol")
    reference = declaration["camera_config"]
    config_path = camera._regular_file(Path(reference["path"]), root=Path(root),
                                       label="automatic camera config")
    if screen._sha256(config_path) != reference["sha256"]:
        raise ValueError("automatic camera config hash differs")
    camera_config = yaml.safe_load(config_path.read_text())
    output = Path(declaration["bundle"]).resolve(strict=True)
    expected_output = (Path(root) / "outputs/icra2027" / camera_config["freeze_id"]
                       / "harness/automatic_camera_scorer").resolve()
    if output != expected_output or Path(declaration["bundle"]).is_symlink():
        raise ValueError("automatic camera bundle path differs")
    if screen._sha256(output / "gate.json") != declaration["gate_sha256"]:
        raise ValueError("automatic camera gate hash differs")
    chain = camera.validate_automatic_chain(camera_config, root=Path(root))
    checked = camera.validate_automatic_camera_output(config_path=config_path,
        output=output, expected_code_commit=declaration["source_commit"])
    if (checked["gate"]["code"]["commit"] != declaration["source_commit"]
            or checked["gate"]["code"]["dirty"]
            or checked["gate"]["upstream"] != chain["summary"]):
        raise ValueError("automatic eligibility source/upstream differs")
    is_full=declaration.get("kind")=="automatic_full"
    if is_full:
        matrix,canonical,pilot_ids,chain,camera_cells=full_qualified_policy_selection(camera_config,chain,checked)
        if (matrix["status"]!="READY_FOR_FROZEN_PILOT" or config.get("full_qualification_selection")!=matrix
                or config.get("pilot_reset_ids")!=pilot_ids
                or config.get("study_scope") not in {"automatic_full_policy_engineering","e4_full_scripted_smoke"}):
            raise ValueError("full qualified matrix or immutable pilot selection differs")
    else:
        if camera_config.get("study_scope")!=camera.AUTOMATIC_SCOPE:
            raise ValueError("compact eligibility cannot consume full camera scope")
        protocol = compact.checked_protocol(camera_config["protocol"]["path"],
                                             camera_config["protocol"]["sha256"])
        canonical = compact.definitions(protocol)
        camera_cells=checked["cells"]
    expected_resets=80 if is_full else 20
    identity = lambda state: (state.scene_id, state.task_id, state.ep,
                             state.base_seed, state.reset_seed, state.reset_state_id)
    if states is None:
        states = canonical
    if len(states) != expected_resets or {identity(s) for s in states} != {identity(s) for s in canonical}:
        raise ValueError("automatic eligibility canonical reset bank differs")
    treatments = {t.scene: t.id for t in spec.treatments.values()}
    if (set(treatments) != {"fixed_single_path", "agentic"} or len(spec.treatments) != 2
            or any(t.collision != "full_room" or t.observation != "raster" or t.options
                   for t in spec.treatments.values())):
        raise ValueError("automatic eligibility construction arms differ")
    scenes = {scene["id"]: scene for scene in config["scenes"]}
    if len(scenes) != len(config["scenes"]) or set(scenes) != set(chain["scenes"]):
        raise ValueError("automatic eligibility planned scenes differ")
    for scene_id, data in chain["scenes"].items():
        scene, bundle = scenes[scene_id], data["task_bundle"]
        if (Path(scene["menagerie_root"]).resolve() != Path(camera_config["menagerie_root"]).resolve()
                or Path(scene["tasks_json"]).resolve() != Path(bundle["planning_tasks"]).resolve()
                or screen._sha256(Path(scene["task_freeze_manifest"])) != bundle["bundle_manifest_sha256"]
                or set(scene["construction_variants"]) != set(treatments)):
            raise ValueError("automatic eligibility scene/task-bundle/menagerie differs")
        for policy, variant in (("A0", "fixed_single_path"), ("A4", "agentic")):
            configured = scene["construction_variants"][variant]
            if Path(configured.get("menagerie_root", scene["menagerie_root"])).resolve() != Path(camera_config["menagerie_root"]).resolve():
                raise ValueError("automatic eligibility variant menagerie differs")
            for field, expected in (("tasks_json", bundle["variant_tasks"][policy]),
                                    ("factory_dir", bundle["factories"][policy]),
                                    ("scene_xml", bundle["scene_xml"][policy])):
                if Path(configured[field]).resolve() != Path(expected).resolve():
                    raise ValueError("automatic eligibility paired artifact binding differs")
    rows = chain["rows"]
    indexed = index_reset_eligibility(rows, states, treatments, declaration)
    cells = {cell["cell_id"]: cell for cell in camera_cells}
    if len(rows) != 2*expected_resets or len(cells) != 2*expected_resets or len(camera_cells) != 2*expected_resets \
            or set(cells) != {r["cell_id"] for r in rows}:
        raise ValueError("automatic eligibility camera denominator differs")
    state_by_identity = {(s.scene_id, s.task_id, s.ep, s.reset_seed): s for s in states}
    for row in rows:
        cell = cells[row["cell_id"]]
        state = state_by_identity[(row["scene_id"], row["task_id"], row["episode"], row["reset_seed"])]
        if (any(cell.get(key) != row[key] for key in
                ("scene_id", "task_id", "policy_id", "episode", "reset_seed", "target"))
                or cell.get("reset_state_id") != state.reset_state_id):
            raise ValueError("automatic eligibility camera cell identity differs")
        unmeasured = not row["passed"] or cell["outcome"] == "diagnostic_error"
        if unmeasured and (cell.get("executed") is not False or cell.get("passed") is not False
                or any(cell.get(k) is not None for k in ("camera_metrics", "workspace_metrics",
                    "disambiguation_metrics", "scorer_metrics", "reset_provenance", "checks"))):
            raise ValueError("automatic unexecuted diagnostic has fabricated telemetry")
        variant = {"A0": "fixed_single_path", "A4": "agentic"}[row["policy_id"]]
        evidence = indexed[(treatments[variant], state.reset_state_id)]
        if row["passed"] and cell["passed"] is not True:
            evidence.update(passed=False, failure_type="camera_scorer_failure",
                failed_checks=(["diagnostic_error"] if cell["outcome"] == "diagnostic_error"
                    else sorted(k for k, value in cell["checks"].items() if not value)))
        evidence.update(kind=declaration["kind"], camera_config=dict(reference),
            camera_cell_id=cell["cell_id"], camera_manifest_sha256=checked["manifest_sha256"])
    return {"indexed": indexed, "chain": chain, "states": states, "rows": rows}


def _certify_automatic_failure(record, spec, *, root):
    """Certify planned failure without borrowing simulator or server telemetry."""
    from robo.eval import harness_runner as runner
    from robo.eval import paired_runner as legacy
    from robo.manifest import hash as manifest_hash
    from robo.eval.harness_validation import task_definition_hash

    checked = _automatic_eligibility(spec.raw, spec, None, root=root)
    treatment = spec.treatments.get(record.get("treatment_id"))
    if treatment is None or record.get("treatment") != treatment.to_dict():
        raise ValueError("automatic prebuild treatment differs from declared arm")
    expected = checked["indexed"].get((treatment.id, record.get("reset_state_id")))
    if expected is None or expected["passed"] is not False \
            or record.get("construction_validity_evidence") != expected:
        raise ValueError("automatic prebuild failure differs from its original failed cell")
    state = next(s for s in checked["states"] if s.reset_state_id == record["reset_state_id"])
    if any(record.get(key) != getattr(state, key) for key in ("scene_id", "task_id", "reset_seed", "base_seed")):
        raise ValueError("automatic prebuild planned reset identity differs")
    contract = record.get("contract", {})
    if (contract.get("policy_execution") != "not_invoked_prebuild"
            or contract.get("policy", {}).get("server_identity") is not None
            or contract.get("server_identity") is not None
            or record.get("server_identity") is not None
            or record.get("reset_provenance") is not None
            or contract.get("reset_provenance") is not None
            or any(container.get(key) is not None for container in (record, contract)
                   for key in ("camera_metrics", "workspace_metrics",
                               "disambiguation_metrics", "scorer_metrics"))):
        raise ValueError("automatic prebuild must not fabricate server/state execution")
    data = checked["chain"]["scenes"][state.scene_id]
    task = next(t for t in data["suites"]["A0"]["tasks"] if t["task_id"] == state.task_id)
    definition = runner._reset_definition(state, task=task, jitter_xy=screen.JITTER_XY_M)
    controller_hash, camera_hash, action, dimension = legacy._frozen_config_hashes()
    expected_fields = {"reset_definition": definition,
        "reset_definition_hash": manifest_hash.canonical_hash(definition),
        "task_definition_hash": task_definition_hash(task),
        "controller_config_hash": controller_hash, "camera_config_hash": camera_hash,
        "action_convention": action, "action_dim": dimension,
        "task_instruction": task["instructions"][spec.raw.get("variant", "default")], "task_id": state.task_id,
        "reset_state_id": state.reset_state_id, "rollout_seed": state.base_seed,
        "reset_seed": state.reset_seed, "policy_id": spec.raw["policy"],
        "policy_checkpoint_hash": spec.raw["contract"]["policy"]["checkpoint_hash"],
        "runtime_policy_checkpoint_fingerprint": spec.raw["contract"]["policy"]["checkpoint_hash"]}
    for key, value in spec.raw["contract"].items():
        if key not in expected_fields and contract.get(key) != value:
            raise ValueError("automatic prebuild frozen policy/contract fields differ")
    if (any(contract.get(k) != v for k,v in expected_fields.items())
            or contract.get("construction_artifacts", {}).get("task_freeze_manifest_sha256")
                != data["task_bundle"]["bundle_manifest_sha256"]):
        raise ValueError("automatic prebuild task digest or planned reset/contract differs")
    if (record.get("ticks") != 0 or record.get("success") is not False
            or record.get("score") != 0 or any(record.get("stages", {}).values())):
        raise ValueError("automatic prebuild contains fabricated rollout success")
    return True

def write_smoke_config(bundle_path, out_path, source_commit):
    """Generate the canonical ten-cell scripted harness config after CPU sealing."""
    import yaml
    from robo.eval import paired_runner
    from robo.eval.harness_spec import load_harness_spec

    root = screen.evidence_root()
    code = screen._code_snapshot(source_commit)
    bundle_path = Path(bundle_path).resolve()
    gate = json.loads((bundle_path / "gate.json").read_text())
    task_bundle = gate["task_bundle"]
    config = yaml.safe_load((screen.CODE_ROOT /
        "configs/experiments/icra2027/harness.yaml").read_text())
    config.update(policy="scripted_sinusoid", episodes=5, seeds=[0],
                  jitter=screen.JITTER_XY_M, jitter_first_episode=True,
                  variant="default", horizon_s=16.0, video=True,
                  study_scope="one_target_engineering_smoke", paper_mode=False)
    config["contract"]["horizon_s"] = 16.0
    config["treatments"] = [
        {"id": "a0_raster", "scene": "fixed_single_path",
         "collision": "full_room", "observation": "raster"},
        {"id": "a4_raster", "scene": "agentic",
         "collision": "full_room", "observation": "raster"}]
    config["comparisons"] = [{"id": "construction", "axis": "scene",
        "baseline": "a0_raster", "treatments": ["a0_raster", "a4_raster"]}]
    planning = json.loads(Path(task_bundle["planning_tasks"]).read_text())
    config["scenes"] = [{
        "id": planning["scene"], "tasks_json": task_bundle["planning_tasks"],
        "task_freeze_manifest": str(Path(task_bundle["planning_tasks"]).parent / "manifest.json"),
        "menagerie_root": str(screen.EXPECTED_MENAGERIE_ROOT),
        "construction_variants": {
            variant: {"factory_dir": task_bundle["factories"][policy],
                      "tasks_json": task_bundle["variant_tasks"][policy],
                      "scene_xml": task_bundle["scene_xml"][policy]}
            for policy, variant in (("A0", "fixed_single_path"), ("A4", "agentic"))}}]
    config["cpu_reset_eligibility"] = {
        "bundle": str(bundle_path), "gate_sha256": screen._sha256(bundle_path / "gate.json"),
        "source_commit": code["commit"]}
    states = paired_runner.plan_reset_states(config["scenes"], [0], 5)
    config["contract"]["reset_ids"] = [s.reset_state_id for s in states]
    config["out_dir"] = str(bundle_path.parent / "scripted_harness")
    load_reset_eligibility(config, load_harness_spec(config), states, root=root)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    return config


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-bundle", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-code-commit", required=True)
    args = parser.parse_args()
    write_smoke_config(args.cpu_bundle, args.out, args.expected_code_commit)
