import copy
import json

import numpy as np
import pytest

from robo.eval.harness_spec import load_harness_spec
from robo.eval.harness_validation import (
    validate_manifest_pair,
    validate_records,
    validate_reset_provenance,
    validate_saved_treatment_records,
    validate_treatment_manifests,
)
from robo.manifest.hash import canonical_hash, hash_checkpoint_path


def test_validation_requires_every_planned_reset_per_treatment():
    spec = load_harness_spec({
        "treatments": [{"id": "a", "scene": "box_proxy"}, {"id": "b"}],
        "comparisons": [{"id": "scene", "axis": "scene",
                         "treatments": ["a", "b"]}]})
    records = [
        {"treatment_id": "a", "reset_state_id": "r0", "outcome": "task_failure"},
        {"treatment_id": "b", "reset_state_id": "r0", "outcome": "success"}]
    assert validate_records(records, spec, {"r0"})["ok"]
    assert not validate_records(records[:-1], spec, {"r0"})["ok"]


def _contract(policy_hash="ab" * 32):
    return {
        "policy": {
            "id": "real-policy", "kind": "real",
            "checkpoint_hash": policy_hash,
            "checkpoint_hash_kind": "content_sha256",
        },
        "robot": {"arm": "panda", "gripper": "robotiq"},
        "cameras": {"exterior": "ext", "wrist": "wrist"},
        "action_convention": "absolute_joint_position",
        "controller": {"name": "joint-position"},
        "control_rate_hz": 15,
        "horizon_s": 32,
        "task_instruction": "place the object",
        "rubric": {"version": "v1"},
        "reset_ids": ["r0"],
    }


def _paper_config(policy_hash="ab" * 32):
    return {
        "paper_mode": True,
        "contract": _contract(policy_hash),
        "treatments": [
            {"id": "proxy", "scene": "box_proxy"},
            {"id": "sim", "scene": "simany"},
            {
                "id": "harmony", "scene": "simany", "observation": "harmonizer_c",
                "options": {"enhancer": {
                    "backend": "socket", "checkpoint_hash": "cd" * 32,
                    "checkpoint_hash_kind": "content_sha256",
                }},
            },
        ],
        "comparisons": [
            {"id": "scene", "axis": "scene", "treatments": ["proxy", "sim"]},
            {"id": "observation", "axis": "observation",
             "treatments": ["sim", "harmony"]},
        ],
    }


def _manifest(treatment_id, scene, observation="raster"):
    return {
        "schema_version": 1,
        "manifest_kind": "harness_rollout",
        "treatment_id": treatment_id,
        "outcome": "task_failure",
        "treatment": {
            "id": treatment_id, "scene": scene,
            "collision": "full_room", "observation": observation,
            "options": {},
        },
        "contract": {
            "policy_id": "real-policy",
            "policy_checkpoint_hash": "ab" * 32,
            "robot": {"arm": "panda", "gripper": "robotiq"},
            "cameras": {"exterior": "ext", "wrist": "wrist"},
            "action_convention": "absolute_joint_position",
            "controller": {"name": "joint-position"},
            "control_rate_hz": 15,
            "horizon_s": 32,
            "task_instruction": "place the object",
            "rubric": {"version": "v1"},
            "reset_state_id": "r0",
        },
    }


def _reset_provenance(*, post_object_x=0.0):
    state = {
        "time_s": 0.0,
        "robot_arm_qpos_rad": [0.0],
        "robot_arm_qvel_rad_s": [0.0],
        "non_object_joint_states": {"robot": {"qpos": [0.0], "qvel": [0.0]}},
        "object_states": {
            "obj_00": {
                "world_position_m": [0.0, 0.0, 0.7],
                "world_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "joints": {"obj_00_joint": {"qpos": [0.0], "qvel": [0.0]}},
            }
        },
        "actuator_controls": {"arm": 0.0},
    }
    post = copy.deepcopy(state)
    post["time_s"] = 1.5
    post["object_states"]["obj_00"]["world_position_m"][0] = post_object_x
    seed = 123
    draw = np.random.RandomState(seed).random_sample(2)
    jitter = {
        "algorithm": "numpy.random.RandomState.random_sample_then_affine",
        "reset_seed": seed,
        "body": "obj_00",
        "max_abs_xy_m": 0.02,
        "uniform_draw_0_1": [float(value) for value in draw],
        "offset_xy_m": [float(value) for value in (2.0 * draw - 1.0) * 0.02],
        "applied": True,
    }
    settle = {
        "engine": "mujoco",
        "step_function": "mujoco.mj_step",
        "requested_duration_s": 1.5,
        "model_timestep_s": 0.002,
        "step_count": 750,
        "simulated_duration_s": 1.5,
    }
    return {
        "schema_version": 1,
        "pre_jitter_nominal_state": state,
        "pre_jitter_nominal_state_sha256": canonical_hash(state),
        "jitter": jitter,
        "jitter_sha256": canonical_hash(jitter),
        "settle_protocol": settle,
        "settle_protocol_sha256": canonical_hash(settle),
        "post_settle_pre_policy_state": post,
        "post_settle_pre_policy_state_sha256": canonical_hash(post),
    }


def test_undeclared_manifest_drift_reports_exact_frozen_field():
    config = _paper_config()
    config["paper_mode"] = False
    spec = load_harness_spec(config)
    proxy = _manifest("proxy", "box_proxy")
    sim = _manifest("sim", "simany")
    sim["contract"]["cameras"]["wrist"] = "different-camera"
    report = validate_treatment_manifests([proxy, sim], spec, {"r0"})
    assert not report["ok"]
    assert any("contract.cameras" in value for value in report["violations"])


def test_only_declared_axis_is_ignored_by_manifest_diff():
    proxy = _manifest("proxy", "box_proxy")
    sim = _manifest("sim", "simany")
    assert validate_manifest_pair(proxy, sim, {"scene"}) == []
    sim["contract"]["control_rate_hz"] = 10
    assert validate_manifest_pair(proxy, sim, {"scene"}) == [
        "contract.control_rate_hz"]


def test_task_and_reset_hashes_are_paired_frozen_fields():
    config = _paper_config()
    config["paper_mode"] = False
    spec = load_harness_spec(config)
    proxy = _manifest("proxy", "box_proxy")
    sim = _manifest("sim", "simany")
    for manifest in (proxy, sim):
        manifest["contract"].update({
            "task_id": "logical_task",
            "task_definition_hash": "12" * 32,
            "reset_definition_hash": "34" * 32,
            "rollout_seed": 0,
            "reset_seed": 123,
        })
    sim["contract"]["reset_definition_hash"] = "56" * 32
    report = validate_treatment_manifests([proxy, sim], spec, {"r0"})
    assert not report["ok"]
    assert any("contract.reset_definition_hash" in value
               for value in report["violations"])


def test_reset_provenance_replays_seed_and_authenticates_nested_states():
    provenance = _reset_provenance()
    reset_definition = {"jitter": copy.deepcopy(provenance["jitter"])}
    assert validate_reset_provenance(
        provenance, reset_definition=reset_definition) == []

    tampered = copy.deepcopy(provenance)
    tampered["post_settle_pre_policy_state"]["robot_arm_qpos_rad"] = [0.1]
    violations = validate_reset_provenance(
        tampered, reset_definition=reset_definition)
    assert "reset_provenance.post_settle_pre_policy_state_sha256 differs" in violations

    tampered = copy.deepcopy(provenance)
    tampered["jitter"]["uniform_draw_0_1"] = [0.1, 0.2]
    tampered["jitter"]["offset_xy_m"] = [-0.016, -0.012]
    tampered["jitter_sha256"] = canonical_hash(tampered["jitter"])
    violations = validate_reset_provenance(
        tampered, reset_definition={"jitter": tampered["jitter"]})
    assert any("differs from seed" in value for value in violations)


def test_reset_provenance_uses_authoritative_near_integer_step_rounding():
    provenance = _reset_provenance()
    settle = provenance["settle_protocol"]
    settle["model_timestep_s"] = 1.0 / 600.0
    settle["step_count"] = 900
    settle["simulated_duration_s"] = 900 * (1.0 / 600.0)
    provenance["settle_protocol_sha256"] = canonical_hash(settle)
    reset_definition = {"jitter": copy.deepcopy(provenance["jitter"])}
    assert validate_reset_provenance(
        provenance, reset_definition=reset_definition
    ) == []

    truncated = copy.deepcopy(provenance)
    truncated["settle_protocol"]["step_count"] = 899
    truncated["settle_protocol"]["simulated_duration_s"] = 899 * (1.0 / 600.0)
    truncated["settle_protocol_sha256"] = canonical_hash(
        truncated["settle_protocol"]
    )
    violations = validate_reset_provenance(
        truncated, reset_definition=reset_definition
    )
    assert "reset_provenance.settle_protocol step_count differs" in violations
    assert (
        "reset_provenance.settle_protocol simulated_duration_s differs"
        not in violations
    )


def test_reset_state_pairing_is_axis_aware():
    a0 = _manifest("a0", "fixed_single_path")
    a4 = _manifest("a4", "agentic")
    a0["reset_provenance"] = _reset_provenance(post_object_x=0.0)
    a4["reset_provenance"] = _reset_provenance(post_object_x=0.01)
    assert validate_manifest_pair(a0, a4, {"scene"}) == []

    observation_drift = validate_manifest_pair(a0, a4, {"observation"})
    assert any("post_settle_pre_policy_state.object_states" in value
               for value in observation_drift)

    a4["reset_provenance"]["post_settle_pre_policy_state"][
        "robot_arm_qpos_rad"] = [0.1]
    a4["reset_provenance"]["post_settle_pre_policy_state_sha256"] = canonical_hash(
        a4["reset_provenance"]["post_settle_pre_policy_state"])
    construction_drift = validate_manifest_pair(a0, a4, {"scene"})
    assert "reset_provenance.post_settle_pre_policy_state.robot_arm_qpos_rad" in (
        construction_drift)


def test_certified_prebuild_absence_keeps_plan_and_executed_state_checks_strict():
    config = _paper_config()
    config["paper_mode"] = False
    config["contract"]["policy"]["kind"] = "scripted_smoke"
    config["policy"] = config["contract"]["policy"]["id"]
    config.update({"treatments": [
        {"id": "a0", "scene": "fixed_single_path"}, {"id": "a4", "scene": "agentic"}],
        "comparisons": [{"id": "construction", "axis": "scene", "treatments": ["a0", "a4"]}]})
    spec = load_harness_spec(config)
    failed = _manifest("a0", "fixed_single_path")
    failed["outcome"] = "build_failure"
    executed = _manifest("a4", "agentic")
    executed["reset_provenance"] = _reset_provenance()
    definition = {"reset_state_id": "r0", "jitter": executed["reset_provenance"]["jitter"]}
    for value in (failed, executed):
        value["contract"]["reset_definition"] = copy.deepcopy(definition)
        value["contract"]["reset_definition_hash"] = canonical_hash(definition)
    certified = frozenset({("a0", "r0")})
    assert not validate_treatment_manifests([failed, executed], spec, {"r0"})["ok"]
    assert validate_treatment_manifests([failed, executed], spec, {"r0"},
        certified_prebuild_failures=certified)["ok"]
    missing = copy.deepcopy(executed)
    missing.pop("reset_provenance")
    assert not validate_treatment_manifests([failed, missing], spec, {"r0"},
        certified_prebuild_failures=certified)["ok"]
    changed = copy.deepcopy(failed)
    changed["contract"]["controller"]["name"] = "changed"
    assert not validate_treatment_manifests([changed, executed], spec, {"r0"},
        certified_prebuild_failures=certified)["ok"]
    changed = copy.deepcopy(executed)
    changed["reset_provenance"]["jitter"]["offset_xy_m"][0] += .01
    assert not validate_treatment_manifests([failed, changed], spec, {"r0"},
        certified_prebuild_failures=certified)["ok"]
    changed = copy.deepcopy(failed)
    changed["contract"]["reset_definition"]["jitter"]["reset_seed"] += 1
    changed["contract"]["reset_definition_hash"] = canonical_hash(changed["contract"]["reset_definition"])
    assert not validate_treatment_manifests([changed, executed], spec, {"r0"},
        certified_prebuild_failures=certified)["ok"]


def test_observation_backend_options_are_part_of_declared_observation_axis():
    raster = _manifest("raster", "simany")
    harmony = _manifest("harmony", "simany", "harmonizer_c")
    harmony["treatment"]["options"] = {
        "enhancer": {"backend": "socket", "checkpoint_hash": "cd" * 32},
        "restoration": {"dilate_px": 4},
    }
    assert validate_manifest_pair(raster, harmony, {"observation"}) == []


def test_placeholder_policy_checkpoint_is_rejected_in_paper_mode():
    with pytest.raises(ValueError, match="contract.policy.checkpoint_hash is a placeholder"):
        load_harness_spec(_paper_config("PLACEHOLDER"))


def test_identity_harmonizer_is_rejected_in_paper_mode():
    config = _paper_config()
    config["treatments"][2]["options"]["enhancer"] = {
        "backend": "identity", "allow_smoke_only": True,
    }
    with pytest.raises(ValueError, match="identity enhancer in paper mode"):
        load_harness_spec(config)


def test_paper_checkpoint_hashes_are_bound_to_existing_paths(tmp_path):
    policy_path = tmp_path / "policy.bin"
    harmony_path = tmp_path / "harmony.bin"
    policy_path.write_bytes(b"policy")
    harmony_path.write_bytes(b"harmony")
    config = _paper_config(hash_checkpoint_path(policy_path))
    config["contract"]["policy"].update({
        "checkpoint_path": str(policy_path),
        "checkpoint_hash_kind": "path_size_mtime_sha256",
    })
    enhancer = config["treatments"][2]["options"]["enhancer"]
    enhancer.update({
        "checkpoint_path": str(harmony_path),
        "checkpoint_hash": hash_checkpoint_path(harmony_path),
        "checkpoint_hash_kind": "path_size_mtime_sha256",
    })
    assert load_harness_spec(config).raw["paper_mode"] is True
    config["contract"]["policy"]["checkpoint_hash"] = "ef" * 32
    with pytest.raises(ValueError, match="does not match checkpoint_path fingerprint"):
        load_harness_spec(config)


def test_failure_row_counts_as_coverage_and_missing_row_fails():
    config = _paper_config()
    config["paper_mode"] = False
    spec = load_harness_spec(config)
    rows = [
        {"treatment_id": treatment, "reset_state_id": "r0",
         "outcome": "build_failure"}
        for treatment in spec.treatment_ids()
    ]
    complete = validate_records(rows, spec, {"r0"})
    assert complete["ok"]
    assert complete["coverage"] == 1.0
    assert complete["evaluable_coverage"] == 0.0
    missing = validate_records(rows[:-1], spec, {"r0"})
    assert not missing["ok"]
    assert "is missing 1 planned resets" in " | ".join(missing["violations"])


def test_per_treatment_frozen_contract_override_is_rejected():
    config = _paper_config()
    config["paper_mode"] = False
    config["treatments"][1].setdefault("options", {})["contract_overrides"] = {
        "control_rate_hz": 10,
    }
    with pytest.raises(ValueError, match="control_rate_hz"):
        load_harness_spec(copy.deepcopy(config))


def test_unknown_treatment_option_drift_is_rejected():
    config = _paper_config()
    config["paper_mode"] = False
    config["treatments"][1].setdefault("options", {})["horizon_s"] = 10
    with pytest.raises(ValueError, match=r"options\.horizon_s"):
        load_harness_spec(config)


def test_saved_manifests_and_explicit_failure_contracts_are_validated(tmp_path):
    config = {
        "paper_mode": False,
        "contract": _contract(),
        "treatments": [
            {"id": "proxy", "scene": "box_proxy"},
            {"id": "sim", "scene": "simany"},
        ],
        "comparisons": [{
            "id": "scene", "axis": "scene", "treatments": ["proxy", "sim"],
        }],
    }
    spec = load_harness_spec(config)
    proxy = _manifest("proxy", "box_proxy")
    proxy_path = tmp_path / "proxy.json"
    proxy_path.write_text(json.dumps(proxy))
    sim = _manifest("sim", "simany")
    records = [
        {"episode_id": "proxy-r0", "manifest_path": str(proxy_path)},
        {
            "episode_id": "sim-r0", "treatment_id": "sim",
            "scene_id": "scene", "reset_state_id": "r0",
            "treatment": sim["treatment"], "contract": sim["contract"],
            "outcome": "build_failure",
        },
    ]
    report = validate_saved_treatment_records(records, spec, {"r0"})
    assert report["ok"]
    assert report["coverage"] == 1.0
    assert report["loaded_manifests"] == 2

    sim["contract"]["controller"]["name"] = "different-controller"
    records[1]["contract"] = sim["contract"]
    report = validate_saved_treatment_records(records, spec, {"r0"})
    assert not report["ok"]
    assert any("contract.controller.name" in value
               for value in report["violations"])


def test_certified_noninvocation_pair_keeps_live_identity_and_frozen_fields(monkeypatch):
    from robo.eval import harness_validation as validation
    from robo.policy.runtime_identity import PolicyRuntimeIdentityError
    config = _paper_config()
    config['paper_mode'] = False
    config['contract']['policy']['training_config'] = 'pi05_droid'
    config.update(treatments=[{'id':'a0','scene':'fixed_single_path'},
        {'id':'a4','scene':'agentic'}], comparisons=[{'id':'construction',
        'axis':'scene','treatments':['a0','a4']}])
    # This unit isolates saved-manifest validation from registry/filesystem preflight.
    config['contract']['policy']['kind'] = 'scripted_smoke'
    config['policy'] = config['contract']['policy']['id']
    spec = load_harness_spec(config)
    config['contract']['policy']['kind'] = 'real'
    expected_policy = config['contract']['policy']
    monkeypatch.setattr(validation, 'expected_server_identity_from_config', lambda _: {'expected':True})
    def verify(identity, *, expected, verify_runtime_checkpoint_now):
        if identity != {'verified_fixture':True}:
            raise PolicyRuntimeIdentityError('invalid live identity')
        return {'policy':{'checkpoint':{'fingerprint':'ab'*32},'training_config':'pi05_droid'}}
    monkeypatch.setattr(validation, 'validate_server_identity', verify)
    failed = _manifest('a0', 'fixed_single_path')
    failed['outcome'] = 'build_failure'
    executed = _manifest('a4', 'agentic')
    executed['reset_provenance'] = _reset_provenance()
    definition = {'reset_state_id':'r0','jitter':executed['reset_provenance']['jitter']}
    for item in (failed, executed):
        item['contract'].update(policy=copy.deepcopy(expected_policy),
            runtime_policy_checkpoint_fingerprint='ab'*32,
            reset_definition=copy.deepcopy(definition), reset_definition_hash=canonical_hash(definition))
    failed['contract']['policy_execution'] = 'not_invoked_prebuild'
    executed['contract']['policy']['server_identity'] = {'verified_fixture':True}
    certified = frozenset({('a0','r0')})
    def report(a=failed,b=executed):
        return validate_treatment_manifests([a,b],spec,{'r0'},certified_prebuild_failures=certified)
    assert report()['ok'], report()
    for field,value in [('controller',{'name':'changed'}),('policy_checkpoint_hash','cd'*32),
                        ('reset_definition_hash','ef'*32),('action_convention','changed')]:
        changed=copy.deepcopy(failed);changed['contract'][field]=value
        assert not report(a=changed)['ok'], field
    changed=copy.deepcopy(executed)
    changed['contract']['policy']['server_identity']={'forged':True}
    assert any('invalid live identity' in x for x in report(b=changed)['violations'])
    changed=copy.deepcopy(executed);changed.pop('reset_provenance')
    assert not report(b=changed)['ok']
    changed=copy.deepcopy(failed);changed['contract']['policy']['training_config']='changed'
    assert not report(a=changed)['ok']
