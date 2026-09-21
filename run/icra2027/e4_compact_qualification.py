"""Source-bound compact qualification/camera launcher; no policy execution."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
from pathlib import Path
import sys

from robo.eval import e4_candidate_screen as screen
from robo.eval import e4_camera_scorer_gate as camera
from run.icra2027 import e4_compact_harness as compact

MATERIALIZATION_COMMIT = "14964859267ac986cd595344737570b0d733a313"
MATERIALIZATION_FREEZE = "20260905-e8277d2-v1"
MATERIALIZATION_CONTRACT = "bfe82199cd6103514e02025d19dfb7da65a47ef7392b339d2d11bbca9a035230"
SCOPE = "e4_compact_qualification_engineering"


def _api():
    from run.icra2027 import e4_compact_materialization
    return e4_compact_materialization


def _validate_original_variants(requests, runtime):
    """Keep the historical validator at its genuine recorded source/worktree."""
    manifests = [json.loads((Path(r["factory"])/"materialization_manifest.json").read_text()) for r in requests]
    roots = {m["provenance"]["code_root"] for m in manifests}
    if len(roots) != 1 or any(m["provenance"]["validator_commit"] != MATERIALIZATION_COMMIT for m in manifests):
        raise ValueError("original validator source roots differ")
    code_root = Path(next(iter(roots))).resolve(strict=True)
    head = subprocess.check_output(["git","rev-parse","HEAD"],cwd=code_root,text=True).strip()
    dirty = subprocess.check_output(["git","status","--porcelain","--untracked-files=normal"],cwd=code_root,text=True)
    if head != MATERIALIZATION_COMMIT or dirty.strip():
        raise ValueError("original validator worktree is no longer exact and clean")
    script = ("import json,sys; from robo.eval.e3_factory_materializer import validate_materialized_factory; "
        "requests=json.load(sys.stdin); "
        "print(json.dumps([validate_materialized_factory(r['factory'],expected_scene_id=r['scene'],"
        "expected_policy_id=r['arm']) for r in requests]))")
    env = dict(os.environ, SIMANY_EVIDENCE_ROOT=str(screen.evidence_root()))
    env.pop("PYTHONPATH", None)
    result = subprocess.run([runtime["path"],"-c",script],cwd=code_root,env=env,
        input=json.dumps(requests),text=True,capture_output=True,check=True)
    reports = json.loads(result.stdout)
    if len(reports) != len(requests):
        raise ValueError("original validator returned a different population")
    return reports


def inspect_materialization(handoff_path):
    """Reauthenticate original destination-bound products without relocating them."""
    api = _api()
    identity = api.identity(handoff_path)
    handoff = api.read(identity["path"])
    if (handoff.get("status") != "PASS" or handoff.get("paper_ready") is not False
            or handoff.get("materializer_commit") != MATERIALIZATION_COMMIT
            or handoff.get("planned_manipulation_episodes") != 40
            or handoff.get("fixed_task_ids") != api.FIXED_TASKS
            or set(handoff.get("scenes", {})) != set(compact.SCENES)):
        raise ValueError("original materialization handoff scope/source/population differs")
    stage_root = Path(identity["path"]).parent.parent
    contract_path = stage_root / "contract/freeze_manifest.json"
    contract = api.read(contract_path)
    if (api.contract_digest(contract) != MATERIALIZATION_CONTRACT
            or stage_root.name != MATERIALIZATION_FREEZE
            or contract["code"].get("commit") != MATERIALIZATION_COMMIT
            or contract["code"].get("dirty") is not False
            or contract.get("freeze_id") != stage_root.name):
        raise ValueError("original materialization E0 source differs")
    stage_config_identity = api.resource(contract, "e4_compact_materialization_config")
    stage_config = api.read(stage_config_identity["path"])
    reviewed = api.inspect_source(handoff["source"]["e3_root"])
    if reviewed["status"] != "READY":
        return reviewed
    if reviewed["source"] != handoff["source"] or stage_config.get("source") != handoff["source"]:
        raise ValueError("original materialization source closure changed")
    protocol = reviewed["source"]["protocol"]
    compact.checked_protocol(protocol["path"], protocol["sha256"])
    requests = []
    for scene, count in api.SCENES.items():
        source = handoff["scenes"][scene]
        for field in ("receipt", "descriptor"):
            if api.identity(source[field]["path"]) != source[field]:
                raise ValueError("original materialization receipt/descriptor changed")
        receipt = api.read(source["receipt"]["path"])
        if (receipt.get("source") != reviewed["source"] or receipt.get("scene_id") != scene
                or receipt.get("materializer_commit") != MATERIALIZATION_COMMIT
                or receipt.get("status") != "PASS" or receipt.get("paper_ready") is not False
                or receipt.get("planned_objects") != count
                or receipt.get("planned_manipulation_episodes") != 20
                or receipt.get("stage_config") != stage_config_identity):
            raise ValueError("original materialization scene receipt changed")
        if api.read(source["descriptor"]["path"]) != reviewed["source"]["scenes"][scene]["descriptor"]:
            raise ValueError("materialization descriptor differs from canonical source")
        for arm in compact.ARMS:
            member = source["variants"][arm]
            if (api.identity(member["path"]) != member
                    or receipt["variants"][arm]["manifest"] != member):
                raise ValueError("original materialization variant identity changed")
            requests.append(dict(factory=str(Path(member["path"]).parent),scene=scene,arm=arm))
    runtime = api.resource(contract, "materialization_python")
    reports = _validate_original_variants(requests, runtime)
    for request, report in zip(requests,reports):
        if report["roster"]["job_count"] != api.SCENES[request["scene"]]:
            raise ValueError("original materialization population changed")
    return dict(status="READY", source=dict(handoff=identity, materialization_e0=api.identity(contract_path),
        original_validator_runtime=runtime, materialization_config=stage_config_identity, canonical=reviewed["source"]))


def prepare_config(*, handoff_path, freeze_id, menagerie_root, out, export_reuse_root=None):
    """Prepare a new external config for later exact-source E0 sealing."""
    api = _api()
    reviewed = inspect_materialization(handoff_path)
    if reviewed["status"] != "READY":
        return reviewed
    screen._validated_screen_id(freeze_id)
    model = camera._menagerie_snapshot(Path(menagerie_root), camera.EXPECTED_MENAGERIE_COMMIT)
    config = dict(schema_version=1, scope=SCOPE, paper_ready=False, freeze_id=freeze_id,
        source=reviewed["source"], python=str(Path(sys.executable).resolve()),
        menagerie_root=str(Path(menagerie_root).resolve()),
        menagerie={k:v for k,v in model.items() if k != "files"},
        openpi_resize_identity=camera._openpi_snapshot())
    if export_reuse_root is not None:
        from run.icra2027 import e4_automatic_export_reuse as reuse
        config['export_reuse'] = {scene:reuse.inspect_source(export_reuse_root, scene) for scene in compact.SCENES}
    with Path(out).open("x") as stream:
        json.dump(config, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return config


def validate_stage(config_path, stage_root, expected_commit):
    api = _api()
    code = screen._code_snapshot(expected_commit)
    config = api.read(config_path)
    stage_root = Path(stage_root).resolve(strict=True)
    expected_root = screen._experiment_root(screen.evidence_root(), config["freeze_id"])
    contract = api.read(stage_root / "contract/freeze_manifest.json")
    api.contract_digest(contract)
    if (config.get("schema_version") != 1 or config.get("scope") != SCOPE
            or config.get("paper_ready") is not False or stage_root != expected_root
            or contract.get("freeze_id") != config["freeze_id"]
            or contract["code"].get("commit") != code["commit"]
            or contract["code"].get("dirty") is not False):
        raise ValueError("qualification scope/path or exact-source E0 differs")
    if api.resource(contract, "e4_compact_qualification_config") != api.identity(config_path):
        raise ValueError("qualification config differs from E0")
    runtime = api.resource(contract, "qualification_python")
    if (Path(runtime["path"]).resolve() != Path(sys.executable).resolve()
            or Path(config["python"]).resolve() != Path(sys.executable).resolve()):
        raise ValueError("qualification Python differs")
    reviewed = inspect_materialization(config["source"]["handoff"]["path"])
    if reviewed["status"] != "READY":
        return config, reviewed
    if reviewed["source"] != config["source"]:
        raise ValueError("qualification frozen upstream identity changed")
    model = camera._menagerie_snapshot(Path(config["menagerie_root"]), camera.EXPECTED_MENAGERIE_COMMIT)
    if ({k:v for k,v in model.items() if k != "files"} != config["menagerie"]
            or camera._openpi_snapshot() != config["openpi_resize_identity"]):
        raise ValueError("qualification model or policy resize changed")
    if 'export_reuse' in config:
        from run.icra2027 import e4_automatic_export_reuse as reuse
        if set(config['export_reuse']) != set(compact.SCENES):
            raise ValueError('reuse scene population differs')
        for scene, declaration in config['export_reuse'].items():
            reuse.validate_source(declaration)
            if declaration['scene_id'] != scene:
                raise ValueError('reuse scene key differs')
            original_config = _api().read(declaration['identities']['config']['path'])
            if original_config.get('source') != config['source']:
                raise ValueError('reuse canonical source/protocol differs')
    return config, reviewed


def run(*, config_path, stage_root, expected_commit, scene_id=None, camera_phase=False):
    if os.environ.get("SLURM_ARRAY_JOB_ID"):
        raise ValueError("individual jobs required")
    if os.environ.get("SLURM_GPUS_ON_NODE", "0") not in {"", "0"}:
        raise ValueError("CPU-only qualification/camera stage")
    if camera_phase == (scene_id is not None) or (scene_id is not None and scene_id not in compact.SCENES):
        raise ValueError("choose one fixed scene or camera phase")
    config, reviewed = validate_stage(config_path, stage_root, expected_commit)
    if reviewed["status"] != "READY":
        return reviewed
    root = screen.evidence_root(); freeze = config["freeze_id"]
    experiment = screen._experiment_root(root, freeze)
    common = dict(screen_id=freeze, scene_id=scene_id, expected_commit=expected_commit)
    if not camera_phase:
        handoff = _api().read(config["source"]["handoff"]["path"])
        if not (experiment / "automatic_candidates" / scene_id / "population").exists():
            screen.prepare_automatic_candidates(**common,
                e3_root=config["source"]["canonical"]["e3_root"],
                automatic_scene_descriptor=handoff["scenes"][scene_id]["descriptor"]["path"], export=True,
                **({"export_reuse": config["export_reuse"][scene_id]} if "export_reuse" in config else {}))
        if not (experiment / "scene_prepares" / scene_id).exists():
            screen.prepare_automatic_task_suites(**common)
        # Canonical qualifier refuses overwrite; validate an existing completed bundle.
        qualifier = experiment / "scene_qualifiers" / scene_id
        if qualifier.exists():
            result = screen._validate_qualifier_output(root=root, **common)
            result = dict(status="PASS", resumed=True, planned_cells=len(result["metric_rows"]))
        else:
            result = screen.qualify_scene(**common, automatic_population=True,
                menagerie_root=config["menagerie_root"], expected_menagerie_commit=camera.EXPECTED_MENAGERIE_COMMIT)
    else:
        protocol = config["source"]["canonical"]["protocol"]
        camera_config = experiment / "automatic_camera_config.json"
        expected_camera_config = dict(schema_version=1, study_scope=camera.AUTOMATIC_SCOPE,
            paper_ready=False, freeze_id=freeze, qualifier_screen_id=freeze,
            producer_commit=expected_commit, protocol={k:protocol[k] for k in ("path","sha256")},
            menagerie_root=config["menagerie_root"], openpi_resize_identity=config["openpi_resize_identity"],
            render_backend="osmesa")
        if camera_config.exists():
            if _api().read(camera_config) != expected_camera_config:
                raise ValueError("existing automatic camera config drift")
        else:
            camera.write_automatic_camera_config(protocol_path=protocol["path"], protocol_sha256=protocol["sha256"],
                qualifier_screen_id=freeze, freeze_id=freeze, menagerie_root=config["menagerie_root"],
                expected_code_commit=expected_commit, out=camera_config)
        output = experiment / "harness/automatic_camera_scorer"
        if output.exists():
            result = camera.validate_automatic_camera_output(config_path=camera_config,
                output=output, expected_code_commit=expected_commit)["gate"]
        else:
            result = camera.run_automatic_gate(config_path=camera_config, expected_code_commit=expected_commit)
    after_config, after = validate_stage(config_path, stage_root, expected_commit)
    if config != after_config or reviewed != after:
        raise ValueError("qualification inputs changed during execution")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect-handoff")
    parser.add_argument("--prepare-config")
    parser.add_argument("--freeze-id")
    parser.add_argument("--menagerie-root")
    parser.add_argument("--export-reuse-root")
    parser.add_argument("--out")
    parser.add_argument("--config")
    parser.add_argument("--freeze-root")
    parser.add_argument("--expected-code-commit")
    parser.add_argument("--scene-id", choices=compact.SCENES)
    parser.add_argument("--camera", action="store_true")
    args = parser.parse_args()
    if args.inspect_handoff:
        result = inspect_materialization(args.inspect_handoff)
    elif args.prepare_config:
        if not all((args.freeze_id,args.menagerie_root,args.out)):
            parser.error("prepare config requires freeze ID, Menagerie and output")
        result = prepare_config(handoff_path=args.prepare_config, freeze_id=args.freeze_id,
            menagerie_root=args.menagerie_root, out=args.out, export_reuse_root=args.export_reuse_root)
    else:
        if not all((args.config,args.freeze_root,args.expected_code_commit)):
            parser.error("execution requires config, freeze root and exact source")
        result = run(config_path=args.config, stage_root=args.freeze_root,
            expected_commit=args.expected_code_commit, scene_id=args.scene_id, camera_phase=args.camera)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
