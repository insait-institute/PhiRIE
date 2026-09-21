"""One-object DEV sensor-to-TRELLIS bridge; run inside SR1 constructor isolation.

Reuses SAM3's existing image processor, models.s4_trellis, and the E3
registration/probe producer. This bounded diagnostic does not train a room GS,
claim automatic full-room discovery, or report the isolated probe as native
room stability. Constructor inputs are exclusively the sealed TRAIN bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from robo.roundtrip.capture import backproject_camera_z, validate_public_capture


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _config_sha(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def development_config():
    """Resolve existing pinned model wrapper configs, without inventing pins."""
    import yaml
    code = Path(__file__).resolve().parents[2]
    configs = code / "configs/experiments/icra2027"
    auto = yaml.safe_load((configs / "e3_auto_pilot.yaml").read_text())
    trellis = yaml.safe_load((configs / "e3_fresh_trellis_pilot.yaml").read_text())
    return {"schema_version": 1, "scope": "target_only", "tier": "DEV",
        "method": "B0_fixed_trellis", "seed": 42, "source_commit": "RESOLVE_BEFORE_RUN",
        "sam3_python": auto["sam3_python"], "sam3_source": auto["sam3_source"],
        "sam3_checkpoint": auto["sam3_checkpoint"],
        "trellis": {"python": trellis["python"], "models": trellis["models"], "seed": 42},
        "target_prompt": None, "score_min": .45, "minimum_mask_pixels": 64,
        "minimum_views": 2, "matching_distance_m": .02, "matching_fraction": .2,
        "maximum_points_per_mask": 10000, "maximum_observation_points": 20000}


def validate_config(config):
    version = config.get("schema_version", 1)
    if version not in (1, 2):
        raise ValueError("unsupported constructor schema version")
    if (config.get("tier") not in ({"DEV"} if version == 1 else {"DEV", "TEST"}) or config.get("scope") not in ({"target_only"} if version == 1 else {"target_only", "role_asset"})
            or config.get("method") != "B0_fixed_trellis" or config.get("seed") != 42):
        raise ValueError("bounded bridge requires DEV target-only fixed TRELLIS seed42")
    if version == 2:
        if config.get('scope') == 'role_asset':
            if config.get('object_role') not in ('receptacle','support') or config.get('declared_scope') != 'L1_target_destination':
                raise ValueError('role component requires explicit L1 receptacle identity')
            if config.get('component_kind') not in (None, 'sink_basin','cabinet_bottom_shelf'):
                raise ValueError('unsupported public destination component')
            if config.get('component_kind') == 'sink_basin' and config.get('target_prompt') != 'sink basin':
                raise ValueError('sink component requires frozen public sink basin prompt')
            if config.get('component_kind')=='cabinet_bottom_shelf' and (config.get('target_prompt')!='bottom cabinet shelf' or config.get('object_role')!='support'):
                raise ValueError('cabinet support requires explicit bottom shelf prompt and role')
            if config.get('object_role')=='support' and config.get('component_kind')!='cabinet_bottom_shelf':
                raise ValueError('unsupported public support component')
        elif config.get('object_role', 'target') != 'target':
            raise ValueError('target-only build cannot silently construct another role')
        for key in ("canonical_instance_id", "cohort_id"):
            if not isinstance(config.get(key), str) or not config[key].strip():
                raise ValueError(f"versioned constructor requires {key}")
        if not re.fullmatch(r"[0-9a-f]{64}", config.get("capture_manifest_sha256", "")):
            raise ValueError("versioned constructor requires sealed capture identity")
        if config["tier"] == "TEST" and not re.fullmatch(r"[0-9a-f]{64}", config.get("dev_admission_sha256", "")):
            raise ValueError("TEST requires explicit DEV admission receipt identity")
    if 'workspace_inventory' in config:
        from robo.roundtrip.workspace import validate_workspace
        validate_workspace(config['workspace_inventory'])
        if config.get('tier')!='DEV' or config['workspace_inventory'].get('capture_manifest_sha256')!=config.get('capture_manifest_sha256'):
            raise ValueError('workspace discovery requires separate DEV admission and same TRAIN capture')
        if any(k in config for k in ('shared_candidates','observed_surface')):raise ValueError('workspace stage cannot combine producer treatments')
    if not re.fullmatch(r"[0-9a-f]{40}", config.get("source_commit", "")):
        raise ValueError("resolve exact source commit before execution")
    for key in ("minimum_mask_pixels", "minimum_views", "maximum_points_per_mask", "maximum_observation_points"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"positive integer required: {key}")
    for key in ("score_min", "matching_fraction"):
        if not 0 < float(config[key]) <= 1:
            raise ValueError(f"invalid frozen probability threshold: {key}")
    if not np.isfinite(config["matching_distance_m"]) or config["matching_distance_m"] <= 0:
        raise ValueError("positive matching distance required")


def target_prompt(instruction, declared=None):
    # Native task language is an authorized input; native asset/instance IDs are not.
    match = re.search(r"\bpick(?: up)?\s+(?:the\s+)?(.+?)\s+from\b", instruction, re.I)
    if not match:
        raise ValueError("public instruction target noun is unresolved; declare matching target_prompt")
    phrase = match.group(1).strip()
    if declared is not None:
        if not declared.strip() or declared.lower() not in phrase.lower():
            raise ValueError("declared target prompt must occur in public instruction target phrase")
        phrase = declared.strip()
    return phrase


def task_role_prompt(instruction, role='target', declared=None, component_kind=None):
    """Bounded public-language role grounding; unresolved fixtures fail closed.

    A destination component is not a complete L1 build. Native role/asset IDs
    never enter this parser or automatic masks.
    """
    if role == 'target':return target_prompt(instruction, declared)
    if role=='support':
        if (component_kind!='cabinet_bottom_shelf' or declared!='bottom cabinet shelf' or
                not re.search(r"\bplace\s+it\s+(?:in|into)\s+(?:the\s+)?cabinet\b", instruction, re.I)):
            raise ValueError('cabinet bottom support is not grounded in public task language')
        return declared
    if role != 'receptacle':raise ValueError('unsupported public task role')
    if component_kind is not None:
        if (component_kind != 'sink_basin' or declared != 'sink basin' or
                not re.search(r"\bplace\s+it\s+(?:in|into)\s+(?:the\s+)?sink\b", instruction, re.I)):
            raise ValueError('static destination component is not grounded in public task language')
        return 'sink basin'
    match=re.search(r"\bplace\s+it\s+(?:on|in|into)\s+(?:the\s+)?(plate|bowl|tray)\b",instruction,re.I)
    if not match:raise ValueError('public receptacle noun unresolved; no hidden fixture substitution')
    phrase=match.group(1).lower()
    if declared is not None and declared.lower().strip()!=phrase:raise ValueError('declared receptacle differs from public instruction')
    return phrase


def read_train(capture):
    capture = Path(capture).resolve(strict=True)
    manifest = validate_public_capture(capture)
    if manifest["sensor_regime"] != "ideal_rgbd":
        raise ValueError("this diagnostic is ideal RGB-D, never RGB-only")
    rows = [json.loads(line) for line in (capture / "train/cameras.jsonl").read_text().splitlines()]
    for row in rows:
        for field in ("rgb", "depth_m"):
            rel = row[field]
            if (rel not in manifest["files"] or not rel.startswith("train/")
                    or Path(rel).is_absolute() or ".." in Path(rel).parts):
                raise ValueError("camera row references a non-TRAIN or unsealed observation")
    return manifest, rows


def prepare_observations(capture, output, config, predictor):
    """Adapter around automatic SAM3 masks, with a mockable inference boundary.

    predictor(rgb, phrase) yields (boolean HxW mask, confidence). Selection uses
    confidence and TRAIN depth correspondence only; every candidate is logged.
    """
    from scipy.spatial import cKDTree
    validate_config(config)
    capture, output = Path(capture), Path(output)
    manifest, rows = read_train(capture)
    if config.get("schema_version", 1) == 2 and _sha(capture / "capture_manifest.json") != config["capture_manifest_sha256"]:
        raise ValueError("canonical instance capture identity differs before construction")
    phrase = task_role_prompt((capture / "task_instruction.txt").read_text(), config.get("object_role", "target"), config.get("target_prompt"), config.get('component_kind'))
    destination = output / "discovery"
    destination.mkdir(parents=True, exist_ok=False)
    candidates = []
    for row in rows:
        rgb = np.asarray(Image.open(capture / row["rgb"]).convert("RGB"))
        depth = np.load(capture / row["depth_m"], allow_pickle=False)
        points, valid = backproject_camera_z(depth, row["K"], row["T_world_from_camera"])
        for mask_index, (mask, score) in enumerate(predictor(rgb, phrase)):
            mask = np.asarray(mask)
            if mask.dtype != np.bool_ or mask.shape != depth.shape or not np.isfinite(score):
                raise ValueError("automatic mask shape/type/score differs from RGB-D")
            cid = f'{row["frame_id"]}-m{mask_index:04d}'
            mask_path = destination / (cid + ".png")
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
            observed = points[mask & valid]
            record = {"candidate_id": cid, "frame_id": row["frame_id"], "score": float(score),
                "mask_pixels": int(mask.sum()), "depth_points": len(observed),
                "mask_sha256": _sha(mask_path), "status": "eligible"}
            if score < config["score_min"] or len(observed) < config["minimum_mask_pixels"]:
                record["status"] = "insufficient_automatic_mask_evidence"
            if len(observed) > config["maximum_points_per_mask"]:
                observed = observed[np.linspace(0, len(observed)-1, config["maximum_points_per_mask"], dtype=int)]
            candidates.append((record, observed, mask, row))
    eligible = [candidate for candidate in candidates if candidate[0]["status"] == "eligible"]
    if not eligible:
        _write(destination / "candidates.json", [c[0] for c in candidates])
        raise ValueError("no automatic target mask with usable TRAIN depth")
    eligible.sort(key=lambda c: (-c[0]["score"], -c[0]["depth_points"], c[0]["candidate_id"]))
    anchor = eligible[0]
    tree = cKDTree(anchor[1])
    accepted = [anchor]
    used_frames = {anchor[0]["frame_id"]}
    for candidate in eligible[1:]:
        fraction = float(np.mean(tree.query(candidate[1])[0] < config["matching_distance_m"]))
        candidate[0]["fraction_near_anchor"] = fraction
        if candidate[0]["frame_id"] not in used_frames and fraction >= config["matching_fraction"]:
            accepted.append(candidate)
            used_frames.add(candidate[0]["frame_id"])
    _write(destination / "candidates.json", [c[0] for c in candidates])
    if len(used_frames) < config["minimum_views"]:
        raise ValueError("automatic target lacks predeclared multiview depth confirmation")
    observation = np.concatenate([c[1] for c in accepted])
    if len(observation) > config["maximum_observation_points"]:
        observation = observation[np.linspace(0, len(observation)-1, config["maximum_observation_points"], dtype=int)]
    np.save(destination / "observation_points.npy", observation, allow_pickle=False)
    obj = output / "construction/objects/obj_00"
    obj.mkdir(parents=True, exist_ok=False)
    rgb = np.asarray(Image.open(capture / anchor[3]["rgb"]).convert("RGB"))
    Image.fromarray(np.dstack([rgb, anchor[2].astype(np.uint8) * 255]), mode="RGBA").save(obj / "rgba.png")
    _write(obj.parent / "objects.json", [{"index": 0, "label": phrase}])
    result = {"planned_objects": 1, "prepared_objects": 1, "object_id": "observed_0",
        "config_sha256": _config_sha(config),
        "capture_id": manifest["capture_id"], "capture_manifest_sha256": _sha(capture / "capture_manifest.json"),
        "target_prompt": phrase, "prompt_source": "public_task_instruction", "anchor": anchor[0]["candidate_id"],
        "accepted_candidates": [c[0]["candidate_id"] for c in accepted], "confirmed_views": len(used_frames),
        "observation_points": len(observation), "observation_sha256": _sha(destination / "observation_points.npy"),
        "rgba_sha256": _sha(obj / "rgba.png"), "mask_source": "automatic_sam3",
        "anchor_mask_pixel_fraction": float(anchor[2].mean()),
        "source_gaussian": "NOT_RUN; target-only metric RGB-D diagnostic", "read_splits": ["train"]}
    if config.get("schema_version", 1) == 2:
        result.update(schema_version=2, canonical_instance_id=config["canonical_instance_id"],
                      cohort_id=config["cohort_id"], tier=config["tier"])
    _write(destination / "discovery_manifest.json", result)
    return result


def segment(capture, output, config):
    from run.icra2027.e3_trellis_generation_pilot import validate_models
    validate_models({"models": {"sam3_source": config["sam3_source"], "sam3_checkpoint": config["sam3_checkpoint"]}})
    sys.path.insert(0, config["sam3_source"]["path"])
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model(device="cuda", checkpoint_path=config["sam3_checkpoint"]["path"],
                                  load_from_HF=False, compile=False)
    processor = Sam3Processor(model, resolution=1008, device="cuda", confidence_threshold=config["score_min"])
    def predictor(rgb, phrase):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = processor.set_image(Image.fromarray(rgb))
            state = processor.set_text_prompt(prompt=phrase, state=state)
            masks = state["masks"].squeeze(1).cpu().numpy().astype(bool)
            scores = state["scores"].float().cpu().numpy()
        return list(zip(masks, scores))
    if 'workspace_inventory' in config:
        from robo.roundtrip.workspace_inventory import discover_inventory
        return discover_inventory(capture,output,config,predictor)
    return prepare_observations(capture, output, config, predictor)


def generate(output, config):
    from run.icra2027.e3_trellis_generation_pilot import validate_models
    from run.icra2027.e3_generator_backend import environment
    validate_models(config["trellis"])
    output = Path(output)
    discovery = json.loads((output / "discovery/discovery_manifest.json").read_text())
    if discovery["config_sha256"] != _config_sha(config):
        raise ValueError("constructor config changed between discovery and generation")
    records = output / "producer_records"
    records.mkdir(exist_ok=False)
    env = environment(config["trellis"], Path(__file__).resolve().parents[2], output)
    env.update(SIMANY_NO_GT="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
    result = subprocess.run([config["trellis"]["python"], "-m", "models.s4_trellis"], env=env)
    record = records / "object_00.json"
    if result.returncode or not record.is_file() or json.loads(record.read_text())["status"] != "generated":
        raise RuntimeError("existing TRELLIS producer failed; preserve producer_records")
    return json.loads(record.read_text())


def align(output, config):
    from agents.orchestrator.runtime import align_and_probe
    from agents.assets.s6_physics import coacd_parts
    output = Path(output)
    discovery = json.loads((output / "discovery/discovery_manifest.json").read_text())
    if discovery["config_sha256"] != _config_sha(config):
        raise ValueError("constructor config changed between discovery and registration")
    observation_path = output / "discovery/observation_points.npy"
    obj = output / "construction/objects/obj_00"
    if _sha(observation_path) != discovery["observation_sha256"] or _sha(obj / "rgba.png") != discovery["rgba_sha256"]:
        raise ValueError("frozen discovery inputs changed before registration")
    registered = align_and_probe(obj / "mesh_sim.ply", np.load(observation_path, allow_pickle=False),
        label=discovery["target_prompt"], visible_fraction=discovery["anchor_mask_pixel_fraction"], out_dir=output / "registration",
        signed_source_up=False, producer_commit=config["source_commit"],
        input_hashes={"mesh": _sha(obj / "mesh_sim.ply"), "observation": _sha(observation_path)})
    record = registered["alignment"]
    from robo.roundtrip.asset_metadata import exported_dimensions
    _write(obj / "aligned.json", {"T": record["T"], "scale": record["scale"],
        "world_dims": exported_dimensions(obj,record["scale"]), "registration_source": "agents.orchestrator.runtime.align_and_probe"})
    # The E3 isolated hull probe is evidence only. Actual native collision uses
    # the existing factory CoACD producer, so mug/container cavities are not
    # silently replaced by that probe's convex hull.
    collision_files = coacd_parts(obj, world_max_dim=max(record["world_dims_m"]))
    if not collision_files:
        raise ValueError("existing CoACD producer returned no collision parts")
    shutil.copyfile(output / "registration/physical/physics.json", obj / "physics.json")
    result = {"schema_version": config.get("schema_version", 1), "status": "BUILT", "tier": config["tier"], "planned_objects": 1,
        "built_objects": 1, "object_id": "observed_0", "object_dir": str(obj.resolve()),
        "method": "B0_fixed_trellis", "scope": config["scope"], "sensor_regime": "ideal_rgbd",
        "config_sha256": _config_sha(config),
        "model_identities": {"sam3_source": config["sam3_source"], "sam3_checkpoint": config["sam3_checkpoint"],
                             "trellis": config["trellis"]["models"]},
        "appearance": "uniform_color_diagnostic", "collision": "existing_s6_coacd_decomposition",
        "collision_parts": len(collision_files),
        "isolated_probe_collision": "single_convex_hull; diagnostic only, not imported geometry",
        "visibility_fraction_input": "anchor mask pixel fraction; not 3D completeness and not used for selection",
        "native_room_stability": "NOT_RUN", "source_gaussian": "NOT_RUN",
        "source_commit": config["source_commit"], "capture_manifest_sha256": discovery["capture_manifest_sha256"],
        "source_hashes": {str(p.relative_to(output)): _sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
        "native_import": "NOT_RUN", "policy_rollouts": "NOT_RUN"}
    if config.get("schema_version", 1) == 2:
        result.update(canonical_instance_id=config["canonical_instance_id"], cohort_id=config["cohort_id"],
                      object_role=config.get("object_role", "target"), declared_scope=config.get("declared_scope", "L0_target_only"),
                      component_only=config["scope"] == "role_asset")
        if config.get('component_kind') is not None:
            result.update(component_kind=config['component_kind'],
                          retained_reference_components=(['faucet articulation', 'fixture goal metadata', 'fixture child accessories']
                            if config['component_kind']=='sink_basin' else ['cabinet walls and ceiling','native-open doors','upper shelves','all native goal regions']))
    _write(output / "build_manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--phase", choices=["run", "segment", "generate", "align"], default="run")
    args = parser.parse_args()
    if args.template:
        print(json.dumps(development_config(), indent=2)); return
    if not args.config or not args.capture or not args.out:
        parser.error("--config, --capture and --out are required")
    import yaml
    config = yaml.safe_load(args.config.read_text())
    validate_config(config)
    os.environ.update(SIMANY_NO_GT="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_IMPLICIT_TOKEN="1", PYTHONDONTWRITEBYTECODE="1",
                      SIMANY_AUTO="1", SIMANY_MESH_SRC="derived",
                      SIMANY_OUT=str((args.out / "construction").resolve()))
    read_train(args.capture)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase == "run":
        _write(args.out / "build_config.json", config)
        phases=[("segment", config["sam3_python"])] if 'workspace_inventory' in config else [
            ("segment", config["sam3_python"]),("generate", config["trellis"]["python"]),("align", config["trellis"]["python"])]
        for phase, python in phases:
            started = time.monotonic()
            command = [python, "-m", "robo.roundtrip.build", "--phase", phase,
                       "--config", str(args.config), "--capture", str(args.capture), "--out", str(args.out)]
            with (args.out / (phase + ".log")).open("x") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            _write(args.out / (phase + "_runtime.json"), {"command": command, "exit_code": result.returncode,
                                                       "wall_s": time.monotonic() - started})
            if result.returncode:
                _write(args.out / "build_failure.json", {"planned_objects": 1, "built_objects": 0,
                    "phase": phase, "status": "construction_unavailable", "log": phase + ".log"})
                raise SystemExit(result.returncode)
    elif args.phase == "segment":
        segment(args.capture, args.out, config)
    elif args.phase == "generate":
        generate(args.out, config)
    else:
        align(args.out, config)


if __name__ == "__main__":
    main()
