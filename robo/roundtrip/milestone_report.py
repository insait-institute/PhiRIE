"""Package the first native comparison from existing producers, without rescoring."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

from robo.manifest.hash import canonical_hash


FIELDS = ["record_group", "treatment_id", "execution_kind", "planned", "executed",
    "completed", "success_count", "rollout_coverage", "reset_seed", "ticks", "horizon", "wall_s",
    "relative_displacement_rmse_cm", "relative_rotation_increment_mean_deg",
    "relative_full_horizon_final_displacement_error_cm", "position_rmse_cm", "rotation_error_deg"]
LIMITATIONS = [
    "Development milestone: one reconstructed target and one paired reset; not a TEST preservation estimate.",
    "The five native pilot resets are interface validation, not five reconstructed pairs; native failures remain included.",
    "Target-only replacement retains the original room and destination as oracle context.",
    "Ideal posed RGB-D inputs; uniform-color native rendering; no room Gaussian or RGB-only claim.",
    "Geometry, appearance and physical priors change together; the policy comparison is a joint asset-bundle contrast.",
    "Native CounterToSink success uses live target-body origin in the sink plus gripper distance, not full-mesh containment.",
    "Existing replay metrics describe relative public-marker displacement and rotation increments; absolute pose errors remain null.",
    "Fixed-action replay uses the recorded reference prefix, while closed-loop policies retain the native configured horizon.",
    "Videos are byte-exact copies of individual continuous episodes; no splicing, overlays or speed changes.",
    "Stages have separate recorded source commits; this package binds their lineage and is not a single-source full experiment freeze.",
]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _row(result, group, metric=None):
    executed = result.get("executed") is True
    row = dict.fromkeys(FIELDS)
    row.update(record_group=group, treatment_id=result["treatment_id"], execution_kind=result["execution_kind"],
        planned=1, executed=int(executed), completed=int(executed and result.get("error") is None),
        success_count=int(result.get("success") is True), rollout_coverage=float(executed),
        reset_seed=result["reset_seed"], ticks=result.get("ticks"), horizon=result.get("horizon"), wall_s=result.get("wall_s"))
    if metric:
        for name in FIELDS[12:]:
            row[name] = metric.get(name)
    return row


def package(*, pilot, identity, build, comparison, reference_replay, diagnosis, out, include_replay_video=False):
    pilot, identity, build, comparison, reference_replay, diagnosis, out = map(Path,
        (pilot, identity, build, comparison, reference_replay, diagnosis, out))
    if out.exists():
        raise FileExistsError("immutable report destination already exists")
    sources = {}
    def track(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError(f"source hash differs: {path}")
        sources[str(path)] = {"sha256": digest, "bytes": path.stat().st_size}
        return path
    def load(path):
        return json.loads(track(path).read_text())
    def episode(directory):
        result = load(directory / "result.json")
        ledger = [json.loads(line) for line in track(directory / "harness_ledger.jsonl").read_text().splitlines()]
        if ledger != [result]:
            raise ValueError("canonical ledger/result disagreement")
        for name in ("initial_state.json", "actions.json", "trace.json.gz"):
            track(directory / name)
        return result

    run = load(pilot / "run_manifest.json")
    config = run["config"]
    config_hash = canonical_hash(config)
    if run["config_sha256"] != config_hash:
        raise ValueError("native config hash changed")
    seeds = config["reset_seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("invalid native planned reset roster")
    native = [episode(pilot / f"episode_seed{seed}") for seed in seeds]
    if canonical_hash(load(pilot / "pilot_results.json")) != canonical_hash(native):
        raise ValueError("pilot aggregation changed or dropped a native outcome")
    for seed, result in zip(seeds, native):
        if (result["reset_seed"] != seed or result["config_sha256"] != config_hash
                or result["treatment_id"] != "REF_NATIVE" or result["execution_kind"] != "closed_loop_visual_policy"
                or canonical_hash(result["policy_identity"]) != canonical_hash(native[0]["policy_identity"])):
            raise ValueError("native pilot episode contract differs")
    control = load(identity)
    if (control.get("passed") is not True or control.get("initial_observation_diff")
            or control.get("u1_initial_observation_diff") or control.get("initial_objects_equal") is not True
            or control.get("u1_import", {}).get("body_xml_equal") is not True):
        raise ValueError("native wrapper/import identity did not pass")
    for key in ("u0", "u1"):
        if not control.get(key) or any(row.get("observation_diff") or row.get("state_max_abs") != 0
                                     or row.get("predicate_equal") is not True for row in control[key]):
            raise ValueError("native identity trace contains a mismatch")
    if (identity.parent / "run_manifest.json").exists():
        identity_run = load(identity.parent / "run_manifest.json")
        if identity_run["config_sha256"] != config_hash:
            raise ValueError("identity control native config differs from pilot")
    construction = load(build / "build_manifest.json")
    build_config = load(build / "build_config.json")
    # This producer hashes the complete constructor config without exclusions.
    build_config_hash = hashlib.sha256(json.dumps(build_config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if construction["config_sha256"] != build_config_hash:
        raise ValueError("constructor config hash changed")
    for name, digest in construction["source_hashes"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe build source member")
        track(build / relative, digest)
    pair_dir = comparison / "paired/FIXED_NATIVE_seed0"
    pair = load(pair_dir / "pair_receipt.json")
    if pair.get("state") != "RECORDED" or pair.get("planned_comparison_episodes") != 1:
        raise ValueError("first native comparison is not terminal")
    reference = pair["reference"]
    seed = reference["reset_seed"]
    ref_dir = pilot / f"episode_seed{seed}"
    if Path(reference["reference_episode"]).resolve() != ref_dir.resolve() or reference["config_sha256"] != config_hash:
        raise ValueError("paired reference episode/config differs")
    ref = native[seeds.index(seed)]
    canonical = pilot / f"canonical_seed{seed}"
    if Path(reference["canonical_reference"]).resolve() != canonical.resolve():
        raise ValueError("canonical instance is not the native pilot sibling")
    for key, directory in (("reference_files", ref_dir), ("canonical_files", canonical)):
        for name, digest in reference[key].items():
            track(directory / name, digest)
    fixed = episode(pair_dir / "episode")
    track(pair_dir / "episode/result.json", pair["comparison_result_sha256"])
    if (fixed["config_sha256"] != config_hash or fixed["reset_seed"] != seed
            or fixed["execution_kind"] != "closed_loop_visual_policy"
            or canonical_hash(ref["policy_identity"]) != canonical_hash(fixed["policy_identity"])
            or canonical_hash(ref["policy_identity"]) != canonical_hash(pair["policy_identity"])):
        raise ValueError("native policy/checkpoint/config pairing differs")
    imported = load(pair_dir / "import_receipt.json")
    for path, digest in imported["source_hashes"].items():
        tracked = track(path, digest)
        if not tracked.is_relative_to(build.resolve()):
            raise ValueError("imported target is not from the declared build")

    replays, replay_rows = [], []
    for label, directory in (("fixed_action_replay", comparison / "replay/FIXED_NATIVE_seed0"),
                             ("reference_import_replay_control", reference_replay)):
        metric = load(directory / "replay_metrics.json")
        result = episode(directory / "episode")
        source = result["policy_identity"]["source"]
        if (result["execution_kind"] != "fixed_action_replay"
                or source["reference_result_sha256"] != sha(ref_dir / "result.json")
                or source["actions_sha256"] != sha(ref_dir / "actions.json")
                or sha(directory / "episode/actions.json") != sha(ref_dir / "actions.json")
                or result["policy_identity"].get("object_pose_playback") is not False):
            raise ValueError("replay does not execute byte-identical native robot actions")
        for key, source_dir in (("reference_files", ref_dir), ("comparison_files", directory / "episode")):
            for name, digest in metric["provenance"][key].items():
                track(source_dir / name, digest)
        if metric.get("absolute_frame_correspondence") == "NOT_ESTABLISHED" and any(
                metric.get(key) is not None for key in ("position_rmse_cm", "rotation_error_deg", "final_position_error_cm")):
            raise ValueError("unestablished absolute frame has nonnull headline pose errors")
        if (metric.get("native_replay_outcome") != result or metric["planned_action_steps"] != ref["ticks"]
                or metric["comparison_recorded_steps"] != result["ticks"]):
            raise ValueError("existing replay metrics are not bound to these episodes")
        replay_rows.append(_row(result, label, metric))
        replays.append({"label": label, "metrics": {key: value for key, value in metric.items()
            if not isinstance(value, (dict, list))}, "result": result})

    findings = load(diagnosis)
    if not isinstance(findings, dict) or not findings:
        raise ValueError("specific diagnosis JSON is required")
    diagnosed = {str(ref_dir.resolve()): ref, str((pair_dir / "episode").resolve()): fixed,
                 str((comparison / "replay/FIXED_NATIVE_seed0/episode").resolve()): replays[0]["result"]}
    if {str(Path(row["episode"]).resolve()) for row in findings["rows"]} != set(diagnosed):
        raise ValueError("diagnosis must retain all three compared native outcomes")
    for row in findings["rows"]:
        directory = Path(row["episode"]).resolve()
        track(directory / "trace.json.gz", row["source_trace_sha256"])
        if row["logged_native_success"] != diagnosed[str(directory)]["success"] or row["recomputed_native_success"] != row["logged_native_success"]:
            raise ValueError("diagnosis native predicate differs from recorded outcome")
    action_control_path = reference_replay.parent.parent / "full_action_identity.json"
    action_control = load(action_control_path)
    if any(action_control.get(key) is not True for key in ("all_actions_equal", "all_qpos_equal", "all_qvel_equal",
            "all_observation_hashes_equal", "all_native_predicates_equal")) or any(
            action_control.get(key) != ref["ticks"] for key in ("planned_steps", "reference_steps", "replay_steps")):
        raise ValueError("full recorded-action native import identity did not pass")
    rows = [_row(result, "native_pilot") for result in native] + [_row(fixed, "paired_closed_loop")] + replay_rows
    counts = [{"population": "native_pilot", "planned": len(seeds), "executed": sum(r["executed"] is True for r in native),
               "successes": sum(r["success"] is True for r in native)},
              {"population": "paired_reference", "planned": 1, "executed": int(ref["executed"]), "successes": int(ref["success"] is True)},
              {"population": "paired_reconstructed", "planned": 1, "executed": int(fixed["executed"]), "successes": int(fixed["success"] is True)}]
    for row in replay_rows:
        counts.append({"population": row["record_group"], "planned": row["planned"], "executed": row["executed"], "successes": row["success_count"]})
    for row in counts:
        row["rollout_coverage"] = row["executed"] / row["planned"]
    videos = [(ref, ref_dir, "REF_NATIVE_seed0.mp4"), (fixed, pair_dir / "episode", "FIXED_NATIVE_seed0.mp4")]
    if include_replay_video:
        videos.append((replays[0]["result"], comparison / "replay/FIXED_NATIVE_seed0/episode", "FIXED_NATIVE_replay_seed0.mp4"))
    for result, directory, _ in videos:
        video = track(directory / "continuous.mp4")
        if result.get("video_error") is not None or Path(result["video_path"]).resolve() != video:
            raise ValueError("episode continuous video binding failed")
    out.mkdir(parents=True, exist_ok=False)
    video_receipts = []
    for result, directory, name in videos:
        source = directory / "continuous.mp4"
        shutil.copyfile(source, out / name)
        if sha(source) != sha(out / name):
            raise ValueError("video byte-exact copy failed")
        video_receipts.append({"name": name, "source": str(source.resolve()), "sha256": sha(source),
            "episode_id": result["episode_id"], "continuous_single_episode": True, "transcoding": False})
    report = {"schema_version": 1, "scope": "DEV_target_only_oracle_context", "identity_passed": True,
        "native_task": ref["task_id"], "native_config_sha256": config_hash, "policy_identity": ref["policy_identity"],
        "counts": counts, "episodes": rows, "native_pilot_outcomes": native, "paired_reference": ref,
        "paired_reconstructed": fixed, "replays": replays, "build": construction,
        "full_action_identity": action_control,
        "source_lineage": {"native_pilot": run.get("code"), "identity": identity_run.get("code") if (identity.parent / "run_manifest.json").exists() else None,
            "construction_commit": construction["source_commit"], "comparison": pair["source_code"], "diagnosis_commit": findings.get("source_commit")},
        "diagnosis": findings, "videos": video_receipts, "limitations": LIMITATIONS,
        "claim": "one measured native closed-loop reconstruction comparison; no population preservation or full-room claim"}
    def write(name, value):
        (out / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    write("milestone_report.json", report)
    shutil.copyfile(diagnosis, out / "diagnosis.json")
    for name, records, fields in (("episodes.csv", rows, FIELDS), ("coverage.csv", counts, list(counts[0]))):
        with (out / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(records)
    lines = [f"# First native comparison: {ref['task_id']}", "",
        f"Native pilot: {counts[0]['successes']}/{counts[0]['planned']} successful; all outcomes retained.",
        f"Paired seed {seed}: REF success={ref['success']}, {ref['ticks']} ticks; reconstructed success={fixed['success']}, {fixed['ticks']} ticks.",
        "", "| Replay | Native success | Matched/planned steps | Relative displacement RMSE cm |", "|---|---:|---:|---:|"]
    for replay in replays:
        m = replay["metrics"]
        lines.append(f"| {replay['label']} | {replay['result']['success']} | {m['matched_steps']}/{m['planned_action_steps']} | {m['relative_displacement_rmse_cm']} |")
    lines += ["", "Absolute pose errors: unmeasured (common object frame not established).", ""]
    lines += [f"Diagnosis ({row['treatment']}): target-origin-inside-sink={row['target_origin_inside_sink']}; "
              f"gripper distance={row['gripper_to_target_origin_m']} m; native retreat threshold={row['native_retreat_threshold_m']} m; "
              f"gripper-far={row['gripper_far']}." for row in findings["rows"]]
    lines += [""]
    lines += [f"- [{video['name']}]({video['name']})" for video in video_receipts]
    lines += ["", "[Specific diagnosis](diagnosis.json) · [Machine-readable results](milestone_report.json) · [Source manifest](source_manifest.json)", ""]
    lines += ["- " + limitation for limitation in LIMITATIONS]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    write("source_manifest.json", {"sources": sources, "videos": video_receipts,
        "outputs": {path.name: {"sha256": sha(path), "bytes": path.stat().st_size} for path in sorted(out.iterdir()) if path.is_file()}})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot", "identity", "build", "comparison", "reference-replay", "diagnosis", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--include-replay-video", action="store_true")
    args = vars(parser.parse_args())
    report = package(**args)
    print(json.dumps({"counts": report["counts"], "videos": report["videos"]}, indent=2))


if __name__ == "__main__":
    main()
