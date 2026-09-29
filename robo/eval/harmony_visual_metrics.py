"""Evaluate visual quality, task-mask preservation, temporal consistency, and
latency for Harmonizer observation treatments.

A manifest declares one or more conditions. Each condition can provide aligned
render/GT directories, task-object mask pairs, robot mask pairs, explicit
warped temporal pairs, and latency records. Metrics are reported only when their
required evidence exists; absent evidence stays null rather than being guessed.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from robo.eval.fidelity_metrics import LPIPSEvaluator, evaluate_images, load_rgb
from robo.eval.metric_utils import write_csv, write_json

LATEX_NEWLINE = r"\\"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
REQUIRED_CONDITIONS = (
    "raw_composite", "color_match", "harmonizer_non_temporal",
    "harmonizer_temporal", "harmonizer_robot_restore_c")


def validate_planned_frames(manifest: dict) -> dict[str, dict]:
    """Validate schema-2 coverage before loading models or evaluating pixels.

    This is a visual-input contract, not service/model provenance certification.
    Every condition has one explicit terminal record per planned frame. Successful
    image bytes are hashed; failed records cannot leave an image under that label.
    Option C's invariant is recomputed from original bytes and a nonempty core.
    """
    if manifest.get("schema_version") != 2:
        raise ValueError("planned-frame validation requires schema_version 2")
    conditions = manifest["conditions"]
    if set(conditions) != set(REQUIRED_CONDITIONS):
        raise ValueError("exactly the five declared visual conditions are required")
    frames = manifest["planned_frames"]
    if not frames:
        raise ValueError("planned frame roster is empty")
    names, identities = set(), set()

    def verified(ref):
        path = Path(ref["path"])
        if not path.is_absolute() or not path.is_file():
            raise ValueError("frame reference must name an existing absolute file")
        if hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
            raise ValueError(f"frame hash differs: {path}")
        return path

    sources = {}
    for frame in frames:
        name = frame["filename"]
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError("invalid planned relative filename")
        key = tuple(frame[k] for k in ("run_id", "scene_id", "task_id", "reset_id", "camera_id", "frame_index"))
        if name in names or key in identities:
            raise ValueError("duplicate planned filename or episode/camera/frame identity")
        if any(v is None or v == "" for v in key) or type(key[-1]) is not int or key[-1] < 0:
            raise ValueError("incomplete episode/camera/frame identity")
        names.add(name); identities.add(key)
        sources[name] = (verified(frame["raw"]), verified(frame["gt"]), frame)
    coverage = {}
    for condition, spec in conditions.items():
        records = spec["frame_records"]
        if len(records) != len(frames) or {r["filename"] for r in records} != names:
            raise ValueError(f"{condition}: terminal records differ from planned frames")
        successful = {r["filename"] for r in records if r["status"] == "success"}
        render_dir, gt_dir = Path(spec["render_dir"]), Path(spec["gt_dir"])
        actual = {str(p.relative_to(render_dir)) for p in render_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
        if actual != successful:
            raise ValueError(f"{condition}: render filenames differ from successful records")
        evaluation_masks = spec.get("evaluation_mask_dir")
        if evaluation_masks:
            mask_root = Path(evaluation_masks)
            mask_names = {str(p.relative_to(mask_root)) for p in mask_root.rglob("*")
                          if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
            if mask_names != successful:
                raise ValueError("evaluation masks differ from successful frames; no full-frame fallback")
            for name in mask_names:
                with Image.open(sources[name][0]) as im: expected_size = im.size
                with Image.open(mask_root / name) as im:
                    if im.size != expected_size or not (np.asarray(im.convert("L")) >= 128).any():
                        raise ValueError("evaluation mask must be aligned and nonempty")
        for reference_key, prediction_key in (
                ("target_reference_mask_dir", "target_prediction_mask_dir"),
                ("robot_reference_mask_dir", "robot_prediction_mask_dir"),
                ("temporal_reference_dir", "temporal_prediction_dir")):
            left, right = spec.get(reference_key), spec.get(prediction_key)
            if bool(left) != bool(right):
                raise ValueError("preservation/temporal pairs require both directories")
            if left:
                sets = [{str(p.relative_to(Path(d))) for p in Path(d).rglob("*")
                         if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES} for d in (left, right)]
                if sets[0] != sets[1] or not sets[0] <= successful:
                    raise ValueError("preservation/temporal filenames differ or include unsuccessful frames")
                for name in sets[0]:
                    with Image.open(sources[name][0]) as im: expected_size = im.size
                    for directory in (left, right):
                        with Image.open(Path(directory) / name) as im:
                            if im.size != expected_size:
                                raise ValueError("preservation/temporal shape differs; resizing is forbidden")
        counts = {status: 0 for status in ("success", "enhancer_failure", "missing", "not_run")}
        core, latencies = [], []
        for record in records:
            name, status = record["filename"], record["status"]
            if status not in counts:
                raise ValueError(f"{condition}: invalid frame terminal status")
            counts[status] += 1
            latency = record.get("end_to_end_ms")
            if status in {"success", "enhancer_failure"}:
                if type(latency) not in {float, int} or not np.isfinite(latency) or latency < 0:
                    raise ValueError("attempted frames require finite nonnegative end-to-end latency")
                latencies.append(latency)
            elif latency is not None:
                raise ValueError("unattempted frames cannot contain latency measurements")
            if status != "success":
                if not record.get("failure_reason") or record.get("render") is not None:
                    raise ValueError("unavailable frames require a reason and no fallback render")
                continue
            render = verified(record["render"])
            raw, gt, frame = sources[name]
            if render.resolve() != (render_dir / name).resolve() or gt.resolve() != (gt_dir / name).resolve():
                raise ValueError("condition render/GT path differs from planned aligned frame")
            with Image.open(render) as im: output = np.asarray(im.convert("RGB"))
            with Image.open(raw) as im: original = np.asarray(im.convert("RGB"))
            with Image.open(gt) as im: reference = np.asarray(im.convert("RGB"))
            if output.shape != original.shape or output.shape != reference.shape:
                raise ValueError("aligned frame shape mismatch; resizing is forbidden")
            if condition == "raw_composite" and not np.array_equal(output, original):
                raise ValueError("raw condition differs from original composite")
            if condition == "harmonizer_robot_restore_c":
                mask = _binary_mask(verified(frame["robot_core_mask"]))
                if mask.shape != original.shape[:2] or not mask.any():
                    raise ValueError("robot core mask must be aligned and nonempty")
                core.append(bool(np.array_equal(output[mask], original[mask])))
        coverage[condition] = {
            "planned_frames": len(frames), "processed_frames": counts["success"],
            "frame_coverage": counts["success"] / len(frames),
            "enhancer_failures": counts["enhancer_failure"],
            "missing_frames": counts["missing"], "not_run_frames": counts["not_run"],
            "robot_core_exact_pixels": all(core) if core else None,
            "robot_core_checked_frames": len(core),
            "p95_ms_per_camera": float(np.quantile(latencies, .95)) if latencies else None,
            "latency_samples": len(latencies),
            "latency_scope": "one_end_to_end_sample_per_attempted_camera_frame_including_enhancer_failures",
        }
    return coverage


def _read_json_or_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines()
                if line.strip()]
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    return payload.get("rows", payload.get("records", [payload]))


def _binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) >= 128


def _matched_files(a_dir: str | Path, b_dir: str | Path):
    a_dir, b_dir = Path(a_dir), Path(b_dir)
    a = {p.relative_to(a_dir): p for p in a_dir.rglob("*")
         if p.suffix.lower() in IMAGE_SUFFIXES}
    b = {p.relative_to(b_dir): p for p in b_dir.rglob("*")
         if p.suffix.lower() in IMAGE_SUFFIXES}
    for relative in sorted(set(a) & set(b)):
        yield relative, a[relative], b[relative]


def mask_iou(reference_dir: str | Path | None,
             prediction_dir: str | Path | None) -> tuple[float | None, int]:
    if not reference_dir or not prediction_dir:
        return None, 0
    values = []
    for _, reference_path, prediction_path in _matched_files(
            reference_dir, prediction_dir):
        reference = _binary_mask(reference_path)
        prediction = _binary_mask(prediction_path)
        if prediction.shape != reference.shape:
            prediction = np.asarray(Image.fromarray(prediction.astype(np.uint8) * 255)
                                    .resize((reference.shape[1], reference.shape[0]),
                                            Image.Resampling.NEAREST)) >= 128
        union = np.logical_or(reference, prediction).sum()
        intersection = np.logical_and(reference, prediction).sum()
        values.append(1.0 if union == 0 else intersection / union)
    return (float(np.mean(values)) if values else None), len(values)


def temporal_lpips(reference_dir: str | Path | None,
                   prediction_dir: str | Path | None,
                   evaluator: LPIPSEvaluator) -> tuple[float | None, int]:
    """LPIPS between explicitly warped temporal predictions and references.

    The caller must provide pixel-aligned temporal residual/warped-frame pairs;
    this function does not invent optical flow or compare unrelated frames.
    """
    if not reference_dir or not prediction_dir or evaluator.model is None:
        return None, 0
    values = []
    for _, reference_path, prediction_path in _matched_files(
            reference_dir, prediction_dir):
        reference = load_rgb(reference_path)
        prediction = load_rgb(prediction_path)
        if prediction.shape != reference.shape:
            prediction = np.asarray(
                Image.fromarray(np.uint8(np.clip(prediction * 255, 0, 255)))
                .resize((reference.shape[1], reference.shape[0]),
                        Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        value = evaluator(prediction, reference)
        if value is not None:
            values.append(value)
    return (float(np.mean(values)) if values else None), len(values)


def latency_p95(path: str | Path | None, condition: str | None = None):
    if not path:
        return None, 0
    values = []
    for row in _read_json_or_jsonl(path):
        if condition and row.get("treatment_id") not in {None, condition}:
            continue
        for key in ("latency_ms", "observation_latency_ms", "round_trip_ms"):
            value = row.get(key)
            if isinstance(value, list):
                values.extend(float(v) for v in value if v is not None)
            elif value not in {None, ""}:
                values.append(float(value))
        for camera in row.get("per_camera_metadata", []) or []:
            for metadata in camera.values():
                if metadata.get("round_trip_ms") is not None:
                    values.append(float(metadata["round_trip_ms"]))
                elif metadata.get("latency_ms") is not None:
                    values.append(float(metadata["latency_ms"]))
    return (float(np.quantile(values, 0.95)) if values else None), len(values)


def deterministic_core_check(path: str | Path | None, condition: str | None = None):
    if not path:
        return None
    seen = False
    for row in _read_json_or_jsonl(path):
        if condition and row.get("treatment_id") not in {None, condition}:
            continue
        for camera_set in row.get("per_camera_metadata", []) or []:
            for metadata in camera_set.values():
                if "robot_core_equal" in metadata:
                    seen = True
                    if not bool(metadata["robot_core_equal"]):
                        return False
                    if int(metadata.get("max_robot_core_error", 0)) != 0:
                        return False
    return True if seen else None


def evaluate_condition(name: str, spec: dict[str, Any],
                       evaluator: LPIPSEvaluator) -> dict:
    appearance = evaluate_images(
        spec["render_dir"], spec["gt_dir"],
        mask_dir=spec.get("evaluation_mask_dir"),
        lpips_evaluator=evaluator) if spec.get("render_dir") and spec.get("gt_dir") else {
            "n_images": 0, "psnr": None, "ssim": None, "lpips": None}
    target_iou, n_target = mask_iou(
        spec.get("target_reference_mask_dir"),
        spec.get("target_prediction_mask_dir"))
    robot_iou, n_robot = mask_iou(
        spec.get("robot_reference_mask_dir"),
        spec.get("robot_prediction_mask_dir"))
    tlpips, n_temporal = temporal_lpips(
        spec.get("temporal_reference_dir"),
        spec.get("temporal_prediction_dir"), evaluator)
    p95, n_latency = latency_p95(spec.get("latency_records"), name)
    core_equal = deterministic_core_check(spec.get("latency_records"), name)
    return {
        "condition": name, **appearance,
        "target_iou": target_iou, "target_pairs": n_target,
        "robot_iou": robot_iou, "robot_pairs": n_robot,
        "tlpips": tlpips, "temporal_pairs": n_temporal,
        "p95_ms_per_camera": p95, "latency_samples": n_latency,
        "robot_core_exact": core_equal,
    }


def _fmt(value, digits=3):
    if value is None or not np.isfinite(value):
        return "--"
    return f"{float(value):.{digits}f}"


def render_latex(rows: list[dict]) -> str:
    coverage = any("planned_frames" in row for row in rows)
    if coverage and not all("planned_frames" in row for row in rows):
        raise ValueError("cannot mix planned-frame and legacy visual metric rows")
    lines = [
        r"\begin{table*}[t]",
        r"\caption{\textbf{Visual quality and preservation under observation treatments.} Pixel metrics use held-out aligned policy-camera frames. Target and robot IoU measure task-geometry preservation, tLPIPS uses explicitly warped temporal pairs, and latency is measured end to end. Option C must additionally pass the byte-exact robot-core invariant.}",
        r"\label{tab:harmony-visual}", r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{4.4pt}",
        r"\begin{tabular}{l" + ("cc" if coverage else "") + "ccccccc}", r"\toprule",
        "Observation condition" + (" & Processed/planned & Enhancer failures" if coverage else "") + r" & PSNR $\uparrow$ & SSIM $\uparrow$ & LPIPS $\downarrow$ & target IoU $\uparrow$ & robot IoU $\uparrow$ & tLPIPS $\downarrow$ & p95 ms/cam $\downarrow$ \\",
        r"\midrule",
    ]
    for row in rows:
        label = row["condition"].replace("_", " ")
        if row.get("robot_core_exact") is True:
            label = r"\textbf{" + label + "}"
        support = (f" & {row['processed_frames']}/{row['planned_frames']} & {row['enhancer_failures']}"
                   if coverage else "")
        lines.append(
            f"{label}{support} & {_fmt(row['psnr'], 2)} & {_fmt(row['ssim'])} & "
            f"{_fmt(row['lpips'])} & {_fmt(row['target_iou'])} & "
            f"{_fmt(row['robot_iou'])} & {_fmt(row['tlpips'])} & "
            f"{_fmt(row['p95_ms_per_camera'], 1)} {LATEX_NEWLINE}")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


@contextlib.contextmanager
def _atomic_directory(destination: str | Path):
    """Publish ``destination`` atomically: write into ``<destination>.staging``
    and rename it into place once every file has been written."""
    final = Path(destination)
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable output: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.with_name(f"{final.name}.staging")
    if staging.is_dir() and not staging.is_symlink():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        yield staging
        if final.exists() or final.is_symlink():
            raise FileExistsError(f"output appeared during publication: {final}")
        staging.rename(final)
    except Exception:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise


def generate(manifest_path: str | Path, out_dir: str | Path,
             lpips_device="cpu") -> dict:
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") not in {None, 1, 2}:
        raise ValueError("unsupported visual manifest schema_version")
    strict = manifest.get("schema_version") == 2
    coverage = validate_planned_frames(manifest) if strict else {}
    if strict and Path(out_dir).exists():
        raise FileExistsError("refusing to overwrite a planned-frame evaluation")
    evaluator = LPIPSEvaluator(lpips_device)
    rows = [evaluate_condition(name, spec, evaluator)
            for name, spec in manifest["conditions"].items()]
    for row in rows:
        if strict:
            row.update(coverage[row["condition"]])
            if row["n_images"] != row["processed_frames"]:
                raise ValueError("metric support differs from successful frame roster")
            if row["condition"] == "harmonizer_robot_restore_c":
                row["robot_core_exact"] = row["robot_core_exact_pixels"]
    payload = {
        "rows": rows, "lpips_backend_error": evaluator.error,
        "manifest": str(manifest_path), **({
            "schema_version": 2,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "scope": "planned_aligned_visual_frames_not_service_or_policy_certification",
        } if strict else {})}
    if strict:
        publication = _atomic_directory(out_dir)
    else:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        publication = contextlib.nullcontext(Path(out_dir))
    with publication as destination:
        write_json(destination / "harmony_visual_table.json", payload)
        write_csv(destination / "harmony_visual_table.csv", rows)
        (destination / "harmony_visual_table.tex").write_text(render_latex(rows))
        if strict:
            (destination / "input_manifest.json").write_bytes(manifest_bytes)
    return {"rows": rows, "lpips_backend_error": evaluator.error}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lpips-device", default="cpu")
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.manifest, args.out, args.lpips_device), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
