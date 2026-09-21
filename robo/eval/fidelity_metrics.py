"""Strict held-out appearance and metric-geometry evaluation for Table II.

The evaluator consumes source artifacts, never pre-computed paper numbers.  A
non-empty record is fail-closed: image pairs are exact, object masks are
non-empty, generator inputs are disjoint from evaluation views, and geometry
registration evidence is independent from the evaluation surface.  Empty
method record lists remain representable so an audit can publish explicit
missing cells, but such a bundle is never paper-ready.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import importlib.metadata
import json
import math
import os
import re
import shutil
import subprocess
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
SCHEMA_VERSION = 2
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ROOM_METHODS = (
    "Input scene Gaussian, reconstruction ceiling",
    "Factorized composite, GT discovery",
    "Factorized composite, automatic discovery",
    "Composite + Harmonizer Option C",
)
OBJECT_METHODS = (
    "TRELLIS, best single view",
    "ReconViaGen, multi-view",
    "Evidence-selected proposal",
    "Evaluation-only oracle candidate",
)
UNIT_SCALE_TO_METERS = {"m": 1.0, "cm": 0.01, "mm": 0.001}


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def load_mask(
    path: str | Path | None,
    shape: tuple[int, int],
    *,
    allow_resize: bool = True,
    require_nonempty: bool = False,
) -> np.ndarray:
    if path is None:
        value = np.ones(shape, dtype=bool)
    else:
        with Image.open(path) as image:
            raw = np.asarray(image.convert("L"))
            if raw.shape != tuple(shape):
                if not allow_resize:
                    raise ValueError(
                        f"mask shape {raw.shape} differs from image shape {tuple(shape)}: "
                        f"{path}"
                    )
                raw = np.asarray(
                    Image.fromarray(raw).resize(
                        (shape[1], shape[0]), Image.Resampling.NEAREST
                    )
                )
        value = raw >= 128
    if require_nonempty and not bool(value.any()):
        raise ValueError(f"empty evaluation mask: {path}")
    return value


def psnr(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    error = (
        np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    ) ** 2
    if mask is not None:
        error = error[np.asarray(mask, dtype=bool)]
    mse = float(error.mean()) if error.size else float("nan")
    return float("inf") if mse == 0 else float(-10.0 * math.log10(mse))


def ssim(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Match scikit-image's default SSIM and isolate optional mask support.

    This is the ``win_size=7``, uniform-window, sample-covariance definition
    used by :func:`skimage.metrics.structural_similarity`.  It is implemented
    locally because the frozen E2 metrics environment contains SciPy but not
    scikit-image.  The unmasked scalar intentionally crops the three-pixel
    window border exactly as scikit-image does.
    """
    from scipy.ndimage import uniform_filter

    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.shape != target.shape:
        raise ValueError(f"SSIM shape mismatch: {pred.shape} != {target.shape}")
    if pred.ndim != 3 or pred.shape[2] != 3:
        raise ValueError(f"SSIM expects HxWx3 RGB arrays, got {pred.shape}")
    smallest_side = min(pred.shape[:2])
    win_size = 7 if smallest_side >= 7 else (
        smallest_side if smallest_side % 2 else smallest_side - 1
    )
    if win_size < 3:
        raise ValueError(
            f"SSIM requires image dimensions of at least 3 pixels: {pred.shape}"
        )
    float_type = (
        np.float32
        if max(pred.dtype.itemsize, target.dtype.itemsize) <= 4
        else np.float64
    )
    pred = pred.astype(float_type, copy=False)
    target = target.astype(float_type, copy=False)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != pred.shape[:2]:
            raise ValueError(f"SSIM mask shape mismatch: {mask.shape} != {pred.shape[:2]}")
        # Prevent values outside the evaluation support from bleeding into a
        # local window whose centre is inside the mask.
        pred = np.where(mask[..., None], pred, target)
    npixels = win_size**2
    covariance_normalization = npixels / (npixels - 1)
    c1, c2 = 0.01**2, 0.03**2
    maps = []
    scores = []
    pad = (win_size - 1) // 2
    for channel in range(3):
        x, y = target[..., channel], pred[..., channel]
        ux = uniform_filter(x, size=win_size)
        uy = uniform_filter(y, size=win_size)
        uxx = uniform_filter(x * x, size=win_size)
        uyy = uniform_filter(y * y, size=win_size)
        uxy = uniform_filter(x * y, size=win_size)
        vx = covariance_normalization * (uxx - ux * ux)
        vy = covariance_normalization * (uyy - uy * uy)
        vxy = covariance_normalization * (uxy - ux * uy)
        value = ((2 * ux * uy + c1) * (2 * vxy + c2)) / (
            (ux * ux + uy * uy + c1) * (vx + vy + c2)
        )
        maps.append(value)
        cropped = value[pad:-pad, pad:-pad] if pad else value
        scores.append(float(cropped.mean(dtype=np.float64)))
    if mask is None:
        return float(np.mean(scores, dtype=np.float64))
    channel_map = np.stack(maps, axis=-1)
    selected = channel_map[mask]
    return float(selected.mean(dtype=np.float64)) if selected.size else float("nan")


class LPIPSEvaluator:
    def __init__(self, device: str = "cpu"):
        self.device, self.model, self.error = device, None, None
        self.provenance: dict[str, Any] = {
            "backend_class": None,
            "package": None,
            "package_version": None,
            "network": "alex",
            "backbone_checkpoint": None,
            "linear_checkpoint": None,
        }
        try:
            import torch

            try:
                import lpips

                network = lpips.LPIPS(net="alex").to(device).eval()
                linear_path = Path(lpips.__file__).resolve().parent / "weights/v0.1/alex.pth"
                self.provenance = {
                    "backend_class": f"{type(network).__module__}.{type(network).__qualname__}",
                    "package": "lpips",
                    "package_version": importlib.metadata.version("lpips"),
                    "network": "alex",
                    "backbone_checkpoint": _lpips_backbone_identity(),
                    "linear_checkpoint": _artifact_identity(linear_path),
                }
                self.model = (network, torch, False)
            except ImportError:
                from torchmetrics.image.lpip import (
                    LearnedPerceptualImagePatchSimilarity,
                )

                network = LearnedPerceptualImagePatchSimilarity(
                    net_type="alex", normalize=True
                ).to(device).eval()
                self.provenance = {
                    "backend_class": f"{type(network).__module__}.{type(network).__qualname__}",
                    "package": "torchmetrics",
                    "package_version": importlib.metadata.version("torchmetrics"),
                    "network": "alex",
                    "backbone_checkpoint": _lpips_backbone_identity(),
                    "linear_checkpoint": None,
                }
                self.model = (network, torch, True)
        except Exception as exc:  # optional dependency / unavailable CUDA
            self.error = f"{type(exc).__name__}: {exc}"

    def __call__(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> float | None:
        if self.model is None:
            return None
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            ys, xs = np.where(mask)
            if not len(ys):
                return None
            pred = np.where(mask[..., None], pred, target)
            pred = pred[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
            target = target[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        model, torch, normalized = self.model
        a = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        b = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).to(self.device)
        if not normalized:
            a, b = a * 2.0 - 1.0, b * 2.0 - 1.0
        with torch.no_grad():
            return float(model(a, b).item())


def _image_map(directory: str | Path) -> dict[Path, Path]:
    directory = Path(directory)
    return {
        path.relative_to(directory): path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def matched_images(render_dir: str | Path, gt_dir: str | Path):
    """Yield the legacy intersection; strict callers use ``_pair_images``."""
    render_dir, gt_dir = Path(render_dir), Path(gt_dir)
    renders, targets = _image_map(render_dir), _image_map(gt_dir)
    for relative in sorted(set(renders) & set(targets)):
        yield relative, renders[relative], targets[relative]


def _coverage_set(coverage: dict[str, Any], *keys: str) -> set[str]:
    found: list[Any] = [coverage[key] for key in keys if key in coverage]
    if not found:
        return set()
    if len(found) > 1 and any(value != found[0] for value in found[1:]):
        raise ValueError(f"conflicting coverage aliases: {keys}")
    if not isinstance(found[0], list):
        raise ValueError(f"coverage {keys[0]} must be a list")
    return {Path(str(value)).as_posix() for value in found[0]}


def _pair_images(
    render_dir: Path,
    gt_dir: Path,
    *,
    strict: bool,
    coverage: dict[str, Any] | None,
) -> tuple[list[tuple[Path, Path, Path]], dict[str, list[str]]]:
    if strict:
        if not render_dir.is_dir():
            raise ValueError(f"render_dir is not a directory: {render_dir}")
        if not gt_dir.is_dir():
            raise ValueError(f"gt_dir is not a directory: {gt_dir}")
    renders, targets = _image_map(render_dir), _image_map(gt_dir)
    render_only = {path.as_posix() for path in set(renders) - set(targets)}
    gt_only = {path.as_posix() for path in set(targets) - set(renders)}
    if strict:
        declared = coverage or {}
        if not isinstance(declared, dict):
            raise ValueError("record coverage must be an object")
        declared_render = _coverage_set(
            declared, "render_only", "missing_in_gt"
        )
        declared_gt = _coverage_set(
            declared, "gt_only", "missing_in_render"
        )
        if declared_render != render_only or declared_gt != gt_only:
            raise ValueError(
                "predicted and GT relative filenames differ without an exact "
                "coverage record; "
                f"render_only={sorted(render_only)}, gt_only={sorted(gt_only)}, "
                f"declared_render_only={sorted(declared_render)}, "
                f"declared_gt_only={sorted(declared_gt)}"
            )
    pairs = [
        (relative, renders[relative], targets[relative])
        for relative in sorted(set(renders) & set(targets))
    ]
    if strict and not pairs:
        raise ValueError(f"no paired evaluation images in {render_dir} and {gt_dir}")
    return pairs, {
        "render_only": sorted(render_only),
        "gt_only": sorted(gt_only),
    }


def _sha256_file(path: Path, cache: dict[Path, str] | None = None) -> str:
    canonical = path.resolve()
    if cache is not None and canonical in cache:
        return cache[canonical]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    if cache is not None:
        cache[canonical] = value
    return value


def _display_repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _artifact_identity(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"LPIPS checkpoint is missing or symlinked: {path}")
    return {
        "path": _display_repo_path(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _lpips_backbone_identity() -> dict[str, Any]:
    declared = os.environ.get("E2_LPIPS_WEIGHTS_PATH")
    candidates: list[Path] = []
    if declared:
        path = Path(declared)
        candidates.append(path if path.is_absolute() else REPOSITORY_ROOT / path)
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        cache = Path(torch_home) / "hub/checkpoints"
        if cache.is_dir():
            candidates.extend(sorted(cache.glob("alexnet-*.pth")))
    unique = []
    for candidate in candidates:
        if candidate not in unique and candidate.is_file():
            unique.append(candidate)
    if len(unique) != 1:
        raise ValueError(
            "LPIPS AlexNet provenance requires exactly one declared/local "
            f"checkpoint, found {[str(path) for path in unique]}"
        )
    identity = _artifact_identity(unique[0])
    expected_hash = os.environ.get("E2_LPIPS_WEIGHTS_SHA256")
    expected_size = os.environ.get("E2_LPIPS_WEIGHTS_SIZE")
    if expected_hash and identity["sha256"] != expected_hash:
        raise ValueError("LPIPS AlexNet checkpoint hash differs from declared provenance")
    if expected_size and identity["size_bytes"] != int(expected_size):
        raise ValueError("LPIPS AlexNet checkpoint size differs from declared provenance")
    return identity


def _combined_hash(values: Iterable[str]) -> str | None:
    clean = sorted(str(value) for value in values if value)
    if not clean:
        return None
    digest = hashlib.sha256()
    for value in clean:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _pair_source_hash(
    relative: Path,
    render_path: Path,
    gt_path: Path,
    mask_path: Path | None,
    cache: dict[Path, str],
) -> str:
    values = [
        f"render:{relative.as_posix()}:{_sha256_file(render_path, cache)}",
        f"gt:{relative.as_posix()}:{_sha256_file(gt_path, cache)}",
    ]
    if mask_path is not None:
        values.append(f"mask:{relative.as_posix()}:{_sha256_file(mask_path, cache)}")
    return _combined_hash(values) or ""


def evaluate_images(
    render_dir: str | Path,
    gt_dir: str | Path,
    *,
    mask_dir: str | Path | None = None,
    lpips_evaluator: LPIPSEvaluator | None = None,
    strict: bool = False,
    require_masks: bool = False,
    coverage: dict[str, Any] | None = None,
    _hash_cache: dict[Path, str] | None = None,
    _return_details: bool = False,
) -> dict[str, Any]:
    """Evaluate aligned images while retaining the historical public API.

    ``strict=True`` disables implicit resizing, requires exact coverage
    declarations for filename differences, and fails on absent/empty masks.
    Infinite PSNR values are never averaged and are counted separately.
    """
    render_dir, gt_dir = Path(render_dir), Path(gt_dir)
    mask_root = Path(mask_dir) if mask_dir is not None else None
    pairs, coverage_loss = _pair_images(
        render_dir, gt_dir, strict=strict, coverage=coverage
    )
    values: dict[str, list[float]] = {"psnr": [], "ssim": [], "lpips": []}
    observations: dict[str, list[dict[str, Any]]] = {
        "psnr": [],
        "ssim": [],
        "lpips": [],
    }
    duplicate_count = 0
    cache = _hash_cache if _hash_cache is not None else {}
    frame_details = []
    for relative, render_path, gt_path in pairs:
        pred, target = load_rgb(render_path), load_rgb(gt_path)
        if pred.shape != target.shape:
            if strict:
                raise ValueError(
                    f"image shape mismatch for {relative.as_posix()}: "
                    f"{pred.shape} != {target.shape}"
                )
            pred = (
                np.asarray(
                    Image.fromarray(np.uint8(np.clip(pred * 255, 0, 255))).resize(
                        (target.shape[1], target.shape[0]), Image.Resampling.BILINEAR
                    ),
                    dtype=np.float32,
                )
                / 255.0
            )
        mask_path = mask_root / relative if mask_root is not None else None
        if strict and mask_path is not None and not mask_path.is_file():
            raise ValueError(f"missing evaluation mask for {relative.as_posix()}: {mask_path}")
        if strict and require_masks and mask_path is None:
            raise ValueError(f"object evaluation requires mask_dir: {relative.as_posix()}")
        mask = (
            load_mask(
                mask_path if mask_path is not None and mask_path.exists() else None,
                target.shape[:2],
                allow_resize=not strict,
                require_nonempty=strict and require_masks,
            )
            if mask_path is not None or require_masks
            else None
        )
        source_hash = _pair_source_hash(
            relative,
            render_path,
            gt_path,
            mask_path if mask_path is not None and mask_path.exists() else None,
            cache,
        )
        p_value = psnr(pred, target, mask)
        exact_duplicate = math.isinf(p_value)
        if exact_duplicate:
            duplicate_count += 1
        elif math.isfinite(p_value):
            values["psnr"].append(p_value)
            observations["psnr"].append(
                {"frame": relative.as_posix(), "value": p_value, "source_hash": source_hash}
            )
        s_value = ssim(pred, target, mask)
        if not math.isfinite(s_value):
            raise ValueError(f"non-finite SSIM for {relative.as_posix()}")
        values["ssim"].append(s_value)
        observations["ssim"].append(
            {"frame": relative.as_posix(), "value": s_value, "source_hash": source_hash}
        )
        l_value = lpips_evaluator(pred, target, mask) if lpips_evaluator else None
        if l_value is not None:
            if not math.isfinite(l_value):
                raise ValueError(f"non-finite LPIPS for {relative.as_posix()}")
            values["lpips"].append(l_value)
            observations["lpips"].append(
                {"frame": relative.as_posix(), "value": l_value, "source_hash": source_hash}
            )
        frame_details.append(
            {
                "frame": relative.as_posix(),
                "source_artifact_hash": source_hash,
                "exact_duplicate": exact_duplicate,
            }
        )
    result: dict[str, Any] = {
        "n_images": len(pairs),
        "psnr": float(np.mean(values["psnr"])) if values["psnr"] else None,
        "ssim": float(np.mean(values["ssim"])) if values["ssim"] else None,
        "lpips": float(np.mean(values["lpips"])) if values["lpips"] else None,
        "exact_duplicate_images": duplicate_count,
        "metric_samples": {key: len(value) for key, value in values.items()},
        "metric_source_hashes": {
            key: _combined_hash(item["source_hash"] for item in observations[key])
            for key in values
        },
        "coverage": coverage_loss,
    }
    if _return_details:
        result["observations"] = observations
        result["frames"] = frame_details
    return result


def load_surface(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        points = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            points = archive["points" if "points" in archive else list(archive)[0]]
    else:
        import trimesh

        geometry = trimesh.load(path, process=False)
        if hasattr(geometry, "vertices"):
            points = np.asarray(geometry.vertices)
        else:
            chunks = [
                np.asarray(value.vertices)
                for value in geometry.geometry.values()
                if len(value.vertices)
            ]
            points = np.concatenate(chunks) if chunks else np.empty((0, 3))
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        raise ValueError(f"empty point set: {path}")
    return points


def nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    return cKDTree(target).query(source, k=1, workers=-1)[0]


def geometry_metrics(
    predicted: np.ndarray, target: np.ndarray, threshold_m: float = 0.02
) -> dict[str, Any]:
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if not len(predicted) or not len(target):
        raise ValueError("geometry metrics require two non-empty surfaces")
    p_to_t = nearest_distances(predicted, target)
    t_to_p = nearest_distances(target, predicted)
    precision = float(np.mean(p_to_t <= threshold_m))
    recall = float(np.mean(t_to_p <= threshold_m))
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "cd_cm": 100.0 * float(0.5 * (p_to_t.mean() + t_to_p.mean())),
        "precision20": precision,
        "recall20": recall,
        "f1_20": f1,
        # The threshold is deliberately strict: exactly 0.1 is not collapse.
        "collapse": bool(f1 < 0.1),
    }


def _required_text(record: dict[str, Any], key: str, identity: str) -> str:
    value = record.get(key)
    if value is None or not str(value).strip():
        raise ValueError(f"{identity} is missing required {key}")
    return str(value).strip()


def _checked_repository_path(path: str | Path, identity: str) -> Path:
    """Reject lexical escapes and every symlink component before access."""
    root = Path(os.path.abspath(REPOSITORY_ROOT))
    lexical = Path(os.path.abspath(path))
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{identity} is outside the SimAny repository: {path}") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{identity} uses a symlink component: {current}")
    return lexical


def _nonnegative_int(value: Any, identity: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{identity} must be a non-negative integer")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{identity} must be a non-negative integer") from exc
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError(f"{identity} must be a non-negative integer")
    return int(number)


def _resolve_path(value: Any, manifest_dir: Path, identity: str, kind: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"{identity} is missing required {kind}")
    raw = Path(str(value))
    if raw.is_absolute():
        path = raw
    else:
        local = manifest_dir / raw
        repository = REPOSITORY_ROOT / raw
        path = local if local.exists() else repository
    path = _checked_repository_path(path, f"{identity} {kind}")
    if kind.endswith("dir"):
        if not path.is_dir():
            raise ValueError(f"{identity} {kind} is not a directory: {path}")
    elif not path.is_file():
        raise ValueError(f"{identity} {kind} is not a file: {path}")
    return path


def _method_records(spec: Any, identity: str) -> tuple[str | None, list[dict[str, Any]]]:
    if isinstance(spec, list):
        return None, spec
    if not isinstance(spec, dict):
        raise ValueError(f"{identity} method specification must be an object or list")
    source_id = spec.get("source_id")
    if source_id is not None and not str(source_id).strip():
        raise ValueError(f"{identity} source_id must be non-empty")
    if "records" in spec:
        records = spec["records"]
    elif "render_dir" in spec or "gt_dir" in spec:  # legacy one-record shape
        records = [spec]
    else:
        records = []
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError(f"{identity} records must be a list of objects")
    return str(source_id).strip() if source_id is not None else None, records


def _declared_frames(record: dict[str, Any], identity: str) -> set[str] | None:
    frames = record.get("evaluation_frames")
    if frames is None:
        return None
    if not isinstance(frames, list):
        raise ValueError(f"{identity} evaluation_frames must be a list")
    normalized = [Path(str(value)).as_posix() for value in frames]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{identity} evaluation_frames contains duplicates")
    return set(normalized)


def _frame_tokens(value: str | Path) -> set[str]:
    path = Path(str(value))
    text = path.as_posix()
    return {text, path.name, path.stem}


def _check_generator_leakage(
    generator_inputs: Any, evaluation_frames: Iterable[str], identity: str
) -> list[str]:
    if not isinstance(generator_inputs, list):
        raise ValueError(f"{identity} generator_input_frames must be a list")
    evaluation_tokens = set().union(*(_frame_tokens(value) for value in evaluation_frames))
    leaks = [
        str(value)
        for value in generator_inputs
        if _frame_tokens(str(value)) & evaluation_tokens
    ]
    if leaks:
        raise ValueError(
            f"generator/evaluation frame leakage for {identity}: {sorted(leaks)}"
        )
    return [str(value) for value in generator_inputs]


def _validate_declared_room_views(
    record: dict[str, Any],
    *,
    manifest_dir: Path,
    render_dir: Path,
    gt_dir: Path,
    actual_frames: set[str],
    identity: str,
    hash_cache: dict[Path, str],
) -> None:
    """Bind evaluator inputs to the inventory's per-view paths and hashes."""
    views = record.get("views")
    if not isinstance(views, list) or len(views) != len(actual_frames):
        raise ValueError(
            f"{identity} views must contain exactly {len(actual_frames)} records"
        )
    declared_frames: set[str] = set()
    for index, view in enumerate(views):
        view_identity = f"{identity}/view-{index}"
        if not isinstance(view, dict):
            raise ValueError(f"{view_identity} must be an object")
        view_id = _required_text(view, "view_id", view_identity)
        frame = f"{Path(view_id).stem}.png"
        if frame in declared_frames:
            raise ValueError(f"{identity} views contain duplicate frame {frame}")
        declared_frames.add(frame)
        render_path = _resolve_path(
            view.get("render_path"), manifest_dir, view_identity, "render_file"
        )
        gt_path = _resolve_path(
            view.get("gt_path"), manifest_dir, view_identity, "gt_file"
        )
        if render_path != render_dir / frame or gt_path != gt_dir / frame:
            raise ValueError(
                f"{view_identity} paths do not match the paired evaluation directories"
            )
        render_hash = _required_text(view, "render_sha256", view_identity)
        gt_hash = _required_text(view, "gt_sha256", view_identity)
        if SHA256_RE.fullmatch(render_hash) is None or SHA256_RE.fullmatch(gt_hash) is None:
            raise ValueError(f"{view_identity} requires lowercase SHA-256 hashes")
        if _sha256_file(render_path, hash_cache) != render_hash:
            raise ValueError(f"{view_identity} render_sha256 does not match its file")
        if _sha256_file(gt_path, hash_cache) != gt_hash:
            raise ValueError(f"{view_identity} gt_sha256 does not match its file")
    if declared_frames != actual_frames:
        raise ValueError(
            f"{identity} views differ from paired files; "
            f"missing={sorted(actual_frames - declared_frames)}, "
            f"extra={sorted(declared_frames - actual_frames)}"
        )


def _fixed_sample(points: np.ndarray, count: int, identity: str) -> np.ndarray:
    if len(points) < count:
        raise ValueError(
            f"{identity} has {len(points)} surface samples, fewer than declared {count}"
        )
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    ordered = points[order]
    if len(ordered) == count:
        return ordered
    indices = np.floor(np.arange(count, dtype=np.float64) * len(ordered) / count).astype(int)
    return ordered[indices]


def _stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    return (seed + int.from_bytes(digest[:8], "big")) % (2**63 - 1)


def _cluster_bootstrap(
    observations: list[dict[str, Any]],
    *,
    group_key: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    for observation in observations:
        value = observation.get("value")
        if value is not None and math.isfinite(float(value)):
            groups[str(observation[group_key])].append(float(value))
    keys = sorted(groups)
    if not keys:
        return {"unit": group_key, "n_units": 0, "samples": samples, "seed": seed, "ci95": [None, None]}
    if len(keys) == 1:
        value = float(np.mean(groups[keys[0]]))
        return {"unit": group_key, "n_units": 1, "samples": samples, "seed": seed, "ci95": [value, value]}
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, len(keys), size=len(keys))
        values = [item for selected_index in selected for item in groups[keys[selected_index]]]
        estimates[index] = float(np.mean(values))
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return {"unit": group_key, "n_units": len(keys), "samples": samples, "seed": seed, "ci95": [float(lo), float(hi)]}


def _aggregate_observations(
    observations: list[dict[str, Any]],
    *,
    balance_key: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    finite = [
        row for row in observations
        if row.get("value") is not None and math.isfinite(float(row["value"]))
    ]
    groups: dict[str, list[float]] = defaultdict(list)
    for row in finite:
        groups[str(row[balance_key])].append(float(row["value"]))
    balanced = float(np.mean([np.mean(values) for values in groups.values()])) if groups else None
    return {
        "value": float(np.mean([row["value"] for row in finite])) if finite else None,
        "n": len(finite),
        "source_artifact_hash": _combined_hash(row["source_hash"] for row in finite),
        f"{balance_key}_balanced": balanced,
        "bootstrap": _cluster_bootstrap(
            finite,
            group_key=balance_key,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
    }


def _method_minimum(
    coverage: dict[str, Any],
    *,
    mapping_key: str,
    fallback_key: str,
    method: str,
    source_id: str | None,
) -> int:
    fallback = _nonnegative_int(coverage.get(fallback_key, 0), fallback_key)
    mapping = coverage.get(mapping_key, {})
    if not isinstance(mapping, dict):
        raise ValueError(f"coverage.{mapping_key} must be an object")
    matches = []
    for key in (method, source_id):
        if key is not None and key in mapping:
            matches.append((key, _nonnegative_int(mapping[key], f"{mapping_key}.{key}")))
    if len(matches) == 2 and matches[0][1] != matches[1][1]:
        raise ValueError(
            f"conflicting {mapping_key} minima for method label and source_id: {matches}"
        )
    return matches[0][1] if matches else fallback


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    output = {
        key: value
        for key, value in row.items()
        if not isinstance(value, (dict, list))
    }
    for metric, detail in row.get("metrics", {}).items():
        output[metric] = detail.get("value")
        output[f"{metric}_n"] = detail.get("n")
        output[f"{metric}_source_artifact_hash"] = detail.get("source_artifact_hash")
        ci = detail.get("bootstrap", {}).get("ci95", [None, None])
        output[f"{metric}_ci95_low"] = ci[0]
        output[f"{metric}_ci95_high"] = ci[1]
    return output


def _csv_text(rows: list[dict[str, Any]]) -> str:
    flat = [_flatten_row(row) for row in rows]
    fieldnames: list[str] = []
    for row in flat:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(flat)
    return stream.getvalue()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bundle(out_dir: Path, outputs: dict[str, str]) -> None:
    out_dir = _checked_repository_path(out_dir, "fidelity output directory")
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite fidelity table: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = out_dir.with_name(f".{out_dir.name}.staging-{uuid.uuid4().hex}")
    try:
        staging.mkdir()
        for name, contents in outputs.items():
            path = staging / name
            with path.open("x", encoding="utf-8", newline="") as handle:
                handle.write(contents)
                handle.flush()
                os.fsync(handle.fileno())
        _fsync_directory(staging)
        if out_dir.exists() or out_dir.is_symlink():
            raise FileExistsError(f"fidelity output appeared during publication: {out_dir}")
        staging.rename(out_dir)
        _fsync_directory(out_dir.parent)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def evaluate_manifest(
    manifest_path: str | Path,
    out_dir: str | Path,
    *,
    lpips_device: str = "cpu",
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    manifest_path = _checked_repository_path(manifest_path, "fidelity manifest")
    if not manifest_path.is_file():
        raise ValueError(f"fidelity manifest is not a file: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("fidelity manifest must be a JSON object")
    bootstrap_samples = _nonnegative_int(bootstrap_samples, "bootstrap_samples")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    freeze_id = _required_text(manifest, "freeze_id", "manifest")
    coverage = manifest.get("coverage", {})
    if not isinstance(coverage, dict):
        raise ValueError("manifest coverage must be an object")
    room_split = manifest.get("room_split", {})
    object_split = manifest.get("object_split", {})
    if not isinstance(room_split, dict) or not isinstance(object_split, dict):
        raise ValueError("room_split and object_split must be objects")
    if room_split.get("unit", "held_out_view") != "held_out_view":
        raise ValueError("room_split.unit must be 'held_out_view'")
    if float(object_split.get("f1_threshold_mm", 20)) != 20.0:
        raise ValueError("object_split.f1_threshold_mm is fixed at exactly 20")
    if float(object_split.get("catastrophic_collapse_below", 0.1)) != 0.1:
        raise ValueError("catastrophic collapse threshold is fixed at exactly 0.1")
    if object_split.get("require_independent_registration_surface", True) is not True:
        raise ValueError("independent registration/evaluation surfaces are required")
    room_methods = manifest.get("room_methods")
    object_methods = manifest.get("object_methods")
    if not isinstance(room_methods, dict) or not isinstance(object_methods, dict):
        raise ValueError("room_methods and object_methods must be objects")

    evaluator = LPIPSEvaluator(lpips_device)
    hash_cache: dict[Path, str] = {}
    rows: list[dict[str, Any]] = []
    room_evidence: dict[str, list[dict[str, Any]]] = {}
    room_frame_sets: dict[tuple[str, str], set[str]] = {}
    room_gt_hashes: dict[tuple[str, str], str] = {}
    coverage_losses: list[dict[str, Any]] = []

    for method, spec in room_methods.items():
        source_id, records = _method_records(spec, f"room method {method!r}")
        seen_scenes: set[str] = set()
        evidence = []
        for index, record in enumerate(records):
            identity = f"room/{method}/record-{index}"
            if _required_text(record, "freeze_id", identity) != freeze_id:
                raise ValueError(f"{identity} freeze_id differs from manifest")
            scene_id = _required_text(record, "scene_id", identity)
            _required_text(record, "source_build", identity)
            if scene_id in seen_scenes:
                raise ValueError(f"duplicate room scene for {method}: {scene_id}")
            seen_scenes.add(scene_id)
            render_dir = _resolve_path(record.get("render_dir"), manifest_path.parent, identity, "render_dir")
            gt_dir = _resolve_path(record.get("gt_dir"), manifest_path.parent, identity, "gt_dir")
            result = evaluate_images(
                render_dir,
                gt_dir,
                mask_dir=(
                    _resolve_path(record.get("mask_dir"), manifest_path.parent, identity, "mask_dir")
                    if record.get("mask_dir") else None
                ),
                lpips_evaluator=evaluator,
                strict=True,
                coverage=record.get("coverage"),
                _hash_cache=hash_cache,
                _return_details=True,
            )
            actual_frames = {row["frame"] for row in result["frames"]}
            declared_frames = _declared_frames(record, identity)
            if declared_frames is not None and declared_frames != actual_frames:
                raise ValueError(
                    f"{identity} evaluation_frames differ from paired directory files; "
                    f"missing={sorted(actual_frames - declared_frames)}, "
                    f"extra={sorted(declared_frames - actual_frames)}"
                )
            if "n_views" in record and _nonnegative_int(record["n_views"], f"{identity}.n_views") != result["n_images"]:
                raise ValueError(f"{identity} n_views differs from paired image count")
            _validate_declared_room_views(
                record,
                manifest_dir=manifest_path.parent,
                render_dir=render_dir,
                gt_dir=gt_dir,
                actual_frames=actual_frames,
                identity=identity,
                hash_cache=hash_cache,
            )
            if record.get("optimization_input_frames") is None:
                raise ValueError(
                    f"{identity} is missing required optimization_input_frames"
                )
            _check_generator_leakage(
                record["optimization_input_frames"], actual_frames, identity
            )
            for metric in result["observations"]:
                for observation in result["observations"][metric]:
                    observation["scene"] = scene_id
            frame_hash_by_name = {row["frame"]: row["source_artifact_hash"] for row in result["frames"]}
            for frame in actual_frames:
                # The pair hash includes GT; equality across methods therefore
                # also checks the exact held-out reference content.
                room_gt_hashes.setdefault((scene_id, frame), _sha256_file(gt_dir / frame, hash_cache))
                if room_gt_hashes[(scene_id, frame)] != _sha256_file(gt_dir / frame, hash_cache):
                    raise ValueError(f"room methods use different GT content for {scene_id}/{frame}")
            room_frame_sets[(method, scene_id)] = actual_frames
            if result["coverage"]["render_only"] or result["coverage"]["gt_only"]:
                coverage_losses.append({"unit": "room", "method": method, "scene_id": scene_id, **result["coverage"]})
            evidence.append({
                "scene_id": scene_id,
                "n_images": result["n_images"],
                "exact_duplicate_images": result["exact_duplicate_images"],
                "metric_samples": result["metric_samples"],
                "metric_source_hashes": result["metric_source_hashes"],
                "coverage": result["coverage"],
                "frame_source_hashes": frame_hash_by_name,
                "observations": result["observations"],
            })
        minimum = _method_minimum(
            coverage,
            mapping_key="minimum_room_views_by_method",
            fallback_key="minimum_room_views_per_method",
            method=method,
            source_id=source_id,
        )
        observed = sum(row["n_images"] for row in evidence)
        if observed < minimum:
            raise ValueError(f"room method {method!r} has {observed} views, below declared minimum {minimum}")
        minimum_scenes = _method_minimum(
            coverage,
            mapping_key="minimum_room_scenes_by_method",
            fallback_key="minimum_room_scenes_per_method",
            method=method,
            source_id=source_id,
        )
        if len(evidence) < minimum_scenes:
            raise ValueError(
                f"room method {method!r} has {len(evidence)} scenes, "
                f"below declared minimum {minimum_scenes}"
            )
        metric_details = {}
        for metric in ("psnr", "ssim", "lpips"):
            observations = [item for row in evidence for item in row["observations"][metric]]
            metric_details[metric] = _aggregate_observations(
                observations,
                balance_key="scene",
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=_stable_seed(bootstrap_seed, "room", method, metric),
            )
        row = {
            "unit": "room",
            "method": method,
            "source_id": source_id,
            "n_images": observed,
            "n_scenes": len(evidence),
            "exact_duplicate_images": sum(item["exact_duplicate_images"] for item in evidence),
            "psnr": metric_details["psnr"]["value"],
            "ssim": metric_details["ssim"]["value"],
            "lpips": metric_details["lpips"]["value"],
            "cd_cm": None,
            "f1_20": None,
            "collapses": None,
            "metrics": metric_details,
            "metric_samples": {name: value["n"] for name, value in metric_details.items()},
            "metric_source_hashes": {name: value["source_artifact_hash"] for name, value in metric_details.items()},
        }
        rows.append(row)
        room_evidence[method] = evidence

    # Identical paired cameras are required across methods that contain a scene.
    by_scene: dict[str, list[tuple[str, set[str]]]] = defaultdict(list)
    for (method, scene), frames in room_frame_sets.items():
        by_scene[scene].append((method, frames))
    for scene, method_frames in by_scene.items():
        union = set().union(*(frames for _, frames in method_frames))
        for method, frames in method_frames:
            if frames != union:
                raise ValueError(
                    f"room methods do not use the exact same camera set for scene {scene}; "
                    f"{method} missing={sorted(union - frames)}"
                )

    object_evidence: dict[str, list[dict[str, Any]]] = {}
    geometry_contract: tuple[str, str, int] | None = None
    object_reference: dict[tuple[str, str], tuple[str, str]] = {}
    for method, spec in object_methods.items():
        source_id, records = _method_records(spec, f"object method {method!r}")
        seen_objects: set[tuple[str, str]] = set()
        evidence = []
        for index, record in enumerate(records):
            identity = f"object/{method}/record-{index}"
            if _required_text(record, "freeze_id", identity) != freeze_id:
                raise ValueError(f"{identity} freeze_id differs from manifest")
            scene_id = _required_text(record, "scene_id", identity)
            object_id = _required_text(record, "object_id", identity)
            _required_text(record, "source_build", identity)
            object_key = (scene_id, object_id)
            if object_key in seen_objects:
                raise ValueError(f"duplicate object record for {method}: {scene_id}/{object_id}")
            seen_objects.add(object_key)
            render_dir = _resolve_path(record.get("render_dir"), manifest_path.parent, identity, "render_dir")
            gt_dir = _resolve_path(record.get("gt_dir"), manifest_path.parent, identity, "gt_dir")
            mask_dir = _resolve_path(record.get("mask_dir"), manifest_path.parent, identity, "mask_dir")
            image_result = evaluate_images(
                render_dir,
                gt_dir,
                mask_dir=mask_dir,
                lpips_evaluator=None,
                strict=True,
                require_masks=True,
                coverage=record.get("coverage"),
                _hash_cache=hash_cache,
                _return_details=True,
            )
            actual_frames = {row["frame"] for row in image_result["frames"]}
            declared_frames = _declared_frames(record, identity)
            if declared_frames is not None and declared_frames != actual_frames:
                raise ValueError(f"{identity} evaluation_frames differ from paired directory files")
            _check_generator_leakage(record.get("generator_input_frames"), actual_frames, identity)
            pred_path = _resolve_path(record.get("pred_surface"), manifest_path.parent, identity, "pred_surface")
            gt_path = _resolve_path(
                record.get("gt_surface", record.get("independent_gt_surface")),
                manifest_path.parent,
                identity,
                "gt_surface",
            )
            registration_hash = _required_text(record, "registration_input_surface_hash", identity)
            if SHA256_RE.fullmatch(registration_hash) is None:
                raise ValueError(f"{identity} registration_input_surface_hash must be lowercase SHA-256")
            evaluation_hash = _sha256_file(gt_path, hash_cache)
            if registration_hash == evaluation_hash:
                raise ValueError(f"registration and evaluation surface hashes are identical for {identity}")
            if record.get("evaluation_surface_hash") is not None and record["evaluation_surface_hash"] != evaluation_hash:
                raise ValueError(f"{identity} evaluation_surface_hash does not match gt_surface")
            if record.get("registration_input_surface") is not None:
                registration_path = _resolve_path(record["registration_input_surface"], manifest_path.parent, identity, "registration_input_surface")
                if _sha256_file(registration_path, hash_cache) != registration_hash:
                    raise ValueError(f"{identity} registration_input_surface_hash does not match its file")
            units = str(record.get("units", object_split.get("units", ""))).strip().lower()
            if units not in UNIT_SCALE_TO_METERS:
                raise ValueError(f"{identity} requires units in {sorted(UNIT_SCALE_TO_METERS)}")
            coordinate_frame = str(record.get("coordinate_frame", object_split.get("coordinate_frame", ""))).strip()
            if not coordinate_frame:
                raise ValueError(f"{identity} is missing required coordinate_frame")
            alignment = str(record.get("alignment_convention", object_split.get("alignment_convention", ""))).strip()
            if not alignment:
                raise ValueError(f"{identity} is missing required alignment_convention")
            sample_value = record.get("surface_sample_count", object_split.get("surface_sample_count"))
            sample_count = _nonnegative_int(sample_value, f"{identity}.surface_sample_count")
            if sample_count < 1:
                raise ValueError(f"{identity} surface_sample_count must be positive")
            contract = (units, alignment, sample_count)
            if geometry_contract is None:
                geometry_contract = contract
            elif geometry_contract != contract:
                raise ValueError(
                    "units, alignment convention, and surface sample count must be identical across proposals"
                )
            reference_contract = (coordinate_frame, evaluation_hash)
            if object_key in object_reference and object_reference[object_key] != reference_contract:
                raise ValueError(f"methods use different coordinate frame or evaluation surface for {scene_id}/{object_id}")
            object_reference[object_key] = reference_contract
            scale = UNIT_SCALE_TO_METERS[units]
            predicted = _fixed_sample(load_surface(pred_path), sample_count, f"{identity} predicted surface") * scale
            target = _fixed_sample(load_surface(gt_path), sample_count, f"{identity} evaluation surface") * scale
            geometry = geometry_metrics(predicted, target, threshold_m=0.02)
            geometry_source_hash = _combined_hash([
                f"pred:{_sha256_file(pred_path, hash_cache)}",
                f"gt:{evaluation_hash}",
                f"registration:{registration_hash}",
                f"units:{units}",
                f"frame:{coordinate_frame}",
                f"alignment:{alignment}",
                f"samples:{sample_count}",
            ])
            for metric in image_result["observations"]:
                for observation in image_result["observations"][metric]:
                    observation["object"] = f"{scene_id}/{object_id}"
                    observation["category"] = str(record.get("category_id", "unlabeled"))
            evidence.append({
                "scene_id": scene_id,
                "object_id": object_id,
                "object": f"{scene_id}/{object_id}",
                "category": str(record.get("category_id", "unlabeled")),
                "n_images": image_result["n_images"],
                "image_observations": image_result["observations"],
                "geometry": geometry,
                "geometry_source_hash": geometry_source_hash,
                "evaluation_surface_hash": evaluation_hash,
                "registration_input_surface_hash": registration_hash,
                "surface_sample_count": sample_count,
            })
        minimum_objects = _method_minimum(
            coverage,
            mapping_key="minimum_objects_by_method",
            fallback_key="minimum_objects_per_method",
            method=method,
            source_id=source_id,
        )
        if len(evidence) < minimum_objects:
            raise ValueError(f"object method {method!r} has {len(evidence)} objects, below declared minimum {minimum_objects}")
        minimum_views = _method_minimum(
            coverage,
            mapping_key="minimum_object_views_by_method",
            fallback_key="minimum_object_views_per_method",
            method=method,
            source_id=source_id,
        )
        n_images = sum(item["n_images"] for item in evidence)
        if n_images < minimum_views:
            raise ValueError(f"object method {method!r} has {n_images} views, below declared minimum {minimum_views}")
        metrics: dict[str, Any] = {}
        for metric in ("psnr", "ssim"):
            # Object appearance is object-balanced: first aggregate held-out
            # views within each object, then treat each object as one sample.
            observations = []
            for item in evidence:
                frame_values = item["image_observations"][metric]
                if frame_values:
                    observations.append({
                        "object": item["object"],
                        "category": item["category"],
                        "value": float(np.mean([row["value"] for row in frame_values])),
                        "source_hash": _combined_hash(row["source_hash"] for row in frame_values),
                    })
            detail = _aggregate_observations(
                observations,
                balance_key="object",
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=_stable_seed(bootstrap_seed, "object", method, metric),
            )
            categories: dict[str, list[float]] = defaultdict(list)
            for observation in observations:
                categories[observation["category"]].append(observation["value"])
            detail["category_balanced"] = (
                float(np.mean([np.mean(values) for values in categories.values()]))
                if categories else None
            )
            metrics[metric] = detail
        for metric in ("cd_cm", "f1_20"):
            observations = [
                {
                    "object": item["object"],
                    "category": item["category"],
                    "value": item["geometry"][metric],
                    "source_hash": item["geometry_source_hash"],
                }
                for item in evidence
            ]
            detail = _aggregate_observations(
                observations,
                balance_key="object",
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=_stable_seed(bootstrap_seed, "object", method, metric),
            )
            categories: dict[str, list[float]] = defaultdict(list)
            for observation in observations:
                categories[observation["category"]].append(observation["value"])
            detail["category_balanced"] = (
                float(np.mean([np.mean(values) for values in categories.values()]))
                if categories else None
            )
            metrics[metric] = detail
        row = {
            "unit": "object",
            "method": method,
            "source_id": source_id,
            "n_images": n_images,
            "n_objects": len(evidence),
            "psnr": metrics["psnr"]["value"],
            "ssim": metrics["ssim"]["value"],
            "lpips": None,
            "cd_cm": metrics["cd_cm"]["value"],
            "f1_20": metrics["f1_20"]["value"],
            "collapses": int(sum(item["geometry"]["collapse"] for item in evidence)) if evidence else None,
            "metrics": metrics,
            "metric_samples": {name: value["n"] for name, value in metrics.items()},
            "metric_source_hashes": {name: value["source_artifact_hash"] for name, value in metrics.items()},
        }
        rows.append(row)
        object_evidence[method] = evidence

    complete_room = set(room_methods) == set(ROOM_METHODS) and all(
        next(row for row in rows if row["unit"] == "room" and row["method"] == method)["n_images"] > 0
        for method in ROOM_METHODS
    )
    complete_object = set(object_methods) == set(OBJECT_METHODS) and all(
        next(row for row in rows if row["unit"] == "object" and row["method"] == method)["n_objects"] > 0
        for method in OBJECT_METHODS
    )
    all_room_metrics = all(
        row["metrics"][metric]["n"] > 0
        for row in rows if row["unit"] == "room"
        for metric in ("psnr", "ssim", "lpips")
    )
    declared_paper_ready = manifest.get("paper_ready", False) is True
    paper_ready = bool(
        declared_paper_ready and complete_room and complete_object
        and all_room_metrics and not coverage_losses
    )
    manifest_sha = _sha256_file(manifest_path)
    provenance_spec = manifest.get("provenance", {})
    if not isinstance(provenance_spec, dict):
        raise ValueError("manifest provenance must be an object")
    code_commit = provenance_spec.get("code_commit", manifest.get("code_commit")) or _git_commit()
    validation = {
        "valid": True,
        "paper_ready": paper_ready,
        "declared_paper_ready": declared_paper_ready,
        "room_method_count": len(room_methods),
        "object_method_count": len(object_methods),
        "room_record_count": sum(len(value) for value in room_evidence.values()),
        "object_record_count": sum(len(value) for value in object_evidence.values()),
        "coverage_losses": coverage_losses,
        "generator_leakage_count": 0,
        "registration_evaluation_hash_collision_count": 0,
        "lpips_provenance_complete": bool(
            evaluator.provenance.get("backend_class")
            and evaluator.provenance.get("backbone_checkpoint")
            and (
                evaluator.provenance.get("package") != "lpips"
                or evaluator.provenance.get("linear_checkpoint")
            )
        ),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "freeze_id": freeze_id,
        "provenance": {
            "freeze_id": freeze_id,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "code_commit": code_commit,
        },
        "validation": validation,
        "paper_ready": paper_ready,
        "rows_count": len(rows),
        "rows": rows,
        "lpips_backend_error": evaluator.error,
        "lpips_provenance": evaluator.provenance,
    }
    outputs = {
        "fidelity_table.json": json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        "fidelity_table.csv": _csv_text(rows),
    }
    _atomic_write_bundle(Path(out_dir), outputs)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lpips-device", default="cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args(argv)
    payload = evaluate_manifest(
        args.manifest,
        args.out,
        lpips_device=args.lpips_device,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
