"""Export leakage-checked held-out room renders for the E2 fidelity table.

This deliberately does not modify the legacy ``factory_eval_render`` outputs.
For one ScanNet++ scene it reads the frozen ``*_factory`` and ``*_auto``
object builds, renders the same eight official test cameras for all three E2
conditions, and atomically publishes a self-contained scene directory.

The module must run in the gsplat environment (``run_gs`` from ``run/env.sh``).
All outputs are required to live inside this checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
N_EVAL_FRAMES = 8
METHOD_INPUT = "input_scene_gaussian"
METHOD_GT = "factorized_gt_discovery"
METHOD_AUTO = "factorized_auto_discovery"


def _inside_repo(path: str | os.PathLike[str], *, must_exist: bool) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    candidate = candidate.resolve(strict=must_exist)
    try:
        candidate.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"path must stay inside {REPO_ROOT}: {candidate}") from exc
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc


def select_official_frames(
    split_payload: dict[str, Any],
    registered_names: set[str],
    *,
    count: int = N_EVAL_FRAMES,
    excluded_names: set[str] | None = None,
) -> list[str]:
    """Select fixed official-test frames disjoint from all optimization inputs."""
    test = split_payload.get("test")
    train = split_payload.get("train")
    if not isinstance(test, list) or not all(isinstance(v, str) and v for v in test):
        raise ValueError("train_test_lists.json must contain a non-empty string list 'test'")
    if not isinstance(train, list) or not all(isinstance(v, str) and v for v in train):
        raise ValueError("train_test_lists.json must contain a non-empty string list 'train'")
    if len(test) != len(set(test)) or len(train) != len(set(train)):
        raise ValueError("train/test frame lists must not contain duplicates")
    overlap = sorted(set(test) & set(train))
    if overlap:
        raise ValueError(f"official train/test split overlaps: {overlap[:5]}")
    excluded = excluded_names or set()
    usable = [
        name for name in test
        if name in registered_names and name not in excluded
    ]
    if len(usable) < count:
        raise ValueError(
            f"need at least {count} registered official-test frames disjoint from "
            f"optimization inputs; found {len(usable)}"
        )
    indices = np.linspace(0, len(usable) - 1, count).astype(int).tolist()
    selected = [usable[index] for index in indices]
    if len(selected) != count or len(set(selected)) != count:
        raise ValueError("official-test frame selection is not unique")
    basenames = [Path(name).stem for name in selected]
    if len(set(basenames)) != count:
        raise ValueError("selected frame stems collide in the canonical PNG layout")
    for name in selected:
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"test frame must be a plain filename: {name!r}")
    return selected


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _plain_frame(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).name != value
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(f"{label} must be a plain non-empty filename: {value!r}")
    return value


def _checked_artifact(
    base: Path, raw: Any, *, label: str, repo_relative: bool = False,
) -> Path:
    """Re-hash one declared artifact and reject symlink/path traversal."""
    if not isinstance(raw, dict):
        raise ValueError(f"{label} artifact must be an object")
    relative = raw.get("path")
    size = raw.get("size_bytes")
    digest = raw.get("sha256")
    if (
        not isinstance(relative, str)
        or not relative
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise ValueError(f"invalid {label} artifact declaration")
    raw_path = Path(relative)
    if raw_path.is_absolute():
        raise ValueError(f"{label} artifact path must be relative")
    candidate = (REPO_ROOT / raw_path) if repo_relative else (base / raw_path)
    lexical = Path(os.path.abspath(candidate))
    allowed = REPO_ROOT if repo_relative else base
    try:
        parts = lexical.relative_to(allowed).parts
    except ValueError as exc:
        raise ValueError(f"{label} artifact escapes its declared root: {relative}") from exc
    current = allowed
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} artifact uses a symlink: {current}")
    if not lexical.is_file() or lexical.stat().st_size != size:
        raise ValueError(f"{label} artifact size/path mismatch: {lexical}")
    if _sha256(lexical) != digest:
        raise ValueError(f"{label} artifact hash mismatch: {lexical}")
    return lexical


def _load_replacement_overlay(
    *,
    config_path: str | os.PathLike[str] | None,
    replacement_root: str | os.PathLike[str] | None,
    freeze_id: str,
    scene_id: str,
    mode: str,
    code_commit: str,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any] | None]:
    """Load the exact frozen train-only remediation bundle for one source."""
    if (config_path is None) != (replacement_root is None):
        raise ValueError("replacement config and root must be supplied together")
    if config_path is None:
        return {}, None
    config_file = _inside_repo(config_path, must_exist=True)
    config = _read_json(config_file)
    remediation = config.get("leakage_remediation") if isinstance(config, dict) else None
    if not isinstance(remediation, dict) or remediation.get("schema_version") != 1:
        raise ValueError("E2 config lacks leakage_remediation schema 1")
    bundles = remediation.get("bundles")
    if not isinstance(bundles, list):
        raise ValueError("E2 leakage_remediation.bundles must be a list")
    expected = [
        item for item in bundles
        if isinstance(item, dict)
        and item.get("scene_id") == scene_id
        and item.get("mode") == mode
    ]
    if len(expected) > 1:
        raise ValueError(f"duplicate replacement bundle for {scene_id}/{mode}")
    if not expected:
        return {}, None

    root = _inside_repo(replacement_root, must_exist=True)
    canonical_root = _inside_repo(
        f"outputs/icra2027/{freeze_id}/fidelity/replacements", must_exist=True
    )
    if root != canonical_root or root.is_symlink() or not root.is_dir():
        raise ValueError("replacement root is not the canonical frozen E2 directory")
    expected_bundle = expected[0]
    bundle_id = expected_bundle.get("bundle_id")
    if bundle_id != f"{scene_id}-{mode}":
        raise ValueError(f"invalid replacement bundle identity: {bundle_id!r}")
    bundle_dir = _inside_repo(root / bundle_id, must_exist=True)
    if bundle_dir.is_symlink() or not bundle_dir.is_dir():
        raise ValueError(f"replacement bundle is not a regular directory: {bundle_dir}")
    manifest_path = _inside_repo(bundle_dir / "replacement_manifest.json", must_exist=True)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"missing regular replacement manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    fixed = {
        "schema_version": 1,
        "task": "E2 leakage remediation",
        "freeze_id": freeze_id,
        "bundle_id": bundle_id,
        "dataset_id": "scannetpp_v2",
        "split_id": "nvs_sem_val",
        "scene_id": scene_id,
        "mode": mode,
        "legacy_source": f"outputs/{scene_id}_{mode}",
        "legacy_source_immutable": True,
        "paper_ready": False,
    }
    if not isinstance(manifest, dict) or any(
        manifest.get(key) != value for key, value in fixed.items()
    ):
        raise ValueError(f"replacement manifest identity mismatch: {manifest_path}")
    git = manifest.get("git")
    if not isinstance(git, dict) or git.get("commit") != code_commit or git.get("dirty") is not False:
        raise ValueError(f"replacement bundle Git provenance mismatch: {manifest_path}")
    config_artifact = manifest.get("config_artifact")
    if (
        not isinstance(config_artifact, dict)
        or config_artifact.get("sha256") != _sha256(config_file)
        or config_artifact.get("canonical_sha256") != _canonical_hash(config)
    ):
        raise ValueError(f"replacement bundle config provenance mismatch: {manifest_path}")
    validation = manifest.get("validation")
    required_validation = (
        "fixed_target_population",
        "mapped_by_gt_object_id",
        "preserved_legacy_indices",
        "generation_frames_official_train_only",
        "all_regenerated_alignments_accepted",
        "legacy_sources_unchanged",
        "atomic_fresh_publish",
    )
    if not isinstance(validation, dict) or any(
        validation.get(field) is not True for field in required_validation
    ) or validation.get("generation_eval_overlap") != []:
        raise ValueError(f"replacement validation gates failed: {manifest_path}")

    expected_targets = expected_bundle.get("targets")
    observed_targets = manifest.get("targets")
    if not isinstance(expected_targets, list) or not isinstance(observed_targets, list):
        raise ValueError("replacement targets must be lists")
    expected_by_index = {item.get("legacy_index"): item for item in expected_targets}
    if len(expected_by_index) != len(expected_targets):
        raise ValueError("replacement config repeats a legacy index")
    overlay: dict[int, dict[str, Any]] = {}
    public_targets: list[dict[str, Any]] = []
    for target in observed_targets:
        if not isinstance(target, dict):
            raise ValueError("replacement target must be an object")
        index = target.get("legacy_index")
        expected_target = expected_by_index.get(index)
        if expected_target is None:
            raise ValueError(f"unexpected replacement target index: {index!r}")
        for observed_key, expected_key in (
            ("gt_object_id", "gt_object_id"),
            ("label", "label"),
            ("old_test_frame", "old_frame"),
        ):
            if target.get(observed_key) != expected_target.get(expected_key):
                raise ValueError(f"replacement target contract mismatch: {index}/{observed_key}")
        new_frame = _plain_frame(target.get("new_train_frame"), label="new train frame")
        if target.get("optimization_input_frames") != [new_frame]:
            raise ValueError(f"replacement target frame provenance mismatch: {index}")
        if target.get("alignment_tier") not in {"A", "B"}:
            raise ValueError(f"replacement target alignment is not accepted: {index}")
        legacy_artifacts = target.get("legacy_artifacts")
        if not isinstance(legacy_artifacts, list) or not legacy_artifacts:
            raise ValueError(f"replacement target has no legacy artifact closure: {index}")
        for artifact in legacy_artifacts:
            _checked_artifact(
                REPO_ROOT, artifact, label="replacement legacy", repo_relative=True
            )
        generated = target.get("generated_artifacts")
        if not isinstance(generated, list) or not generated:
            raise ValueError(f"replacement target has no artifact closure: {index}")
        generated_by_path: dict[str, dict[str, Any]] = {}
        for artifact in generated:
            if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
                raise ValueError(f"invalid replacement artifact for target {index}")
            if artifact["path"] in generated_by_path:
                raise ValueError(f"duplicate replacement artifact path: {artifact['path']}")
            _checked_artifact(bundle_dir, artifact, label="replacement generated")
            generated_by_path[artifact["path"]] = artifact
        prefix = f"objects/obj_{index:02d}/"
        required_paths = {
            "meta": prefix + "meta.json",
            "alignment": prefix + "aligned.json",
            "gaussian": prefix + "trellis_gs.ply",
        }
        if not set(required_paths.values()) <= set(generated_by_path):
            raise ValueError(f"replacement target lacks consumed artifacts: {index}")
        overlay[index] = {
            "frame": new_frame,
            **{
                key + "_path": bundle_dir / relative
                for key, relative in required_paths.items()
            },
        }
        # Preserve the exact frozen target record. The inventory compares this
        # projection byte-for-byte with replacement_manifest.json rather than
        # accepting a weaker hand-written subset.
        public_targets.append(dict(target))
    if set(overlay) != set(expected_by_index):
        raise ValueError(f"replacement target population mismatch: {bundle_id}")
    declared_frames = sorted({target["new_train_frame"] for target in public_targets})
    if sorted(manifest.get("optimization_input_frames", [])) != declared_frames:
        raise ValueError(f"replacement bundle frame union mismatch: {bundle_id}")
    provenance = {
        "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_sha256": _sha256(manifest_path),
        "bundle_id": bundle_id,
        "targets": public_targets,
    }
    return overlay, provenance


def _accepted_generation_input_frames(
    source: Path, *, use_trellis_snapshot: bool,
    replacements: dict[int, dict[str, Any]] | None = None,
) -> list[str]:
    """Read the actual single-view generation input for every accepted object."""
    objects_path = _inside_repo(source / "objects" / "objects.json", must_exist=True)
    objects = _read_json(objects_path)
    if not isinstance(objects, list):
        raise ValueError(f"object inventory must be a list: {objects_path}")
    frames: set[str] = set()
    replacements = replacements or {}
    used_replacements: set[int] = set()
    seen: set[int] = set()
    for record in objects:
        if not isinstance(record, dict) or not isinstance(record.get("index"), int):
            raise ValueError(f"invalid object record in {objects_path}: {record!r}")
        index = record["index"]
        if index in seen:
            raise ValueError(f"duplicate object index {index} in {objects_path}")
        seen.add(index)
        object_dir = source / "objects" / f"obj_{index:02d}"
        aligned_path = _inside_repo(object_dir / "aligned.json", must_exist=True)
        aligned = _read_json(aligned_path)
        if aligned.get("rejected"):
            continue
        if use_trellis_snapshot:
            snapshot_alignment = _read_json(
                _inside_repo(
                    object_dir / "trellis" / "aligned.json", must_exist=True
                )
            )
            if snapshot_alignment.get("rejected"):
                raise ValueError(
                    "canonical accepted object has a rejected TRELLIS snapshot: "
                    f"{object_dir}"
                )
        elif aligned.get("vmesh_source") not in (None, "trellis"):
            raise ValueError(
                "automatic construction asset is not a single-view TRELLIS "
                f"asset: {aligned_path}"
            )
        replacement = replacements.get(index)
        if replacement is not None:
            used_replacements.add(index)
        meta_path = _inside_repo(
            replacement["meta_path"] if replacement is not None
            else object_dir / "meta.json",
            must_exist=True,
        )
        meta = _read_json(meta_path)
        if not isinstance(meta, dict) or meta.get("index") != index:
            raise ValueError(f"generation metadata index mismatch: {meta_path}")
        frame = meta.get("frame")
        if (
            not isinstance(frame, str)
            or not frame
            or Path(frame).name != frame
            or frame in {".", ".."}
        ):
            raise ValueError(f"invalid generation input frame in {meta_path}: {frame!r}")
        if replacement is not None and frame != replacement["frame"]:
            raise ValueError(f"replacement generation frame mismatch: {meta_path}")
        frames.add(frame)
    if used_replacements != set(replacements):
        raise ValueError(
            "replacement indices are not canonical accepted objects: "
            f"{sorted(set(replacements) - used_replacements)}"
        )
    return sorted(frames)


def _load_composite(
    common,
    background,
    source: Path,
    *,
    use_trellis_snapshot: bool,
    replacements: dict[int, dict[str, Any]] | None = None,
    replacement_provenance: dict[str, Any] | None = None,
    public_fill: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    public = None
    root = REPO_ROOT
    if public_fill is not None:
        from agents.edit.inpaint_fill import validate_public_fill
        from robo.eval import agentic_ablation as e3
        if use_trellis_snapshot or replacements or replacement_provenance:
            raise ValueError("public selected assets cannot use legacy replacement aliases")
        public = validate_public_fill(public_fill)
        if not public['result']['cleaned_background_created'] or public['source']['blocked']:
            raise ValueError("public composite requires a complete clean background")
        if Path(source) != public['source']['factory']:
            raise ValueError("public composite factory differs from its sealed fill")
        root = e3.REPOSITORY_ROOT
        background = common.load_gaussians(public['clean_background']['path'])
    def inside(path, *, must_exist):
        if public is not None:
            return e3.checked_repo_path(path, 'public composite artifact', must_exist=must_exist)
        return _inside_repo(path, must_exist=must_exist)
    def artifact(path):
        return {'path': str(path.relative_to(root)), 'size_bytes': path.stat().st_size,
                'sha256': _sha256(path)}
    objects_path = inside(source / "objects" / "objects.json", must_exist=True)
    if not objects_path.is_file():
        raise FileNotFoundError(f"missing object inventory: {objects_path}")
    objects = _read_json(objects_path)
    if not isinstance(objects, list):
        raise ValueError(f"object inventory must be a list: {objects_path}")

    parts = [background]
    artifacts = [artifact(objects_path)]
    if public is not None:
        artifacts += [artifact(Path(public['seal_identity']['path'])),
                      artifact(Path(public['clean_background']['path']))]
    accepted_ids: list[int] = []
    optimization_input_frames: set[str] = (set(public['source']['train_images'])
                                           if public is not None else set())
    seen: set[int] = set()
    replacements = replacements or {}
    used_replacements: set[int] = set()
    for record in objects:
        if not isinstance(record, dict) or not isinstance(record.get("index"), int):
            raise ValueError(f"invalid object record in {objects_path}: {record!r}")
        index = record["index"]
        if index in seen:
            raise ValueError(f"duplicate object index {index} in {objects_path}")
        seen.add(index)
        object_dir = source / "objects" / f"obj_{index:02d}"
        aligned_path = inside(object_dir / "aligned.json", must_exist=True)
        if not aligned_path.is_file():
            raise FileNotFoundError(f"missing alignment record: {aligned_path}")
        aligned = _read_json(aligned_path)
        artifacts.append(artifact(aligned_path))
        if public is not None:
            if (record.get('automatic_instance_id') != index or record.get('instance_namespace') != 'automatic'
                    or 'gt_object_id' in record or aligned.get('index') != index):
                raise ValueError('public composite requires unchanged automatic object identities')
            if aligned.get('terminal_action') in {'reject', 'abstain'}:
                continue
            if aligned.get('terminal_action') != 'accept' or aligned.get('rejected'):
                raise ValueError('public composite asset has inconsistent terminal state')
            selected_path = inside(object_dir/'selected_asset.json', must_exist=True)
            selected = _read_json(selected_path)
            if (selected.get('terminal_action') != 'accept' or selected.get('object_slot') != object_dir.name
                    or selected.get('job_id') != aligned.get('job_id')
                    or selected.get('policy_id') != aligned.get('policy_id')
                    or selected.get('selected_proposal_id') != aligned.get('proposal_id')):
                raise ValueError('public selected asset/alignment identities differ')
            gaussian_path = inside(object_dir/'trellis_gs.ply', must_exist=True)
            asset = selected['selected_asset']
            if (_sha256(gaussian_path) != asset['artifact_hashes']['raw_gaussian']
                    or gaussian_path.stat().st_size != asset['artifact_sizes']['raw_gaussian']):
                raise ValueError('public native Gaussian differs from selected proposal')
            artifacts.append(artifact(selected_path))
            asset_alignment = aligned
        else:
            if aligned.get("rejected"):
                continue
            replacement = replacements.get(index)
            if replacement is not None:
                used_replacements.add(index)
            meta_path = inside(
                replacement["meta_path"] if replacement is not None
                else object_dir / "meta.json",
                must_exist=True,
            )
            meta = _read_json(meta_path)
            if not isinstance(meta, dict) or meta.get("index") != index:
                raise ValueError(f"generation metadata index mismatch: {meta_path}")
            frame = meta.get("frame")
            if (
                not isinstance(frame, str)
                or not frame
                or Path(frame).name != frame
                or frame in {".", ".."}
            ):
                raise ValueError(f"invalid generation input frame in {meta_path}: {frame!r}")
            if replacement is not None and frame != replacement["frame"]:
                raise ValueError(f"replacement generation frame mismatch: {meta_path}")
            artifacts.append(artifact(meta_path))
            optimization_input_frames.add(frame)
            if replacement is not None:
                asset_alignment_path = inside(
                    replacement["alignment_path"], must_exist=True
                )
                asset_alignment = _read_json(asset_alignment_path)
                if asset_alignment.get("rejected"):
                    raise ValueError(f"replacement alignment is rejected: {asset_alignment_path}")
                gaussian_path = inside(
                    replacement["gaussian_path"], must_exist=True
                )
                artifacts.append(artifact(asset_alignment_path))
            elif use_trellis_snapshot:
                asset_alignment_path = inside(
                    object_dir / "trellis" / "aligned.json", must_exist=True
                )
                asset_alignment = _read_json(asset_alignment_path)
                if asset_alignment.get("rejected"):
                    raise ValueError(
                        "canonical accepted object has a rejected TRELLIS snapshot: "
                        f"{object_dir}"
                    )
                gaussian_path = inside(
                    object_dir / "trellis" / "trellis_gs.ply", must_exist=True
                )
                artifacts.append(artifact(asset_alignment_path))
            else:
                if aligned.get("vmesh_source") not in (None, "trellis"):
                    raise ValueError(
                        "automatic construction asset is not a single-view TRELLIS "
                        f"asset: {aligned_path}"
                    )
                asset_alignment = aligned
                gaussian_path = inside(
                    object_dir / "trellis_gs.ply", must_exist=True
                )
        transform = np.asarray(asset_alignment.get("T"), dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(f"invalid 4x4 alignment transform: {asset_alignment_path if use_trellis_snapshot else aligned_path}")
        if not gaussian_path.is_file() or gaussian_path.stat().st_size == 0:
            raise FileNotFoundError(f"missing accepted-object Gaussian: {gaussian_path}")
        gaussian = common.pad_sh(
            common.load_gaussians(gaussian_path), background["sh_degree"]
        )
        parts.append(common.transform_gaussians(gaussian, transform))
        artifacts.append(artifact(gaussian_path))
        accepted_ids.append(index)

    if used_replacements != set(replacements):
        raise ValueError(
            "replacement indices are not canonical accepted objects: "
            f"{sorted(set(replacements) - used_replacements)}"
        )

    if public is not None and len(accepted_ids) != public['source']['accepted_objects']:
        raise ValueError('public composite dropped or added accepted construction assets')
    composite = common.cat_gaussians(parts) if accepted_ids else background
    return composite, {
        "source": str(source.relative_to(root)),
        "accepted_object_ids": accepted_ids,
        "n_objects_composited": len(accepted_ids),
        "asset_variant": ("public_evidence_selected_native" if public is not None else (
            ("trellis_snapshot" if use_trellis_snapshot else "trellis_canonical")
            + ("_with_train_only_replacements" if replacements else "")
        )),
        "optimization_input_frames": sorted(optimization_input_frames),
        "artifacts": artifacts,
        **(
            {"replacement_bundle": replacement_provenance}
            if replacement_provenance is not None else {}
        ),
    }


def _write_json_fsync(path: Path, payload: Any) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _git_snapshot(root: Path) -> dict[str, Any]:
    """Return full-SHA provenance without importing the pydantic package.

    The dedicated gsplat Python intentionally has a small dependency set and
    does not install pydantic, so importing ``robo.manifest`` here would make
    rendering fail before CUDA is reached.
    """
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    commit = run("rev-parse", "HEAD")
    if len(commit) != 40 or not all(c in "0123456789abcdef" for c in commit):
        raise RuntimeError(f"invalid full Git commit from checkout: {commit!r}")
    dirty = bool(run("status", "--porcelain", "--untracked-files=normal"))
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    return {"commit": commit, "dirty": dirty, "branch": branch}


def _export_scene_impl(
    *,
    freeze_id: str,
    scene_id: str,
    factory_source: str | os.PathLike[str],
    auto_source: str | os.PathLike[str],
    out: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
    replacement_root: str | os.PathLike[str] | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    if not freeze_id or "/" in freeze_id or freeze_id in {".", ".."}:
        raise ValueError(f"invalid freeze ID: {freeze_id!r}")
    if not scene_id or "/" in scene_id or scene_id in {".", ".."}:
        raise ValueError(f"invalid scene ID: {scene_id!r}")
    factory_source = _inside_repo(factory_source, must_exist=True)
    auto_source = _inside_repo(auto_source, must_exist=True)
    if factory_source.name != f"{scene_id}_factory":
        raise ValueError(
            f"factory source does not match scene {scene_id}: {factory_source.name}"
        )
    if auto_source.name != f"{scene_id}_auto":
        raise ValueError(
            f"automatic source does not match scene {scene_id}: {auto_source.name}"
        )
    out = _inside_repo(out, must_exist=False)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"refusing to overwrite E2 room output: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(f".{out.name}.partial-{os.getpid()}")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(f"staging output already exists: {partial}")

    snapshot = _git_snapshot(REPO_ROOT)
    if snapshot["dirty"] and not allow_dirty:
        raise RuntimeError(
            "E2 render export requires a clean Git snapshot; commit the tested code first"
        )
    factory_replacements, factory_replacement_provenance = _load_replacement_overlay(
        config_path=config_path,
        replacement_root=replacement_root,
        freeze_id=freeze_id,
        scene_id=scene_id,
        mode="factory",
        code_commit=snapshot["commit"],
    )
    auto_replacements, auto_replacement_provenance = _load_replacement_overlay(
        config_path=config_path,
        replacement_root=replacement_root,
        freeze_id=freeze_id,
        scene_id=scene_id,
        mode="auto",
        code_commit=snapshot["commit"],
    )

    # common.py resolves these constants at import time, so set them first.
    os.environ["SIMANY_ROOT"] = str(REPO_ROOT)
    os.environ["SIMANY_SCENE"] = scene_id
    os.environ["SIMANY_OUT"] = str(factory_source)
    from agents.core import common

    if common.SCENE_ID != scene_id:
        raise RuntimeError(
            f"agents.core.common was imported for {common.SCENE_ID}, not {scene_id}"
        )
    split_path = common.SCENE_DIR / "dslr" / "train_test_lists.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"official train/test split is required: {split_path}")
    split_payload = _read_json(split_path)
    world_to_camera = common.load_colmap_w2c()
    factory_generation_frames = _accepted_generation_input_frames(
        factory_source,
        use_trellis_snapshot=True,
        replacements=factory_replacements,
    )
    auto_generation_frames = _accepted_generation_input_frames(
        auto_source,
        use_trellis_snapshot=False,
        replacements=auto_replacements,
    )
    optimization_input_frames = sorted(
        set(split_payload["train"])
        | set(factory_generation_frames)
        | set(auto_generation_frames)
    )
    names = select_official_frames(
        split_payload,
        set(world_to_camera),
        excluded_names=set(optimization_input_frames),
    )
    K, width, height, _ = common.load_intrinsics()
    for name in names:
        image_path = common.IMAGES_DIR / name
        if not image_path.is_file():
            raise FileNotFoundError(f"missing official held-out image: {image_path}")
    if not common.SPLAT_PLY.is_file() or common.SPLAT_PLY.stat().st_size == 0:
        raise FileNotFoundError(f"missing input-scene Gaussian: {common.SPLAT_PLY}")

    partial.mkdir()
    method_ids = [METHOD_INPUT, METHOD_GT, METHOD_AUTO]
    for directory in ["gt", *method_ids]:
        (partial / directory).mkdir()

    source_images = []
    output_artifacts = []
    for name in names:
        target = np.asarray(
            Image.open(common.IMAGES_DIR / name).convert("RGB"), dtype=np.uint8
        )
        if target.shape[:2] != (height, width):
            raise ValueError(
                f"held-out image shape {target.shape[:2]} disagrees with calibration "
                f"{(height, width)}: {common.IMAGES_DIR / name}"
            )
        output_name = f"{Path(name).stem}.png"
        gt_path = partial / "gt" / output_name
        Image.fromarray(target).save(gt_path)
        source_images.append({
            "frame": name,
            "path": str((common.IMAGES_DIR / name)),
            "size_bytes": (common.IMAGES_DIR / name).stat().st_size,
            "sha256": _sha256(common.IMAGES_DIR / name),
            "exported_png": output_name,
        })
        output_artifacts.append({
            "relative_path": str(gt_path.relative_to(partial)),
            "sha256": _sha256(gt_path),
        })
    # Keep only one factorized scene resident at a time. Some rooms have many
    # accepted object Gaussians; holding both factory and automatic composites
    # at once can needlessly double peak VRAM.
    background = common.load_gaussians(common.SPLAT_PLY)

    def render_condition(method_id: str, gaussians: dict[str, Any]) -> None:
        for name in names:
            output_name = f"{Path(name).stem}.png"
            rgb, _, _ = common.render_view(
                gaussians, world_to_camera[name], K, width, height
            )
            render_path = partial / method_id / output_name
            Image.fromarray(
                np.uint8(np.clip(rgb, 0.0, 1.0) * 255.0)
            ).save(render_path)
            output_artifacts.append({
                "relative_path": str(render_path.relative_to(partial)),
                "sha256": _sha256(render_path),
            })

    render_condition(METHOD_INPUT, background)
    factory, factory_provenance = _load_composite(
        common,
        background,
        factory_source,
        use_trellis_snapshot=True,
        replacements=factory_replacements,
        replacement_provenance=factory_replacement_provenance,
    )
    render_condition(METHOD_GT, factory)
    del factory
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    automatic, auto_provenance = _load_composite(
        common,
        background,
        auto_source,
        use_trellis_snapshot=False,
        replacements=auto_replacements,
        replacement_provenance=auto_replacement_provenance,
    )
    render_condition(METHOD_AUTO, automatic)
    del automatic

    manifest = {
        "schema_version": 1,
        "freeze_id": freeze_id,
        "dataset_id": "scannetpp_v2",
        "split_id": "nvs_sem_val:official-test",
        "scene_id": scene_id,
        "evaluation_unit": "held_out_view",
        "n_eval_frames": len(names),
        "eval_frames": names,
        "methods": method_ids,
        "image_width": width,
        "image_height": height,
        "color_space": "sRGB decoded to RGB; canonical outputs are 8-bit PNG",
        "source_scene_gaussian": {
            "path": str(common.SPLAT_PLY),
            "size_bytes": common.SPLAT_PLY.stat().st_size,
            "sha256": _sha256(common.SPLAT_PLY),
        },
        "split_artifact": {
            "path": str(split_path),
            "size_bytes": split_path.stat().st_size,
            "sha256": _sha256(split_path),
        },
        "camera_artifacts": {
            "intrinsics": {
                "path": str(common.TRANSFORMS_JSON),
                "size_bytes": common.TRANSFORMS_JSON.stat().st_size,
                "sha256": _sha256(common.TRANSFORMS_JSON),
            },
            "poses": {
                "path": str(common.COLMAP_IMAGES_TXT),
                "size_bytes": common.COLMAP_IMAGES_TXT.stat().st_size,
                "sha256": _sha256(common.COLMAP_IMAGES_TXT),
            },
        },
        "source_images": source_images,
        "official_train_frames": split_payload["train"],
        "optimization_input_frames": optimization_input_frames,
        "construction_sources": {
            METHOD_GT: factory_provenance,
            METHOD_AUTO: auto_provenance,
        },
        "output_artifacts": sorted(
            output_artifacts, key=lambda item: item["relative_path"]
        ),
        "git": snapshot,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "paper_ready": False,
        "paper_ready_reason": "scene export; canonical E2 validation/aggregation pending",
    }
    _write_json_fsync(partial / "manifest.json", manifest)
    os.replace(partial, out)
    return manifest


def export_scene(
    *,
    freeze_id: str,
    scene_id: str,
    factory_source: str | os.PathLike[str],
    auto_source: str | os.PathLike[str],
    out: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
    replacement_root: str | os.PathLike[str] | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Publish one scene and remove only this call's staging path on failure."""
    try:
        return _export_scene_impl(
            freeze_id=freeze_id,
            scene_id=scene_id,
            factory_source=factory_source,
            auto_source=auto_source,
            out=out,
            config_path=config_path,
            replacement_root=replacement_root,
            allow_dirty=allow_dirty,
        )
    except Exception:
        # `_export_scene_impl` derives this same path after validating `out`.
        # Re-resolve it here so cleanup remains repository-confined and never
        # touches a published destination or another process's staging path.
        try:
            output = _inside_repo(out, must_exist=False)
            partial = output.with_name(f".{output.name}.partial-{os.getpid()}")
            if partial.is_dir() and not partial.is_symlink():
                shutil.rmtree(partial)
        except Exception:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-id", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--factory-source", required=True)
    parser.add_argument("--auto-source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--config",
        help="frozen schema-v2 E2 config (required with --replacement-root)",
    )
    parser.add_argument(
        "--replacement-root",
        help="canonical frozen replacement bundle root (required with --config)",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="developer-only escape hatch; full Slurm launchers must never set this",
    )
    args = parser.parse_args(argv)
    result = export_scene(
        freeze_id=args.freeze_id,
        scene_id=args.scene_id,
        factory_source=args.factory_source,
        auto_source=args.auto_source,
        out=args.out,
        config_path=args.config,
        replacement_root=args.replacement_root,
        allow_dirty=args.allow_dirty,
    )
    print(json.dumps({
        "scene_id": result["scene_id"],
        "n_eval_frames": result["n_eval_frames"],
        "methods": result["methods"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
