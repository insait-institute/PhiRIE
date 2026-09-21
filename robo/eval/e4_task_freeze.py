"""Freeze one canonical paired E4 task bundle for sealed A0/A4 factories.

The per-factory task generator is useful for discovering feasible tasks, but
it derives table and region geometry independently from each construction.
Those independently generated suites are not a paired intervention.  This
module selects a canonical planning source explicitly, retains only logical
tasks supported by both sealed factories, normalizes the logical scene ID,
and publishes two suites which differ only in ``scene_xml``. Authenticated
automatic population mode retains every declared pair so missing construction
endpoints can be recorded as failures before environment creation. It does not
certify task feasibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from agents.orchestrator.artifact import sha256_file
from robo.eval import agentic_ablation as e3
from robo.eval import e3_factory_materializer as materializer


SCHEMA_VERSION = 1
CODE_ROOT = Path(__file__).resolve().parents[2]
if CODE_ROOT != materializer.CODE_ROOT:
    raise RuntimeError("E4 task freezer and factory validator code roots differ")
REPOSITORY_ROOT = materializer.REPOSITORY_ROOT
SCENE_ID_RE = materializer.SCENE_ID_RE
COMMIT_RE = materializer.COMMIT_RE
validate_materialized_factory = materializer.validate_materialized_factory
POLICY_TO_VARIANT = {"A0": "fixed_single_path", "A4": "agentic"}
SUITE_KEYS = frozenset(
    {
        "exclude_objects",
        "ext_cam",
        "robot",
        "scene",
        "scene_xml",
        "table",
        "tasks",
        "time_limit_s",
    }
)
MANIFEST_KEYS = frozenset(
    {
        "candidate_suites",
        "e3_claim_status",
        "e3_code_commit",
        "e3_freeze_id",
        "factories",
        "files",
        "logical_task_ids",
        "manifest_kind",
        "max_tasks",
        "planning_source",
        "provenance",
        "scene_id",
        "schema_version",
        "variant_task_paths",
    }
)
PROVENANCE_KEYS = frozenset(
    {
        "code_root",
        "evidence_root",
        "freezer_commit",
        "freezer_dirty",
        "validator_commit",
        "validator_dirty",
    }
)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _git_snapshot() -> dict[str, Any]:
    """Return the exact Git identity of the executing task-freeze code."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=CODE_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        top_level = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=CODE_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=CODE_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("cannot obtain the task-freezer Git snapshot") from exc
    observed_root = Path(top_level).resolve(strict=True)
    if observed_root != CODE_ROOT:
        raise ValueError(
            "executing task freezer is not rooted at its declared code root: "
            f"expected={CODE_ROOT}, observed={observed_root}"
        )
    if COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("task-freezer Git commit is not a full lowercase SHA")
    return {
        "code_root": str(observed_root),
        "commit": commit,
        "dirty": bool(status),
        "status": status,
    }


def _require_clean_code_snapshot() -> dict[str, Any]:
    snapshot = _git_snapshot()
    if snapshot.get("code_root") != str(CODE_ROOT):
        raise ValueError("task-freezer Git snapshot code root differs")
    commit = snapshot.get("commit")
    if not isinstance(commit, str) or COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("task-freezer Git snapshot lacks a full lowercase commit")
    if snapshot.get("dirty") is not False or snapshot.get("status") != []:
        raise ValueError(
            "task freezer requires a clean code worktree; Git status is "
            f"{snapshot.get('status')!r}"
        )
    return dict(snapshot)


def _evidence_root(repository_root: str | Path | None) -> Path:
    root = Path(REPOSITORY_ROOT if repository_root is None else repository_root).resolve(
        strict=True
    )
    if root != REPOSITORY_ROOT:
        raise ValueError(
            "task-freeze evidence root differs from SIMANY_EVIDENCE_ROOT: "
            f"expected={REPOSITORY_ROOT}, observed={root}"
        )
    return root


def _provenance_record(
    *, code_snapshot: Mapping[str, Any], evidence_root: Path
) -> dict[str, Any]:
    commit = code_snapshot.get("commit")
    if not isinstance(commit, str) or COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("task-freezer commit must be a full lowercase Git SHA")
    if code_snapshot.get("dirty") is not False:
        raise ValueError("task-freezer Git snapshot is dirty")
    return {
        "code_root": str(CODE_ROOT),
        "evidence_root": str(evidence_root),
        "freezer_commit": commit,
        "freezer_dirty": False,
        # Creation and standalone revalidation are shipped by this module.
        "validator_commit": commit,
        "validator_dirty": False,
    }


def _validate_provenance(
    value: Any, *, evidence_root: Path, validator_snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != PROVENANCE_KEYS:
        raise ValueError("E4 task-freeze provenance schema differs")
    freezer_commit = value["freezer_commit"]
    recorded_validator = value["validator_commit"]
    executing_validator = validator_snapshot.get("commit")
    for label, commit in (
        ("freezer", freezer_commit),
        ("recorded validator", recorded_validator),
        ("executing validator", executing_validator),
    ):
        if not isinstance(commit, str) or COMMIT_RE.fullmatch(commit) is None:
            raise ValueError(f"E4 task-freeze {label} commit is invalid")
    if freezer_commit != recorded_validator:
        raise ValueError("E4 task-freeze freezer/validator commits differ")
    if recorded_validator != executing_validator:
        raise ValueError("E4 task-freeze validator commit differs from executing code")
    if value["freezer_dirty"] is not False:
        raise ValueError("E4 task-freeze provenance records a dirty freezer")
    if value["validator_dirty"] is not False:
        raise ValueError("E4 task-freeze provenance records a dirty validator")
    if value["code_root"] != str(CODE_ROOT):
        raise ValueError("E4 task-freeze code root differs from executing code root")
    if value["evidence_root"] != str(evidence_root):
        raise ValueError("E4 task-freeze evidence root differs from validation root")
    return dict(value)


def _identity(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _canonical_repo_path(
    value: str | Path, *, root: Path, label: str, kind: str
) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be a non-empty canonical POSIX path") from exc
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise ValueError(f"{label} must be a non-empty canonical POSIX path")
    pure = PurePosixPath(raw)
    if raw != pure.as_posix() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{label} contains a dot segment or non-canonical spelling")
    path = Path(raw)
    candidate = path if path.is_absolute() else root / path
    candidate = candidate.absolute()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the repository") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component")
    if kind == "file" and not candidate.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {candidate}")
    if kind == "dir" and not candidate.is_dir():
        raise FileNotFoundError(f"{label} is not a directory: {candidate}")
    if candidate.resolve(strict=True) != candidate:
        raise ValueError(f"{label} is not canonical")
    return candidate


def _fresh_destination(value: str | Path, *, root: Path) -> Path:
    """Validate a fresh repository-local output without requiring an arm parent."""
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise ValueError(
            "task-freeze output must be a non-empty canonical POSIX path"
        ) from exc
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise ValueError("task-freeze output must be a non-empty canonical POSIX path")
    pure = PurePosixPath(raw)
    if raw != pure.as_posix() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("task-freeze output contains a dot segment or non-canonical spelling")
    path = Path(raw)
    destination = path if path.is_absolute() else root / path
    destination = destination.absolute()
    try:
        relative = destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("task-freeze output escapes the repository") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("task-freeze output contains a symlink component")
        if current != destination and current.exists() and not current.is_dir():
            raise ValueError("task-freeze output parent is not a directory")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite task-freeze output: {destination}")
    return destination


def _read_suite(path: Path, *, scene_id: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, Mapping) or set(value) != SUITE_KEYS:
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(f"{label} schema differs: {observed}")
    if value["scene"] not in {scene_id, f"{scene_id}_factory"}:
        raise ValueError(f"{label} has a different scene binding")
    if not isinstance(value["tasks"], list) or not value["tasks"]:
        raise ValueError(f"{label} has no tasks")
    return dict(value)


def _normalize_task(task: Any, *, scene_id: str, label: str) -> dict[str, Any]:
    if not isinstance(task, Mapping):
        raise ValueError(f"{label} task must be an object")
    task = json.loads(json.dumps(task, allow_nan=False))
    task_id = task.get("task_id")
    if not isinstance(task_id, str) or "__" not in task_id:
        raise ValueError(f"{label} has an invalid task_id")
    prefix, suffix = task_id.split("__", 1)
    if prefix not in {scene_id, f"{scene_id}_factory"} or not suffix:
        raise ValueError(f"{label} has a different logical scene binding")
    task["task_id"] = f"{scene_id}__{suffix}"
    target = task.get("target")
    if not isinstance(target, str) or not target:
        raise ValueError(f"{label} task has no target")
    receptacle = task.get("receptacle")
    if receptacle is not None and (not isinstance(receptacle, str) or not receptacle):
        raise ValueError(f"{label} task has an invalid receptacle")
    return task


def _task_map(suite: Mapping[str, Any], *, scene_id: str, label: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for index, task in enumerate(suite["tasks"]):
        normalized = _normalize_task(task, scene_id=scene_id, label=f"{label}[{index}]")
        task_id = normalized["task_id"]
        if task_id in result:
            raise ValueError(f"{label} contains duplicate logical task_id {task_id!r}")
        result[task_id] = normalized
    return result


def _accepted_slots(report: Mapping[str, Any], *, policy: str) -> set[str]:
    roster = report.get("roster")
    if not isinstance(roster, Mapping):
        raise ValueError(f"{policy} materialization roster is invalid")
    object_slots = roster.get("object_slots")
    accepted_slots = roster.get("accepted_slots")
    if (
        not isinstance(object_slots, list)
        or not isinstance(accepted_slots, list)
        or any(not isinstance(slot, str) or not slot for slot in object_slots)
        or any(not isinstance(slot, str) or not slot for slot in accepted_slots)
        or len(object_slots) != len(set(object_slots))
        or len(accepted_slots) != len(set(accepted_slots))
        or not set(accepted_slots) <= set(object_slots)
    ):
        raise ValueError(f"{policy} materialization accepted-slot roster is invalid")
    return set(accepted_slots)


def _factory_report(factory, *, scene_id, policy, root):
    report = validate_materialized_factory(factory, expected_scene_id=scene_id,
        expected_policy_id=policy, repository_root=root)
    if 'automatic_scene_descriptor' not in report:
        return report
    # These are validation views of actual sealed fields, not newly asserted E3 claims.
    manifest = json.loads((Path(factory)/'materialization_manifest.json').read_text())
    if manifest.get('manifest_kind') != 'e4_automatic_construction_variant_materialization':
        raise ValueError('automatic task materialization kind differs')
    digest = lambda value: hashlib.sha256(_json_bytes(value)).hexdigest()
    return {**report, 'automatic': True, 'evidence_root': str(root),
            'e3_claim_status': None, 'e3_code_commit': manifest['e3_code_commit'],
            'e3_freeze_id': manifest['e3_freeze_id'], 'e3_root': manifest['e3_root'],
            'study_scope': manifest['study_scope'],
            'input_identities_sha256': digest(manifest['input_identities']),
            'source_scene_sha256': digest(manifest['source_scene'])}


def _automatic_population_tasks(path, *, root, scene_id, factories, reports):
    """Authenticate the complete declared population; accepted intersection is not coverage."""
    from robo.eval import e4_candidate_screen as screen
    path = _canonical_repo_path(path, root=root, label='automatic population gate', kind='file')
    if path.name != 'gate.json':
        raise ValueError('automatic population must name its sealed gate')
    screen._validate_bundle(path.parent, root=root, expected_kind='e4_automatic_candidate_population')
    gate = json.loads(path.read_text())
    if gate.get('scene_id') != scene_id or gate.get('paper_ready') is not False \
            or gate.get('static_export_state') != 'PASS' or gate.get('qualified_tasks') is not None:
        raise ValueError('automatic population is not a completed static engineering declaration')
    if any(report.get('automatic') is not True for report in reports.values()):
        raise ValueError('automatic all-planned mode cannot consume GT materializations')
    before = screen.automatic_candidate_population(e3_root=gate['e3_root'], scene_id=scene_id,
        automatic_scene_descriptor=materializer.e3.checked_repo_path(gate['descriptor']['path'],
            'automatic population descriptor', kind='file'))
    for key,value in before.items():
        if key != 'pairs' and gate.get(key) != value:
            raise ValueError('automatic population source or denominator changed')
    expected = {}
    for pair in before['pairs']:
        task_id = f"{scene_id}__{pair['target']}_to_"+(pair['receptacle'] or 'region')
        expected[task_id] = (pair['target'], pair['receptacle'])
    if len(expected) != before['counts']['semantic_pairs'] or gate.get('planned_pair_arm_rows') != 2*len(expected):
        raise ValueError('automatic population task denominator differs')
    if len(gate['pairs']) != len(before['pairs']):
        raise ValueError('automatic population dropped semantic pairs')
    for actual,original in zip(gate['pairs'], before['pairs'], strict=True):
        if {key:actual.get(key) for key in original} != original:
            raise ValueError('automatic population pair identity changed')
    context = screen._automatic_export_context(factories, scene_id=scene_id, root=root)
    for policy in ('A0','A4'):
        if reports[policy]['roster'] != context['rosters'][policy] \
                or gate['materializations'][policy]['manifest_sha256'] != reports[policy]['manifest_sha256']:
            raise ValueError('automatic population factory binding differs')
        observed = screen.validate_full_room_export(factories[policy], scene_id=scene_id,
            policy=policy, root=root, expected_object_slots=context['rosters'][policy]['accepted_slots'],
            expected_discovered_slots=before['object_slots'], automatic_factories=factories)
        if observed != gate['exports'][policy]:
            raise ValueError('automatic population static export changed')
    if gate['exports']['A0'].get('shared_static_package') != gate['exports']['A4'].get('shared_static_package'):
        raise ValueError('automatic paired static package identity differs')
    if gate['exports']['A0'].get('static_collision') is None or gate['exports']['A0']['static_collision'] != gate['exports']['A4'].get('static_collision'):
        raise ValueError('automatic paired static collision bytes differ')
    if 'qualification_selection' in gate:
        declared=gate['qualification_selection']
        protocol=_check_path_identity(declared['protocol'],root=root,label='qualification budget protocol')
        replayed=screen.qualification_selection(protocol,before,root=root)
        if declared!=replayed:raise ValueError('sealed qualification selection differs from complete protocol')
        return {task_id:expected[task_id] for task_id in replayed['selected_task_ids']}
    return expected


def _validate_paired_factory_reports(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    evidence_root: Path,
    code_snapshot: Mapping[str, Any],
    prefix: str,
) -> None:
    for policy in ("A0", "A4"):
        report = reports[policy]
        if report.get("code_root") != str(CODE_ROOT):
            raise ValueError(f"{policy} factory validator code root differs")
        if report.get("evidence_root") != str(evidence_root):
            raise ValueError(f"{policy} factory evidence root differs")
        if report.get("validator_commit") != code_snapshot.get("commit"):
            raise ValueError(f"{policy} factory validator commit differs")
        _accepted_slots(report, policy=policy)
    for field in (
        "e3_claim_status",
        "e3_code_commit",
        "e3_freeze_id",
        "e3_producer_commit",
        "e3_root",
        "evidence_root",
        "input_identities_sha256",
        "materializer_commit",
        "source_scene_sha256",
        "study_scope",
    ):
        if reports["A0"].get(field) != reports["A4"].get(field):
            raise ValueError(f"{prefix} differ at {field}")
    if reports["A0"]["roster"]["object_slots"] != reports["A4"]["roster"][
        "object_slots"
    ]:
        raise ValueError(f"{prefix} use different object-job rosters")


def _paired_eligible_ids(
    tasks: Mapping[str, Mapping[str, Mapping[str, Any]]],
    reports: Mapping[str, Mapping[str, Any]],
    automatic_tasks: Mapping[str, tuple[str, str | None]] | None = None,
) -> list[str]:
    """Replay the logical pairing and accepted endpoint intersection."""
    if automatic_tasks is not None:
        for policy in ('A0', 'A4'):
            if set(tasks[policy]) != set(automatic_tasks):
                raise ValueError('automatic task bundle must retain every declared pair')
            planned = set(reports[policy]['roster']['object_slots'])
            for task_id, endpoints in automatic_tasks.items():
                task = tasks[policy][task_id]
                if (task['target'], task.get('receptacle')) != endpoints or any(
                        slot not in planned for slot in endpoints if slot is not None):
                    raise ValueError('automatic task endpoints differ from source population')
        return sorted(automatic_tasks)
    accepted = {
        policy: _accepted_slots(reports[policy], policy=policy)
        for policy in ("A0", "A4")
    }
    eligible: list[str] = []
    for task_id in sorted(set(tasks["A0"]) & set(tasks["A4"])):
        a0_task = tasks["A0"][task_id]
        a4_task = tasks["A4"][task_id]
        a0_endpoints = (a0_task["target"], a0_task.get("receptacle"))
        a4_endpoints = (a4_task["target"], a4_task.get("receptacle"))
        if a0_endpoints != a4_endpoints:
            raise ValueError(
                f"paired task {task_id!r} has different target/receptacle bindings"
            )
        required = {a0_task["target"]}
        if a0_task.get("receptacle") is not None:
            required.add(a0_task["receptacle"])
        if all(required <= accepted[policy] for policy in ("A0", "A4")):
            eligible.append(task_id)
    return eligible


def _validated_max_tasks(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("max_tasks must be positive or null")
    return value


def _shared_suite(
    suite: Mapping[str, Any],
    *,
    scene_id: str,
    tasks: Mapping[str, Mapping[str, Any]],
    logical_task_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "exclude_objects": list(suite["exclude_objects"]),
        "ext_cam": suite["ext_cam"],
        "robot": suite["robot"],
        "scene": scene_id,
        "scene_xml": None,
        "table": suite["table"],
        "tasks": [tasks[task_id] for task_id in logical_task_ids],
        "time_limit_s": suite["time_limit_s"],
    }


def _write(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def _check_identity(path: Path, value: Any, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {"sha256", "size_bytes"}:
        raise ValueError(f"{label} identity schema differs")
    digest = value["sha256"]
    size = value["size_bytes"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(ch not in "0123456789abcdef" for ch in digest)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
    ):
        raise ValueError(f"{label} identity is invalid")
    if path.stat().st_size != size or sha256_file(path) != digest:
        raise ValueError(f"{label} bytes drifted")


def _check_path_identity(
    value: Any, *, root: Path, label: str
) -> Path:
    if not isinstance(value, Mapping) or set(value) != {
        "path", "sha256", "size_bytes"
    }:
        raise ValueError(f"{label} path identity schema differs")
    path = _canonical_repo_path(
        value["path"], root=root, label=label, kind="file"
    )
    _check_identity(
        path,
        {"sha256": value["sha256"], "size_bytes": value["size_bytes"]},
        label,
    )
    return path


def validate_task_bundle(
    manifest_path: str | Path,
    *,
    expected_scene_id: str,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Revalidate a task-freeze bundle and both sealed construction inputs."""
    validator_snapshot = _require_clean_code_snapshot()
    root = _evidence_root(repository_root)
    path = _canonical_repo_path(
        manifest_path, root=root, label="E4 task-freeze manifest", kind="file"
    )
    bundle = path.parent
    seal_path = _canonical_repo_path(
        bundle / "seal.json", root=root, label="E4 task-freeze seal", kind="file"
    )
    seal = json.loads(seal_path.read_text())
    if (
        not isinstance(seal, Mapping)
        or set(seal) != {"manifest_kind", "members", "schema_version"}
        or seal.get("manifest_kind") != "e4_paired_task_freeze"
        or seal.get("schema_version") not in (SCHEMA_VERSION, 2)
        or not isinstance(seal.get("members"), Mapping)
        or set(seal["members"]) != {"manifest.json"}
    ):
        raise ValueError("E4 task-freeze seal schema differs")
    _check_identity(path, seal["members"]["manifest.json"], "E4 task-freeze manifest")
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, Mapping) or set(manifest) != (MANIFEST_KEYS | {"automatic_population"} if manifest.get("schema_version") == 2 else MANIFEST_KEYS):
        raise ValueError("E4 task-freeze manifest schema differs")
    if (
        manifest.get("manifest_kind") != "e4_paired_task_freeze"
        or manifest.get("schema_version") != seal.get("schema_version")
        or manifest.get("scene_id") != expected_scene_id
        or manifest.get("planning_source") not in POLICY_TO_VARIANT
    ):
        raise ValueError("E4 task-freeze manifest binding differs")
    provenance = _validate_provenance(
        manifest["provenance"],
        evidence_root=root,
        validator_snapshot=validator_snapshot,
    )
    max_tasks = _validated_max_tasks(manifest["max_tasks"])
    files = manifest["files"]
    if not isinstance(files, Mapping) or set(files) != {
        "planning_tasks.json", "a0_tasks.json", "a4_tasks.json"
    }:
        raise ValueError("E4 task-freeze file inventory differs")
    actual_names = sorted(child.name for child in bundle.iterdir())
    if actual_names != [
        "a0_tasks.json", "a4_tasks.json", "manifest.json", "planning_tasks.json", "seal.json"
    ]:
        raise ValueError("E4 task-freeze directory has missing or extra files")
    for name, identity in files.items():
        _check_identity(
            _canonical_repo_path(
                bundle / name, root=root, label=f"E4 task-freeze {name}", kind="file"
            ),
            identity,
            f"E4 task-freeze {name}",
        )
    candidate_suites = manifest["candidate_suites"]
    if not isinstance(candidate_suites, Mapping) or set(candidate_suites) != {
        "A0", "A4"
    }:
        raise ValueError("E4 task-freeze candidate suite inventory differs")
    candidate_paths = {
        policy: _check_path_identity(
            identity, root=root, label=f"{policy} candidate task suite"
        )
        for policy, identity in candidate_suites.items()
    }
    candidate_values = {
        policy: _read_suite(
            candidate_paths[policy],
            scene_id=expected_scene_id,
            label=f"{policy} candidate suite",
        )
        for policy in ("A0", "A4")
    }
    candidate_tasks = {
        policy: _task_map(
            candidate_values[policy],
            scene_id=expected_scene_id,
            label=f"{policy} candidate tasks",
        )
        for policy in ("A0", "A4")
    }

    suites = {
        "planning": _read_suite(
            bundle / "planning_tasks.json", scene_id=expected_scene_id, label="planning suite"
        ),
        "A0": _read_suite(
            bundle / "a0_tasks.json", scene_id=expected_scene_id, label="A0 frozen suite"
        ),
        "A4": _read_suite(
            bundle / "a4_tasks.json", scene_id=expected_scene_id, label="A4 frozen suite"
        ),
    }
    if suites["planning"]["scene_xml"] is not None:
        raise ValueError("planning suite scene_xml must be null")
    shared_a0 = {key: value for key, value in suites["A0"].items() if key != "scene_xml"}
    shared_a4 = {key: value for key, value in suites["A4"].items() if key != "scene_xml"}
    shared_planning = {
        key: value for key, value in suites["planning"].items() if key != "scene_xml"
    }
    if shared_a0 != shared_a4 or shared_a0 != shared_planning:
        raise ValueError("E4 frozen task suites drift outside scene_xml")
    logical_ids = [task["task_id"] for task in suites["planning"]["tasks"]]
    if logical_ids != manifest["logical_task_ids"] or logical_ids != sorted(set(logical_ids)):
        raise ValueError("E4 task-freeze logical task roster differs")

    variant_paths = manifest["variant_task_paths"]
    if not isinstance(variant_paths, Mapping) or set(variant_paths) != {"A0", "A4"}:
        raise ValueError("E4 task-freeze variant path mapping differs")
    factories = manifest["factories"]
    if not isinstance(factories, Mapping) or set(factories) != {"A0", "A4"}:
        raise ValueError("E4 task-freeze factory inventory differs")
    reports: dict[str, dict[str, Any]] = {}
    factory_paths: dict[str, str] = {}
    xml_paths: dict[str, str] = {}
    for policy, filename in (("A0", "a0_tasks.json"), ("A4", "a4_tasks.json")):
        declared_suite = _canonical_repo_path(
            variant_paths[policy],
            root=root,
            label=f"{policy} frozen task suite path",
            kind="file",
        )
        if declared_suite != bundle / filename:
            raise ValueError(f"{policy} frozen task suite path differs")
        factory_row = factories.get(policy)
        if not isinstance(factory_row, Mapping) or set(factory_row) != {
            "factory_dir", "materialization_manifest_sha256", "policy_id", "scene_xml"
        }:
            raise ValueError(f"{policy} task-freeze factory schema differs")
        factory = _canonical_repo_path(
            factory_row["factory_dir"],
            root=root,
            label=f"{policy} task-freeze factory",
            kind="dir",
        )
        report = _factory_report(factory, scene_id=expected_scene_id, policy=policy, root=root)
        if (
            report["manifest_sha256"] != factory_row["materialization_manifest_sha256"]
            or factory_row["policy_id"] != policy
        ):
            raise ValueError(f"{policy} task-freeze materialization binding differs")
        xml_identity = factory_row["scene_xml"]
        if not isinstance(xml_identity, Mapping) or set(xml_identity) != {
            "path", "sha256", "size_bytes"
        }:
            raise ValueError(f"{policy} task-freeze scene XML identity differs")
        xml_path = _canonical_repo_path(
            xml_identity["path"], root=root, label=f"{policy} scene XML", kind="file"
        )
        _check_identity(
            xml_path,
            {"sha256": xml_identity["sha256"], "size_bytes": xml_identity["size_bytes"]},
            f"{policy} scene XML",
        )
        candidate_xml = _canonical_repo_path(
            candidate_values[policy]["scene_xml"],
            root=root,
            label=f"{policy} candidate scene XML",
            kind="file",
        )
        if candidate_xml != xml_path:
            raise ValueError(f"{policy} candidate suite scene_xml differs from its factory")
        if suites[policy]["scene_xml"] != str(xml_path):
            raise ValueError(f"{policy} frozen suite scene_xml differs")
        reports[policy] = report
        factory_paths[policy] = str(factory)
        xml_paths[policy] = str(xml_path)
    _validate_paired_factory_reports(
        reports,
        evidence_root=root,
        code_snapshot=validator_snapshot,
        prefix="E4 task-freeze construction inputs",
    )
    if (
        manifest["e3_claim_status"] != reports["A0"]["e3_claim_status"]
        or manifest["e3_code_commit"] != reports["A0"]["e3_code_commit"]
        or manifest["e3_freeze_id"] != reports["A0"]["e3_freeze_id"]
    ):
        raise ValueError("E4 task-freeze E3 provenance binding differs")
    automatic_tasks = None
    if manifest['schema_version'] == 2:
        if max_tasks is not None:
            raise ValueError('automatic population cannot truncate its task denominator')
        population_path = _check_path_identity(manifest['automatic_population'], root=root,
                                               label='automatic population')
        automatic_tasks = _automatic_population_tasks(population_path, root=root,
            scene_id=expected_scene_id, factories={p:Path(v) for p,v in factory_paths.items()}, reports=reports)
    eligible_ids = _paired_eligible_ids(candidate_tasks, reports, automatic_tasks)
    replayed_ids = eligible_ids if max_tasks is None else eligible_ids[:max_tasks]
    if not replayed_ids:
        raise ValueError("no common task is supported by both A0 and A4")
    if logical_ids != replayed_ids:
        raise ValueError("E4 task-freeze planning-source task selection replay differs")
    expected_planning = _shared_suite(
        candidate_values[manifest["planning_source"]],
        scene_id=expected_scene_id,
        tasks=candidate_tasks[manifest["planning_source"]],
        logical_task_ids=replayed_ids,
    )
    if suites["planning"] != expected_planning:
        raise ValueError("E4 task-freeze planning-source replay differs")
    return {
        **({'automatic_population': manifest['automatic_population'],
            'construction_rosters': {p: reports[p]['roster'] for p in ('A0', 'A4')},
            'planned_pair_arm_rows': 2 * len(logical_ids)} if manifest['schema_version'] == 2 else {}),
        "bundle_manifest_sha256": sha256_file(path),
        "code_root": str(CODE_ROOT),
        "e3_claim_status": reports["A0"]["e3_claim_status"],
        "e3_code_commit": reports["A0"]["e3_code_commit"],
        "e3_freeze_id": reports["A0"]["e3_freeze_id"],
        "evidence_root": str(root),
        "factories": factory_paths,
        "freezer_commit": provenance["freezer_commit"],
        "logical_task_ids": logical_ids,
        "max_tasks": max_tasks,
        "planning_source": manifest["planning_source"],
        "planning_tasks": str(bundle / "planning_tasks.json"),
        "provenance": provenance,
        "scene_xml": xml_paths,
        "validator_commit": validator_snapshot["commit"],
        "variant_tasks": {
            "A0": str(bundle / "a0_tasks.json"),
            "A4": str(bundle / "a4_tasks.json"),
        },
    }


def freeze_task_bundle(
    *,
    scene_id: str,
    a0_factory: str | Path,
    a4_factory: str | Path,
    a0_candidates: str | Path,
    a4_candidates: str | Path,
    planning_source: str,
    out: str | Path,
    max_tasks: int | None = None,
    repository_root: str | Path | None = None,
    automatic_population: str | Path | None = None,
) -> dict[str, Any]:
    """Validate inputs and atomically publish one paired task bundle."""
    code_snapshot = _require_clean_code_snapshot()
    if not isinstance(scene_id, str) or not SCENE_ID_RE.fullmatch(scene_id):
        raise ValueError("scene_id must be a ten-character lowercase hex ID")
    if planning_source not in POLICY_TO_VARIANT:
        raise ValueError("planning_source must be A0 or A4")
    max_tasks = _validated_max_tasks(max_tasks)
    root = _evidence_root(repository_root)
    factories = {
        "A0": _canonical_repo_path(a0_factory, root=root, label="A0 factory", kind="dir"),
        "A4": _canonical_repo_path(a4_factory, root=root, label="A4 factory", kind="dir"),
    }
    reports = {
        policy: _factory_report(factory, scene_id=scene_id, policy=policy, root=root)
        for policy, factory in factories.items()
    }
    _validate_paired_factory_reports(
        reports,
        evidence_root=root,
        code_snapshot=code_snapshot,
        prefix="A0/A4 materializations",
    )

    candidate_paths = {
        "A0": _canonical_repo_path(
            a0_candidates, root=root, label="A0 candidate suite", kind="file"
        ),
        "A4": _canonical_repo_path(
            a4_candidates, root=root, label="A4 candidate suite", kind="file"
        ),
    }
    suites = {
        policy: _read_suite(path, scene_id=scene_id, label=f"{policy} candidate suite")
        for policy, path in candidate_paths.items()
    }
    factory_xmls: dict[str, Path] = {}
    for policy in ("A0", "A4"):
        xml = _canonical_repo_path(
            factories[policy] / "sim_export" / "scene.xml",
            root=root,
            label=f"{policy} scene XML",
            kind="file",
        )
        candidate_xml = _canonical_repo_path(
            suites[policy]["scene_xml"],
            root=root,
            label=f"{policy} candidate scene XML",
            kind="file",
        )
        if candidate_xml != xml:
            raise ValueError(f"{policy} candidate suite scene_xml differs from its factory")
        factory_xmls[policy] = xml
    tasks = {
        policy: _task_map(suite, scene_id=scene_id, label=f"{policy} candidate tasks")
        for policy, suite in suites.items()
    }
    automatic_tasks = None
    population_path = None
    if automatic_population is not None:
        if max_tasks is not None:
            raise ValueError('automatic population cannot truncate its task denominator')
        population_path = _canonical_repo_path(automatic_population, root=root,
                                               label='automatic population', kind='file')
        automatic_tasks = _automatic_population_tasks(population_path, root=root,
            scene_id=scene_id, factories=factories, reports=reports)
    eligible_ids = _paired_eligible_ids(tasks, reports, automatic_tasks)
    if max_tasks is not None:
        eligible_ids = eligible_ids[:max_tasks]
    if not eligible_ids:
        raise ValueError("no common task is supported by both A0 and A4")

    shared = _shared_suite(
        suites[planning_source],
        scene_id=scene_id,
        tasks=tasks[planning_source],
        logical_task_ids=eligible_ids,
    )
    destination = _fresh_destination(out, root=root)

    with e3._atomic_directory(destination) as staging:
        identities: dict[str, dict[str, Any]] = {}
        planning_path = staging / "planning_tasks.json"
        identities["planning_tasks.json"] = _write(planning_path, _json_bytes(shared))
        variant_paths: dict[str, str] = {}
        for policy in ("A0", "A4"):
            variant = json.loads(json.dumps(shared))
            variant["scene_xml"] = str(factory_xmls[policy])
            filename = f"{policy.lower()}_tasks.json"
            identities[filename] = _write(staging / filename, _json_bytes(variant))
            variant_paths[policy] = (destination / filename).relative_to(root).as_posix()
        manifest = {
            "candidate_suites": {
                policy: _identity(path, root) for policy, path in candidate_paths.items()
            },
            "e3_claim_status": reports["A0"]["e3_claim_status"],
            "e3_code_commit": reports["A0"]["e3_code_commit"],
            "e3_freeze_id": reports["A0"]["e3_freeze_id"],
            "factories": {
                policy: {
                    "factory_dir": factory.relative_to(root).as_posix(),
                    "materialization_manifest_sha256": reports[policy]["manifest_sha256"],
                    "policy_id": policy,
                    "scene_xml": _identity(factory / "sim_export" / "scene.xml", root),
                }
                for policy, factory in factories.items()
            },
            "files": dict(identities),
            "logical_task_ids": eligible_ids,
            "manifest_kind": "e4_paired_task_freeze",
            "max_tasks": max_tasks,
            "planning_source": planning_source,
            "provenance": _provenance_record(
                code_snapshot=code_snapshot, evidence_root=root
            ),
            "scene_id": scene_id,
            "schema_version": 2 if population_path else SCHEMA_VERSION,
            **({"automatic_population": _identity(population_path, root)} if population_path else {}),
            "variant_task_paths": variant_paths,
        }
        manifest_identity = _write(staging / "manifest.json", _json_bytes(manifest))
        _write(
            staging / "seal.json",
            _json_bytes(
                {
                    "manifest_kind": "e4_paired_task_freeze",
                    "members": {"manifest.json": manifest_identity},
                    "schema_version": 2 if population_path else SCHEMA_VERSION,
                }
            ),
        )
        e3._fsync_directory(staging)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--a0-factory", required=True)
    parser.add_argument("--a4-factory", required=True)
    parser.add_argument("--a0-candidates", required=True)
    parser.add_argument("--a4-candidates", required=True)
    parser.add_argument("--planning-source", choices=("A0", "A4"), required=True)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--automatic-population", help="sealed complete automatic static population gate")
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = freeze_task_bundle(
        scene_id=args.scene_id,
        a0_factory=args.a0_factory,
        a4_factory=args.a4_factory,
        a0_candidates=args.a0_candidates,
        a4_candidates=args.a4_candidates,
        planning_source=args.planning_source,
        out=args.out,
        max_tasks=args.max_tasks,
        automatic_population=args.automatic_population,
    )
    print(
        f"[e4-task-freeze] scene={manifest['scene_id']} "
        f"tasks={len(manifest['logical_task_ids'])} source={manifest['planning_source']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
