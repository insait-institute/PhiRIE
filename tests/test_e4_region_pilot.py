import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from robo.eval import e4_region_pilot as pilot


def _target_slots(scene_id: str) -> list[str]:
    return [
        task_id.split("__", 1)[1].rsplit("_to_region", 1)[0]
        for task_id in pilot.expected_task_ids(scene_id)
    ]


def _write_export(
    root: Path,
    scene_id: str,
    policy: str,
    *,
    accepted_slots: list[str] | None = None,
    xml_body_slots: list[str] | None = None,
) -> Path:
    factory = root / "outputs" / policy / f"{scene_id}_factory"
    sim = factory / "sim"
    export = factory / "sim_export"
    sim.mkdir(parents=True)
    export.mkdir(parents=True)
    (sim / "background.obj").write_text("o background\nv 0 0 0\n")
    target_slots = _target_slots(scene_id)
    accepted_slots = accepted_slots or target_slots
    xml_body_slots = accepted_slots if xml_body_slots is None else xml_body_slots
    bodies = "".join(f'<body name="{slot}"/>' for slot in xml_body_slots)
    meshes = []
    for slot in accepted_slots:
        mesh = factory / "objects" / slot / "mesh_sim.obj"
        mesh.parent.mkdir(parents=True)
        mesh.write_text(f"o {slot}\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        meshes.append(
            f'<mesh name="{slot}_vis" file="objects/{slot}/mesh_sim.obj"/>'
        )
    xml = export / "scene.xml"
    xml.write_text(
        f'<mujoco><compiler meshdir="{factory}"/><asset>{"".join(meshes)}</asset>'
        f"<worldbody>{bodies}</worldbody></mujoco>"
    )
    (export / "room_collision_report.json").write_text(json.dumps({
        "mode": "room",
        "benchmark": {
            "finite": True,
            "steps": 1000,
            "state_hash": "a" * 64,
        },
    }))
    drifts = {slot: 0.01 + index * 0.001 for index, slot in enumerate(target_slots)}
    (export / "mujoco_settle.json").write_text(json.dumps({
        "drift_m": drifts,
        "stable_3cm": len(drifts),
        "n": len(drifts),
    }))
    tasks = []
    for slot in target_slots:
        tasks.append({
            "task_id": f"{scene_id}_factory__{slot}_to_region",
            "target": slot,
            "receptacle": None,
            "region": {"cx": 0.0, "cy": 0.0, "hx": 0.2, "hy": 0.2,
                       "zlo": 0.0, "zhi": 1.0},
        })
    (export / "pi05_tasks.json").write_text(json.dumps({
        "scene": f"{scene_id}_factory",
        "scene_xml": str(xml),
        "tasks": tasks,
    }))
    (export / "isaac_manifest.json").write_text(json.dumps({
        "objects": [{"name": slot} for slot in accepted_slots],
    }))
    return factory


@pytest.mark.parametrize("scene_id", pilot.SCENE_IDS)
def test_export_gate_accepts_exact_region_targets(monkeypatch, tmp_path, scene_id):
    factory = _write_export(tmp_path, scene_id, "A4")
    monkeypatch.setattr(
        pilot,
        "_compile_scene_xml",
        lambda _path, target_slots: {
            "compiled": True,
            "nbody": len(target_slots) + 1,
            "ngeom": 1,
            "target_body_ids": {slot: i + 1 for i, slot in enumerate(target_slots)},
        },
    )

    report = pilot.validate_export_outputs(
        factory,
        scene_id=scene_id,
        policy="A4",
        root=tmp_path,
        expected_object_slots=_target_slots(scene_id),
    )

    assert report["collision_mode"] == "room"
    assert report["expected_task_ids"] == list(pilot.expected_task_ids(scene_id))
    assert max(report["target_drift_m"].values()) < 0.03
    assert report["compile_check"]["compiled"] is True
    assert set(report["generated_trees"]) == {"sim", "sim_export"}


def test_export_gate_rejects_target_at_exact_30mm(monkeypatch, tmp_path):
    scene_id = "b0a08200c9"
    factory = _write_export(tmp_path, scene_id, "A0")
    settle_path = factory / "sim_export" / "mujoco_settle.json"
    settle = json.loads(settle_path.read_text())
    target = "obj_00"
    settle["drift_m"][target] = 0.03
    settle["stable_3cm"] = sum(v < 0.03 for v in settle["drift_m"].values())
    settle_path.write_text(json.dumps(settle))
    monkeypatch.setattr(pilot, "_compile_scene_xml", lambda *_args, **_kwargs: {})

    with pytest.raises(pilot.PilotGateError, match="not <30 mm"):
        pilot.validate_export_outputs(
            factory,
            scene_id=scene_id,
            policy="A0",
            root=tmp_path,
            expected_object_slots=_target_slots(scene_id),
        )


def test_export_gate_rejects_shim_mode_before_compile(monkeypatch, tmp_path):
    scene_id = "825d228aec"
    factory = _write_export(tmp_path, scene_id, "A0")
    collision_path = factory / "sim_export" / "room_collision_report.json"
    collision = json.loads(collision_path.read_text())
    collision["mode"] = "shim"
    collision_path.write_text(json.dumps(collision))
    monkeypatch.setattr(
        pilot,
        "_compile_scene_xml",
        lambda *_args, **_kwargs: pytest.fail("compile should not be reached"),
    )

    with pytest.raises(pilot.PilotGateError, match="mode=room"):
        pilot.validate_export_outputs(
            factory,
            scene_id=scene_id,
            policy="A0",
            root=tmp_path,
            expected_object_slots=_target_slots(scene_id),
        )


def test_export_gate_rejects_receptacle_task(monkeypatch, tmp_path):
    scene_id = "b0a08200c9"
    factory = _write_export(tmp_path, scene_id, "A4")
    tasks_path = factory / "sim_export" / "pi05_tasks.json"
    suite = json.loads(tasks_path.read_text())
    suite["tasks"][0]["receptacle"] = "obj_99"
    tasks_path.write_text(json.dumps(suite))
    monkeypatch.setattr(pilot, "_compile_scene_xml", lambda *_args, **_kwargs: {})

    with pytest.raises(pilot.PilotGateError, match="not region-only"):
        pilot.validate_export_outputs(
            factory,
            scene_id=scene_id,
            policy="A4",
            root=tmp_path,
            expected_object_slots=_target_slots(scene_id),
        )


def test_export_gate_rejects_omitted_non_target_accepted_body(monkeypatch, tmp_path):
    scene_id = "b0a08200c9"
    accepted = [*_target_slots(scene_id), "obj_04"]
    factory = _write_export(
        tmp_path,
        scene_id,
        "A0",
        accepted_slots=accepted,
        xml_body_slots=_target_slots(scene_id),
    )
    monkeypatch.setattr(pilot, "_compile_scene_xml", lambda *_args, **_kwargs: {})

    with pytest.raises(pilot.PilotGateError, match="object-body roster differs"):
        pilot.validate_export_outputs(
            factory,
            scene_id=scene_id,
            policy="A0",
            root=tmp_path,
            expected_object_slots=accepted,
        )


def test_export_gate_rejects_omitted_non_target_settle_body(monkeypatch, tmp_path):
    scene_id = "b0a08200c9"
    accepted = [*_target_slots(scene_id), "obj_04"]
    factory = _write_export(tmp_path, scene_id, "A0", accepted_slots=accepted)
    settle_path = factory / "sim_export/mujoco_settle.json"
    settle = json.loads(settle_path.read_text())
    # The fixture's task targets are present, but the accepted non-target body
    # is absent.  A target-subset-only gate would incorrectly accept this.
    settle["n"] = len(settle["drift_m"])
    settle["stable_3cm"] = len(settle["drift_m"])
    settle_path.write_text(json.dumps(settle))
    monkeypatch.setattr(pilot, "_compile_scene_xml", lambda *_args, **_kwargs: {})

    with pytest.raises(pilot.PilotGateError, match="settle body roster differs"):
        pilot.validate_export_outputs(
            factory,
            scene_id=scene_id,
            policy="A0",
            root=tmp_path,
            expected_object_slots=accepted,
        )


def test_recorded_export_rejects_referenced_mesh_tamper(monkeypatch, tmp_path):
    scene_id = "825d228aec"
    slots = _target_slots(scene_id)
    factory = _write_export(tmp_path, scene_id, "A4", accepted_slots=slots)
    monkeypatch.setattr(
        pilot,
        "_compile_scene_xml",
        lambda _path, target_slots: {
            "compiled": True,
            "target_body_ids": {slot: i + 1 for i, slot in enumerate(target_slots)},
        },
    )
    recorded = pilot.validate_export_outputs(
        factory,
        scene_id=scene_id,
        policy="A4",
        root=tmp_path,
        expected_object_slots=slots,
    )
    (factory / "objects" / slots[0] / "mesh_sim.obj").write_text("tampered\n")

    with pytest.raises(pilot.PilotGateError, match="changed after scene gate"):
        pilot.validate_export_against_recorded(
            factory,
            scene_id=scene_id,
            policy="A4",
            root=tmp_path,
            expected_object_slots=slots,
            recorded=recorded,
        )


def test_recorded_export_rejects_unreferenced_generated_file_tamper(
    monkeypatch, tmp_path
):
    scene_id = "b0a08200c9"
    slots = _target_slots(scene_id)
    factory = _write_export(tmp_path, scene_id, "A0", accepted_slots=slots)
    unreferenced = factory / "sim" / "generator_notes.json"
    unreferenced.write_text('{"version": 1}\n')
    monkeypatch.setattr(pilot, "_compile_scene_xml", lambda *_args, **_kwargs: {})
    recorded = pilot.validate_export_outputs(
        factory,
        scene_id=scene_id,
        policy="A0",
        root=tmp_path,
        expected_object_slots=slots,
    )
    unreferenced.write_text('{"version": 2}\n')

    with pytest.raises(pilot.PilotGateError, match="changed after scene gate"):
        pilot.validate_export_against_recorded(
            factory,
            scene_id=scene_id,
            policy="A0",
            root=tmp_path,
            expected_object_slots=slots,
            recorded=recorded,
        )


def test_export_gate_rejects_symlink_in_generated_tree(monkeypatch, tmp_path):
    scene_id = "825d228aec"
    slots = _target_slots(scene_id)
    factory = _write_export(tmp_path, scene_id, "A4", accepted_slots=slots)
    (factory / "sim" / "host-link").symlink_to("/etc/hosts")
    monkeypatch.setattr(pilot, "_compile_scene_xml", lambda *_args, **_kwargs: {})

    with pytest.raises(pilot.PilotGateError, match="contains a symlink"):
        pilot.validate_export_outputs(
            factory,
            scene_id=scene_id,
            policy="A4",
            root=tmp_path,
            expected_object_slots=slots,
        )


def test_submitter_is_seven_ordinary_cpu_jobs_without_gpu_or_array():
    source = (
        pilot.CODE_ROOT / "run" / "icra2027" / "submit_e4_region_pilot_noarray.sh"
    ).read_text()
    calls = [
        line for line in source.splitlines()
        if line.startswith("submit_job ")
    ]
    assert len(calls) == 7
    assert sum(line.startswith("submit_job materialize ") for line in calls) == 4
    assert sum(line.startswith("submit_job prepare ") for line in calls) == 2
    assert sum(line.startswith("submit_job final ") for line in calls) == 1
    assert "--no-requeue" in source
    assert "ACCOUNT=${SLURM_ACCOUNT:-phirie}" in source
    assert "SUBMIT_USER=$(id -un)" in source
    assert "--dependency=\"$dependency\"" in source
    assert "--kill-on-invalid-dep=yes" in source
    assert 'local export_spec="E4_FREEZE_ID=' in source
    assert 'local export_spec="ALL,' not in source
    assert "--array" not in source
    assert "--gpus" not in source
    assert "--gres" not in source
    assert "sof1-cpu" in source
    assert "sbatch_receipts" in source
    assert 'sync -f "$LEDGER_TEMP"' in source


def test_launcher_checks_clean_exact_commit_for_every_phase():
    source = (
        pilot.CODE_ROOT / "run" / "slurm" / "icra2027_e4_region_cpu.sbatch"
    ).read_text()
    assert source.index("observed_commit=$(git") < source.index("case \"$phase\"")
    assert '"$observed_commit" == "$E4_CODE_COMMIT"' in source
    assert "status --porcelain --untracked-files=normal" in source
    assert "SIMANY_EVIDENCE_ROOT=\"$EVIDENCE_ROOT\"" in source
    assert 'export PYTHONPATH="$CODE_ROOT"' in source
    assert 'cd "$CODE_ROOT"' in source
    assert "wrong E4 import origin" in source
    assert '"${SLURM_JOB_ACCOUNT:-}" == "$ACCOUNT"' in source
    assert "Requeue=0" in source
    assert "MinMemoryNode=" in source
    assert "Command=$CODE_ROOT/run/slurm/icra2027_e4_region_cpu.sbatch" in source
    prepare_block = source.rsplit("  prepare-scene)", 1)[1].split("    ;;", 1)[0]
    assert prepare_block.count('--freeze-id "$E4_FREEZE_ID"') == 1
    assert "--array" not in source
    assert "--gpus" not in source
    assert "--gres" not in source


def test_inside_rejects_parent_traversal(tmp_path):
    with pytest.raises(pilot.PilotGateError, match="dot segment"):
        pilot._inside(
            Path("outputs") / ".." / "outside.json",
            root=tmp_path,
            label="hostile path",
        )


def test_inside_rejects_symlink_component(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    with pytest.raises(pilot.PilotGateError, match="symlink component"):
        pilot._inside(link / "artifact.json", root=tmp_path, label="hostile path")


def test_external_geometry_identity_detects_source_tamper(monkeypatch, tmp_path):
    scene_id = "b0a08200c9"
    scans = tmp_path / "data" / scene_id / "scans"
    scans.mkdir(parents=True)
    for name in ("mesh_aligned_0.05.ply", "segments.json", "segments_anno.json"):
        (scans / name).write_text(name + "\n")
    colmap = tmp_path / "data" / scene_id / "dslr" / "colmap" / "images.txt"
    transforms = (
        tmp_path
        / "data"
        / scene_id
        / "dslr"
        / "nerfstudio"
        / "transforms_undistorted.json"
    )
    colmap.parent.mkdir(parents=True)
    transforms.parent.mkdir(parents=True)
    colmap.write_text("# camera poses\n")
    transforms.write_text('{"frames": []}\n')
    monkeypatch.setattr(pilot, "SCANNETPP_ROOT", tmp_path)

    recorded = pilot.external_geometry_identities(scene_id)
    assert set(recorded) == {
        "mesh",
        "segments",
        "segments_anno",
        "camera_colmap_images",
        "camera_transforms_undistorted",
    }
    transforms.write_text('{"frames": [{"tampered": true}]}\n')

    assert pilot.external_geometry_identities(scene_id) != recorded


def test_recorded_scene_gate_identity_rejects_byte_tamper(tmp_path):
    gate = tmp_path / "outputs" / "scene_gates" / "b0a08200c9.json"
    gate.parent.mkdir(parents=True)
    gate.write_text('{"status": "validated"}\n')
    recorded = pilot._identity(gate, root=tmp_path)
    gate.write_text('{"status": "tampered"}\n')

    with pytest.raises(pilot.PilotGateError, match="scene gate identity changed"):
        pilot._validate_recorded_identity(
            gate,
            root=tmp_path,
            recorded=recorded,
            label="scene gate",
        )


def test_final_gate_chains_both_scene_gate_identities():
    import inspect

    source = inspect.getsource(pilot.finalize)
    assert '"scene_gate_identities": scene_gate_identities' in source
    assert "for scene_id in SCENE_IDS" in source


def test_harness_config_pins_planning_tasks_to_split_evidence_root(
    monkeypatch, tmp_path
):
    freeze_id = "split-root-pilot"
    bundle_reports = {}
    for scene_id in pilot.SCENE_IDS:
        bundle = (
            tmp_path / "outputs" / "icra2027" / freeze_id / "task_freezes" / scene_id
        )
        bundle.mkdir(parents=True)
        (bundle / "manifest.json").write_text("{}\n")
        planning = bundle / "planning_tasks.json"
        planning.write_text(
            json.dumps(
                {
                    "scene": scene_id,
                    "tasks": [
                        {"task_id": task_id}
                        for task_id in pilot.expected_task_ids(scene_id)
                    ],
                }
            )
        )
        factories = {}
        variant_tasks = {}
        scene_xml = {}
        for policy in pilot.POLICIES:
            factory = tmp_path / "factories" / policy / scene_id
            factory.mkdir(parents=True)
            tasks = bundle / f"{policy.lower()}_tasks.json"
            tasks.write_text("{}\n")
            xml = factory / "scene.xml"
            xml.write_text("<mujoco/>\n")
            factories[policy] = str(factory)
            variant_tasks[policy] = str(tasks)
            scene_xml[policy] = str(xml)
        bundle_reports[scene_id] = {
            "factories": factories,
            "planning_tasks": str(planning),
            "scene_xml": scene_xml,
            "variant_tasks": variant_tasks,
        }

    config = pilot._harness_config(bundle_reports, root=tmp_path, freeze_id=freeze_id)
    for scene in config["scenes"]:
        tasks_path = Path(scene["tasks_json"])
        assert tasks_path.is_absolute()
        assert tmp_path in tasks_path.parents

    # The legacy reset planner uses its module ROOT for relative paths.  The
    # sealed absolute paths must therefore replay even when code and evidence
    # live under disjoint roots.
    from robo.eval import paired_runner

    monkeypatch.setattr(paired_runner, "ROOT", tmp_path / "unrelated-code-root")
    states = paired_runner.plan_reset_states(
        config["scenes"], config["seeds"], config["episodes"]
    )
    assert {state.reset_state_id for state in states} == set(
        config["contract"]["reset_ids"]
    )


def _write_executable(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    path.chmod(0o755)


def _init_clean_repo(path: Path) -> str:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=E4 Test",
            "-c",
            "user.email=e4-test@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _with_fixture_roots(source: str, code_root: Path, evidence_root: Path) -> str:
    # Read the launcher's pinned roots, not this importing worktree's path.
    # Sentinels also update embedded Python checks without prefix collisions.
    assignments = dict(line.split("=", 1) for line in source.splitlines()
                       if line.startswith(("CODE_ROOT=", "EVIDENCE_ROOT=")))
    return (source.replace(assignments["CODE_ROOT"], "__FIXTURE_CODE__")
            .replace(assignments["EVIDENCE_ROOT"], "__FIXTURE_EVIDENCE__")
            .replace("__FIXTURE_CODE__", str(code_root))
            .replace("__FIXTURE_EVIDENCE__", str(evidence_root)))


def test_submitter_fake_sbatch_builds_exact_4_to_2_to_1_ledger(tmp_path):
    submit_user = subprocess.check_output(["id", "-un"], text=True).strip()
    code_root = tmp_path / "code"
    evidence_root = tmp_path / "evidence"
    launcher = code_root / "run" / "slurm" / "icra2027_e4_region_cpu.sbatch"
    _write_executable(launcher, "#!/usr/bin/env bash\nexit 0\n")
    _init_clean_repo(code_root)
    e3_root = (
        evidence_root
        / "outputs/icra2027/icra2027-contract-v1-e3-48fa807844ef-prelim-full-hala-r2/agentic"
    )
    e3_root.mkdir(parents=True)

    calls_path = tmp_path / "sbatch.calls"
    state_path = tmp_path / "sbatch.state"
    fake_sbatch = tmp_path / "bin" / "sbatch"
    _write_executable(
        fake_sbatch,
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"calls={shlex.quote(str(calls_path))}\n"
        f"state={shlex.quote(str(state_path))}\n"
        "if [[ -f \"$state\" ]]; then read -r job_id < \"$state\"; "
        "else job_id=9101; fi\n"
        "printf '%q ' \"$@\" >> \"$calls\"\n"
        "printf '\\n' >> \"$calls\"\n"
        "printf '%d\\n' \"$((job_id + 1))\" > \"$state\"\n"
        "printf '%d\\n' \"$job_id\"\n",
    )
    fake_sacctmgr = tmp_path / "bin" / "sacctmgr"
    _write_executable(
        fake_sacctmgr,
        "#!/usr/bin/env bash\n"
        f"printf 'phirie|{submit_user}||normal\\n'\n",
    )

    source = (
        pilot.CODE_ROOT / "run/icra2027/submit_e4_region_pilot_noarray.sh"
    ).read_text()
    copied_submitter = tmp_path / "submit.sh"
    copied_submitter.write_text(_with_fixture_roots(source, code_root, evidence_root))
    freeze_id = "e4-fake-noarray"
    environment = dict(os.environ)
    for name in (
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_ARRAY_TASK_COUNT",
        "SLURM_ARRAY_TASK_MIN",
        "SLURM_ARRAY_TASK_MAX",
        "SLURM_ARRAY_TASK_STEP",
        "SBATCH_ARRAY_INX",
        "SBATCH_ARRAY",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "E4_E3_ROOT": (
                "outputs/icra2027/"
                "icra2027-contract-v1-e3-48fa807844ef-prelim-full-hala-r2/agentic"
            ),
            "E4_FREEZE_ID": freeze_id,
            "E4_SUBMIT_USER": submit_user,
            "E4_TEST_TOOL_OVERRIDES": "1",
            "SBATCH_BIN": str(fake_sbatch),
            "SACCTMGR_BIN": str(fake_sacctmgr),
        }
    )
    completed = subprocess.run(
        ["bash", str(copied_submitter)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    calls = [shlex.split(line) for line in calls_path.read_text().splitlines()]
    assert len(calls) == 7
    assert [
        next((arg for arg in call if arg.startswith("--dependency=")), None)
        for call in calls
    ] == [
        None,
        None,
        None,
        None,
        "--dependency=afterok:9101:9102",
        "--dependency=afterok:9103:9104",
        "--dependency=afterok:9105:9106",
    ]
    for call in calls:
        assert "--no-requeue" in call
        assert "--account=phirie" in call
        assert not any(arg.startswith("--array") for arg in call)
        assert not any(arg.startswith("--gpus") for arg in call)
        assert not any(arg.startswith("--gres") for arg in call)
        export_arg = next(arg for arg in call if arg.startswith("--export="))
        assert export_arg.startswith("--export=E4_FREEZE_ID=")
        assert f"E4_SUBMIT_USER={submit_user}" in export_arg
        assert "ALL," not in export_arg
    assert all("--kill-on-invalid-dep=yes" not in call for call in calls[:4])
    assert all("--kill-on-invalid-dep=yes" in call for call in calls[4:])

    ledger_path = (
        evidence_root
        / "outputs/icra2027/submissions"
        / f"{freeze_id}-cpu-prereq/jobs.tsv"
    )
    with ledger_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 7
    assert [row["job_id"] for row in rows] == [str(value) for value in range(9101, 9108)]
    assert [row["stage"] for row in rows] == [
        "materialize", "materialize", "materialize", "materialize",
        "prepare", "prepare", "final",
    ]
    assert [row["dependency"] for row in rows[4:]] == [
        "afterok:9101:9102",
        "afterok:9103:9104",
        "afterok:9105:9106",
    ]
    assert [row["kill_on_invalid_dep"] for row in rows] == [
        "no", "no", "no", "no", "yes", "yes", "yes",
    ]
    assert all(row["profile"] == "sof1-cpu" and row["gres"] == "none" for row in rows)
    assert all(row["account"] == "phirie" for row in rows)
    assert all(row["submit_user"] == submit_user for row in rows)
    assert all(row["code_commit"] == rows[0]["code_commit"] for row in rows)
    for index, row in enumerate(rows, start=9101):
        receipt = ledger_path.parent / row["sbatch_receipt"]
        assert receipt.read_text() == f"{index}\n"
        assert row["sbatch_receipt_sha256"] == pilot._sha256(receipt)
        assert receipt.stat().st_mode & 0o777 == 0o444


def test_launcher_sanitizes_env_and_imports_only_fixture_code(tmp_path):
    submit_user = subprocess.check_output(["id", "-un"], text=True).strip()
    code_root = tmp_path / "isolated-code"
    evidence_root = tmp_path / "evidence"
    capture_path = tmp_path / "materializer-capture.json"
    package = code_root / "robo" / "eval"
    package.mkdir(parents=True)
    (code_root / "robo/__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "e4_region_pilot.py").write_text(
        "from pathlib import Path\nCODE_ROOT = Path(__file__).resolve().parents[2]\n"
    )
    (package / "e3_factory_materializer.py").write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "CODE_ROOT = Path(__file__).resolve().parents[2]\n"
        "REPOSITORY_ROOT = Path(os.environ.get('SIMANY_EVIDENCE_ROOT', CODE_ROOT))\n"
        "if __name__ == '__main__':\n"
        "    payload = {'cwd': os.getcwd(), 'module_file': str(Path(__file__).resolve()), "
        "'argv': sys.argv[1:], 'environment': dict(os.environ), "
        "'optimize': sys.flags.optimize}\n"
        f"    Path({str(capture_path)!r}).write_text(json.dumps(payload, sort_keys=True))\n"
    )
    launcher_source = (
        pilot.CODE_ROOT / "run/slurm/icra2027_e4_region_cpu.sbatch"
    ).read_text()
    fake_bin = tmp_path / "bin"
    launcher_source = _with_fixture_roots(
        launcher_source, code_root, evidence_root
    ).replace(
        "export PATH=/usr/local/bin:/usr/bin:/bin",
        f"export PATH={fake_bin}:/usr/local/bin:/usr/bin:/bin",
    )
    launcher = code_root / "run/slurm/icra2027_e4_region_cpu.sbatch"
    _write_executable(launcher, launcher_source)
    code_commit = _init_clean_repo(code_root)

    python_link = evidence_root / ".venv/bin/python"
    python_link.parent.mkdir(parents=True)
    python_link.symlink_to(Path(sys.executable).resolve())
    e3_relative = (
        "outputs/icra2027/"
        "icra2027-contract-v1-e3-48fa807844ef-prelim-full-hala-r2/agentic"
    )
    (evidence_root / e3_relative).mkdir(parents=True)
    _write_executable(
        fake_bin / "scontrol",
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' 'JobId=77 JobName=e4 UserId={submit_user}(1000) "
        "Account=phirie QOS=normal Requeue=0 TimeLimit=01:00:00 "
        "Partition=batch NumNodes=1 NumCPUs=8 NumTasks=1 CPUs/Task=8 "
        "ReqTRES=cpu=8,mem=64G,node=1 MinMemoryNode=64G "
        f"Command={launcher} WorkDir={code_root}'\n",
    )

    environment = dict(os.environ)
    for name in (
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_ARRAY_TASK_COUNT",
        "SLURM_ARRAY_TASK_MIN",
        "SLURM_ARRAY_TASK_MAX",
        "SLURM_ARRAY_TASK_STEP",
        "SBATCH_ARRAY_INX",
        "SBATCH_ARRAY",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "SLURM_JOB_GPUS",
        "SLURM_STEP_GPUS",
        "SLURM_GPUS",
        "SLURM_GPUS_ON_NODE",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "E4_CODE_COMMIT": code_commit,
            "E4_E3_ROOT": e3_relative,
            "E4_FREEZE_ID": "e4-launcher-fixture",
            "E4_SUBMIT_USER": submit_user,
            "CUDA_VISIBLE_DEVICES": "NoDevFiles",
            "PYTHONOPTIMIZE": "2",
            "PYTHONPATH": "/hostile/main-checkout",
            "PYTHONUSERBASE": "/hostile/userbase",
            "SIMANY_AUTO": "1",
            "SIMANY_MESH_SRC": "derived",
            "SIMF_SCENE": "wrong-scene",
            "SLURMD_NODENAME": "sof1-h200-0",
            "SLURM_CPUS_PER_TASK": "8",
            "SLURM_JOB_ACCOUNT": "phirie",
            "SLURM_JOB_ID": "77",
            "SLURM_JOB_PARTITION": "batch",
            "SLURM_JOB_QOS": "normal",
            "SLURM_JOB_USER": submit_user,
            "SLURM_MEM_PER_NODE": "65536",
            "SLURM_NNODES": "1",
            "SLURM_NTASKS": "1",
        }
    )
    completed = subprocess.run(
        ["bash", str(launcher), "materialize", "b0a08200c9", "A0"],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    capture = json.loads(capture_path.read_text())
    assert capture["cwd"] == str(code_root)
    assert Path(capture["module_file"]) == package / "e3_factory_materializer.py"
    assert capture["optimize"] == 0
    env = capture["environment"]
    assert env["PYTHONPATH"] == str(code_root)
    assert env["SIMANY_ROOT"] == str(code_root)
    assert env["SIMANY_EVIDENCE_ROOT"] == str(evidence_root)
    assert env["SIMANY_SCANNETPP_ROOT"] == "/data/ScanNetpp"
    assert env["SIMANY_SPLATS_ROOT"] == "/data/ScanNetppv2_gsplat/splats"
    assert "SIMANY_SCENE" not in env  # scene is an explicit materializer CLI argument
    for forbidden in (
        "PYTHONOPTIMIZE",
        "PYTHONUSERBASE",
        "SIMANY_AUTO",
        "SIMANY_MESH_SRC",
        "SIMF_SCENE",
        "CUDA_VISIBLE_DEVICES",
    ):
        assert forbidden not in env
