import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "experiments" / "icra_contract_v1.yaml"


def _run(args, cwd=ROOT):
    return subprocess.run(
        [sys.executable, "-m", "agents.eval.validate_contract"] + args,
        cwd=cwd, capture_output=True, text=True,
    )


def test_contract_validates_clean(tmp_path):
    report_path = tmp_path / "contract_report.json"
    proc = _run([str(CONTRACT), "--strict", "--out", str(report_path)])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(report_path.read_text())
    assert report["pass"] is True
    assert report["errors"] == []


def test_missing_gate_fails(tmp_path):
    contract = yaml.safe_load(CONTRACT.read_text())
    del contract["gates"]["G3"]
    mutated = tmp_path / "mutated_contract.yaml"
    mutated.write_text(yaml.safe_dump(contract))
    proc = _run([str(mutated), "--strict", "--out", str(tmp_path / "report.json")])
    assert proc.returncode == 1
    report = json.loads((tmp_path / "report.json").read_text())
    assert any("G3" in e for e in report["errors"])


def test_undeclared_protocol_in_column_map_fails(tmp_path):
    contract = yaml.safe_load(CONTRACT.read_text())
    contract["table_column_metric_map"]["bogus_column"] = "protocol_that_does_not_exist"
    mutated = tmp_path / "mutated_contract.yaml"
    mutated.write_text(yaml.safe_dump(contract))
    proc = _run([str(mutated), "--strict", "--out", str(tmp_path / "report.json")])
    assert proc.returncode == 1


def test_mutated_frozen_field_fails_after_lock_exists(tmp_path, monkeypatch):
    # Run once against the real repo root so the lock file exists with the
    # real frozen_fields hash (idempotent if it already exists).
    _run([str(CONTRACT), "--out", str(tmp_path / "report0.json")])

    # Now mutate a copy of the repo's frozen_fields.yaml file itself and
    # re-point the module's paths at a scratch copy to simulate drift
    # without touching the real frozen fields.
    scratch_root = tmp_path / "scratch_repo"
    (scratch_root / "configs" / "experiments").mkdir(parents=True)
    real_frozen = (ROOT / "configs" / "experiments" / "frozen_fields.yaml").read_text()
    frozen = yaml.safe_load(real_frozen)
    frozen["control"]["rate_hz"] = 999  # mutate a frozen camera/control field
    (scratch_root / "configs" / "experiments" / "frozen_fields.yaml").write_text(yaml.safe_dump(frozen))
    # seed a stale lock recorded against the ORIGINAL content
    import hashlib
    orig_hash = hashlib.sha256(
        json.dumps(yaml.safe_load(real_frozen), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (scratch_root / "configs" / "experiments" / ".frozen_fields.lock.json").write_text(
        json.dumps({"hash": orig_hash, "reason": "seed"})
    )

    sys.path.insert(0, str(ROOT))
    import importlib
    import agents.eval.validate_contract as vc
    importlib.reload(vc)
    monkeypatch.setattr(vc, "ROOT", scratch_root)
    monkeypatch.setattr(vc, "FROZEN_FIELDS_PATH", scratch_root / "configs" / "experiments" / "frozen_fields.yaml")
    monkeypatch.setattr(vc, "LOCK_PATH", scratch_root / "configs" / "experiments" / ".frozen_fields.lock.json")
    errors = vc.check_frozen_fields(bump=False, reason=None)
    assert any("changed since last validated run" in e for e in errors)
    importlib.reload(vc)  # restore module-level constants for any later test in this session
