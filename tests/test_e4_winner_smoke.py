import json
import hashlib
from pathlib import Path

import pytest

from robo.eval import e4_candidate_screen as screen


def test_cached_winner_copy_preserves_bytes_and_rejects_source_drift(tmp_path):
    source = tmp_path / "source.xml"
    source.write_bytes(b"<mujoco/>\n")
    identity = {"sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "size_bytes": source.stat().st_size}
    output = tmp_path / "fresh.xml"
    recorded = screen._copy_verified_winner_member(
        source, output, identity, root=tmp_path)
    assert output.read_bytes() == source.read_bytes()
    assert recorded["path"] == "source.xml"
    with pytest.raises(FileExistsError):
        screen._copy_verified_winner_member(source, output, identity, root=tmp_path)
    source.write_bytes(b"<mujoco changed='true'/>\n")
    missing_output = tmp_path / "must_not_exist.xml"
    with pytest.raises(screen.CandidateScreenError, match="member drift"):
        screen._copy_verified_winner_member(
            source, missing_output, identity, root=tmp_path)
    assert not missing_output.exists()


def test_empty_frozen_suite_does_not_build_environment(monkeypatch):
    from robo.eval import e4_camera_scorer_gate as camera

    def fail(**kwargs):
        raise AssertionError("empty preparation must not construct a simulator")

    monkeypatch.setattr(camera, "_build_headless_droid_env", fail)
    assert screen._qualify_task_suites(
        scene_id="d755b3d9d8", task_bundle=None,
        factories={}, menagerie_root=Path("unused"),
    ) == []


def test_winner_smoke_refuses_existing_output(tmp_path, monkeypatch):
    monkeypatch.setattr(screen, "_code_snapshot", lambda sha: {"commit": sha})
    monkeypatch.setattr(screen, "evidence_root", lambda: tmp_path)
    (tmp_path / "outputs/icra2027/fresh/harness/winner_smoke").mkdir(parents=True)
    with pytest.raises(screen.CandidateScreenError, match="overwrite"):
        screen.prepare_winner_smoke(
            screen_id="fresh", expected_commit="a" * 40,
            contract_manifest=tmp_path / "missing.json",
        )


@pytest.mark.parametrize("dirty,commit", [(True, "a" * 40), (False, "b" * 40)])
def test_winner_smoke_rejects_dirty_or_drifted_e0(tmp_path, monkeypatch, dirty, commit):
    monkeypatch.setattr(screen, "_code_snapshot", lambda sha: {"commit": sha})
    monkeypatch.setattr(screen, "evidence_root", lambda: tmp_path)
    path = tmp_path / "contract.json"
    path.write_text(json.dumps({"code": {"commit": commit, "dirty": dirty}}))
    with pytest.raises(screen.CandidateScreenError, match="clean exact-commit"):
        screen.prepare_winner_smoke(
            screen_id="fresh", expected_commit="a" * 40, contract_manifest=path,
        )


def test_winner_smoke_rejects_tampered_contract_digest(tmp_path, monkeypatch):
    monkeypatch.setattr(screen, "_code_snapshot", lambda sha: {"commit": sha})
    monkeypatch.setattr(screen, "evidence_root", lambda: tmp_path)
    path = tmp_path / "contract.json"
    path.write_text(json.dumps({"code": {"commit": "a" * 40, "dirty": False},
                                "contract_sha256": "f" * 64}))
    with pytest.raises(screen.CandidateScreenError, match="digest differs"):
        screen.prepare_winner_smoke(
            screen_id="fresh", expected_commit="a" * 40, contract_manifest=path,
        )
