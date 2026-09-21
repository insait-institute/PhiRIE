"""Task 15 acceptance test: baselines/release_status.yaml is honest.

Checks three things the paper's construction-baseline table depends on:
1. release_status.yaml parses and every row carries all required fields
   with a `status` from the declared enum.
2. Every baseline plan/15_CONSTRUCTION_BASELINES.md names (plus the two
   already-implemented rows agents/baselines/README.md documents) has a
   row -- nothing gets silently dropped.
3. Any baseline marked `unavailable` or `literature_only` has NO
   numeric-producing adapter file under baselines/ that could be
   accidentally invoked to fabricate a result: the two declared stub
   modules (SimRecon, ReplicateAnyScene) must exist and every public
   callable in them must raise NotImplementedError unconditionally; every
   other unreleased baseline (RoboSnap, PolaRiS-manual-construction) must
   have no file under baselines/ whose name plausibly matches it at all.
"""
from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = ROOT / "baselines"
STATUS_PATH = BASELINES_DIR / "release_status.yaml"

ALLOWED_STATUS = {"implemented", "partial", "literature_only", "unavailable"}
REQUIRED_FIELDS = {"name", "status", "commit_or_version", "license",
                    "release_date", "notes"}

# plan/15_CONSTRUCTION_BASELINES.md's baseline list, plus the two rows
# already implemented in agents/baselines/ (per its README) and this task's
# own new raw_reconstruction.py.
EXPECTED_BASELINE_NAMES = {
    "MaskClustering",
    "FlashSplat",
    "SimFoundry-reproduction",
    "raw_reconstruction",
    "SimRecon",
    "ReplicateAnyScene",
    "RoboSnap",
    "PolaRiS-manual-construction",
}

# Explicit, audited mapping: baseline name -> the stub module that is
# EXPECTED to exist for it. Any unreleased baseline NOT in this mapping
# (RoboSnap, PolaRiS-manual-construction) must instead have no file at all
# under baselines/ whose name plausibly matches, checked via
# FORBIDDEN_FILENAME_FRAGMENTS below.
KNOWN_STUB_MODULES = {
    "SimRecon": "simrecon_adapter",
    "ReplicateAnyScene": "replicate_adapter",
}

FORBIDDEN_FILENAME_FRAGMENTS = {
    "RoboSnap": ["robosnap"],
    "PolaRiS-manual-construction": ["polaris"],
}


def _load_status() -> dict:
    assert STATUS_PATH.exists(), f"missing {STATUS_PATH}"
    doc = yaml.safe_load(STATUS_PATH.read_text())
    assert isinstance(doc, dict), "release_status.yaml must parse to a mapping"
    assert isinstance(doc.get("baselines"), list) and doc["baselines"], (
        "release_status.yaml must have a non-empty 'baselines' list")
    return doc


def test_release_status_yaml_parses_and_declares_the_enum():
    doc = _load_status()
    assert doc.get("schema_version") == 1
    assert set(doc.get("allowed_status", [])) == ALLOWED_STATUS


def test_every_row_has_required_fields_and_valid_status():
    doc = _load_status()
    seen = set()
    for row in doc["baselines"]:
        name = row.get("name", "<unnamed>")
        missing = REQUIRED_FIELDS - set(row)
        assert not missing, f"{name} missing fields: {sorted(missing)}"
        assert row["status"] in ALLOWED_STATUS, (
            f"{name}: status {row['status']!r} not in {sorted(ALLOWED_STATUS)}")
        assert name not in seen, f"duplicate baseline row: {name}"
        seen.add(name)
        for field in REQUIRED_FIELDS:
            val = row[field]
            assert isinstance(val, str) and val.strip(), (
                f"{name}.{field} must be a non-empty string (use the "
                f"literal string 'N/A -- <reason>' when a field truly does "
                f"not apply; never leave it blank)")


def test_all_task_15_baselines_are_present():
    doc = _load_status()
    names = {row["name"] for row in doc["baselines"]}
    missing = EXPECTED_BASELINE_NAMES - names
    assert not missing, f"release_status.yaml is missing rows for: {missing}"


def test_implemented_or_partial_rows_cite_a_concrete_commit_or_reason():
    """`implemented`/`partial` rows must either point at a real commit/
    version, or (for our own ablations/reimplementations with no external
    release to pin) explicitly say why via an 'N/A --' explanation, never a
    bare 'N/A' with no reason."""
    doc = _load_status()
    for row in doc["baselines"]:
        if row["status"] not in ("implemented", "partial"):
            continue
        cov = row["commit_or_version"]
        assert cov.strip(), f"{row['name']}: empty commit_or_version"
        if cov.strip().upper().startswith("N/A"):
            assert "--" in cov or "-" in cov, (
                f"{row['name']}: commit_or_version is N/A but gives no "
                "reason")


def test_unreleased_baselines_have_no_numeric_producing_adapter():
    doc = _load_status()
    filenames_lower = [p.name.lower() for p in BASELINES_DIR.glob("*.py")]

    for row in doc["baselines"]:
        if row["status"] not in ("unavailable", "literature_only"):
            continue
        name = row["name"]
        stub_mod = KNOWN_STUB_MODULES.get(name)

        if stub_mod is not None:
            assert f"{stub_mod}.py" in filenames_lower, (
                f"{name} ({row['status']}) is declared in this test's "
                f"KNOWN_STUB_MODULES as baselines/{stub_mod}.py, but that "
                "file does not exist")
            mod = importlib.import_module(f"baselines.{stub_mod}")
            public_funcs = [
                (fname, fn) for fname, fn in vars(mod).items()
                if inspect.isfunction(fn) and not fname.startswith("_")
                and inspect.getmodule(fn) is mod
            ]
            assert public_funcs, (
                f"baselines.{stub_mod} exposes no public functions -- "
                "cannot confirm it is inert")
            for fname, fn in public_funcs:
                with pytest.raises(NotImplementedError):
                    fn()
        else:
            forbidden = FORBIDDEN_FILENAME_FRAGMENTS.get(name)
            assert forbidden, (
                f"{name} is {row['status']} but this test has no stub-module "
                "mapping AND no forbidden-filename check registered for it "
                "-- add one to KNOWN_STUB_MODULES or "
                "FORBIDDEN_FILENAME_FRAGMENTS so it stays enforced")
            for frag in forbidden:
                hits = [fn for fn in filenames_lower if frag in fn]
                assert not hits, (
                    f"{name} is {row['status']} (no runnable release) but "
                    f"baselines/ contains a file matching {frag!r}: {hits} "
                    "-- this could be invoked to fabricate a result")


def test_release_status_module_docstring_matches_directory_contents():
    """Guard against the yaml and the actual baselines/ directory drifting
    apart: every *_adapter.py / raw_reconstruction.py file must be
    mentioned by name in baselines/__init__.py's docstring."""
    init_doc = (BASELINES_DIR / "__init__.py").read_text()
    for py in BASELINES_DIR.glob("*.py"):
        if py.name == "__init__.py":
            continue
        assert py.name in init_doc, (
            f"baselines/__init__.py's docstring does not mention {py.name} "
            "-- update it so the package overview stays accurate")
