"""Release source checks must reject a corrupted or incomplete bundled viewer."""

import hashlib
import json

import pytest

from tools.release.verify import verify_bundled_source


def test_bundled_source_detects_modified_and_missing_files(tmp_path):
    source = tmp_path / "viewer.py"
    source.write_text("original source\n")
    (tmp_path / "SOURCE_PROVENANCE.json").write_text(
        json.dumps(
            {
                "commit": "upstream-commit",
                "files": {"viewer.py": hashlib.sha256(source.read_bytes()).hexdigest()},
            }
        )
    )
    assert verify_bundled_source(tmp_path)["files"] == 1
    source.write_text("modified source\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_bundled_source(tmp_path)
    source.unlink()
    with pytest.raises(FileNotFoundError):
        verify_bundled_source(tmp_path)


def test_bundled_source_rejects_path_escape(tmp_path):
    (tmp_path / "SOURCE_PROVENANCE.json").write_text(
        json.dumps({"commit": "upstream-commit", "files": {"../outside": "unused"}})
    )
    with pytest.raises(ValueError, match="invalid bundled source path"):
        verify_bundled_source(tmp_path)
