"""Offline packaging verifies content hashes and derives coverage from its input."""

import hashlib
import json
import zipfile

from physicalview.paper_download_pack import main


def test_offline_pack_uses_actual_coverage_and_review(tmp_path):
    root = tmp_path / "campaign"
    folder = root / "behavior" / "fixture" / "a_original"
    folder.mkdir(parents=True)
    (root / ".paper-pack.lock").touch()
    row = dict(
        dataset="behavior",
        scene="fixture",
        feature="a_original",
        status="captured",
        pngs=1,
        review="approved",
        error="",
    )
    (root / "coverage.json").write_text(json.dumps([row]))
    (folder.parent / "features.json").write_text(
        json.dumps(
            {"a_original": {"status": "captured", "frames": ["a_original/frame.png"]}}
        )
    )
    # Byte fixtures test archive integrity, not rendering or paper quality.
    payload = b"fixture PNG bytes"
    (folder / "frame.png").write_bytes(payload)
    (folder / "frame.json").write_text(
        json.dumps({"sha256": hashlib.sha256(payload).hexdigest()})
    )
    (folder / "review-preview.jpg").write_bytes(b"fixture JPEG bytes")
    (folder / "panel-layout.json").write_text(json.dumps({"frames": ["frame.png"]}))
    (folder / "panel.svg").write_text(
        '<svg><image href="data:image/png;base64,ZmFrZQ=="/></svg>'
    )
    out = tmp_path / "download"
    main(["--root", str(root), "--out", str(out)])
    assert len(list(out.glob("*.zip"))) == 2
    for archive in out.glob("*.zip"):
        prefix = "phiview-full" if "-full-" in archive.name else "phiview-preview"
        with zipfile.ZipFile(archive) as z:
            manifest = json.loads(z.read(prefix + "/manifest.json"))
            assert manifest["groups"] == 1 and manifest["datasets"] == {"behavior": 1}
            assert manifest["publication_approved_groups"] == 1
            for item in manifest["files"]:
                assert (
                    hashlib.sha256(z.read(prefix + "/" + item["file"])).hexdigest()
                    == item["sha256"]
                )
            readme = z.read(prefix + "/README.txt").decode()
            assert "420" not in readme and "BEHAVIOR 尚无" not in readme
            assert "1 组获论文质量验收" in readme
