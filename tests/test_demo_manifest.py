import json
from pathlib import Path
import sys
import time

from PIL import Image
import pytest
import yaml

from interface.demo_agentic import DemoError, assemble, present, render, resolve
from agents.orchestrator.artifact import sha256_file


def fixture(tmp_path):
    image = tmp_path / "fixture.png"
    Image.new("RGB", (960, 540), (48, 74, 96)).save(image)
    data = {"schema_version": 1, "mode": "smoke", "freeze_id": "fixture-v1",
            "source_root": str(tmp_path), "resolution": [960, 540], "fps": 15,
            "sources": [{"id": "fixture", "kind": "image", "path": str(image),
                         "sha256": sha256_file(image), "freeze_id": "fixture-v1"}],
            "shots": [{"shot_id": f"shot_{i}", "source_id": "fixture", "duration_s": 3,
                       "overlay_text": ["Synthetic framework test"]} for i in range(4)],
            "limitations": ["Synthetic fixtures are not experiment evidence"]}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path, data


def test_source_hash_and_mixed_freeze_fail_closed(tmp_path):
    path, data = fixture(tmp_path)
    assert resolve(path)["frame_count"] == 180
    data["sources"][0]["freeze_id"] = "other-v1"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(DemoError, match="mixed freeze"):
        resolve(path)
    data["sources"][0]["freeze_id"] = "fixture-v1"
    data["sources"][0]["sha256"] = "0" * 64
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(DemoError, match="SHA256 mismatch"):
        resolve(path)


def test_mixed_episodes_cannot_be_one_policy_block(tmp_path):
    path, data = fixture(tmp_path)
    data["sources"][0]["episode_id"] = "episode_a"
    data["sources"].append({**data["sources"][0], "id": "other", "episode_id": "episode_b"})
    for shot in data["shots"]:
        shot["episode_id"] = "episode_a"
    data["shots"][-1].update(source_id="other", episode_id="episode_b")
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(DemoError, match="multiple episodes"):
        resolve(path)


def test_no_fabricated_numeric_cards_or_paper_mode(tmp_path):
    path, data = fixture(tmp_path)
    data["shots"][0]["overlay_text"] = ["Success 100 percent"]
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(DemoError, match="numeric overlays"):
        resolve(path)
    data["mode"] = "paper"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(DemoError, match="final paper gates"):
        resolve(path)


def test_retry_cannot_reread_same_proposal(tmp_path):
    path, data = fixture(tmp_path)
    retry_path = tmp_path / "retry.json"
    retry_path.write_text(json.dumps({"action": "reread", "proposal_id": "same", "parent_proposal_id": "same"}))
    event = {"freeze_id": "fixture-v1", "proposal_id": "same", "reason_codes": ["bounded_retry"],
             "retry_produced": True, "retry_invoked": True,
             "selected_asset": {"artifact_paths": {"retry": str(retry_path)},
                                "artifact_hashes": {"retry": sha256_file(retry_path)}}}
    source = tmp_path / "event.json"
    source.write_text(json.dumps(event))
    data["sources"][0].update(kind="selection", path=str(source), sha256=sha256_file(source))
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(DemoError, match="distinct proposal"):
        resolve(path)


def test_render_and_assemble_exact_draft_and_refuse_overwrite(tmp_path):
    path, data = fixture(tmp_path)
    output = tmp_path / data["freeze_id"] / "demo"
    render(path, output)
    assert json.loads((output / "qa.json").read_text())["frames"] == 180
    assert (output / "poster.png").is_file()
    assert len((output / "engineering_draft_12s.srt").read_text().split("-->")) == 5
    assert assemble(output)["draft_package_valid"] is True
    with pytest.raises(DemoError, match="overwrite"):
        render(path, output)
    (output / "poster.png").write_bytes(b"changed")
    with pytest.raises(DemoError, match="SHA256 mismatch"):
        assemble(output)


def playback_config(tmp_path):
    fallback = tmp_path / "fallback.mp4"
    fallback.write_bytes(b"preflight-verified-fixture")
    marker = tmp_path / "video_started"
    config = {"fallback": {"path": str(fallback), "size_bytes": fallback.stat().st_size,
                           "mtime_ns": fallback.stat().st_mtime_ns, "sha256": sha256_file(fallback)},
              "video_command": [sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).touch()", str(marker)],
              "live_command": [sys.executable, "-c", "import time; time.sleep(20)"],
              "health_commands": [[sys.executable, "-c", "import time; time.sleep(20)"]]}
    path = tmp_path / "presentation.json"
    path.write_text(json.dumps(config))
    return path, config, marker


def test_unhealthy_service_falls_back_within_three_seconds(tmp_path):
    path, _, marker = playback_config(tmp_path)
    start = time.monotonic()
    proc = present(path)
    proc.wait(timeout=1)
    assert marker.exists()
    assert time.monotonic() - start < 3


def test_service_failure_after_live_start_falls_back(tmp_path):
    path, config, marker = playback_config(tmp_path)
    counter = tmp_path / "counter"
    config["health_commands"] = [[sys.executable, "-c",
        "from pathlib import Path; import sys; p=Path(sys.argv[1]); existed=p.exists(); p.touch(); sys.exit(int(existed))", str(counter)]]
    path.write_text(json.dumps(config))
    start = time.monotonic()
    proc = present(path, "live")
    proc.wait(timeout=1)
    assert marker.exists()
    assert time.monotonic() - start < 3


def test_modified_fallback_is_rejected(tmp_path):
    path, config, marker = playback_config(tmp_path)
    Path(config["fallback"]["path"]).write_bytes(b"not the preflight video")
    with pytest.raises(DemoError, match="fallback changed"):
        present(path)
    assert not marker.exists()
