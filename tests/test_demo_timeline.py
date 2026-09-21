import pytest
import yaml

from interface.demo_agentic import DemoError, resolve
from tests.test_demo_manifest import fixture


def test_draft_requires_exact_duration(tmp_path):
    path, config = fixture(tmp_path)
    config["shots"][0]["duration_s"] = 2
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(DemoError, match="exactly 12 seconds"):
        resolve(path)


def test_template_cannot_pass_as_a_finished_demo():
    with pytest.raises(DemoError, match="assign an immutable freeze"):
        resolve("configs/demo/icra2027.yaml")
