from pathlib import Path

from robo.eval.harness_smoke import run


def test_harness_smoke_generates_valid_main_table(tmp_path):
    config = Path("configs/experiments/harness_smoke.yaml")
    summary = run(config, tmp_path, resets=4)
    assert summary["validation"]["ok"]
    assert summary["validation"]["coverage"] == 1.0
    assert (tmp_path / "paper_tables" / "main_table.tex").exists()
    assert len(summary["table"]["rows"]) == 5
