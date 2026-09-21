import csv

from robo.eval.real_world_pairs import generate, load_pairs


def test_strict_paired_real_world_analysis(tmp_path):
    path = tmp_path / "pairs.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "source", "scene_id", "task_id", "policy_id", "reset_id",
            "sim_success", "real_success", "audit_accept"])
        writer.writeheader()
        writer.writerows([
            {"source": "phone", "scene_id": "s", "task_id": "t",
             "policy_id": "p", "reset_id": "r0", "sim_success": 1,
             "real_success": 1, "audit_accept": True},
            {"source": "phone", "scene_id": "s", "task_id": "t",
             "policy_id": "p", "reset_id": "r1", "sim_success": 1,
             "real_success": 0, "audit_accept": False},
        ])
    payload = generate(path, tmp_path / "out")
    all_row = next(row for row in payload["rows"] if row["subset"] == "all")
    accepted = next(row for row in payload["rows"] if row["subset"] == "audit_accepted")
    assert all_row["outcome_agreement"] == 0.5
    assert accepted["outcome_agreement"] == 1.0
    assert accepted["audit_coverage"] == 0.5
    assert len(load_pairs(path)) == 2
