from robo.eval.real_world_metrics import summarize_construction, summarize_trials


def test_real_world_construction_and_paired_gap():
    construction = summarize_construction([
        {"source": "phone", "workspace_id": "r0", "full_build": True,
         "alignment_pass": True, "translation_cm": 1.0,
         "rotation_deg": 2.0, "queries": 3},
        {"source": "phone", "workspace_id": "r1", "full_build": False,
         "alignment_pass": False, "translation_cm": 3.0,
         "rotation_deg": 4.0, "queries": 2},
    ])[0]
    assert construction["workspaces"] == 2
    assert construction["full_builds"] == 1
    assert construction["alignment_pass_rate"] == 0.5

    trial = summarize_trials([
        {"source": "phone", "subset": "all", "scene_id": "r0",
         "task_id": "t", "policy_id": "p", "sim_success": 1,
         "real_success": 0, "audit_accept": True},
        {"source": "phone", "subset": "all", "scene_id": "r0",
         "task_id": "t", "policy_id": "p", "sim_success": 1,
         "real_success": 1, "audit_accept": False},
    ])[0]
    assert trial["task_pairs"] == 1
    assert trial["sim_success"] == 1.0
    assert trial["real_success"] == 0.5
    assert trial["absolute_gap"] == 0.5
