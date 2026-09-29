"""Aggregate the 7-task x 2-arm OmniGibson validation results into a single
summary comparison table (JSON + Markdown).

Run with any Python (no OmniGibson/GPU needed):
  python3 aggregate_results.py
"""
import json
from pathlib import Path

RESULTS_DIR = (Path(__file__).resolve().parents[3]
               / "outputs" / "omnigibson_export" / "results")
OUT_JSON = RESULTS_DIR.parent / "comparison_table.json"
OUT_MD = RESULTS_DIR.parent / "comparison_table.md"

TASKS = [
    "loading_the_dishwasher", "putting_dirty_dishes_in_sink", "store_beer",
    "clearing_table_after_coffee", "changing_dogs_water", "clean_clear_plastic",
    "fold_a_plastic_bag",
]


def load(task, arm):
    p = RESULTS_DIR / f"{task}__{arm}.json"
    return json.loads(p.read_text()) if p.exists() else None


def n_objects_placed(d):
    """Count of movable objects whose real-checker literal (or, for tasks
    with no real literal at all, whose placement sampler) succeeded - the
    per-object granularity a results table needs beyond just the aggregate
    goal pass/fail bit."""
    if d["placement_sampler_log"]:
        return sum(1 for v in d["placement_sampler_log"].values() if v), len(d["placement_sampler_log"])
    real = [l for l in d["literal_log"] if l["kind"] == "real_checker"]
    if not real:
        return None, None
    return sum(1 for l in real if l["result"]), len(real)


def main():
    rows = []
    for task in TASKS:
        ours = load(task, "ours")
        base = load(task, "baseline")
        assert ours and base, f"missing results for {task}"

        ours_n, ours_total = n_objects_placed(ours)
        base_n, base_total = n_objects_placed(base)

        rows.append({
            "task": task,
            "goal_kind": ours["goal_kind"],
            "n_real_checker_literals": ours["n_real_checker_literals"],
            "n_scripted_literals": ours["n_scripted_literals"],
            "ours_goal_satisfied": ours["all_goal_literals_satisfied"],
            "baseline_goal_satisfied": base["all_goal_literals_satisfied"],
            "ours_objects_placed": f"{ours_n}/{ours_total}" if ours_n is not None else "n/a",
            "baseline_objects_placed": f"{base_n}/{base_total}" if base_n is not None else "n/a",
            "ours_settle_s": round(ours["settle_wall_s"], 1),
            "baseline_settle_s": round(base["settle_wall_s"], 1),
        })

    n_tasks = len(rows)
    n_ours_pass = sum(1 for r in rows if r["ours_goal_satisfied"])
    n_base_pass = sum(1 for r in rows if r["baseline_goal_satisfied"])
    n_real_total = sum(1 for task in TASKS for l in load(task, "ours")["literal_log"] if l["kind"] == "real_checker")
    n_scripted_total = sum(1 for task in TASKS for l in load(task, "ours")["literal_log"] if l["kind"] == "scripted")

    summary = {
        "n_tasks": n_tasks,
        "ours_tasks_passed": n_ours_pass,
        "baseline_tasks_passed": n_base_pass,
        "n_real_checker_literals_total": n_real_total,
        "n_scripted_literals_total": n_scripted_total,
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=1))

    # ---- Markdown table -----------------------------------------------
    lines = [
        "# OmniGibson / BEHAVIOR-1K validation - summary comparison",
        "",
        f"**{n_ours_pass}/{n_tasks} tasks pass with ours (CoACD collision) vs "
        f"{n_base_pass}/{n_tasks} with baseline (single convex-hull collision)**, "
        f"real BDDL goal checker across {n_real_total} geometric (Inside/OnTop) "
        f"literals + {n_scripted_total} scripted (substance/cloth, out of rigid-body "
        f"reconstruction scope) literals total.",
        "",
        "| Task | Goal type | Ours: goal | Ours: objects placed | Baseline: goal | "
        "Baseline: objects placed | Real / Scripted literals |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['task']} | {r['goal_kind']} | "
            f"{'✅' if r['ours_goal_satisfied'] else '❌'} | {r['ours_objects_placed']} | "
            f"{'✅' if r['baseline_goal_satisfied'] else '❌'} | {r['baseline_objects_placed']} | "
            f"{r['n_real_checker_literals']} / {r['n_scripted_literals']} |"
        )
    lines += [
        "",
        "**Methodology notes** (see omnigibson_bridge/import_and_run.py for the exact logic):",
        "- \"Real\" literals are evaluated by OmniGibson's own "
        "`evaluate_bddl_predicate()` (pure kinematic/pose checks - Inside/OnTop), "
        "identical code path whether the object came from the official curated "
        "BEHAVIOR-1K dataset or our custom reconstruction pipeline.",
        "- \"Scripted\" literals (Filled/Covered/Contains/Folded) are substance/cloth "
        "predicates outside our rigid-body reconstruction's scope by design (matches "
        "a stated limitation of this pipeline) - their truth value is asserted, not "
        "measured, and is identical for both arms (not a comparison axis).",
        "- Objects are placed via OmniGibson's own `object_states.Inside/OnTop.set_value()` "
        "rejection-sampling primitive (the same one BehaviorTask's init-sampler uses), "
        "not a hand-rolled drop position - fixture doors are opened via the `Open` state "
        "before sampling where applicable.",
        "- `loading_the_dishwasher` baseline: the convex-hull-only bowl's sampler found "
        "*zero* valid poses across every attempt (see placement_sampler_log in the raw "
        "JSON) - it isn't a near-miss, the naive collision geometry never fit.",
        "- `putting_dirty_dishes_in_sink`/`changing_dogs_water`/`clean_clear_plastic` use "
        "`commercial_kitchen_sink` (not `pedestal_sink`) after the latter's basin proved "
        "too small for any of our objects regardless of arm.",
        "- `changing_dogs_water`'s OnTop(bowl, floor) is scripted rather than real: our "
        "scenes' floor is a bare synthetic ground plane with no StatefulObject/contact "
        "API to check against, not a substance/cloth scope exclusion.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_JSON}\nwrote {OUT_MD}")
    print(f"\n{n_ours_pass}/{n_tasks} ours pass, {n_base_pass}/{n_tasks} baseline pass")


if __name__ == "__main__":
    main()
