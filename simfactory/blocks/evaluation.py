"""Evaluation adapters over the stable backend modules."""
from __future__ import annotations

from simfactory.registry import register
from simfactory.runner import sh

@register("evaluation", "geometry",
          help="yield/tier report + drop test (agents.eval.factory_report; "
               "GT-free in AUTO mode)")
def eval_geometry(ctx, entry):
    sh(ctx, "agents.eval.factory_report")


@register("evaluation", "geometry_vs_gt",
          help="F1@20/40mm vs GT scan (agents.eval.eval_vs_gt; ScanNet++)")
def eval_geometry_gt(ctx, entry):
    sh(ctx, "agents.eval.eval_vs_gt", ctx.out)


@register("evaluation", "render",
          help="held-out composite render metrics "
               "(agents.eval.factory_eval_render)")
def eval_render(ctx, entry):
    sh(ctx, "agents.eval.factory_eval_render", env_kind="gsplat")


@register("evaluation", "policy_pi05",
          help="closed-loop pi0.5 in the twin (needs a policy server; see "
               "run/slurm/pi05_closedloop.sbatch)")
def eval_pi05(ctx, entry):
    print("[simfactory] policy_pi05 needs a live policy server + GPU; "
          "submit run/slurm/pi05_closedloop.sbatch with "
          f"SCENES={ctx.scene} OBS=composite instead of running inline")


@register("evaluation", "traj_replay",
          help="replay real robot trajectories in the twin "
               "(robo.eval.traj_replay; DROID joints / BEHAVIOR puppet)")
def eval_replay(ctx, entry):
    args = entry.get("args", [])
    sh(ctx, "robo.eval.traj_replay", *args)
