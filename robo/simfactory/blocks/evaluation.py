"""Evaluation adapters over the stable backend modules."""
from __future__ import annotations

from robo.simfactory.registry import register
from robo.simfactory.runner import sh


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
          help="closed-loop pi0.5 in the twin (needs a running policy server: "
               "bash run/pi05_serve.sh)")
def eval_pi05(ctx, entry):
    print("[simfactory] policy_pi05 needs a live policy server + GPU: start "
          "`bash run/pi05_serve.sh` on a GPU machine, then run "
          f"`python -m robo.eval.pi05_eval --tasks {ctx.out}/sim_export/pi05_tasks.json "
          "--obs composite`")
