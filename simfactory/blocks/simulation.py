"""Simulation adapters over the stable backend modules."""
from __future__ import annotations

from simfactory.registry import register
from simfactory.runner import sh

@register("simulation", "mujoco",
          help="MJCF export + settle test + Isaac manifest "
               "(robo.sim.export_mjcf --test)")
def sim_mujoco(ctx, entry):
    sh(ctx, "robo.sim.export_mjcf", "--test")
    sh(ctx, "robo.tasks.pi05_tasks", "--out-dir", ctx.out)


@register("simulation", "pybullet",
          help="PyBullet settle + dynamics (robo.sim.s7_sim)")
def sim_pybullet(ctx, entry):
    sh(ctx, "robo.sim.s7_sim")


@register("simulation", "omnigibson",
          help="OmniGibson/BEHAVIOR asset export (robo.sim.export_omnigibson)")
def sim_omnigibson(ctx, entry):
    sh(ctx, "robo.sim.export_omnigibson")
