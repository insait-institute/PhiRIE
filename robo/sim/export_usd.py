"""USD/Isaac export -- OUT OF SCOPE, stub only.

`docs/ICRA_RESEARCH_CONTRACT.md` (Decision 1, frozen 2026-08-16) drops
PolaRiS/Isaac Lab entirely: "the robot stack is MuJoCo + pi0.5 only" (see
docs/ROBOT.md). plan/06_FULL_ROOM_COLLISION_EXPORT.md's task file lists
`robo/sim/export_usd.py` as an output because the 22-task plan predates that
scope decision -- the plan describes the idealized program, the contract
states what is actually executed. Task 06's real effort went into
`robo/sim/room_collision.py` + `export_mjcf.py`'s `--collision-mode`, which
is what the paper's `mujoco_paired` track actually needs.

This module exists so any caller that still expects a USD export path gets
a loud, honest failure instead of a silent no-op or a half-working stub
that looks like it did something. Nothing in this repo currently imports or
calls `export_to_usd` -- `export_omnigibson.py` + `omnigibson_bridge/
import_and_run.py` are the actual (pre-existing, still in scope)
URDF->USD/OmniGibson bridge for BEHAVIOR-1K, and they are unaffected by
this stub.

If USD/Isaac Lab export is ever un-dropped, implement it here for real
(the room/shim MJCF collision groups in `room_collision.py` would need a
straightforward USD-physics-schema translation) rather than resurrecting
ambition this stub deliberately does not provide.
"""


def export_to_usd(*_args, **_kwargs):
    """Not implemented -- USD/Isaac export is out of scope per
    docs/ICRA_RESEARCH_CONTRACT.md Decision 1. Raises unconditionally so a
    caller cannot mistake silence for success."""
    raise NotImplementedError(
        "USD/Isaac export is out of scope per docs/ICRA_RESEARCH_CONTRACT.md "
        "Decision 1 (no PolaRiS/Isaac Lab) -- the robot stack is MuJoCo + "
        "pi0.5 only (docs/ROBOT.md). Use robo.sim.export_mjcf's "
        "--collision-mode {room,shim} for the MJCF/MuJoCo path instead, or "
        "robo.sim.export_omnigibson.py for the pre-existing (unaffected) "
        "BEHAVIOR-1K/OmniGibson URDF bridge.")


if __name__ == "__main__":
    export_to_usd()
