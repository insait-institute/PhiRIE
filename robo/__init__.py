"""Robot layer: closed-loop policy evaluation (pi0.5) inside exported
scenes, plus the simulator export backends.

envs/        DROID-style MuJoCo gym environment
rigs/        robot rig construction (Panda + Robotiq 2F-85)
tasks/       pick-and-place task-suite generation and staged scoring
rendering/   photoreal composite observations (splat + posed assets + robot)
policy/      policy registry, frozen control contract, and policy clients
eval/        pi0.5 closed-loop episode runner and fidelity/visual metrics
sim/         PyBullet / MJCF / OmniGibson export and settle utilities
simfactory/  YAML-driven block runner over the construction/sim/eval stages
manifest/    content-addressed provenance manifests for builds and rollouts
"""
