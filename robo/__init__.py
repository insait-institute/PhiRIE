"""Robot layer (RoboLab-style): closed-loop policy evaluation (pi0.5)
inside exported scenes, plus the simulator export backends.

envs/       DROID-style MuJoCo gym environment
rigs/       robot rig construction (Panda + Robotiq 2F-85)
tasks/      pick-and-place task-suite generation and staged scoring
rendering/  photoreal composite observations (splat + posed assets + robot)
eval/       websocket policy client and episode runner
sim/        PyBullet / MJCF / OmniGibson / Isaac export, settle utilities
"""
