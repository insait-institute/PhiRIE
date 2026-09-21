"""Task-conditioned twin certification (Task 11 -> Task 12 -> Task 13).

grounding.py    role resolution: task/rubric language -> ranked scene-instance
                hypotheses, plus the geometric primitives (support edges,
                swept-workspace capsule, camera frustum) role resolution and
                the graph builder both need. Reads only the frozen scene and
                task manifests -- never an evaluation-result field.
task_graph.py   eligibility gating, node/edge assembly, JSON serialization,
                the `python -m robo.certification.task_graph` CLI, and the
                optional top-down matplotlib overlay.

Task 12 (features/model/calibrate/report) and Task 13 (repair) are separate,
not-yet-built packages that consume this module's graph JSON; they are out
of scope here.
"""
