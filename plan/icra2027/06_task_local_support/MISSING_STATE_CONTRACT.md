# Missing public construction state

Owner: E6 agent; branch `agent/icra-e6-public-grounding`. This extends the existing task graph and feature/label producers. It performs no 3D role association, virtual robot placement, model invocation or reference-data access.

A missing robot, base position or explicitly null yaw produces a robot node with `transform: null`, unresolved construction status and `robot_frame` missing evidence. No origin is inserted. The legacy omitted-yaw convention remains unchanged for existing declared frames; new public inputs must use null for an unknown yaw. Reach, robot-relative spatial hints and swept-workspace evidence cannot use a missing frame.

Absent or incomplete policy cameras remain missing references. A declared camera identity whose frame is incomplete keeps an unresolved camera node with a null transform. Visibility features are null with their existing missingness indicators; absence of a camera does not become a measured invisible target. Malformed/nonfinite frames and degenerate camera directions are rejected.

Tasks may explicitly declare `unresolved_roles: {<requested role>: <concrete reason>}`. Such roles retain empty hypotheses instead of receiving a category-prior substitute. Support and obstacle overrides remove their stale edges/role assignments. An explicitly unresolved support no longer excludes its former support hypothesis from the obstacle calculation. The public 3D association producer remains a separate required stage.

The canonical feature producer retains every predeclared query, merged unresolved references, and construction-only missingness indicators. Distinct target-region query identities and their source manifests remain distinct even if their feature values match. Five legacy complete-frame graph JSON outputs remain byte-identical.

Only the separate evaluation join treats an explicitly absent required robot/camera frame as an unavailable invariant, analogous to an actual failed build. Ordinary or explicitly declared role-grounding uncertainty is not used as an oracle validity label; independent invariant measurements still determine that label. Degenerate folds preserve all joined rows and diagnostics before reporting that LOSO cannot run.

CPU command: `python -m pytest -q tests/test_e6_missing_graph_state.py tests/test_task_graph.py tests/test_certificate.py tests/test_task_support_real_inputs.py tests/test_task_support_smoke.py tests/test_audit_loso.py tests/test_audit_metrics.py`. Negative tests include absent robot/cameras/target/support, malformed coordinates, unsupported spatial hints, no stale support/obstacle edges, distinct region queries, forbidden reference channels and degenerate-fold row retention.

Scientific status: no new real feature rows or labels; `LOCAL_SUPPORT_GATE=NOT_RUN`. The earlier RGB-only reconstruction smoke and three-condition pilot are independent sealed stages at source `7c76b2c`; their sources and environments are unchanged.
