# Input examples

`libero-roster.json` is a schema example with placeholder paths. Replace them with a native
MuJoCo XML whose mesh/texture paths are resolved and an NPZ containing a `state` vector
in the order `[time, qpos..., qvel...]`. The adapter checks its length against the model,
freezes articulated fixtures at the recorded state and omits the native robot.

BEHAVIOR uses the task/shard mapping from the configured SimAny backend's
`oracle.capture_generator` dataset preparation. Supply that actual tasks YAML through
`--tasks-config`; synthetic datasets and placeholder captures are not shipped.
