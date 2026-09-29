"""robo.manifest: immutable, content-addressed provenance manifests for
scene builds and policy rollouts.

Every scene-construction and policy-rollout output must carry
`scene_build_commit`, `scene_manifest_hash`, `policy_checkpoint_hash`,
`controller_config_hash`, `camera_config_hash`, `task_id`,
`initial_state_id`, `rollout_seed`, so that any later comparison between
runs can mechanically check which declared frozen fields actually matched
(configs/policies/frozen_fields.yaml).

    from robo.manifest.schema import SceneBuildManifest, RolloutManifest
    from robo.manifest import io as manifest_io
    from robo.manifest import hash as manifest_hash
"""
from robo.manifest.schema import (
    SCHEMA_VERSION,
    ManifestBase,
    RolloutManifest,
    SceneBuildManifest,
)

__all__ = [
    "SCHEMA_VERSION",
    "ManifestBase",
    "RolloutManifest",
    "SceneBuildManifest",
]
