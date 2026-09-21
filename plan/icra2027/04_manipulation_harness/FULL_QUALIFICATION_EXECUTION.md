# Full-source CPU qualification launcher

`run/icra2027/e4_full_qualification.py` binds the existing canonical materializer,
full-room exporter, task freezer and CPU qualifier to the preregistered full E4
budget. It computes no new qualification metric and creates no rollout ledger.

Before configuration preparation, copy the exact full protocol into the new
experiment evidence root. `--inspect` validates the original construction E0,
clean source commit, complete inventory and original discovery identities. It
requires the existing `full50_control_integrity_gate.json` and
`full50_observation_integrity_gate.json`; missing audits return WAITING. The
control audit implementation is pinned, all 50 control seal/shard/ledger and
observation seal hashes are rebound, and each canonical scene producer rechecks
its actual construction inputs. No evaluation or GT artifact is required.

```bash
python run/icra2027/e4_full_qualification.py --inspect \
  --protocol "$PROTOCOL_COPY" --e3-root "$ORIGINAL_E3_ROOT"
python run/icra2027/e4_full_qualification.py --prepare-config \
  --protocol "$PROTOCOL_COPY" --e3-root "$ORIGINAL_E3_ROOT" \
  --freeze-id "$NEW_FREEZE_ID" --menagerie-root "$FROZEN_MENAGERIE_ROOT" \
  --out "$TRACKED_STAGE_CONFIG"
```

Commit the stage config and bind it in E0 as `e4_full_qualification_config`, plus
`qualification_python`. Freeze the code, original audit/source references,
Menagerie and OpenPI resize identity before execution. Run the first source-order
scene with a selected query as the fixed real CPU pilot. It must complete the
canonical qualifier and replay integrity; no positive pass count is required to
release the remaining scenes. The pilot is never repeated as part of the full
stage. Selected scene failures remain in the 269-query / 2,690-cell denominator.

```bash
python run/icra2027/e4_full_qualification.py --config "$TRACKED_STAGE_CONFIG" \
  --freeze-root "$NEW_FREEZE_ROOT" --expected-code-commit "$EXACT_SOURCE_SHA" \
  --scene-id "$DECLARED_SCENE"
```

Use ordinary CPU Slurm jobs, 64 GiB for exporters, no GPU, no array, and the
established OSMesa/pinned OpenPI environment. Keep all reset replay on one host.
Pin TMPDIR and XDG_CACHE_HOME within the primary evidence root; use literal
`--export=ALL` and `--no-requeue`. Scene execution resumes only missing canonical
products; a completed scene handoff cannot be overwritten.

Nine zero-query scenes receive source-only NOT_RUN handoffs. The specific
existing no-size-admissible-target planning failure receives a typed
PLANNING_UNQUALIFIED handoff with null reset/camera/policy telemetry. Unexpected
errors propagate with original logs and partial products; they are never
relabeled as measured task failures. A task that fails prerequisites may produce
canonical qualification records without a simulator episode. Handoffs therefore
distinguish prerequisite cells from actual 900-step simulator cells.

The 6,155 original semantic queries and 5,886 predeclared budget exclusions remain
in the authenticated source protocol. This stage does not authorize policy
execution. The compact-only camera path must be extended and validated for this
explicit full scope before downstream camera/scripted/policy phases. Later
matrix selection must use the existing frozen `e4_full_protocol.select_matrix`
after complete qualification and camera authentication; no policy-outcome
substitution is permitted.
