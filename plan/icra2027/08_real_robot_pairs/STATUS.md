# E8 matched physical trials status

Updated: 2026-09-04 UTC. Owner: root orchestrator.
Branch: `agent/icra-e3-agentic`; audited code: `c3b97ffe4db235cff8728bb5a836ae5f6f02a2c0`.
State: NOT_RUN. `PHYSICAL_TRIALS_AVAILABLE=false`. `SIM_REAL_CLAIM=NOT_RUN`.

- Robot available: no confirmed hardware; `docs/CONTRIBUTIONS.md` explicitly states “We have no robot.”
- Policy checkpoints: local pi0.5 DROID and sim-co-trained checkpoint directories exist; autonomous deployment on hardware is not established.
- Matched cameras/calibration, repeatable physical reset protocol, safety operator and physical trial records: unavailable/unverified.
- Smoke, pilot and full: NOT_RUN. Slurm job IDs: none. Hardware/runtime/output trials: none. Checkpoint hashes for physical trial pairing: unavailable.
- Required ingest command once genuine data exists: `python -m robo.eval.real_world_pairs --sim-ledger <freeze>/harness/harness_ledger.jsonl --real-records <freeze>/real_world/real_trial_records.jsonl --config configs/experiments/icra2027/real_robot.yaml --out <freeze>/real_world/paired_trials.csv`.
- Passed criterion: no proxy or demonstration record is treated as autonomous physical success.
- Blocker classification: unavailable physical hardware and genuine matched trials. This does not block E9.
- Claim handoff: remove autonomous real-success/sim-real-gap statements; physical-success table cells must stay empty or be removed with their unsupported claim.
