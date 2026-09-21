# N0 protocol and identity

- Owner: e4_endpoints; branch `agent/icra-n0-contract`.
- Tested implementation commit: `7d6df95` (N0/N5 contracts, cabinet dispatcher, source aggregation); status commit contains this file.
- State: SMOKE_PASSED; real three-family DEV engineering controls passed. This does not by itself admit the complete TEST matrix.
- Smoke: `/group/worldcept/code/SimAny-wt/sr0-native/.venv-native/bin/python -m pytest -q tests/test_roundtrip*.py --basetemp outputs/n0-final-tests`: **190 passed, 1 skipped**, 3.73 s. Includes negative source/config/reset/force-remap/scorer tests.
- Schema v1 remains supported; v2 binds actual canonical XML/assets/full integration state, reset-bank identity and independent policy RNG. CounterToSink H600, SinkToCounter H900, prospective CounterToCabinet H750 are pinned native registry values.
- Cabinet native scorer uses full object bbox, `partial_check=False`, upstream default `th=0.05`. Because fixture axes are not normalized, this is not a uniform 5 cm margin. Cabinet opens in native setup; room/cabinet context remains oracle.
- Integration repair `9f13ce6`: decode canonical integration with the metadata-reset source model, name-map only fixed-body external-force slots, preserve all robot/joint/controller/warmstart fields; reject dynamic topology drift and discarded nonzero force. Real first whisk v2 B0 reached 600 policy ticks without errors after this repair; its native failure is retained.

## Real controls and provenance

- Policy interleaving: client `ba96402`, service `839484` / `fc88da7`, Hala A6000. `python -m robo.roundtrip.policy_interleaving --host hala --port 8017 --out <stage>/contract/policy_interleaving_test.json`: PASS, 12 real model chunks, two streams, serial/reversed concurrent/reconnected outputs byte-identical. 65.616 s including cold compilation; zero native episodes.
- Interleaving receipt: `/group/worldcept/code/SimAny/outputs/icra2027/20260906-f73426b-v1/sim_recon_sim/scale_up/contract/policy_interleaving_test.json`.
- Checkpoint receipt SHA256 `e4ef1d5f325841510ae31a4078cbdc10d3669b140e6397afebed08fb45f381ac`; normalization SHA256 `4aed1af411bd0e0f49d2b0e6d6832b11b8682917231a07173fbc30fd493bbdee`.
- SinkToCounter real U0/U1: source `ba96402`, serial step in service allocation `839484`, 10 actions per control, initial observations identical, every qpos difference zero and every native predicate equal. Receipt `/group/worldcept/code/SimAny/outputs/icra2027/20260906-6ed894a-v1/sim_recon_sim/scale_up/sink_to_counter_identity/identity_report.json`. CounterToSink prior real identity remains preserved.
- Actual generated mesh scorer audit: source `417c54a`, step `839484.14`, exit 0. Command saved at `/group/worldcept/code/SimAny/outputs/icra2027/20260906-c11a41f-v2/sim_recon_sim/scale_up/dispatch/run_scorer.sh`; result `scorer_audit/scorer_diagnosis.json` in that stage.
- All three generated assets (original DEV seeds 1, 2, 4) pass known local-frame reexpression: unchanged mesh asset bytes; world visual vertices differ by at most 4.44e-16 m; fixed evaluator-frame gripper distances by at most 1.11e-16 m. These are static representation controls, not additional policy episodes or unknown physical correspondence estimates.
- Four final-state native predicate recomputations match original logs. Seeds 2/4 B0 have entire declared visual world envelope outside sink. Seed 1 B0 and seed 2 REF raw successes remain origin-bound ambiguous. Native thresholds and original outcomes are unchanged.
- Full-H native importer controls: source `fb10ee2`, step `839484.15`; both original DEV seeds 1/2 reproduce all 600 qpos and native-predicate records exactly. See N5 STATUS.
- Preserved launcher failures: jobs 839505, 839540, 839541 failed before policy execution due inherited CUDA/EGL device-ID mismatch (environment). Corrected pending jobs 839557/839558/839559 were cancelled before execution with receipts in `20260906-6ed894a-v1/.../dispatch`; successful serial steps supersede them only as new outputs.
- Hardware: one shared allocated Hala A6000; policy about 8.5 GiB, one native renderer observed total peak 9.49 GiB / 46.07 GiB. No job arrays. Shared `run/roundtrip/native_env.sh` normalizes GPU IDs.

## Claim gates and limits

`PAIR_CONTRACT_READY=true` for the admitted two-family binding-specific DEV path after per-family controls and real v2 import pilot. N1 owns complete matrix accounting. Full TEST admission is separate; third-family CounterToCabinet U0/U1 now passed at the declared DEV instance.

Raw official native success can be reported explicitly as **binding-specific**, as the frozen matrix allows. Actual generated/native physical correspondence is **NOT_ESTABLISHED**; absolute pose metrics remain NULL; strong task preservation and no-easier-origin claims are **NOT_RUN**. No GT snapping, scorer threshold changes or exclusion of failed builds is allowed.

## CounterToCabinet real gate, completed

- Source `8c30c51`, step `839484.18`, Hala A6000, DEV layout11/style11/seed0, official H750. U0/U1 each 10 control steps: initial/step observations exact, every qpos maximum difference 0, all native predicates equal. Typed full-bbox plus retreat decomposition equals the actual native function; zero learned policy episodes.
- Receipt `/group/worldcept/code/SimAny/outputs/icra2027/20260906-0594848-v1/sim_recon_sim/scale_up/cabinet_identity/identity_report.json`; same stage `cabinet_scorer.json` and `dispatch/run_cabinet.sh`. Step exit0.
- Prior step839484.17 failed before control steps because the diagnostic wrapper omitted output-directory creation. Classification: launcher code bug. Preserved `20260906-cba3908-v1/.../dispatch/cabinet.log`; corrected wrapper ran in a new immutable stage.
- Engineering readiness does not assert target graspability, policy success, physical correspondence or TEST performance on the new family.

## Prospective per-canonical process binding (2026-09-06)

- Owner: `/root/e4_endpoints`, branch `agent/icra-n0-engine`. Legacy policy metadata and frozen episodes remain unchanged.
- Strict cross-process gate remains **FAIL**: same GPU A/B/A diagnostic preserved at `/group/worldcept/code/SimAny/outputs/icra2027/20260906-2c197a2-v1/sim_recon_sim/scale_up/diagnosis.json`, source2830e98, allocation839484. All6 original-process repeats and input hashes match exactly; 0/6 independent-process comparisons match, maximum difference0.0110527861. Eighteen complete50x12 arrays archived. Classification: cross-process runtime realization/environment issue; JAX/XLA operator-level cause remains unproven. No native episodes were run by the diagnostic second server; it was stopped afterward.
- Parent-approved prospective protocol `per_canonical_engine_v1` blocks all arms/resets of one canonical instance on one policy process. This is a declared protocol revision, not a relaxed numerical gate. No native REF may be reused across engine processes.
- Server flags: `--engine-id <planned-id> --engine-protocol per_canonical_engine_v1 --canonical-instance-id <actual-instance-id>`. Metadata adds `policy_engine` with a fresh process UUID, canonical ID, source hashes/commit, package/device/environment runtime record and fingerprint. Config declares `policy_engine_protocol`; optional `policy_engine_id` pins the planned engine name.
- Canonical paired runner validates engine binding before native import/reset. Existing exact policy identity comparison rejects a REF from another process. Canonical episode ledger and episode identity retain the engine; fixed-action replay retains its source engine. Restarting the same named engine creates another process UUID and cannot reuse old REF.
- Smoke: native Python `-m pytest -q tests/test_roundtrip*.py --basetemp outputs/engine-integrated-tests`:271 passed,3 skipped,15.37s. Added paired closed-loop/replay regression suite: `-m pytest -q tests/test_roundtrip_native_v2_runner.py tests/test_roundtrip_policy_engine.py --basetemp outputs/engine-pair-tests`:12 passed,1.20s; rejects process restart before native mutation and preserves legacy behavior.
- Real versioned-server pilot: NOT_RUN, owned jointly with N1 local wrapper. No TEST execution admission until same-process interleaving and real DEV paired pilot pass. Checkpoint receipt remains `e4ef1d5f325841510ae31a4078cbdc10d3669b140e6397afebed08fb45f381ac`.
