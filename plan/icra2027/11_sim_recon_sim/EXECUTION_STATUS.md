# Sim-recon-sim: first native comparison complete

Updated2026-09-06 18:26UTC. See [STATUS.md](STATUS.md) for measured results, exact commands, hashes, negative stages and limits. Current result package20260906-6c50f76-v1; native pilot and scientific stage commits are explicitly separate. Job837413 released after all required GPU work. Prior publication unchanged.

| task_id | owner_or_agent | branch | state | dependency_state | smoke_state | pilot_state | full_state | latest_commit | running_job_ids | output_path | blocker | claim_gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|SR0|root/e4_endpoints|main|PILOT_PASSED|First milestone satisfied; broader dependencies remain|SMOKE_PASSED|PILOT_PASSED|NOT_RUN|67ee364|none|20260906-4ee6462-v2/reference/pilot|Broader declared matrix not scheduled|PASS native interface/identity|
|SR1|e4_planning|main|SMOKE_PASSED|First milestone satisfied; broader dependencies remain|SMOKE_PASSED|NOT_RUN|NOT_RUN|67ee364|none|20260906-67ee364-v2|Broader declared matrix not scheduled|PASS static6TRAIN/2TEST|
|SR2|sr2_import|main|PILOT_PASSED|First milestone satisfied; broader dependencies remain|SMOKE_PASSED|PILOT_PASSED|NOT_RUN|85051d4|none|20260906-85051d4-v1|Broader declared matrix not scheduled|PASS one target build/import|
|SR3|root|main|NOT_RUN|First milestone satisfied; broader dependencies remain|NOT_RUN|NOT_RUN|NOT_RUN|85051d4|none|none|Broader declared matrix not scheduled|NOT_RUN verification/repair|
|SR4|root|main|NOT_RUN|First milestone satisfied; broader dependencies remain|NOT_RUN|NOT_RUN|NOT_RUN|85051d4|none|none|Broader declared matrix not scheduled|NOT_RUN GS intervention|
|SR5|e4_endpoints|main|PILOT_PASSED|First milestone satisfied; broader dependencies remain|SMOKE_PASSED|PILOT_PASSED|NOT_RUN|85051d4|none|20260906-85051d4-v2/replay|Broader declared matrix not scheduled|PASS execution;FAIL reconstructed native predicate at392steps|
|SR6|e4_endpoints|main|PILOT_PASSED|First milestone satisfied; broader dependencies remain|SMOKE_PASSED|PILOT_PASSED|NOT_RUN|85051d4|none|20260906-85051d4-v2/paired|Broader declared matrix not scheduled|PASS one same-policy pair;NOT_RUN broad claim|
|SR7|root|main|NOT_RUN|First milestone satisfied; broader dependencies remain|NOT_RUN|NOT_RUN|NOT_RUN|85051d4|none|none|Broader declared matrix not scheduled|NOT_RUN BEHAVIOR|
|SR8|sr2_import/root|main|SMOKE_PASSED|First milestone satisfied; broader dependencies remain|SMOKE_PASSED|NOT_RUN|NOT_RUN|6c50f76|none|20260906-6c50f76-v1|Broader declared matrix not scheduled|PASS milestone package;NOT_RUN paper/demo full|
