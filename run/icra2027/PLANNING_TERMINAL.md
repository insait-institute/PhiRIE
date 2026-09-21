# Planning applicability audit

`e4_planning_terminal` validates the original source/E0/config and both sealed
automatic populations using their genuine original code checkout. It invokes
the existing pure AABB planner, accepts only the known no-size-admissible-target
failure, and propagates every unrelated validation or planning error. Existing
qualifier rows are authenticated separately; an unavailable qualifier is null.

The result's `status=PASS` means audit integrity. Its applicability gate is
`FAIL` and execution is `NOT_RUN`. It supplies no new task definition, reset
bank, camera measurement, policy outcome, or rollout ledger. The existing
compact planned-coverage producer may consume this receipt while retaining all
fixed episode identities and the complete semantic population.

The audit runs read-only original validators. Publishing uses exclusive create;
validation replays the source closure and exact payload. Use a new output path
for every audit. Neither the historical source nor experiment artifacts are
modified.
