# coding_agents/

This repository was built in collaboration with an AI coding agent
([Claude Code](https://claude.ai/code)). This directory makes that part of the
project traceable: the agent's persistent **memory**, the project **history**,
and the reusable operational **skills** it accumulated while building and
debugging the system on the cluster.

```
memory/    point-in-time snapshots of the agent's persistent memory notes
           (facts about the code, data layouts, tuned thresholds, and the
           hard-won environment gotchas behind design decisions)
history/   chronological project log (HISTORY.md) plus raw working notes
skills/    distilled playbooks: how to verify refactors, run cluster jobs,
           and launch every pipeline mode without rediscovering the traps
```

## How this worked

The agent keeps one markdown file per durable fact in a memory directory that
persists across sessions (`~/.claude/projects/<workspace>/memory/`). Each file
has a `name`, a one-line `description` used for recall, and a body that records
not just *what* was decided but *why* and *how to apply it*. Files link to each
other with `[[wiki-style]]` references.

`memory/` here is a **snapshot taken 2026-08-04** of every note relevant to
this repository. They are historical documents: claims about code paths reflect
the moment they were written (several predate the SimFoundry→SimAny rename and
the PhiRoom reorganization, and use old module paths). They are kept verbatim
because they explain decisions the polished docs only state.

## Reading order

1. [history/HISTORY.md](history/HISTORY.md) — what happened, in order.
2. [memory/simfoundry-repro-status.md](memory/simfoundry-repro-status.md) —
   the main technical log: pipeline results, audits, and fixes.
3. [memory/pi05-droid-sim-eval.md](memory/pi05-droid-sim-eval.md) — the robot
   closed-loop: recipes, failures, and honest numbers.
4. [skills/](skills/) — when you need to actually run something.
