# Skill: verifying a repo-wide refactor

Used for the 07-27 package refactor and the 08-04 PhiRoom reorganization;
both times it caught real regressions that review alone missed.

## Import-smoke baseline diff

1. **Before** touching anything, snapshot which modules import cleanly:
   `bash run/smoke_imports.sh > /tmp/smoke_pre.txt` (one interpreter process
   per module — stage modules mutate `sys.path`, so importing them all in one
   process cross-contaminates).
2. Refactor.
3. Re-run and diff against the baseline **by mapped module name**. The success
   criterion is not "everything imports" — some failures are environmental
   (viser and omnigibson are not in `.venv`; one bridge script parses args at
   import). The criterion is: *the failure set is exactly the baseline's*.

The 07-27 run caught a `C.env` → `_C.env` typo and a repo-root
`parents[1]` → `parents[2]` depth bug. The 08-04 run confirmed 76/80 with the
same 4 environmental failures.

## What greps miss

- **`Path(__file__).parents[N]`** — audit every file whose *directory depth
  changed*, not just files whose imports changed.
- **`from pkg import name` forms** — a `pkg.module` → `newpkg.module` regex
  does not rewrite `from pkg import module`; list those forms explicitly.
- **Prefixes baked into helpers** — `env.sh`'s `run()` prepended the package
  name, so call sites looked clean while being wrong.
- **.gitignore inline comments** — `weights/   # comment` is NOT a valid
  gitignore pattern; the rule silently matches nothing.
- **One-time cache markers** — `.snapshot_done`-style markers keep serving
  stale pre-fix outputs after a fix; delete the cache dirs before rerunning.
- **cwd assumptions** — outputs metadata may hard-code relative paths
  (`../outputs/...`), pinning what the working directory must be.

## Shell scripts

`bash -n` every `.sh`/`.sbatch` after mechanical edits, and grep for the old
tokens with word boundaries (`\bsimany\b`) rather than substrings.
