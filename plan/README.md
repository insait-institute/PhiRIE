# SimAnyRoom Execution Plans

The authoritative plan for the current paper is:

## [`plan/icra2027/README.md`](icra2027/README.md)

Paper title:

> **SimAnyRoom: Making Rooms Simulatable through Agentic Real-to-Sim**

The current plan is organized as one self-contained `README.md` per coding-agent task:

- shared experiment freeze;
- 50-room construction evaluation;
- held-out PSNR/SSIM/LPIPS/CD/F1 evaluation;
- controlled agentic-construction ablation;
- paired closed-loop manipulation;
- Harmonizer Option C;
- task-local support and risk-coverage;
- DROID and phone capture-to-simulator evaluation;
- matched physical trials, when hardware exists;
- reproducible paper-table freeze;
- hero demo, teaser, loop, and live presentation package.

Start every new agent at `plan/icra2027/README.md`, then assign exactly one task directory. The agent must create a `STATUS.md` in that directory before handoff.

## Legacy design documents

The flat numbered files in this directory (`00_...md` through `23_...md`) record earlier predictive-simulator, phone-capture, PolaRiS, certificate, repair, and Harmonizer planning. They remain useful technical references, but they are **not** the current task index. Do not dispatch an agent from a legacy file unless the corresponding ICRA 2027 task README links to it.

## Non-negotiable rule

No paper value is entered manually. All experiment tasks feed the canonical producers under `robo/eval/`, and `robo.eval.paper_pipeline` creates the final table/provenance freeze.