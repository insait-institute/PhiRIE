# Skill: running jobs on this cluster

Accumulated slurm/GPU facts that repeatedly decided whether a job worked.

## Partitions and nodes

- Interactive debugging: `srun -p debug --gres=gpu:a6000:1 --time=240 ...`.
  The debug QoS caps **4 concurrent GPU jobs per user** (AssocGrpGRES).
- `hala` is the A6000/sm_86 node. The prebuilt gsplat JIT cache holds
  **sm_86 + sm_90 only** — pin `TORCH_CUDA_ARCH_LIST="8.6;9.0+PTX"`
  (exactly; a login profile leaking `9.0+PTX` alone changes the extension
  hash and forces a silent full rebuild). `run/env.sh` sets this.
- **batch/H200 nodes (msp3-\*) do not mount /group/worldcept** nor shared
  /home — unusable for this project except self-contained rescues.
- Qwen inpainting exceeds the 4 h debug wall — use `-p batch -w hala`
  (`run/slurm/inpaint_scene.sbatch`).

## Memory and storage

- **Always set `#SBATCH --mem`** in the sbatch file, not the command line.
  The cgroup default is 2 GB and OOM-kills large restores with a misleading
  exit (job 645991: `OUT_OF_MEMORY`, code 0:125). The pi0.5 server needs
  `--mem=100G` (12 GB checkpoint staged in host RAM by jax).
- **orbax/zarr restore vs CephFS**: many-small-scattered-reads can hang for
  an hour with zero progress and zero errors, while a plain rsync of the
  same 12 GB finishes in ~9 s. Fix: copy to node-local `/scratch/<user>/...`
  gated on a `.copy_complete` marker (`run/pi05_serve.sh`); restore drops to
  ~15 s. /scratch is node-local — the copy only helps on that node.
- gsplat JIT fallback wants gcc ≤ 13; `run/env.sh` probes g++-12/13 and sets
  `NVCC_APPEND_FLAGS=-allow-unsupported-compiler`.

## Long-running services

- Concurrent jobs can share a node: derive ports from the job id
  (`PORT=$((8000 + SLURM_JOB_ID % 1000))`), and note tyro wants `--port`
  *before* the `policy:checkpoint` subcommand.
- First inference after connect triggers jax jit (minutes); websocket
  keepalive (20 s) kills the connection during it. The client must
  reconnect-retry — first infer typically needs **3** reconnects; the
  server-side handshake tracebacks in that window are noise. Retry cap 5.
- When waiting on a spawned process from an agent, use a real blocking wait
  tied to a PID (`while kill -0 $PID; do sleep N; done`) — assuming a
  background notification will arrive at a turn boundary silently stalls.

## Scene-scale planning numbers

11–17 min/scene on one A6000 (GT vs auto mode), ~40–55 s/object; the
202-book library scene needs ~3 h. Full 1018-scene extrapolation ≈ 215
A6000 h — decided against; the 50-scene val set is the protocol.
