#!/bin/bash
#SBATCH --job-name=simany-harness
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/harness-%j.out
#SBATCH --error=logs/harness-%j.err

set -euo pipefail
mkdir -p logs

CONFIG=${CONFIG:-configs/experiments/harness_paper.yaml}
OUT=${OUT:-outputs/harness_runs/paper_main}

# Start the policy server and Harmonizer service separately, or add the
# site-specific launch commands here. The harness itself remains fail-closed:
# unavailable services become explicit terminal outcomes.
srun python -m robo.eval.harness_runner \
  --config "$CONFIG" \
  --out "$OUT"
