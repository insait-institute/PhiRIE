#!/bin/bash
# Run run/selftest_gpu.py once on each GPU type of the cluster (via srun,
# all four in parallel) and print a PASS/FAIL matrix.
#
#   run/selftest_all_gpus.sh            # all four
#   run/selftest_all_gpus.sh a6000 h200  # subset
#
# Reports: outputs/selftest/<gpu>.json (+ <gpu>.log with the srun output).
# Batch-partition jobs may queue; each job is given $WAIT_MIN (default 25) min
# wall including queue time and is reported TIMEOUT otherwise.
set -uo pipefail

STUDIO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$STUDIO_DIR/env.sh"
OUT_DIR=${STUDIO_SELFTEST_DIR:-$PHYSICALVIEW_ROOT/outputs/selftest}
WAIT_MIN=${WAIT_MIN:-25}
mkdir -p "$OUT_DIR"

# gpu -> "partition gres extra-srun-flags"
declare -A SPEC=(
  [a6000]="debug a6000"                          # hala, CC 8.6, 48 GB
  [a100]="batch a100-80g"                        # gcp-eu1-a100-80g-*, CC 8.0, 80 GB
  [h200]="batch h200 --exclude=msp3-[0-7]"       # sof1-h200-*, CC 9.0 (msp3-* lack /group/worldcept)
  [rtx6000]="batch rtx6000"                      # gcp-eu1-rtx6000-*, RTX PRO 6000 Blackwell, CC 12.0
)
GPUS=("$@"); [ ${#GPUS[@]} -eq 0 ] && GPUS=(a6000 a100 h200 rtx6000)

declare -A PIDS
for g in "${GPUS[@]}"; do
  [ -n "${SPEC[$g]:-}" ] || { echo "unknown gpu '$g' (known: ${!SPEC[*]})"; exit 2; }
  read -r part gres extra <<<"${SPEC[$g]}"
  rm -f "$OUT_DIR/$g.json" "$OUT_DIR/$g.rc"
  (
    cd "$PHYSICALVIEW_ROOT"
    # shellcheck disable=SC2086
    timeout --signal=INT --kill-after=30 "$((WAIT_MIN * 60))" \
      srun --partition="$part" --gres="gpu:$gres:1" --ntasks=1 --cpus-per-task=4 --mem=32G \
           --time=00:20:00 --job-name="studio-selftest-$g" $extra \
           bash -lc "source '$STUDIO_DIR/env.sh' && \"\$STUDIO_PY\" '$STUDIO_DIR/selftest_gpu.py' --out '$OUT_DIR/$g.json'"
    rc=$?; echo "srun exit: $rc"; echo "$rc" > "$OUT_DIR/$g.rc"
  ) > "$OUT_DIR/$g.log" 2>&1 &
  PIDS[$g]=$!
  echo "submitted $g  (partition=$part gres=gpu:$gres:1 $extra) -> $OUT_DIR/$g.log"
done

declare -A RC
for g in "${GPUS[@]}"; do wait "${PIDS[$g]}"; RC[$g]=$?; done

# ---- matrix -----------------------------------------------------------------
printf '\n%-8s %-8s %-34s %-4s %-6s %-6s %-10s %-9s %-9s  %s\n' \
  GPU STATUS DEVICE CC TORCH GSPLAT "SO_ARCHS" "RENDER_ms" "MJ_ms" "FAILED/NOTES"
fail=0
for g in "${GPUS[@]}"; do
  f="$OUT_DIR/$g.json"
  if [ -f "$f" ]; then
    "$STUDIO_PY" - "$f" "$g" <<'PY' || fail=1
import json, sys
r = json.load(open(sys.argv[1])); c = r["checks"]
def okc(k): v = c.get(k, {}).get("ok"); return "ok" if v else ("skip" if v is None else "FAIL")
gs = c.get("gsplat", {})
archs = ",".join(a.replace("sm_", "") for a in gs.get("so_archs", [])) or "-"
print(f"{sys.argv[2]:<8s} {'PASS' if r['pass'] else 'FAIL':<8s} {c['gpu'].get('name','?')[:34]:<34s} "
      f"{c['gpu'].get('compute_capability','?'):<4s} {okc('torch'):<6s} {okc('gsplat'):<6s} {archs:<10s} "
      f"{str(gs.get('render_640x360_ms','-')):<9s} {str(c.get('mujoco',{}).get('render_320x240_ms','-')):<9s}  "
      f"{','.join(r['failed_critical']) or r['host']}")
sys.exit(0 if r["pass"] else 1)
PY
  else
    rc=$(cat "$OUT_DIR/$g.rc" 2>/dev/null || echo "${RC[$g]}")
    status=TIMEOUT; [ "$rc" -ne 124 ] && status="NO-REPORT(rc=$rc)"   # 124 = `timeout` fired
    printf '%-8s %-8s %s\n' "$g" "$status" "see $OUT_DIR/$g.log"
    fail=1
  fi
done
exit $fail
