#!/usr/bin/env bash
# Monitor an E3 ordinary-job ledger produced by submit_e3_noarray.sh.
# This script is read-only: it queries Slurm and never changes jobs or files.
set -euo pipefail

ROOT=/group/worldcept/PhiRIE/code/SimAny
SQUEUE_BIN=${SQUEUE_BIN:-squeue}
SACCT_BIN=${SACCT_BIN:-sacct}
WATCH=false
LEDGER=""
INTERVAL=${E3_MONITOR_INTERVAL_S:-30}

usage() {
  echo "usage: bash run/icra2027/monitor_e3_noarray.sh --ledger PATH [--watch]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ledger)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      LEDGER="$2"
      shift 2
      ;;
    --watch)
      WATCH=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown monitor argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

[[ -n "$LEDGER" ]] || { usage; exit 2; }
if [[ "$LEDGER" != /* ]]; then
  LEDGER="$ROOT/$LEDGER"
fi
case "$LEDGER" in
  "$ROOT"/outputs/icra2027/submissions/*/jobs.tsv) ;;
  *)
    echo "ledger must be a jobs.tsv below the repository E3 submission root: $LEDGER" >&2
    exit 2
    ;;
esac
RELATIVE="${LEDGER#"$ROOT"/}"
CURRENT="$ROOT"
IFS=/ read -r -a PARTS <<< "$RELATIVE"
for PART in "${PARTS[@]}"; do
  [[ -n "$PART" && "$PART" != . && "$PART" != .. ]] || {
    echo "unsafe ledger path component: $LEDGER" >&2
    exit 2
  }
  CURRENT="$CURRENT/$PART"
  [[ ! -L "$CURRENT" ]] || {
    echo "ledger path uses a symlink component: $CURRENT" >&2
    exit 2
  }
done
[[ -s "$LEDGER" && -f "$LEDGER" ]] || {
  echo "missing, empty, or non-regular E3 ledger: $LEDGER" >&2
  exit 2
}
EXPECTED_HEADER=$'sequence\tsubmitted_utc\tstage\tphase\tscene_id\tjob_id\tdependency\tprofile\tcpus\tmem\ttime\tpartition\tqos\tnodelist\tgres\tlauncher'
[[ "$(head -n 1 "$LEDGER")" == "$EXPECTED_HEADER" ]] || {
  echo "unexpected E3 ledger header: $LEDGER" >&2
  exit 2
}
awk -F '\t' '
  NR == 1 {next}
  NF != 16 || $1 != NR - 1 {bad=1}
  END {exit bad}
' "$LEDGER" || {
  echo "E3 ledger has a malformed row or non-contiguous sequence" >&2
  exit 2
}

mapfile -t JOB_IDS < <(awk -F '\t' 'NR > 1 {print $6}' "$LEDGER")
[[ ${#JOB_IDS[@]} -ge 1 && ${#JOB_IDS[@]} -le 154 ]] || {
  echo "E3 ledger must contain between 1 and 154 jobs; found ${#JOB_IDS[@]}" >&2
  exit 2
}
declare -A SEEN=()
for JOB_ID in "${JOB_IDS[@]}"; do
  [[ "$JOB_ID" =~ ^[0-9]+$ ]] || {
    echo "E3 ledger contains a non-ordinary job ID: $JOB_ID" >&2
    exit 2
  }
  [[ ! -v "SEEN[$JOB_ID]" ]] || {
    echo "E3 ledger contains duplicate job ID: $JOB_ID" >&2
    exit 2
  }
  SEEN[$JOB_ID]=1
done
JOB_CSV=$(IFS=,; printf '%s' "${JOB_IDS[*]}")

command -v "$SQUEUE_BIN" >/dev/null || {
  echo "squeue executable not found: $SQUEUE_BIN" >&2
  exit 2
}
command -v "$SACCT_BIN" >/dev/null || {
  echo "sacct executable not found: $SACCT_BIN" >&2
  exit 2
}
[[ "$INTERVAL" =~ ^[0-9]+$ && "$INTERVAL" -ge 5 && "$INTERVAL" -le 60 ]] || {
  echo "E3_MONITOR_INTERVAL_S must be an integer from 5 to 60" >&2
  exit 2
}

is_terminal() {
  case "$1" in
    COMPLETED|CANCELLED|FAILED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|\
    BOOT_FAIL|DEADLINE|REVOKED|SPECIAL_EXIT) return 0 ;;
    *) return 1 ;;
  esac
}

show_once() {
  local accounting
  local queue
  local raw_state
  local state
  local terminal=0
  local failed=0
  local missing=0
  local id
  declare -A STATES=()

  printf '\nE3 status at %s UTC (%d ordinary jobs)\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%S')" "${#JOB_IDS[@]}"
  if ! queue=$("$SQUEUE_BIN" -h -j "$JOB_CSV" \
    -o '%.18i %.12P %.42j %.10T %.10M %.10l %.24R'); then
    echo "squeue query failed" >&2
    return 13
  fi
  [[ -z "$queue" ]] || printf '%s\n' "$queue"

  if ! accounting=$("$SACCT_BIN" -X -n -P -j "$JOB_CSV" \
    -o JobIDRaw,JobName,State,ExitCode,Elapsed,NodeList); then
    echo "sacct query failed" >&2
    return 13
  fi
  printf '%s\n' "$accounting" | awk -F '|' '
    NF >= 3 {
      state=$3
      sub(/ .*/, "", state)
      sub(/\+.*/, "", state)
      count[state]++
    }
    END {
      printf "summary:"
      for (state in count) printf " %s=%d", state, count[state]
      print ""
    }'

  while IFS='|' read -r id _ raw_state _; do
    [[ -n "$id" ]] || continue
    state="${raw_state%% *}"
    state="${state%%+*}"
    STATES[$id]="$state"
  done <<< "$accounting"
  for id in "${JOB_IDS[@]}"; do
    state="${STATES[$id]:-}"
    if [[ -z "$state" ]]; then
      missing=$((missing + 1))
    elif is_terminal "$state"; then
      terminal=$((terminal + 1))
      [[ "$state" == COMPLETED ]] || failed=$((failed + 1))
    fi
  done
  printf 'terminal=%d/%d failed_terminal=%d accounting_pending=%d\n' \
    "$terminal" "${#JOB_IDS[@]}" "$failed" "$missing"

  if [[ $terminal -eq ${#JOB_IDS[@]} ]]; then
    [[ $failed -eq 0 ]] && return 10
    return 11
  fi
  return 12
}

while true; do
  set +e
  show_once
  STATUS=$?
  set -e
  case "$STATUS" in
    10)
      echo "E3 DAG complete: all ordinary jobs completed successfully"
      exit 0
      ;;
    11)
      echo "E3 DAG reached terminal state with one or more failures" >&2
      exit 1
      ;;
    12)
      [[ "$WATCH" == true ]] || exit 0
      sleep "$INTERVAL"
      ;;
    13)
      echo "E3 monitor could not query Slurm" >&2
      exit 2
      ;;
    *)
      echo "unexpected monitor status: $STATUS" >&2
      exit 2
      ;;
  esac
done
