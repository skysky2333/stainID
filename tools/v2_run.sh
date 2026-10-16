#!/usr/bin/env bash
# The v2 detection run, stage by stage. Every stage is resumable and redoes only core images that are missing or out of
# date (a changed input, parameter, model or upstream stage), so the same command continues a stopped run.
#
#   tools/v2_run.sh all        stage 2 proposals (CPU, $WORKERS core images at once), stage 4 calls, stage 3 outlines
#                              (one process per GPU, $GPUS GPUs), stage 5 measures
#   tools/v2_run.sh decide     only stages 4 and 5: after a new model, cut-off or rule (minutes, no image is read)
#   tools/v2_run.sh proposals | calls | outlines | measures | status    one stage (or how far each has got)
#
# Extra arguments go to every stage, e.g. `tools/v2_run.sh all --core cores.txt` or `--stain AT8`.
# Settings (environment): STAINID_PROJECT (the project folder), STAINID_PYTHON (python with stainid installed),
# STAINID_SRC (a source tree to run instead of the installed one), WORKERS (default 10: ~2.2 GB each), GPUS (default 2),
# OUTLINES=0 to skip Segment Anything. Logs: <project>/data/analysis/v2/logs/.
set -euo pipefail
what=${1:-all}
shift || true
PROJECT=${STAINID_PROJECT:-$(pwd)}
PY=${STAINID_PYTHON:-python}
WORKERS=${WORKERS:-10}
GPUS=${GPUS:-2}
OUTLINES=${OUTLINES:-1}
LOGS="$PROJECT/data/analysis/v2/logs"
mkdir -p "$LOGS"
if [[ -n "${STAINID_SRC:-}" ]]; then export PYTHONPATH="$STAINID_SRC${PYTHONPATH:+:$PYTHONPATH}"; fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

v2() {  # v2 <stage> [args…]: one stage, its output appended to the stage's log
  local stage=$1
  shift
  echo "== $(date '+%F %T') stage $stage $*" | tee -a "$LOGS/$stage.log"
  "$PY" -m stainid.cli --project "$PROJECT" v2 "$stage" "$@" 2>&1 | grep --line-buffered -v Warning | tee -a "$LOGS/$stage.log"
}

outlines() {  # stage 3 on every GPU at once, one process each
  local pids=()
  for ((i = 0; i < GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i "$PY" -m stainid.cli --project "$PROJECT" v2 outlines --device cuda --shard-index "$i" --shard-count "$GPUS" "$@" \
      > >(grep --line-buffered -v Warning >> "$LOGS/outlines_gpu$i.log") 2>&1 &
    pids+=($!)
  done
  echo "== $(date '+%F %T') stage outlines on $GPUS GPUs (logs: $LOGS/outlines_gpu*.log)"
  local failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  return $failed
}

case $what in
  all)
    v2 proposals --workers "$WORKERS" "$@"
    v2 calls "$@"
    if [[ $OUTLINES == 1 ]]; then outlines "$@"; fi
    v2 measures
    v2 status ;;
  decide)
    v2 calls "$@"
    v2 measures
    v2 status ;;
  proposals) v2 proposals --workers "$WORKERS" "$@" ;;
  outlines) outlines "$@" ;;
  calls | measures | status) v2 "$what" "$@" ;;
  *) echo "unknown: $what (all, decide, proposals, calls, outlines, measures, status)" >&2; exit 2 ;;
esac
