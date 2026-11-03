#!/usr/bin/env bash
# The v2 detection run, stage by stage. Every stage is resumable and redoes only core images that are missing or out of
# date (a changed input, parameter, model or upstream stage), so the same command continues a stopped run.
#
#   tools/v2_run.sh all        stage 2 proposals (CPU, $PROCESSES tile processes), stage 4 calls, stage 3 outlines
#                              ($PER_GPU processes per GPU, $GPUS GPUs), white matter and grey-matter burdens (only with
#                              v2.white_matter.analysis: exclude), stage 5 measures
#   tools/v2_run.sh decide     only stages 4 and 5 (and, with exclude, white matter and burdens, which redo only what the
#                              new calls changed): after a new model, cut-off or rule (minutes without exclude, no image read)
#   tools/v2_run.sh proposals | calls | outlines | white-matter | grey-matter | measures | status    one stage (or how far each has got)
#
# Extra arguments go to every stage, e.g. `tools/v2_run.sh all --core cores.txt` or `--stain AT8`.
# Settings (environment): STAINID_PROJECT (the project folder), STAINID_PYTHON (python with stainid installed),
# STAINID_SRC (a source tree to run instead of the installed one), PROCESSES (stage 2 tile processes, default -1: a quarter
# of the processors, where memory bandwidth tops out, ~1.3 GB each; 0: the older layout, one process per core image), WORKERS (stage 2
# core images in flight sharing the tile processes, default 3, ~1.4 GB each; with PROCESSES=0 processes of ~2.2 GB),
# THREADS (stage 2 threads per tile process, default 1), TILES_AT_ONCE (with PROCESSES=0: tiles in threads per process),
# GPUS (default 2), PER_GPU (stage 3 processes per GPU, default 3: ~3.3 GB of memory each), GREY_WORKERS (grey-matter burdens:
# core images at once, default 4, ~2.5 GB each). None of them changes any result.
# OUTLINES=0 to skip Segment Anything. Logs: <project>/data/analysis/v2/logs/.
set -euo pipefail
what=${1:-all}
shift || true
PROJECT=${STAINID_PROJECT:-$(pwd)}
PY=${STAINID_PYTHON:-python}
WORKERS=${WORKERS:-3}
PROCESSES=${PROCESSES:--1}
GPUS=${GPUS:-2}
PER_GPU=${PER_GPU:-3}
OUTLINES=${OUTLINES:-1}
GREY_WORKERS=${GREY_WORKERS:-4}
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

outlines() {  # stage 3 on every GPU at once, PER_GPU processes each (a process is mostly CPU work between GPU calls)
  local pids=() shards=$((GPUS * PER_GPU))
  for ((i = 0; i < shards; i++)); do
    CUDA_VISIBLE_DEVICES=$((i % GPUS)) "$PY" -m stainid.cli --project "$PROJECT" v2 outlines --device cuda --shard-index "$i" --shard-count "$shards" "$@" \
      > >(grep --line-buffered -v Warning >> "$LOGS/outlines_gpu$((i % GPUS))_$((i / GPUS)).log") 2>&1 &
    pids+=($!)
  done
  echo "== $(date '+%F %T') stage outlines on $GPUS GPUs, $PER_GPU processes each (logs: $LOGS/outlines_gpu*.log)"
  local failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  return $failed
}

white_matter() {  # what stage 5 reads when the project leaves white matter out; every core image, redoing what is out of date
  if "$PY" -c 'import sys; from stainid.project import load_project; from stainid.v2 import white_matter; sys.exit(0 if white_matter.excluded(load_project(sys.argv[1])) else 1)' "$PROJECT"; then
    v2 white-matter
    v2 grey-matter --workers "$GREY_WORKERS"
  fi
}

case $what in
  all)
    v2 proposals --workers "$WORKERS" --processes "$PROCESSES" --threads "${THREADS:-0}" --tiles-at-once "${TILES_AT_ONCE:-1}" "$@"
    v2 calls "$@"
    if [[ $OUTLINES == 1 ]]; then outlines "$@"; fi
    white_matter
    v2 measures
    v2 status ;;
  decide)
    v2 calls "$@"
    white_matter
    v2 measures
    v2 status ;;
  proposals) v2 proposals --workers "$WORKERS" --processes "$PROCESSES" --threads "${THREADS:-0}" --tiles-at-once "${TILES_AT_ONCE:-1}" "$@" ;;
  outlines) outlines "$@" ;;
  grey-matter) v2 grey-matter --workers "$GREY_WORKERS" "$@" ;;
  calls | white-matter | measures | status) v2 "$what" "$@" ;;
  *) echo "unknown: $what (all, decide, proposals, calls, outlines, white-matter, grey-matter, measures, status)" >&2; exit 2 ;;
esac
