#!/bin/bash
# Run the Cellpose-SAM part of detection on a CUDA machine, then finish detection on this computer.
# Cellpose is the slow step (about 90 s per tile on an Apple GPU, a few seconds on a data-centre GPU); everything after
# it reads the saved masks, so only the masks travel back.
#
#   tools/gpu_cellpose.sh package /some/folder                  # from the project folder: a self-contained folder to move yourself
#   tools/gpu_cellpose.sh push user@server:/scratch/stainid     # or copy the same files straight to a server
#   tools/gpu_cellpose.sh run [number of GPUs]                  # on the GPU machine, inside that folder
#   tools/gpu_cellpose.sh pull /some/folder                     # back in the project folder: copy the masks home (local folder or user@server:path)
#
# Then run the detection steps here as usual (Workflow page, or stainid neun / fields); they reuse the masks.
set -euo pipefail
action=${1:?push, run or pull}

case "$action" in
  package|push)
    dest=${2:?destination folder, or user@server:/scratch/stainid}
    link=()
    if [ "$action" = package ]; then
      mkdir -p "$dest"
      link=(--link-dest="$PWD")
    fi
    list=$(mktemp)
    .venv/bin/python - "$list" <<'EOF'
import sys
from stainid.project import load_project
from stainid.tables import read_csv
project = load_project(".")
manifest = project.analysis_tiles()
paths = {project.relative(manifest), project.relative(project.model("cellpose")), "stainid.yaml"}
paths |= {row["image_path"] for row in read_csv(manifest)}
open(sys.argv[1], "w").write("\n".join(sorted(paths)) + "\n")
EOF
    rsync -a --exclude __pycache__ src tools pyproject.toml README.md LICENSE "$dest/"
    rsync -a ${link[@]+"${link[@]}"} --files-from="$list" -r ./ "$dest/"
    rsync -a data/analysis/tile_nuclei "$dest/data/analysis/"
    rsync -a data/analysis/cohort_neun/cellpose_masks/ "$dest/data/analysis/tile_nuclei/"
    rm "$list"
    echo "ready: $dest (then on the GPU machine: cd into it and run tools/gpu_cellpose.sh run <number of GPUs>)"
    ;;
  run)
    gpus=${2:-1}
    [ -d .venv ] || { python3 -m venv .venv && .venv/bin/pip install -q -e ".[deep]"; }
    for ((i = 0; i < gpus; i++)); do
      CUDA_VISIBLE_DEVICES=$i nohup .venv/bin/stainid nuclei --stain NeuN --stain AT8 --stain 6E10 --device cuda --batch-size 32 \
        --shard-index "$i" --shard-count "$gpus" > "cellpose_gpu_$i.log" 2>&1 &
    done
    wait
    echo "done: $(ls data/analysis/tile_nuclei | wc -l) mask files"
    ;;
  pull)
    src=${2:?source, e.g. user@server:/scratch/stainid}
    rsync -a "$src/data/analysis/tile_nuclei/" data/analysis/tile_nuclei/
    .venv/bin/python - <<'EOF'
import shutil
from stainid.project import load_project
from stainid.tables import read_csv
project = load_project(".")
masks = project.output("neun") / "cellpose_masks"
masks.mkdir(parents=True, exist_ok=True)
copied = 0
for row in read_csv(project.analysis_tiles()):
    source = project.output("nuclei") / f"{row['tile_id']}.npz"
    if row["stain"] == "NeuN" and source.exists() and not (masks / source.name).exists():
        shutil.copy2(source, masks / source.name)
        copied += 1
print(f"{copied} NeuN Cellpose masks placed for the NeuN step")
EOF
    ;;
  *)
    echo "usage: $0 push|run|pull ..." >&2
    exit 1
    ;;
esac
