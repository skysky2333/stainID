# The job system

Every long piece of work on this computer goes through one queue: the steps people start in the app, the commands
scripts and agents submit with `stainid jobs submit`, and the steps typed in a terminal (they register themselves).
Code: `stainid.jobs` (records), `stainid.jobs.queue` (scheduler, budget, launchers), `stainid.jobs.units` (pause
points), `stainid.jobs.cli` (`stainid jobs …`), `stainid.api.jobs` (one project's view), `webapp/src/queue/QueuePanel.tsx`
(Run page).

## Two levels of priority

1. **Interactive requests** (opening a field, saving labels, the Home page) never wait for the queue. They run in the
   app's request threads and its small background pool (`stainid.api.work`), and the app's service has CPUWeight and
   IOWeight 1000.
2. **Queued jobs**, by class, highest first:

   | Class | What | Nice / CPUWeight / IOWeight | Grace to reach a unit boundary |
   |---|---|---|---|
   | annotation | label-field preparation, applying area edits | 0 / 400 / 400 | 300 s |
   | pipeline | the main study's stages | 5 / 100 / 100 | 300 s |
   | models | training, model comparison, segmentation check | 10 / 50 / 50 | 120 s |
   | public | any project with a `dataset:` section (or `jobs: {class: public}` in its stainid.yaml) | 15 / 20 / 20 | 30 s |
   | downloads | network only; takes no processors, so it runs alongside | 19 / 10 / 10 | 10 s |

   Within a class jobs start first in, first out; the Run page's **Move up** and `stainid jobs priority` change that.

## How scheduling works

The queue follows the batch schedulers people know (Slurm, Kubernetes): jobs declare resources and time, run side by
side when they fit, and small ones flow around big ones without delaying them. Code: `stainid.jobs.policy` (rules),
`stainid.jobs.arrays` (chunks), `Scheduler._schedule` (one round, about once a second).

1. **Resources and size.** All jobs together share a budget: processors, GPUs, memory (default: this computer's
   processors and GPUs, its memory less the 18 GB the app may take and 3 GB), at most 2 io-heavy jobs per disk and at
   most 1 big public-dataset job at a time (`stainid jobs budget`, or `~/.stainid/queue/budget.json`:
   `{"cpus": 24, "ram_gb": 30, "io_per_disk": 2, "caps": {"public": {"big": 1}}}`). Each job declares cpus, gpus,
   ram, io (light or heavy) and network (`STEP_NEEDS` and `policy.IO_HEAVY` per step; `--cpus/--gpus/--ram/--io/--network`
   on submit). Its run time is what it declared (`--time 2h`), else its units (tiles, core images × stains) × the time
   per unit its step took before (this project's, then any project's, then a default), else how long the step took
   when it last finished. Small = at most 4 processors, no GPU, 8 GB and 15 minutes; anything else is big. The Run
   page shows "small · ~2 min" or "big · ~5 h".
2. **Order.** By class, highest first; within a class the next job comes from the project (for annotation work: the
   project and the person) with the fewest jobs running (fair share), then first in, first out.
3. **Side by side.** A job starts as soon as what it needs is free (bin packing over processors, GPUs and memory).
   Two jobs that write the same outputs never run at once: the same project, an output both make, overlapping stains
   and core images (or shards), or a shared `--conflict` key; the later one waits. Readers overlap freely, and a step
   waits for the steps whose outputs it reads (queued before it).
4. **Backfill (EASY).** The first job of each class that does not fit gets a reservation: when it is expected to start
   (the running jobs' expected ends, from their progress or estimates) and what will be left over then. A job further
   down starts now only if it fits now and either ends before that time or uses only what is left over, so it never
   delays the reserved job. A job without an estimate only uses what is left over.
5. **Pre-emption.** When a job would fit once lower-class jobs stopped, those are asked to stop at their next unit
   boundary, stopped hard (SIGTERM, then SIGKILL 30 s later) after their class's grace, and queued again at their
   place. GPUs are handed out by index (`CUDA_VISIBLE_DEVICES`).
6. **Arrays.** Find nuclei and the v2 proposals, calls and outlines over many core images run as a job array: one chunk
   per TMA (a TMA of more than 200 core images in parts), or per 50 core images for image sets and slides, each the same
   command with `--core <the chunk's list>` (`--array tma|N`, `--no-array`). Chunks run side by side when they fit, are
   pre-empted one at a time, and have their own progress and estimate; the parent shows the overall progress, and
   Pause, Resume, Cancel and Move act on all its chunks. A next stage's chunks wait for every chunk of the stage before.

Work outside the queue counts too, looked at every 10 s: a job starts only while the computer has its memory free (plus
2 GB), a GPU on which a program the queue did not start (not the app) uses 1.5 GB or more is not handed out, and the
public and download classes start only while the load average leaves their processors spare.

## Units: pause, resume, crash

Every step works in units (a tile, a core image × stain, a slide, a file, a field) and writes each unit atomically
(the v2 stages also write its manifest), so a new run skips what is done. Between units a step calls
`units.boundary()`: when the queue has written "pause" or "cancel" in the job's control file, the step exits with code
75 there. Pause = stop at the next boundary; Resume = run the same command again; a kill or crash loses at most the
units in flight.

| Step | Unit | Resumes by |
|---|---|---|
| Find nuclei | tile | skipping tiles whose `.npz` exists |
| Calibrate | slide | keeping each slide's threshold as soon as it is known |
| Prepare images | picture at the analysis scale | keeping made pictures |
| v2 proposals / calls / outlines | core image × stain | manifests (a paused `--redo` keeps what it redid: `STAINID_JOB_SINCE`) |
| Prepare fields, segmentation check | field | the saved pools / candidates |
| Download models | 1 MB chunk | an HTTP range request on the `.part` file |
| Train a model, `stainid score` | the whole run (a random forest of under a minute; scoring reads saved detections) | running again |

Training has no epochs (random forests), so there is no per-epoch checkpoint; a new neural-network trainer must save one
each epoch and call `units.boundary()` after it. A command that is not a stainID step (`--exec`) is stopped with SIGTERM
after at most 30 s and must itself skip finished work when run again (the download scripts do).

## Where jobs run, and restarts

Records live in `~/.stainid/queue` (`STAINID_QUEUE_DIR`): `<id>.json`, its log, exit code, pid file and control file.
One process schedules at a time: the one holding `.scheduler.lock`, normally the app (`stainid serve`), else
`stainid jobs serve`. Other processes only read the records and leave requests in `requests/`. Each job runs as a
transient systemd user service `stainid-job-<id>` in `stainid-jobs.slice` (memory capped at the budget), not as a child
of the app, so restarting the app does not stop it; the next scheduler adopts it. A job that vanished without an exit
code (a reboot, a killed unit) is queued again and continues where it stopped; queued and paused jobs simply wait.
Without systemd (macOS, tests) jobs run as processes in their own session.

The user's systemd instance only runs while they are logged in unless lingering is on: for the queue (and the app) to
run after a reboot without a login, run once `sudo loginctl enable-linger sky`.

## Commands

```bash
stainid --project P jobs submit -- nuclei --stain AT8 --device cuda       # a workflow step (class from the project)
stainid --project P jobs submit --class models -- score                   # any stainid command of the project
stainid jobs submit --exec --class downloads --network --cwd DIR --title "Big downloads" -- bash download_big.sh
stainid --project P jobs submit -- v2 proposals --workers 8              # an array: one chunk per TMA (--array 50, --no-array)
A=$(stainid --project P jobs submit --id-only -- v2 proposals --core cores.txt)
stainid --project P jobs submit --after "$A" -- v2 calls --core cores.txt # starts once A has finished (skipped if A fails)
stainid jobs submit --exec --time 20m --io heavy --conflict audit-out --cpus 4 --ram 8 -- python audit.py   # size, disk, conflicts
stainid jobs list [--all] [--json]          stainid jobs show ID
stainid jobs pause ID… | resume ID… | cancel ID…
stainid jobs priority ID --up | --down | --top | --class public
stainid jobs pause-all | resume-all         # the Run page's "Pause all background work"
stainid jobs budget [--cpus N --gpus N --ram GB]
stainid jobs serve                          # run the queue without the app
```

The Run page shows the queue grouped by class with each job's place, progress, time left and Pause / Resume / Cancel /
Move up, its size and expected run time ("small · ~2 min", "big · ~5 h"), an array as one row with its chunks, and the
budget in use.
