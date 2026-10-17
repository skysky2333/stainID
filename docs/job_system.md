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

## Budget and pre-emption

All jobs together share a budget: processors, GPUs and memory (default: this computer's processors and GPUs, its memory
less the 18 GB the app may take and 3 GB; `stainid jobs budget --cpus 24 --ram 30` changes it, in
`~/.stainid/queue/budget.json`). Every job declares what it needs (`STEP_NEEDS` per step; `--cpus/--gpus/--ram/--network`
on submit). A job starts when it fits. One that does not fit keeps what it needs reserved from lower classes, so they
only get what is spare. If it would fit once lower-class jobs stopped, those are **pre-empted**: asked to stop at their
next unit boundary, stopped hard (SIGTERM, then SIGKILL 30 s later) if they have not done so within their class's grace,
and queued again at their place. GPUs are handed out by index (`CUDA_VISIBLE_DEVICES`).

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
A=$(stainid --project P jobs submit --id-only -- v2 proposals --core cores.txt)
stainid --project P jobs submit --after "$A" -- v2 calls --core cores.txt # starts once A has finished (skipped if A fails)
stainid jobs list [--all] [--json]          stainid jobs show ID
stainid jobs pause ID… | resume ID… | cancel ID…
stainid jobs priority ID --up | --down | --top | --class public
stainid jobs pause-all | resume-all         # the Run page's "Pause all background work"
stainid jobs budget [--cpus N --gpus N --ram GB]
stainid jobs serve                          # run the queue without the app
```

The Run page shows the queue grouped by class with each job's place, progress, time left and Pause / Resume / Cancel /
Move up, and the budget in use.
