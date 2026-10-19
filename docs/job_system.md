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
   | annotation | preparing what annotators open (field pictures, crops, Segment Anything features ahead) | 0 / 400 / 400 | 300 s |
   | pipeline | the main study's stages | 5 / 100 / 100 | 300 s |
   | models | training, model comparison, segmentation check | 10 / 50 / 50 | 120 s |
   | public | any project with a `dataset:` section (or `jobs: {class: public}` in its stainid.yaml) | 15 / 20 / 20 | 30 s |
   | downloads | network only; takes no processors, so it runs alongside | 19 / 10 / 10 | 10 s |

   Within a class jobs start first in, first out; the Run page's **Move up** and `stainid jobs priority` change that.

## How scheduling works

The queue is work-conserving: whenever processors, a GPU or memory are free and a waiting job fits, it starts,
whatever its class. Nothing is reserved and no class is capped. Priority decides only the order in which jobs are
looked at, and which jobs make room for which. Code: `Scheduler._schedule`, one round about once a second;
`stainid.jobs.policy` holds sizes, estimates, conflicts and fair share; `stainid.jobs.arrays` holds the chunks.

1. **Resources.**
   - Each job declares cpus, gpus, ram, io (light or heavy) and network: `STEP_NEEDS` and `policy.IO_HEAVY` per step,
     or `--cpus/--gpus/--ram/--io/--network` on submit.
   - Processors and GPUs are the computer's (`stainid jobs budget --cpus 24` limits them, in
     `~/.stainid/queue/budget.json`). A GPU that a program outside the queue uses (1.5 GB or more) is not handed out.
   - At most 2 io-heavy jobs run per disk. A GPU job is held back by its GPU, not its disk.
2. **Memory is measured.**
   - Memory for jobs = the computer's total, less what the app uses now, less a 4 GB headroom for the app to grow, less
     a 3 GB system reserve. On 53 GB with the app at 3 GB, that is 43 GB.
   - A running job counts its highest measured memory plus 10%: its systemd unit's anonymous memory, without page
     cache. Its declared value is only a floor until it has been measured for 2 minutes.
   - A job also starts only while the memory available on the computer, less the headroom and 2 GB, holds what it
     declared.
   - The Run page and `stainid jobs list` show e.g. "RAM: 22 of 54 GB used · 25 free for jobs".
3. **Order.**
   - Classes go highest first.
   - Within a class:
     - jobs that have waited ready (held back only by room) for more than 30 minutes come first, oldest first;
     - then light GPU jobs that fit a free GPU, so no GPU stays idle while one waits;
     - then the others, taking turns between projects (and between people for annotation work), first in, first out.
4. **All-or-nothing starts.**
   - A job starts only when everything it needs is free at once, and it takes all of it in one step.
   - That includes no running or earlier-queued job writing the same outputs. Writers of the same outputs are jobs of
     the same project with an output in common and overlapping stains, core images or shards, or jobs sharing a
     `--conflict` key. Readers overlap freely.
   - A step also waits for the steps queued before it whose outputs it reads, and for its `--after` jobs.
5. **Pre-emption is the only way to get room.**
   - When the next job does not fit, it pauses the fewest jobs it outranks that make it fit: lowest class first, then
     the largest holders of what it lacks. It outranks lower classes, and, once it has waited 30 minutes, newer jobs of
     its own class that started after it was queued.
   - They stop at their next unit boundary. After their class's grace they get SIGTERM, then SIGKILL 30 s later. They
     are queued again, and they resume skipping their finished units.
   - While they stop, what the job will take is not given to lower jobs. This lasts at most their grace, and it is how
     Kubernetes treats a pre-empting pod's node.
6. **Memory pressure.** When the memory available falls under the app's 4 GB headroom, the lowest-class jobs (largest
   first) are paused at once, with at most 20 s to reach a boundary, before the kernel's out-of-memory killer acts. If
   the app grows with many annotators, this is what keeps its headroom.
7. **Arrays.**
   - Find nuclei and the v2 proposals, calls and outlines over many core images run as a job array: one chunk per TMA
     (a TMA of more than 200 core images in parts), or per 50 core images for image sets and slides.
   - Each chunk is the same command with `--core <the chunk's list>`; `--array tma|N` and `--no-array` change the split.
   - Chunks run side by side, are pre-empted one at a time, and have their own progress and estimate.
   - The parent shows the overall progress, and Pause, Resume, Cancel and Move act on all its chunks.
   - A next stage's chunks wait for every chunk of the stage before.
8. **Size and time** are shown and never hold a job back.
   - The run time is what was declared (`--time 2h`), else units × the step's time per unit in earlier runs, else how
     long the step last took.
   - Small = at most 4 processors, no GPU, 8 GB and 15 minutes; anything else is big. The Run page shows
     "small · ~2 min" or "big · ~5 h".

### Why it can't deadlock or starve

- **No hold-and-wait.** A waiting job holds nothing; a job gets everything at once or nothing. Pausing releases
  everything, including its conflict keys.
- **Waits never form a cycle.** A job only ever waits for:
  - running jobs, which end or are stopped within grace + 30 s;
  - jobs queued before it (outputs it reads, writers of the same outputs);
  - its `--after` jobs, which must exist when it is submitted, and a cycle is refused.
  Every wait points backwards in time, so the waits-for graph is acyclic.
- **Nothing waits forever.**
  - A failed or stopped dependency ends its dependents ("Not started because “…” failed").
  - A job that asks for more than this computer can ever give jobs is given the most there is, with a warning
    that says so plainly ("asks for 40 GB of memory; at most 37 GB can be given to jobs on this computer: it runs with
    37 GB of memory"; shown at submit, kept in the job's record, and shown by `stainid jobs list` and the Run page,
    for jobs the app started too). The most is the processors and GPUs of the
    budget, and for memory the total less the system's reserve, the app and its headroom, and no more than is
    available once the jobs' own memory is given back, less the headroom and the 2 GB a job start keeps free. This
    is checked at submit and again whenever the queue is idle (other programs may have taken memory since).
  - A wait for room says who holds it: "held by work of the same or a higher priority" only when such running work
    holds what is short; otherwise lower-priority work that cannot be paused now, or the app and other programs on
    this computer.
- **No starvation.**
  - The first job in priority order can always get room: every job it outranks can be paused.
  - Jobs of its class or above end in finite time, and once it has waited 30 minutes, newer jobs of its own class
    give way too.
  - Ageing only lets older jobs pause newer ones, never back, so two jobs cannot keep pausing each other.
- **Watchdog.** A job that has waited 15 minutes although what it needs is free, or is being freed, is logged
  (`stainid.jobs`) and shown on the Run page and in `stainid jobs list`.
- **Tests.** `tests/test_job_scheduling.py` checks:
  - random mixes of jobs, classes, dependencies, conflicts and failures always drain;
  - no job that fits stays waiting;
  - an idle GPU gets a waiting public GPU job;
  - a high job needing memory pauses the right low jobs, and they resume.

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
stainid jobs budget [--cpus N --gpus N --ram GB]   # shows the real memory too
stainid jobs serve                          # run the queue without the app
```

The Run page shows the queue grouped by class with each job's place, progress, time left and Pause / Resume / Cancel /
Move up, its size and expected run time ("small · ~2 min", "big · ~5 h"), an array as one row with its chunks, and the
budget in use.

### Progress and time left

A running job's line says how far it is, how long it has run, and what is known of the time left:

- `12 of 72 done · 1 h 45 min so far · about 8 h left`: from its progress, at the pace so far;
- `running · 1 h 45 min so far · no estimate`: it reports no progress and nothing tells how long it takes;
- `… · taking longer than expected`: it has run past its expected time (`--time`, or how long the same command took
  before) with no progress to go by. The queue never shows a made-up estimate.

A stainID step reports progress with the `[i/n]` lines it prints. Any other command (`--exec`) can use its **progress
file**, whose path is in `$STAINID_PROGRESS_FILE` (next to its log in the queue folder), in either form:

```bash
echo "12/72" >> "$STAINID_PROGRESS_FILE"           # done/total: the last such line counts
echo "total 72" >> "$STAINID_PROGRESS_FILE"        # or the total once, then one line per finished unit
echo "unit done" >> "$STAINID_PROGRESS_FILE"       # ("done <anything>" counts too)
```

```python
import os
progress = os.environ.get("STAINID_PROGRESS_FILE")   # None outside the queue
...
if progress:
    with open(progress, "a") as f:
        f.write(f"{done}/{total}\n")                   # several workers may append "unit done" lines side by side
```

The progress file wins over `[i/n]` lines in the log; it is kept when a job is paused and resumed, so a script that
skips its finished units can simply go on counting. `stainid jobs list` shows the same: `1h45m in, no estimate`.
