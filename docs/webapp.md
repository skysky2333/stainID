# Using the app

## Starting

Double-click `Start stainID.command` (macOS) or `Start stainID.bat` (Windows), or run:

```bash
stainid serve                              # opens the Projects page, http://127.0.0.1:8765
stainid --project path/to/study serve      # opens that project
stainid serve --projects-dir ~/Studies     # also list every project folder in ~/Studies
stainid serve --no-browser --port 9000
stainid serve --host 127.0.0.1             # reachable from this computer only
stainid serve --allow-origin http://lab-pc:8080   # also answer a copy of the web app served from there (see below)
```

The app opens on the **Projects** page (see [Projects](#projects)), or on the project given with `--project`. If
stainID is already running, starting it again just opens the browser.

**Admin password.** The app asks for the admin password; it is `admin` until you change it under *Settings → Admin
password*. The app listens on your network (so annotators can reach the annotation pages), but everything except
those pages needs the password, and while it is still `admin` you can only sign in on the computer running stainID.

**Several people at once.** Anyone on your network can open `http://<the computer running stainID>:8765` in their
browser. Each person signs in (or, on the annotation pages, gives a name) and picks a project. People in the same
project see the same job list and the same queue: two people can press Run at the same time, and the queue orders their
steps as if one person had pressed both. People in different projects work side by side: the project is part of the
page's address (`/p/<project>/workflow`), so switching project in one browser changes nothing for anyone else, and a
link to a page always opens the right project. Each project keeps its own job list; heavy steps take turns across all
projects (see [How steps run](#how-steps-run)).

## Projects

The **Projects** page (the start page, and *Switch project… → All projects* at the top of the sidebar) lists every study
this server knows: the projects opened in the app before, and every folder with a `stainid.yaml` in the projects folder.
Each shows its status (*Not set up yet*, *Set up*, *Running: …* with the steps that run, *Results ready*) and when it
was last opened.

- **New study**: a study name and where to keep it (the projects folder when one is set); stainID makes a folder with
  the study's name there and opens the guided setup. *Start with the settings of* another study copies, as you choose,
  its diagnostic groups, stains, models and their settings (the new study uses the same model files) and its TMA
  layout and naming (core-name prefix, rows and columns, pixel size, how donor ids and donor-regions are written).
  Slides, maps, donors and results are never copied.
- **Open existing folder…**: choose a folder that holds a `stainid.yaml`.
- **Remove from list**: takes the study off the list only. Its folder is not touched, steps still running carry on,
  and opening the folder again brings it back.
- **Advanced → Projects folder**: the folder whose project folders are listed (`stainid serve --projects-dir DIR` sets
  it for one run and wins over this setting).

The list is kept in `~/.stainid/app.json` (`recent`, the folders opened, most recent first; `opened`, when;
`projects_dir`; `hidden`, folders of the projects folder removed from the list).

## A study, from start to finish

| | Where | What you do |
|---|---|---|
| 1 | **Projects** | *New study*: a study name and where to keep it. |
| 2 | **Guided setup** | Slides, cores, TMA maps, donor information and checks, in six short screens (see [The guided setup](#the-guided-setup)); it ends with *Run everything*. |
| 3 | **Models** | Download the published models. Add the NeuN, 6E10 and AT8 models, or train them (see [Training a model](#training-a-model)). |
| 4 | **Workflow** | Follow the steps as they run; run any step on its own with its options. |
| 5 | **Cohort & cores** | Look at the detections on the tissue, core by core. |
| 6 | **Labels** | Review fields (alone or with others) to measure how accurate the detections are and to collect labels for training. |
| 7 | **Results** | Download the results table (one row per donor and region) and read what every column means. |

## The guided setup

*Guided setup* (in the sidebar; Home leads a new project there, and *Add slides* on Home opens it later) has six short
screens. Each shows what stainID worked out by itself and asks only for what it could not; file locations, full tables,
step options and logs are folded under **Advanced** on each screen. The steps along the top show how far each screen
has got, and any screen can be opened again later.

| Screen | What stainID does | What you do |
|---|---|---|
| **Slides** | Lists the scans in the slides folder, each with a small picture of the slide and its label (the overview the scan file keeps, also for `.vsi`; else the Find cores preview). Where a file name says the TMA and stain clearly (`TMA LIP-3 AT8.vsi` → TMA 3, AT8) they are filled in and marked *from the file name*; an unclear name (two stains, several numbers none of which follows the core-name prefix or the word TMA) leaves the row empty and highlighted. Suggests the core-name prefix (`LIP-`). New scans are marked *new*. | Choose each slide's **TMA** from the study's TMAs (or *New TMA…*: type it once, e.g. `LIP-12` or `12`, and it is in the list) and its **stain** from the study's stains. *Save and continue* waits until every slide has both and no TMA has the same stain twice (“2 slides still need a TMA”). |
| **Cores** | *Find cores* fits the grid of cores on every slide. A slide without its own rows and columns gets the size that fits best (the project's size unless another fits clearly better); the size is written to the slides table. One picture per slide shows the fitted grid, with how well it fits. | *Looks right*, or *Fix*: give the slide its rows and columns, or say it lies turned (180°) or mirrored, and its cores are found again. |
| **Map** | Reads the maps from the files you add: Excel sheets, Word tables, CSV, or cells pasted from a spreadsheet. A map is a grid like the array (rows 1, 2, … columns A, B, …); a cell names the donor and region, and usually the group and pathology (`1673 FR` / `AD` / `C:C - B:6`), on one to three lines or over three table rows per map row. Stray spaces (`17 00 FR`, `OC C`), en dashes, prefixed ids (`JOG3 FR` → `JOG_003`) and control cores (`CBLM`, `STOM`) are read through. Each map is guessed to the TMA in its name. | Check the preview grid (cells it could not read have a yellow edge; type into any cell to correct it) and save it as the map of its TMA. A map that is only a photo: *Type it in* opens an empty grid with a picture of the core at each position. |
| **Donors** (optional) | Reads the case sheet: finds the header row by its column names and matches each column (donor id, group, age, sex, race, PMI/PMD, APOE, CERAD, Braak, TMA, diagnoses …); section rows (“CONTROL <65 (90 cases)”) give the group, a repeated header (“JOG#”) starts its own id series, “~15” and “>90” are read as numbers (kept as written beside them). Keeps the donors on the maps and on the study's TMAs. | Check the preview and *Save*. Under Advanced: change a column match, keep every donor, replace instead of update. |
| **Checks** | Compares the map, the donor table and the slides: TMAs with slides but no map, core positions not on the map (which would stop *Attach TMA map*), donors on the map but not in the sheet and the other way round, TMA, group and pathology disagreements (“Donor 2323: map says CERAD B, donor table says 0. Use the donor table?”), ids used twice in the sheet (keep the row on the map's TMA), donor-regions with one core where replicates are the rule, new group codes. | Press the fix next to each one, or correct the file. |
| **Run** | Lists the steps left and estimates how long they take on this computer (from the number of slides, cores and tiles and whether there is a GPU). | *Run everything*. |

The setup writes the same files as the step cards (`slides.csv`, `tma_layout.csv`, `donor_metadata.csv`): saving a map
replaces that TMA's rows and keeps the others; saving donors updates the table. Donor ids follow one rule for maps and
sheets (*Map → Advanced → How donor ids are written*, `donor_ids` in `stainid.yaml`). Each map cell's donor-region (`sample_region_id`) is named the
way the project's saved maps name them (e.g. `BRC_1673_FR`, `JOG_003_FR`) and shown in the preview before saving; a
project without saved maps uses the rule under *Map → Advanced* (`sample_region_ids`).

## The Workflow page

- **Summary.** How many steps are done, the next one, and **Run all remaining steps** (with a rough time), which queues
  every unfinished step except the optional outlines.
- **Now running** (while something runs): progress, elapsed time, time left and why a step is waiting.
- **Steps**, one row each, in order: what it does, its progress, its state and one button (*Run*, *Continue*,
  *Run again* or *Stop*). A step that cannot start yet says what it needs first; a failed step says why in plain words.
  Click a step for its details:
  - what the step does and how long it takes;
  - **Needs** and **Makes**: every file with its location and whether it exists. A missing file has a button: go to
    the step that makes it, download it, or show the expected format and a template;
  - options, with advanced ones hidden, and *Run with these options*;
  - *Show log*.
- **Pipeline map** (folded): every step as a box in the order steps can run; an arrow means "needs the result of".
  Colours show done, partly done, running, waiting, failed and "made with an older model".
- **All jobs** (folded): every run with its command and log.

The *Register slides* step holds the slides-table editor (the same table as the Slides screen), and *Attach TMA map* the template download and the CSV
upload, for those who prefer them to the guided setup.

## How steps run

Steps run in the background, started by a queue with two rules:

- **Order.** A step waits while a step that makes something it needs (an arrow on the map) is still running or
  waiting. The same step with the same options never runs twice at once.
- **Load.** At most two heavy steps (Cellpose, detection, outlines) run at the same time on the whole server, over
  all projects (there is one GPU). A heavy step that waits for a slot says where the running ones are ("Waits for a
  free slot: 2 heavy steps are already running, 1 in the project “Second cohort” (Find nuclei) …"). Heavy steps
  running from the command line count too, also in projects of the list that nobody has open in the app.

The order rule applies within a project; each project keeps its own job list (in its `outputs.jobs` folder).

So you can press Run on several steps in a row and they start in the right order; each waiting step says what it is
waiting for. Just before a step starts, the queue checks that everything it needs exists. If not, the step is
**skipped** and its card says what is missing and which step makes it. When a step fails or is stopped, the steps
waiting for it are skipped too, instead of failing one after another.

Steps keep running when the browser is closed, and when stainID itself is stopped: started again, it picks up the steps
still running and records how each one ended. A step that was stopped can be run again and continues where it stopped.
When you switch to a different stain model, the detection steps made with the old one are marked **Made with an older
model** until you run them again with *Start over* (under Advanced options).

### Steps run from the command line

Every step is also a `stainid` command (the job list shows it under each job, e.g. `stainid neun --threads 8`).
Running one in a terminal puts it in the same job list, with a small **command line** tag:

```bash
stainid --project path/to/study neun --threads 8 > neun.log 2>&1   # with "> file", the app can show its log and progress
```

The queue treats it like a step started in the app: it counts toward the two heavy steps, and a step queued in the app
that needs its results waits for it ("Waits for “Detect NeuN neurons”, running from the command line, to finish").
*Stop* in the app stops it. When it ends, the job list shows whether it finished, failed (with the error) or was
stopped; if its process disappears without saying (closed terminal, computer restarted), the job shows as
interrupted. The log panel of every job also shows its process number, exit code and log file.

## Serving the web app from another computer

Normally the stainID server also serves the web app, and nothing needs setting up. The web app is only a page that
talks to the server, so it can also come from elsewhere, e.g. a copy of `src/stainid/api/static` on any web server
for static files. Two things are needed:

1. **The server must accept the page.** Start it with the address the page is served from (scheme, host and port):
   `stainid serve --allow-origin http://lab-pc:8080`. Repeat the option for several addresses, or set
   `STAINID_ALLOW_ORIGINS=http://lab-pc:8080,https://stainid.example.org`. Without it, the server answers only pages it
   served itself. *Settings → Server* lists the addresses it accepts.
2. **The page must know the server.** On the sign-in screen, open *Server: … · connect to another server* (folded
   under the password) and enter the server's address, e.g. `http://analysis-pc:8765`; the browser remembers it, and
   *Back to …* returns to the default. For a copy everyone uses, set it once in that copy's `config.js` (next to
   `index.html`): `window.STAINID_SERVER = 'http://analysis-pc:8765'`, no rebuild needed.

Sign-in then works as usual (the default password still only works on the computer running stainID); the page keeps
the sign-in as a token instead of a cookie.

## Other pages

| Page | What it is for |
|---|---|
| **Home** | Your next step (the guided setup while the study is not set up, or new slides to add), a checklist of every step, cohort numbers and recent jobs. |
| **Cohort & cores** | Every core of every TMA; click one to see the whole core, then click any spot for a full-resolution preview (220–660 µm) with the AI detections where detection has run and the DAB layer. Only the preview is sent over the network, never the whole scan. The analysed fields themselves are reviewed on Labels. |
| **Stain thresholds** | The DAB threshold of every slide and why it exists. |
| **Labels** | Labelling progress per stain, how often people and the AI agree, the labelling sessions (create, start, stop, share the link, follow people, download) and all labels as CSV. |
| **Models** | Every model with its status and settings, download buttons, training from all labels, and trained models compared with the current one. |
| **Results** | The main results table (download), every other table, plots by group and region, and the meaning of every column. |
| **Settings** | Study name, TMA layout, diagnostic groups and every file location, with a folder picker. Also leads to the Projects page to open or create another project, changes the admin password and (folded away) chooses the server. |
| **Help** | Getting started, common questions and a glossary. Words with a dotted underline show their definition on hover. |

## Labels

Everything people say about the tissue is a **label**: the class of an object (e.g. *NeuN+ neuron*, *compact plaque*,
*not a tau+ neuron*), its outline, a tag on a whole field (*white matter*, *unusable*, …) or a note. Labels are attached
to the objects the pipeline considers in the analysed fields, accepted or rejected; the AI's own call is shown next to
them but is not a label. An object's label is what most people said; ties stay unresolved. All labels live in one
append-only file, `outputs.reviews/labels/labels.jsonl` (one line per saved label with who, what, where and when;
edits and deletions keep the history), and *Download all labels (CSV)* exports them.

The **Labels** page shows, per stain: how many analysed fields have been fully reviewed, how many objects carry labels
and of which class, objects people added that the AI missed, outlines people drew or corrected, how often people agree
with each other, and how often the AI agrees with people (precision and recall). Labels the current model was trained
on are included in the last figure, which therefore flatters the model; the cross-validated figures on Models are the
fair estimate.

### Labelling sessions

Other people label in their browser; everything is managed on the Labels page.

1. **Create a session**: the stain, then either
   - **Review whole fields**: fields nobody has reviewed yet, spread over TMAs and groups in shuffled order (or a list
     of field IDs), and how many people should review each field (1 shares them out; 2 or more gives repeat reviews
     for agreement). People see the field with the AI's calls and outlines pre-filled: dashed circles are the AI's call,
     solid ones their own decision. They change the class of wrong objects, add objects the AI missed, drag the corners
     of an outline or draw a new one, tag the field, and press *Finished*, which records every AI call they left alone
     as confirmed by them. A blind variant shows the candidates without the AI's calls.
   - **Quick object queue**: one object at a time, labelled with the keyboard. The next object is chosen anew each time
     from all prepared fields: where the AI is unsure, where people disagree with each other or with the AI, objects
     one other person has labelled, or at random; each person gets objects they have not labelled, fewest labels first.
2. **Start** it and send the link shown on the page (`http://<this computer>:8765/p/<project>/annotate`: the link
   names the project, so people label in the right study whatever others have open). People register a name
   and can stop and continue later under the same name. Links of the previous release without a project
   (`…:8765/annotate`) still work and open the project opened last. Only started sessions are listed, and nothing else in stainID
   is reachable without the admin password.
3. Follow progress, the people and their changes; **+10 fields** extends a session; **Stop** closes it but keeps
   everything; **Delete** moves the session definition aside (its labels stay).

Fields have to be **prepared** before people can label them: the *Prepare fields for labelling* step (`stainid labels
prepare`) finds every candidate exactly as the detection step does and stores its measurements and the AI's call in
`outputs.reviews/pool/<stain>/<field>.csv` (10–30 s per field). Sessions prepare their own fields in the background.

Labels made before the label store (training sets, the CNN-outlier adjudication and the white-matter field review)
were imported once with `stainid labels import`, each as its own source (`import: <name>`); the original folders are
unchanged.

## Training a model

*Train a model* (`stainid train --stain X`) uses every labelled object of that stain in the label store:

- **Labels.** Each object's majority label, positive or negative; "unsure" and ties are skipped. Labels on positions no
  candidate covers (objects people added) cannot train the classifier. It needs at least 10 positive and 10 negative
  labels; 100–200 is a good start.
- **Model.** The same random forest the pipeline uses, on the candidate measurements stored when the field was prepared.
- **Thresholds.** The compact/diffuse threshold (6E10) and the tangle/pretangle threshold (AT8) come from your labels
  when there are enough, otherwise from the current model.
- **Validation.** Leave-one-TMA-out; if a fold would train without one class, stratified folds of whole TMAs. Every donor sits on one TMA, so no donor is ever in both training and testing. The current model is scored on the same labels,
  so the Models page can show both side by side.

The model is saved to `outputs.trained_models/<stain>_<time>/bundle.joblib` with `report.json`. It is used only after
you press *Use this model*, which sets `models.<neun|amyloid|tau>` in `stainid.yaml`. Detection steps never mix
results from two models: run them again with *Start over*, which moves the old results aside, then *Make results
tables* again.
