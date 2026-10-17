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
stainid serve --cache-limit 200G           # space each project's pictures for viewing may take (default 100G; Settings can set it per project)
stainid serve --cache-min-free 80G         # free space every drive keeps: below it, the least recently used pictures of all its projects go (default "50G or 10%")
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
| 4 | **Run** | Follow the steps as they run; run any step on its own with its options. |
| 5 | **Cores** | Look at the detections on the tissue, core by core. |
| 6 | **Labels** | Ask colleagues to check and correct detection's calls in annotation rounds (see [Annotation rounds](#annotation-rounds)), to measure how accurate detection is and to collect labels for training. |
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

## The Run page

- **Flow chart** (at the top): every step of Set up, Detect and Results as a box, stage by stage, with arrows labelled
  with the data each step hands on (*Core images*, *Tiles*, *Nuclei*, *Candidates*, *Calls* …). Each box shows where the
  step stands (done, part done, running, waiting, error, out of date, not started), its counts (*621 of 621 tiles*,
  *90 of 90 core images*, out-of-date cores), what it makes, how long its last run took, and while it runs how far it
  is and the time left. Hovering a step lights up what it needs and what needs it; clicking it opens its details
  (the same as its row below) under the chart. On a phone the boxes stack, the arrows pointing down.
- **Summary.** How many steps are done, the next one, and **Run what's left** (with a rough time), which queues
  every unfinished or out-of-date step of Set up, Detect and Results.
- **Now running** (while something runs): progress, elapsed time, time left and why a step is waiting.
- **Steps**, one row each, in order: what it does, its progress, its state and one button (*Run*, *Continue*,
  *Run again* or *Stop*). A step that cannot start yet says what it needs first; a failed step says why in plain words.
  Click a step for its details:
  - what the step does and how long it takes;
  - **Needs** and **Makes**: every file with its location and whether it exists. A missing file has a button: go to
    the step that makes it, download it, or show the expected format and a template;
  - options, with advanced ones hidden, and *Run with these options*;
  - *Show log*.
- **Detect** is the staged detection (*Find nuclei*, then *Find candidates*, *Decide*, *Outline objects*; see
  [pipeline_v2.md](pipeline_v2.md)) and **Results** its *Make results tables*. Each v2 step says how many cores are
  done and how many are **out of date**: something they were made from changed since (a setting, a model, a
  hand-drawn exclusion or an earlier stage). Running the step again redoes only those cores. *Outline objects →
  Advanced → Outline shown for each stain* sets which outline each stain shows first, its stain outline or Segment
  Anything's (`v2.outlines.default_outline`; NeuN Segment Anything, 6E10 and AT8 the stain outline); changing it only
  puts the candidate tables together again, Segment Anything does not run again.
- **Legacy detection (v1)** (folded, under the stages): the first detection (*Detect NeuN neurons*, *Detect plaques and
  tau*, *Outline objects*, *Make results tables*, each marked *legacy v1*). Still runnable one by one; never run by
  *Run what's left* or *Run everything*.
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
| **Cores** | Every core of every TMA (or every image or slide); click one to see the whole core, then click any spot for a full-resolution preview (220–660 µm) with stainID detection's objects where detection has run and the DAB layer. Where the v2 detection has run, the preview reads its saved candidates: the counted objects with their exact outlines, and, as layers to switch on, the rejected ones (each with why, in plain words), the AT8 threads it saved and the stained areas no candidate covers. Only the preview is sent over the network, never the whole scan. The analysed fields themselves are reviewed on Labels. |
| **Stain thresholds** | The DAB threshold of every slide and why it exists. |
| **Labels** | Labelling progress per stain, how often people and detection agree, the labelling sessions (create, open, close, share the link, follow people, download) and all labels as CSV. |
| **Models** | Every model with its status and settings, download buttons, training from all labels, and trained models compared with the current one. |
| **Results** | The main results table (download), every other table, plots by group and region, and the meaning of every column. |
| **Settings** | Study name, TMA layout, diagnostic groups and every file location, with a folder picker. Also leads to the Projects page to open or create another project, changes the admin password and (folded away) chooses the server. |
| **Help** | Getting started, common questions and a glossary. Words with a dotted underline show their definition on hover. |

## Labels

Everything people say about the tissue is a **label**: the class of an object and, where it has types, its type (e.g.
*NeuN+ neuron*, *plaque · compact*, *tau+ neuron · pretangle*, *thread*; see [label_classes.md](label_classes.md) for
the classes, their keys and how to change them in Settings), its outline, a tag on a whole field (*white matter*, *unusable*, …) or a note. Labels are attached
to the objects the pipeline considers in the analysed fields, accepted or rejected; detection's own call is shown next to
them but is not a label. An object's label is what most people said; ties stay unresolved. All labels live in one
append-only file, `outputs.reviews/labels/labels.jsonl` (one line per saved label with who, what, where and when;
edits and deletions keep the history), and *Download all labels (CSV)* exports them.

A **field** is one tile of a core (a square about 560 µm across) shown to people as one picture.

The **Labels** page shows, per stain: how many fields have been fully reviewed, how many objects carry labels
and of which class, objects people added that detection missed, outlines people drew or corrected, how often people agree
with each other, and how often detection agrees with people (precision and recall). Labels the current model was trained
on are included in the last figure, which therefore flatters the model; the cross-validated figures on Models are the
fair estimate.

### Annotation rounds

Labelling in stainID means checking and correcting stainID detection. People start from every object detection considered,
with its call and outline already on the screen, and fix only what is wrong. The rounds read the candidate store of the
staged detection ([pipeline_v2.md](pipeline_v2.md)): every candidate it proposed, accepted and rejected, with its
three Segment Anything outlines. So a stain needs its *Find candidates* and *Decide* steps first. Until then, `stainid labels
store-from-pool` writes a store from the prepared label pool, to try the rounds out. Rounds work the same in TMA
studies, image sets and whole-slide projects.

**Making a session** (Labels → *New labelling session*):

1. **Stain.**
2. **What should people do?**
   - **Fix the field** (round 2, the default). Whole fields with every object detection considered: point at an
     object and press a key to change its call or its outline, click to add what it missed. *How many fields*: fields
     nobody has checked, spread over TMAs and groups.
   - **Check detection's calls** (round 1). One object at a time, accepted and rejected ones, with detection's
     class, type, score and outline. The most informative come first: near the decision threshold (closest first),
     then strong ones detection rejected, then weak ones it accepted. *How many objects* (default 500; 0 = every one).
3. **Which cores**: every ready core, some TMAs (batches, for slides and images), or fields clicked on the cores. Each
   core shows whether it has detection's candidates yet.
4. **Title**, and whether it opens to labellers right away.

*Advanced* holds:

- *How generous*: which candidates are shown, from every candidate detection proposed down to only clearly stained ones
  (1.5 × the stain threshold);
- *People who must finish each field* (default 1; 2 or more gives repeat labels for agreement);
- the session ID, a note for labellers, and *Also ask for the type*;
- the older kinds of session (see below).

**Sharing.** *Link for labellers* on the Labels page (`http://<this computer>:8765/p/<project>/annotate`) opens every
open session of the study. The link names the project, so people label in the right study whatever others have open.
People type their name, with no account or password, and can stop and continue later under the same name. Nothing else
in stainID is reachable without the admin password. Links without a project (`…:8765/annotate`) open the project
opened last.

**The first time.** A name that has not been through the tutorial goes there first, from the list of sessions or from
a session's own link (and back there afterwards; `/api/session/tutorial` names the practice field). It is a practice on a real field
of the project, in both rounds, where nothing is saved (about 3–5 minutes). A coach asks for one thing at a time and
waits until it is done. *Skip tutorial* is always there, and *Tutorial* at the top shows it again. Who finished it is
kept in `outputs.reviews/tutorials.json`. **Try the tutorial** on the Labels page opens the same practice for the
admin, without a name, in a new tab: what annotators see, with nothing saved or remembered.

**On the page.** Every object is always shown, one shape each, with a thin dark edge so it reads on light and dark
tissue:

- solid in its class colour when it has a class: every class its own saturated colour, unsure (0) white;
- dotted grey when it is not active: detection rejected it, someone removed it (Delete), or it was just added and waits
  for its class (with a small diamond). A class key makes any of them solid.

The page fits the window: the picture as large as it can be and the side panel scrolling by itself, so the page never
scrolls while the picture is panned (drag, or hold the middle button; scroll to zoom). The side panel shows, from the
top: the object under the mouse (its class, detection's call and score, why it was left out, its area and which of its
three outlines shows, *n/3*); the class keys with their colours (0 unsure); the keys; this field (checked, changed,
added; tags and a note); *Save* and *Save and continue*; **?** for the longer help, display options and the other
fields; and the *Tutorial* link. Pointing at an object also shows its name beside it on the picture. Every key acts on the object under the mouse (round 2) or on screen (round 1):

| Key | What it does |
|---|---|
| 1–9 | its class. For a class with types, a type number right after it (within 1 s, `labels.subtype_wait_s`) picks the type, and a type's own letter key answers class and type at once ([label_classes.md](label_classes.md)) |
| 0 | unsure: left out of training; that person is not asked again, but the others are |
| Delete (Backspace) | not an object: it turns dotted grey and stays visible (also an object the person added); a class key brings it back |
| Tab / Shift+Tab (or + / −) | a bigger or smaller outline: the three Segment Anything outputs come with the object, so it changes at once |
| Space | precise editing of this object, with everything else locked (tags, note, other fields, saving): right-click a spot to add it, left-click one to leave it out, drag the white dots (at most 48, the outline simplified to them); Z one step back; Space again keeps it, Esc cancels |
| E | the exclusion brush, on and off (Esc also leaves): left-drag paints a bad area (fold, smudge, empty glass) as a red hatch, right-drag or Shift-drag erases, the wheel sets the brush size, the middle button pans; outlines are locked meanwhile. Saved with Save as rings in field pixels (an `exclude` label, `labels/exclusions.py`). Objects whose centre lies in it are that person's "excluded by the annotator": left out of scoring, agreement and training, and not confirmed when the field is finished. The analyses ignore the areas unless *Settings → More study settings → Apply annotators' excluded areas to the analysis* (`labels.apply_exclusions`, off by default) adds them to the hand-drawn exclusions |
| Z | undo the last unsaved change (a brush stroke too) |
| Enter | round 1: keep detection's call |
| Ctrl+S (⌘S) | save, and stay on the field |
| Ctrl+Enter (⌘Enter) | save and continue |

In round 2, a click on empty tissue outlines a missed object with one click (Segment Anything). It is a new
candidate, dotted grey, and Tab / Shift+Tab switch between its three sizes, until a class key makes it solid. *Draw an object by hand…* is the last resort. One click answers at once,
because the server makes the Segment Anything features of the session's fields ahead: on a GPU every field of the
open sessions, and on the CPU the field someone has open and their next ones. They are kept in
`data/cache/sam_embeddings/`, and the page says *Preparing one-click outlines…* until its field is ready.

**Saving.** Nothing is saved until the person presses *Save* or *Save and continue*. The count at the top says how many
unsaved changes there are, and a small orange dot marks each changed object. A draft is kept in the browser in case of
a reload, and leaving the page with unsaved changes asks first. *Save and continue* finishes the field:

- detection's call counts as right unless the person changed it: nothing needs a class or an answer first;
- every candidate of the tile is then recorded as checked by this person, the ones they left as they are with
  detection's call as their answer (also where someone else decided it), and every outline left as it is too;
- objects the person added and gave no class are left out (the panel says so).

A tile's labels are written all at once (`routes.rounds.save_tile`) or not at all.

**Fields are claimed, so people never collide.** The unit people claim is a field (a tile), in both rounds:

- Opening a field holds it for that person while their page is open, so the others are handed other fields.
- A field that enough people finished (*People who must finish each field*) is not handed to anyone else.
- What others decide meanwhile shows within seconds ("checked by Ben").

**One set of answers.** Both rounds, every session and every person write class labels on the same candidates, keyed by
candidate id (`labels.decisions`). A candidate that enough people decided leaves every round-1 list at once and shows
as checked in round 2. Labels made before the rounds (earlier sessions, imported label sets) are matched onto the
candidates by position, so nothing anyone labelled is asked again. The decisions train the models like any other label
([Training a model](#training-a-model)).

**Following a session** (Labels page). Each session shows:

- the fields done, by anyone, and those in progress;
- who is labelling now, and how many fields each person finished;
- whether its pictures are ready (they are made ahead in the background).

*More* opens the labelling page, *Look at the labels* (every labelled object, field by field), the session's labels
as CSV, its fields (round 2), and *Delete session*. Deleting moves the session's settings aside and keeps every label. *Close*
stops people from opening it; nothing is lost.

### Older kinds of session

Under *Advanced* in *New labelling session* (*make one of those instead…*), on the current label pool rather than the
v2 store:

- **Review whole fields**: fields that stainID picks (nobody has reviewed them yet, spread over TMAs and groups, in
  shuffled order) or that you click on the cores, and how many people should review each field. People see the field
  with detection's calls and outlines pre-filled (dashed: detection's call; solid: their own decision). They change the class
  of wrong objects, add missed objects, drag outline corners or draw new ones, tag the field (*white matter*,
  *unusable*, …), and press *Finished*, which records every detection call they left alone as confirmed by them. An option
  under Advanced shows the candidates without detection's calls.
- **Quick object queue**: one object at a time, labelled with the keyboard. The next object is chosen anew each time
  from all prepared fields: where detection is unsure, where people disagree with each other or with detection, objects one
  other person has labelled, or at random.
- **Outline every object**: the segmentation check (below).

These need their fields **prepared**. The *Prepare fields for labelling* step (`stainid labels prepare`) finds every
candidate as the detection step does and stores its measurements and detection's call in
`outputs.reviews/pool/<stain>/<field>.csv` (10–30 s per field). Sessions prepare their own fields in the background.

Labels made before the label store (training sets, the CNN-outlier adjudication and the white-matter field review)
were imported once with `stainid labels import`, each as its own source (`import: <name>`); the original folders are
unchanged. Every label belongs to a session; the imported ones (no session) together form the read-only pseudo-session
*AI reference labels* (`reference`), which the Labels page lists and lets you look inside like any other session.

### The segmentation check

A model's scores (AUC, precision, recall) are measured on the candidates the detection step proposed, so they cannot see
objects that were never proposed, or that its segmentation cut into pieces or merged with a neighbour. The segmentation
check measures that end to end:

1. **Set it up** on *Models → How well does detection outline objects?* (or *New labelling session → Outline every
   object*): one session per stain, "Segmentation check – <stain>", about 40 ready fields chosen from TMAs 1–7 (in an
   image set: every batch; *Advanced* lets you choose), spread over TMA × diagnosis group × region and over sparse,
   medium and dense staining (thirds of detection's objects per mm² of tissue), fields nobody has labelled first. Its
   fields' candidates are found in the background (`stainid score-segmentation --proposals-only`, 10–30 s per field).
2. **People outline every object** of the stain's classes whose middle lies in the field's yellow square, with
   what detection found shown (it can be hidden): *click to outline* (Segment Anything: Tab tries another size, Shift+click adds a spot that belongs to
   the object, right-click one that does not; a click answers in well under a tenth of a second because the server
   prepares each field's picture features ahead — on a GPU, fp16, every field of the open checks as soon as one is made
   or opened; on the CPU the field someone has open and their next two, about 50 s each — kept in
   `data/cache/sam_embeddings/`; the page says *Preparing outlines…* until its field is done (clicks work meanwhile, the
   first in each part of the field slower); `$STAINID_SAM_DEVICE` (`cpu`, `cuda:1`) chooses the device), *draw by hand* (D), drag corners, 1–9 for the
   class ("unsure" is left out of the scores), Delete, Esc. *Show detection* (R) shows what detection counted and
   every candidate it considered, on by default. *Mark field complete* asks to confirm that nothing is left; only
   complete fields are scored. These outlines are kept apart from the class labels models train on (tasks `object`
   and `complete` in the label store).
3. **Score now** (`stainid score-segmentation`) matches every outlined object one to one with (a) every candidate:
   *proposed*, and (b) the candidates the model counted: *found and counted*, and *right* (precision), at IoU ≥ 0.5 and
   0.3. An object that is not found is *never proposed* (no candidate overlaps it), *merged with a neighbour* (the
   candidate covering it covers another outlined object too), *cut into pieces* (two or more candidates each cover part
   of it) or has a *poor outline*; a well-outlined proposal the model did not count is *proposed, not counted*. The
   card shows this per stain in words; *Advanced* has the figures by TMA, group, density third and object size, the
   outline fit (IoU, boundary IoU, and the Segment Anything outlines of counted objects) and the files.

Results go to `data/analysis/segmentation_benchmark/`: `proposals/<stain>/<field>.npz` (the candidates and their masks,
found with the detection step's own segmentation, because whole-core detection keeps only the counted 6E10 and AT8
objects), `report.json`, `objects.csv` (one row per outlined object), `detections.csv` (one row per counted object) and
`fields.csv`. In projects made from a dataset package, the raters who drew polygons are scored the same way, each on
its own (`imported` in the report).

## Training a model

*Train a model* (`stainid train --stain X [--session NAME ...]`) uses the labelled objects of that stain from the sessions
you choose (on the Models page; every session by default, `reference` = the imported labels). The report of each model
records the sessions it learned from:

- **Labels.** Each object's majority label, positive or negative; "unsure" and ties are skipped. The decisions of the
  annotation rounds count like any other label. An object people added where detection proposed nothing is measured
  at the clicked position first and then trains the model like a candidate; the report counts these, and those that
  could not be measured (their labels stay, but cannot train). It needs at least 10 positive and 10 negative labels;
  100–200 is a good start.
- **Model.** The same random forest the pipeline uses, on the candidate measurements: those of the v2 candidate store
  for objects of the annotation rounds, else those stored when the field was prepared.
- **Thresholds.** The compact/diffuse threshold (6E10) and the tangle/pretangle threshold (AT8) come from your labels
  when there are enough, otherwise from the current model.
- **Validation.** Leave-one-TMA-out; if a fold would train without one class, stratified folds of whole TMAs. Every donor sits on one TMA, so no donor is ever in both training and testing. The current model is scored on the same labels,
  so the Models page can show both side by side.

The model is saved to `outputs.trained_models/<stain>_<time>/bundle.joblib` with `report.json`. It is used only after
you press *Use this model*, which sets `models.<neun|amyloid|tau>` in `stainid.yaml`. Detection steps never mix
results from two models: run them again with *Start over*, which moves the old results aside, then *Make results
tables* again.
