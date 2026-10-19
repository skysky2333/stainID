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

**Accounts.** Everyone signs in on the main page (the server's root address) with a name and a password, and lands
on their home: *My labelling sets*, *Add a set* with a key, and both tutorials. One menu for everyone: its labelling
items (Home, Tutorials, Help, Account) for all, the study's pages only for **admins**, whose Home shows their sets
first and then where the study stands. Admins label too: a session an admin opens joins their sets, and their labels
carry their own account like anyone's; they can open every labelling set from the Labels page. An **annotator** gets
only the labelling pages. Addresses of the earlier separate annotation pages (`/annotate`, `/annotate/tutorial`,
`/annotate/account`) lead to the same pages of the one app; a session's link (`/p/<project>/annotate/<name>`) stays a
deep link, and someone not signed in goes to the main page first and back there after signing in. Every admin-only request is refused for annotators (403)
by the server, not only hidden in the page. Anyone can make an annotator account on the sign-in page; admins add
people, promote and demote them, reset forgotten passwords (a one-time password; the person chooses a new one when
signing in with it) and remove accounts under *Settings → People*. There is always at least one admin: the last one
cannot be demoted or removed. Names are unique on the server (case does not matter); passwords are kept as scrypt
hashes with a salt of their own; sign-ins and join keys are rate-limited. Accounts belong to the server
(`~/.stainid/accounts.sqlite`), so one account works in every project.

The first admin is the account **admin** with the admin password of earlier versions: after an update, sign in with
the name `admin` and the password you used before (`admin` on a new server, until you change it under *Settings →
Your password*). The tutorial needs no account: anyone can try it, and nothing is kept.

**API tokens.** Scripts and agents use a token of an account instead of a password: *Settings → API tokens for
scripts*, or `stainid accounts token NAME` on the server (printed once; `stainid accounts list | add | reset | role`
manage accounts there too). Send it as `Authorization: Bearer <token>` (or `?access_token=<token>` on a plain GET). A
token can do what its account can; revoking it, or removing the account, ends it.

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

Sign-in then works as usual; the page keeps the sign-in as a token instead of a cookie.

## Other pages

| Page | What it is for |
|---|---|
| **Home** | Your next step (the guided setup while the study is not set up, or new slides to add), a checklist of every step, cohort numbers and recent jobs. |
| **Cores** | Every core of every TMA (or every image or slide); click one to open the whole core as a zoomable picture (tiles at every level, so only what is on screen is sent, never the whole scan) with stainID detection's objects. Where the v2 detection has run it reads the saved candidates: the counted objects with their exact outlines in their class's colour, and, as layers, the rejected ones (each with why, in plain words), the AT8 threads, the stained areas no candidate covers, *Excluded from analysis* (detection's areas, hatched by type: what the analysis leaves out) and the **Annotations** layer (where people's labels differ from detection: changed, rejected, unsure, added, split and merged objects, fixed outlines on hover, including the outline a person switched an object to, and people's bad areas; labels only, never what the analysis counts). **Tab** moves every object's outline at once to the next: stain outline → SAM small → SAM medium (a badge and the selected object's card say which is shown; the stain's default first; the stain outline where Segment Anything has none; pieces of a split show the outline chosen for them; stage 3 stores the SAM sizes for every counted object and some rejected ones, the others show their stain outline). It changes only the view. The **minimap** shows the whole core with each analysed tile shaded by its review status (reviewed, in progress, not annotated; hatched where no tile was analysed) and the view as a rectangle; a click moves there, a double-click opens that tile for annotation. **Space** hides everything drawn for a clean look, **←** / **→** go to the previous or next core with the same stain and layers, **↑** / **↓** move the picture. The *Acceptance threshold* card previews what would be accepted at another threshold (never saved; the analysis uses detection's threshold). |
| **Stain thresholds** | The DAB threshold of every slide and why it exists. |
| **Labels** | Labelling progress per stain, how often people and detection agree, the labelling sessions (create, open, close, share the link, follow people, download) and all labels as CSV. |
| **Models** | Every model with its status and settings, download buttons, training from all labels, and trained models compared with the current one. |
| **Results** | The main results table (download), every other table, plots by group and region, and the meaning of every column. |
| **Settings** | Study name, TMA layout, diagnostic groups and every file location, with a folder picker. Also leads to the Projects page to open or create another project, manages the people with accounts (*People*), your password, API tokens for scripts, and (folded away) chooses the server. |
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
   - **Fix the field** (objects). Whole fields with every object detection considered: point at an object and
     press a key to change its call or its outline, click to add what it missed. *How many fields*: fields nobody has
     checked, spread over TMAs and groups.
   - **Mark bad areas** (brush only). One core at a time with all its stains: change the areas of folds, artifacts,
     empty glass and blur, or say a stain looks fine. A sample of cores is enough. The brush is the field tool's
     (`BrushMasks`, labels/brush.ts), always on and with the same keys: each type's current area over the whole core
     image (GET `…/cores/{core}/{stain}/areas`), a drag adds to the chosen type, Shift-drag takes out of it
     (detection's part too: marking a part fine is no more than that), Tab / Shift+Tab (or − / +) a smaller or bigger brush (its size on screen: it stays the same as you zoom, so zooming in paints finer; strokes are kept in tissue units),
     the wheel zooms, Z undoes a stroke, [ / ] the other stain, Enter this stain is done.
   - "Check detection's calls" (one object at a time) is no longer made. Its old sessions stay listed as history and
     their labels stay valid object labels for training and evaluation.
3. **Which cores**: every ready core, some TMAs (batches, for slides and images), or fields clicked on the cores. Each
   core shows whether it has detection's candidates yet.
4. **Title**, and whether it opens to labellers right away.

*Advanced* holds:

- *How generous*: which candidates are shown, from every candidate detection proposed down to only clearly stained ones
  (1.5 × the stain threshold);
- *People who must finish each field* (default 1; 2 or more gives repeat labels for agreement);
- the session ID, a note for labellers, and *Also ask for the type*;
- the older kinds of session (see below).

**Sharing a set with a key (admins).** Every labelling session gets a **join key** when it is made: ten characters
from an alphabet without look-alikes (`k7mq2-xr9ta`, about 50 bits; wrong keys are rate-limited). The key names both
the project and the session. Open a session's *More* on the Labels page: *Who can label* shows the key (*Copy*), the
people who joined (*Remove*), and *Joining on*. *New key* stops the old key at once; people who joined keep their
access. Send people *Link for labellers* (the main page, `http://<this computer>:8765/`) and the key. Only the
members of a set, and admins, can open it: the server checks membership on every annotation request. Labels, claims on
fields, look-ahead holds and fair share are by account (labels carry the account id, `account`, besides the name).

**How to join a labelling set (annotators).** Open the link and sign in, or *Make an account* with your name and a
password. Type the key under *Add a set*: the set joins *My labelling sets* (sets of every project on the server, with
their progress), and *Start* or *Continue* opens it. Sign in again next time, on any computer, to carry on.

**Past names.** Labels made before accounts existed keep their names, and those names are reserved: nobody can sign
up as one. Under *Settings → People*, link an account to such a name (*Past labels as*): the account then labels under
that name and sees its history as its own.

**The tutorials.** Everyone's home, the Tutorials page and Help offer both: *Fix the field* and *Mark bad areas*;
nobody is sent into one. Both practise on **the one tutorial of the server**, the same for every project (the study's,
any other, a new empty one), so a tutorial never depends on a project's outputs and never shows a study's tissue: a
public dataset's field and core, made once with `stainid tutorial build --from <project> --field <tile> --core <core>
--credit "<line>"` into `~/.stainid/tutorial` (`$STAINID_TUTORIAL`; `stainid/tutorial.py`). It is a small stainID
project of its own (the two pictures, their tile and core-image rows, detection's outputs for them, and
`tutorial.json`: what to practise on, the credit line shown in the tutorial, the source's licence and citation, and the
"Change the type" step's example); the practice requests of every project run on it (`api.state.TUTORIAL`, the app's
middleware). Only images whose licence allows public display and redistribution go in: the server's tutorial is a
Keiser lab amyloid-YOLO 6E10 field (compact, diffuse, cored and small plaques, CAA along a vessel, rejects to bring
back, a pair to merge and outlines where Tab matters) and an Aβ42 core with a real edge, detached meninges, glass and
two knife streaks (MIT licence; see the README). Before it is built the tutorials say so. *Fix the field* (`/tutorial`,
`/api/tutorial`) takes about 3–5 minutes: step 1, the acceptance threshold (move it, watch objects switch, *Start
editing*), a clean look (Space), then the object changes — reject (Del), bring types back (numbers), unsure (`), add a
missed object, split (S), merge (M) — then the outlines — Tab (stain outline, Segment Anything small, medium), precise
editing (click / Shift+click a spot, then drag the dots) — the brush (B), saving (*Mark bad areas* on a core of one stain leaves out the stain switching). A coach asks for one thing at a time and
waits until it is done. *Skip tutorial* is always there and goes back home. Who finished or skipped each tutorial is kept with
their account (`/api/session/tutorials`, on every project; the `tutorials.json` of earlier releases counts too); a guest's
in the browser. The tutorials are public: someone without an account can try them. Each guest gets an anonymous id
of its own (the HttpOnly cookie `stainid_guest`), so the background work one guest's page waits for (`/api/session/work/{id}`)
is never another guest's. **Nothing of a practice is
kept, whatever is clicked**: a request on a practice session (`practice-2-<stain>`, `practice-areas`) runs with
`stainid.practice.PRACTISING` set, the practice routes answer as if saved, and every store refuses to write
underneath (the label log, the sessions, the candidate and added-object stores, the holds and claims: 403 instead);
`tests/test_practice_keeps_nothing.py` calls every write endpoint for a practice session and checks every store. **Try the tutorial** on the Labels page opens the same practice in a new tab: what
annotators see.

**Step 1: the acceptance threshold.** Every field opens with a slider at the top of the panel, *Acceptance
threshold*, at detection's own threshold for the stain's model (marked on the track). Dragging it (or [ and ] in steps
of 0.01) re-decides, live, where the objects start, as stage 4 would at that threshold: a candidate passes the model
where its probability is at least the threshold (a missing score counts as 0) and then gets stage 4's call, its rules
after the model included (a speck stays rejected, a deposit on a vessel wall or tissue edge is CAA; `labels/starting.py`
runs stage 4 with every candidate passing, so a promoted plaque gets detection's type); below it, dotted grey, or the
look-alike stage 4 names for a candidate the model does not count (an AT8 object whose stain lies along threads is a
thread). A candidate a rule decided keeps detection's call at every value, as does one in an excluded area, and
pointing at it the side panel says which rule ("Why"). A rule that names a class makes a class, not a reject: a
deposit detection's vessel rule calls CAA starts solid in CAA's colour (key Q), counted as CAA at every value ("Why: set
by detection's vessel rule …"), and is saved as CAA if left as it is; a type key or Del overrides it. Only real rejects
(a speck, a duplicate, an excluded area, an edge or fold, the model's own reject) are dotted grey. The Cores view, the
label comparisons and the Mark bad areas context draw them the same way. So raising the threshold never counts more, at 0 every candidate no rule
decided counts and at 1 only those scoring 1; a person overrides any call with a key. The Cores view's preview slider
follows the same rule. The panel counts them ("142
accepted at 0.43 (detection: 128 at 0.50)"). *Start editing* (or Enter) locks it, and so does the first edit of any
kind (a class key, Del, `, an outline, a precise edit, a split, a merge, a brush stroke); locked, it shows a lock and
its value and cannot move. The only way back is *Reset field*, which asks first, discards the field's unsaved edits and
unlocks it (`labels/fieldStart.ts`). The locked value is saved with the field (a `threshold` label, value e.g.
`0.430`, with every Save): on *Save and continue* the objects left as they started are saved as that person's labels at
that threshold, and the value itself is a training and evaluation signal (to tune each stain's cut-off later). Nothing
in the analysis reads it. The Cores view has the same slider as a preview only ("Preview only — the analysis uses
detection's threshold"): never saved, gone on reload.

**On the page.** Every object is always shown, one shape each, with a thin dark edge so it reads on light and dark
tissue:

- solid in its type's colour: every counted type and every named look-alike its own saturated colour (a positive
  identity without a type, as detection may call it, in the identity's colour);
- dotted amber with a "?" when someone is unsure (` or 0);
- dotted grey when it is not active: detection rejected it, someone rejected it (Delete), or it was just added and
  waits for its type (with a small diamond). A number key makes any of them solid;
- faint when it lies in an excluded area.

The page fits the window: the picture as large as it can be and the side panel scrolling by itself, so the page never
scrolls while the picture is panned (drag, or hold the middle button; scroll to zoom). The side panel shows, from the
top: the object under the mouse (its class, detection's call and score, why it was left out, its area and which of its
three outlines shows, *n/3*); the keys in their groups with their colours ("Plaques 1–4 · Not plaques Q–W · Del not a
plaque · ` unsure · B brush"); the keys; this field (checked, changed,
added; tags and a note); *Save* and *Save and continue*; **?** for the longer help, display options and the other
fields; and the *Tutorial* link. Pointing at an object also shows its name beside it on the picture. Every key acts on the object under the mouse:

| Key | What it does |
|---|---|
| 1–4, 5–9 | its counted type: 6E10 1 compact, 2 diffuse, 3 cored-neuritic, 4 small; AT8 1 pretangle, 2 mature tangle, 3 ghost tangle; NeuN 1 neuron (classes people add take 5–9). Solid, each type its own colour ([label_classes.md](label_classes.md)) |
| Q W E R, T Y U … | a named look-alike, solid in its own colour and its own class (kept apart from the counted types): 6E10 Q CAA, W staining inside a neuron; AT8 Q neuritic crown, W tau in an astrocyte, E thread. Detection's own look-alike calls start that way too: a deposit its vessel rule calls CAA, an AT8 object the model does not count whose stain lies along threads a thread |
| ` (or 0) | unsure: dotted amber with a "?", left out of training and evaluation; that person is not asked again, but the others are |
| Delete (Backspace) | the stain's single reject (not a plaque, not tau, not a neuron): it turns dotted grey and stays visible, the same as detection's own rejects (also an object the person added); a number key makes it solid again |
| Tab (or − / +) | move the object's outline to its next: stain outline → Segment Anything small → Segment Anything medium → stain outline (the three SAM ViT-B sizes of its point prompt, which stage 3 stores; asked live with one click where the store has none; SAM large only where it is the one shown first). The stain's default comes first (6E10 and AT8: the stain outline; NeuN: the SAM output stage 3 picked), the panel and a badge say which is shown, and the one an object is left on is saved with the field per candidate (an `outline` label, value `stain`, `sam_small`, `sam_medium` or `sam_large` — `sam` of earlier releases is the output stage 3 kept — with its points; a changed option counts as an outline fix on the Models page, the analysis keeps the default). On a piece of a split: its split outline → SAM small → SAM medium, asked live with a point on the piece, points on its sibling pieces to leave out and a box around it, the siblings cut out of the result (its largest part kept); the choice is saved with the piece's outline (value `split`, `sam_small` or `sam_medium`, the split outline kept with it). With nothing under the mouse Tab does nothing; the badge shows the outline of the object under the mouse. On an object added with one click, Tab / Shift+Tab step through Segment Anything's three sizes. With the brush on: a smaller or bigger brush (Shift+Tab bigger) |
| [ / ] | step 1: the acceptance threshold down / up by 0.01 (until it locks) |
| S | split mode, on and off (Esc leaves): a click on an object splits it in two (a watershed of its outline on its distance transform, the stain breaking ties; `labels/objectedit.py`) as a preview; a second click elsewhere on it moves the split, Enter keeps it, Esc drops it. Or drag a line: every object it crosses right through is cut along it. The pieces keep the object's type and the keys retype them |
| M | merge mode, on and off (Esc leaves): click objects to add them to the merge or take them out (highlighted; "3 selected · Enter to merge"), or drag over objects to add every one the stroke touches; Enter merges them into one object, their outlines' union (bridged by a small closing where they do not touch), with the type of the largest piece |
| Click an object | precise editing of this object, with everything else locked (tags, note, other fields, saving): a click adds a spot to it, Shift+click leaves one out, drag the white dots (at most 48, the outline simplified to them); Z one step back; a click beyond its neighbourhood (its box grown by half its size, at least 24 screen pixels, drawn faintly) or Enter keeps the edit (the view goes back to where it was), Esc cancels. Classes only need the mouse over the object (no click). The right button has no meaning on the field; the middle one pans |
| Space | the field tool: hide / show everything drawn over the picture (outlines, dots, names, the excluded areas, layers), for a clean look at the tissue; "Overlays hidden · Space" shows meanwhile. |
| B | the brush, on and off (Esc also leaves). People change the existing areas rather than drawing separate new ones: for each type (1 fold or tissue damage, 2 artifact: scratch, bubble, debris, linear, 3 no tissue / empty, 4 out of focus or bad staining; the keys 1–4 choose it while the brush is on, and the class keys rest) the current area is detection's area of that type with everyone's saved edits merged in (GET `…/fields/{tile}/areas`, `exclusions.final_say`), and while the brush is on the chosen type's current area is drawn as one merged region (the others faintly). A drag adds the brushed pixels to it (a union: overlapping strokes are one region, a stroke touching a detected fold extends that fold, pixels of another type become this type); Shift-drag takes them out of the chosen type's area, detection's part included. Tab makes the brush smaller, Shift+Tab bigger (its size in the panel and as the ring under the mouse); the wheel always zooms; the middle button pans; outlines are locked meanwhile. Saved with Save as one edit (an `exclude` label, `labels/exclusions.py`: rings per type, "1"–"4" added, "-1"–"-4" taken out), an ordered log resolved per type into masks (`exclusions.annotated` / `final_say`, one region set per type): **people's areas**. Over every field the page draws detection's own areas hatched, *Excluded from analysis* (GET `…/fields/{tile}/excluded`, the automatic layer, the same as the Cores view's: folds, linear artifacts, no tissue, the AT8 edge guard), and people's areas outlined on top (GET `…/fields/{tile}/annotated`: painted ones in their type's colour, the parts marked fine dashed green). People's areas only train and evaluate models: objects in them (by people's final say over detection's areas, `exclusions.final_say`) are drawn faint, are not asked about, and are left out of training and scoring; the bad-area detector is compared with them on the Models page. The analysis never reads them, so marking an area changes no result and makes no step out of date. Space hides them with everything else; *Display options* turns them off |
| Z | undo the last unsaved change (a brush stroke too) |
| Enter | precise editing: keep the edit; drawing by hand: close the outline |
| Ctrl+S (⌘S) | save, and stay on the field |
| Ctrl+Enter (⌘Enter) | save and continue |

In round 2, a click on empty tissue outlines a missed object with one click (Segment Anything). It is a new
candidate, dotted grey, and Tab / Shift+Tab switch to a smaller or bigger one of Segment Anything's three sizes, until a class key makes it solid. *Draw an object by hand…* is the last resort. One click answers at once,
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

**Fields are claimed, so people never collide.** The unit people claim is a field (a tile):

- Opening a field holds it for that person while their page is open, so the others are handed other fields.
- A field that enough people finished (*People who must finish each field*) is not handed to anyone else.
- What others decide meanwhile shows within seconds ("checked by Ben").

**One set of answers.** Every session and every person write class labels on the same candidates, keyed by
candidate id (`labels.decisions`). A candidate that enough people decided shows as checked in every "Fix the field"
session (and leaves the lists of the old "Check detection's calls" sessions). Labels made before the rounds (earlier sessions, imported label sets) are matched onto the
candidates by position, so nothing anyone labelled is asked again. The decisions train the models like any other label
([Training a model](#training-a-model)).

**AI labels are never shown while annotating.** The AI's labels, objects and bad areas alike (the imported reference
sets `import: …`, the earlier "AI reviewer" and AI review sessions of accounts `ai:<name>`; one rule,
`labels.store.is_ai_label` and `webapp/src/labels/aiLabels.ts`), neither show nor set where anything starts in Fix the
field, Mark bad areas, the brush or the tutorial: an object starts from detection's call (or the person's own), with no
marker and no "checked by" for them, and the brush and Mark bad areas start from detection's automatic areas plus
people's edits. Annotators correct the pipeline's output, not the AI's. The AI labels stay on the Labels and Models
pages (training and evaluation take them as their session is included) and in the Cores view's Annotations layer,
marked as AI.

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

These need their fields **prepared**. The *Prepare fields for labelling* step (`stainid labels prepare`) finds every
candidate as the detection step does and stores its measurements and detection's call in
`outputs.reviews/pool/<stain>/<field>.csv` (10–30 s per field). Sessions prepare their own fields in the background.

Labels made before the label store (training sets, the CNN-outlier adjudication and the white-matter field review)
were imported once with `stainid labels import`, each as its own source (`import: <name>`); the original folders are
unchanged. Every label belongs to a session; the imported ones (no session) together form the read-only pseudo-session
*AI reference labels* (`reference`), which the Labels page lists and lets you look inside like any other session.

**AI review batches** (`stainid review ai-batch build|ingest`, `stainid.review.ai_batch` / `ai_ingest`). *build* picks,
per stain, 8 fields from different TMAs as "Fix the field" draws them (none that people, earlier AI rounds, a session
or the tutorial have; cores whose bad areas a person checked first; about half rich in the stain's object, a quarter
sparse, a quarter near the model's threshold). Every own candidate outside the excluded areas is asked; a field with
more than 80 is cut to a sub-field (and, if need be, a stratified sample), recorded in `fields.csv`. It writes
numbered picture sheets (raw crop and the default outline, with the call, probability and rule), an overview per field
on a field-pixel grid for missed objects, `manifest.csv`, `REVIEW_INSTRUCTIONS.md` (the stain's labels from the
vocabulary) and two assignments (halves A and B by field, about 10 % of the objects in both) under
`outputs.reviews/ai_review_round<N>/<stain>/`. *ingest* reads the reviewers' `review_labels_<half>.csv` into the session
*AI review round N · <stain>* (kind `ai_review`, read-only), each reviewer the account `ai:<name>`: class labels on
the candidates, an outline judged "ok" confirmed as the app records it, missed objects as added objects, objects of
both halves kept where the reviewers agree and "unsure" where they do not. Like any session it can be chosen for
training on the Models page; the annotation pages do not show `ai:` labels.

### Outline fixes

How well detection outlines objects is measured on the outlines people fix in *Fix the field* (*Models → Outlines →
Outline fixes*, `stainid.labels.outline_fixes`). Per stain: how many objects people changed the outline of (a precise
edit, a split, a merge), and for those objects each of detection's outlines — the stain outline, Segment Anything's
small, medium and large masks, and the one shown first — against the person's outline: the median IoU, the median
area ratio (detection's area over the person's) and how often each was the closest. Switching to another of
detection's outlines without drawing is counted apart. When one outline is clearly closer than the one shown first
(at least 0.05 higher median IoU over at least 5 objects) a hint says so, e.g. *SAM medium matches people's fixes
better for AT8 (IoU 0.81 vs 0.66)*: the evidence for the stain's default outline on the Run page, which is never
changed for you. It is an indication, not a full accuracy measure: only the objects people chose to fix count, and a
precise edit starts from the outline shown. *Download the fixed outlines (CSV)* gives one row per object with the
person's outline (core pixels) and its comparison, to train a better outline from; the outlines themselves are labels
in the label log like any other (read through `labels.compare`, the Annotations layer's reader).

### The segmentation check

A model's scores (AUC, precision, recall) are measured on the candidates the detection step proposed, so they cannot see
objects that were never proposed, or that its segmentation cut into pieces or merged with a neighbour. The segmentation
check measures that end to end, where every object of some fields was outlined: by a dataset's raters (polygons in a
dataset package, the public-dataset projects) or in an *Outline every object* session of an older project. Such sessions
are no longer made and cannot be worked on; their outlines stay in the label log (tasks `object` and `complete`), are
listed on Labels and are scored here. The card on *Models → How well does detection outline objects?* shows only where
there is something to score.

**Score now** (`stainid score-segmentation`) matches every outlined object one to one with (a) every candidate:
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
