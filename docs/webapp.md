# Using the app

## Starting

Double-click `Start stainID.command` (macOS) or `Start stainID.bat` (Windows), or run:

```bash
stainid serve                              # opens http://127.0.0.1:8765 in the browser
stainid --project path/to/study serve      # open a specific project
stainid serve --no-browser --port 9000
stainid serve --host 127.0.0.1             # reachable from this computer only
```

The app opens the project used last, or the current folder. If that folder has no `stainid.yaml`, a welcome screen
offers to create a new project or open an existing one. If stainID is already running, starting it again just opens the
browser.

**Admin password.** The app asks for the admin password; it is `admin` until you change it under *Settings → Admin
password*. The app listens on your network (so annotators can reach the annotation pages), but everything except
those pages needs the password, and while it is still `admin` you can only sign in on the computer running stainID.

## A study, from start to finish

| | Where | What you do |
|---|---|---|
| 1 | Welcome screen | Create a project: a study name and an empty folder. |
| 2 | **Settings** | Rows and columns of cores per slide, the core-name prefix (`LIP-` gives cores like `LIP-3_B-2`) and the diagnostic groups (code, label, colour, plot order). |
| 3 | **Models** | Download the published models. Add the NeuN, 6E10 and AT8 models, or train them (see [Training a model](#training-a-model)). |
| 4 | **Workflow** | Register the slides, upload the TMA map, then run the steps from top to bottom (or press *Run all remaining steps*). |
| 5 | **Cohort & cores** | Look at the detections on the tissue, core by core. |
| 6 | **Labels** | Review fields (alone or with others) to measure how accurate the detections are and to collect labels for training. |
| 7 | **Results** | Download the results table (one row per donor and region) and read what every column means. |

## The Workflow page

- **Pipeline map.** Every step is a box, arranged in the order steps can run. An arrow means "needs the result of".
  Colours show done, partly done, running, waiting, failed and "made with an older model". Hover over a step to
  highlight what it needs and what needs it; click it to open its card.
- **Now running.** Running and waiting steps, with progress, elapsed time, time left and why a step is waiting.
  *Run all remaining steps* queues every unfinished step (except the optional outlines).
- **Step cards**, one per step, in order. Finished steps fold away. Each card shows:
  - what the step does and how long it takes;
  - **Needs** and **Makes**: every file with its location and whether it exists. A missing file has a button: go to
    the step that makes it, download it, or show the expected format and a template;
  - options, with advanced ones hidden;
  - Run / Continue / Run again, a progress bar, *Show log*, and a plain-language reason when a step fails.

The *Register slides* card holds the slides-table editor, and the *Attach TMA map* card holds the template download and
the CSV upload.

## How steps run

Steps run in the background, started by a queue with two rules:

- **Order.** A step waits while a step that makes something it needs (an arrow on the map) is still running or
  waiting. The same step with the same options never runs twice at once.
- **Load.** At most two heavy steps (Cellpose, detection, outlines) run at the same time.

So you can press Run on several steps in a row and they start in the right order; each waiting step says what it is
waiting for. Just before a step starts, the queue checks that everything it needs exists. If not, the step is
**skipped** and its card says what is missing and which step makes it. When a step fails or is stopped, the steps
waiting for it are skipped too, instead of failing one after another.

Steps keep running when the browser is closed, but stop when the stainID terminal window is closed; running a step
again continues where it stopped. When you switch to a different stain model, the detection steps made with the old one
are marked **Made with an older model** until you run them again with *Start over* (under Advanced options).

## Other pages

| Page | What it is for |
|---|---|
| **Home** | Your next step (with a button to it), a checklist of every step, cohort numbers and recent jobs. |
| **Cohort & cores** | Every core of every TMA; click one to see the whole core, then click any spot for a full-resolution preview (220–660 µm) with the AI detections where detection has run and the DAB layer. Only the preview is sent over the network, never the whole scan. The analysed fields themselves are reviewed on Labels. |
| **Stain thresholds** | The DAB threshold of every slide and why it exists. |
| **Labels** | Labelling progress per stain, how often people and the AI agree, the labelling sessions (create, start, stop, share the link, follow people, download) and all labels as CSV. |
| **Models** | Every model with its status and settings, download buttons, training from all labels, and trained models compared with the current one. |
| **Results** | The main results table (download), every other table, plots by group and region, and the meaning of every column. |
| **Settings** | Study name, TMA layout, diagnostic groups and every file location, with a folder picker. Also opens or creates another project. |
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
2. **Start** it and send the link shown on the page (`http://<this computer>:8765/annotate`). People register a name
   and can stop and continue later under the same name. Only started sessions are listed, and nothing else in stainID
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
- **Validation.** Leave-one-TMA-out (5-fold if there are too few TMAs). The current model is scored on the same labels,
  so the Models page can show both side by side.

The model is saved to `outputs.trained_models/<stain>_<time>/bundle.joblib` with `report.json`. It is used only after
you press *Use this model*, which sets `models.<neun|amyloid|tau>` in `stainid.yaml`. Detection steps never mix
results from two models: run them again with *Start over*, which moves the old results aside, then *Make results
tables* again.
