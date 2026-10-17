# Label classes: identity and type

Every object label names an **identity class** — what the object is — and, where the class has them, a **type**
(subtype). The stain's detection model learns from the identity only, exactly as before types existed; the types are
kept so that a second-stage model can learn them later (for example pretangle against mature tangle).

Each identity has a **role** in its stain's model:

- *the object the model finds* (`positive`): tau+ neuron, plaque, NeuN+ neuron;
- *not the object* (`negative`): everything else people may say it is, which teaches the model what to leave out;
- *left out of training* (`skip`): "unsure".

## The built-in classes and their keys

One key picks a type at once (`stainid.labels.vocabulary.assign_keys`, the one key assignment the labelling pages,
their key lists, Help and the tutorial all read): numbers for the **counted types**, letters for the **named
look-alikes** (solid, in their own colour, not counted), **Del** for the stain's single reject and **`** for unsure.

| Stain | Counted (positive) | Named look-alikes (negative) | Del (negative) | ` |
| --- | --- | --- | --- | --- |
| 6E10 | plaque: **1** compact (dense-core) · **2** diffuse · **3** cored-neuritic · **4** small (too small to type) | **Q** CAA / vascular amyloid · **W** staining inside a neuron (APP/Aβ, which 6E10 also stains) | not a plaque | unsure |
| AT8 | tau+ neuron: **1** pretangle · **2** mature tangle · **3** ghost (extracellular) tangle | **Q** neuritic crown (tau around a plaque) · **W** tau in an astrocyte · **E** thread or dystrophic neurite | not tau | unsure |
| NeuN | **1** NeuN+ neuron | — | not a neuron | unsure |

- Each counted type and each look-alike has its own colour on every page (`Vocabulary.colour_slots`, sent to the web
  app as `vocabulary.slots`); the reject is dotted grey, the same as detection's own rejects, and a number key makes it
  solid again; unsure is dotted amber with a small "?", and is left out of training and evaluation.
- Together the counted types are the stain's object ("plaque", "tau+ neuron"): what the detection model learns and what
  the analysis counts. A label with the identity and no type (an older one, or detection's own call without a type)
  stays valid; the pages draw it in the identity's colour without a type until someone presses a key, and it is saved
  as detection's call if nobody does.
- NeuN has one type: how strongly a neuron is stained is measured, not judged, so *strongly / weakly stained* are
  hidden (kept for older labels), and *large pyramidal neuron* stays hidden until shown in Settings. AT8's astrocyte
  types (tufted, thorn-shaped) are hidden too: **W** names the look-alike.
- *Artifact or unclear* is no longer a class: debris is **Del**, unclear is **`**, and bad areas are painted with the
  brush (**B**). Its old labels read as the stain's reject (negative, as they always were).
- Quiet second keys: **0** for **`** (for keyboards where ` is awkward) and **Backspace** for **Del** (a Mac's delete
  key). **B** toggles the brush on the field page; while it is on, 1–4 are the area types and the class keys rest.
  None of the keys clash with Space, Enter, Esc, Shift, Tab, the arrows, S (save), Z (undo), V A O (the older field
  page's tools), and no Ctrl/Cmd combination is taken.
- Classes people add (Settings) get **5**–**9** for counted types and **T Y U …** for look-alikes, unless they choose a
  free key themselves.

A stain stainID has no classes for (a dataset from another lab) starts from *object* (1), *not an object* (Del) and
*unsure* (`). Every class and type has a plain description, shown on the labelling pages, and may have an example
picture.

## Labels made before types existed

The label log is never rewritten. Older labels stored one value; they are read as identity and type
(`stainid.labels.vocabulary.LEGACY`):

| Stored value | Identity | Type |
| --- | --- | --- |
| tau+ neuron: tangle | tau+ neuron | mature tangle |
| tau+ neuron: pretangle | tau+ neuron | pretangle |
| tau+ neuron, tau+ neuron (type not recorded) | tau+ neuron | — |
| not a tau+ neuron | not tau | — |
| compact plaque · diffuse plaque · small plaque | plaque | compact · diffuse · small |
| plaque, plaque (type not recorded) | plaque | — |
| vascular | CAA / vascular amyloid | — |
| artifact or unclear | the stain's reject (not tau, not a plaque, not a neuron) | — |

The identity models therefore learn from exactly the same positives and negatives as before (checked on the study's
label log: every stain's training truth is identical, and so are the objects that set the tangle-maturity and
plaque-morphotype thresholds). An object's identity is what most people said; its type is what most of the people
who gave one said (a label without a type does not count against one with a type).

New labels store the identity's id as `value` and the type's id as `subtype`. Ids never change: renaming a class or a
type in Settings changes only the name people see. Pages and imports may also name a class as `identity / type`,
`identity/type` or `identity: type` (by id or name), e.g. `plaque/compact`.

## Labelling with types

The person who creates a session chooses whether people are also asked for the type (*Also ask for the type*, on by
default; sessions made before this choice ask for it). With types asked, each counted type has its number (above).
Without, the counted object has one key (1: plaque, tau+ neuron) and detection's confirmed calls (fields sessions,
*Finished*) are stored without a type. With the mouse, the side panel lists the same keys in their groups
("Plaques 1–4 · Not plaques Q–W · Del not a plaque · ` unsure · B brush"); a click picks one.

The annotation rounds use the same keys, acting on the object under the mouse (round 2) or on screen (round 1). Their
other keys are listed in [webapp.md](webapp.md#annotation-rounds): Tab / Shift+Tab for the outline size, a click on the
object for precise editing (Space in round 1; in round 2 Space hides the drawings), B the brush, Z undo and Enter to
keep detection's call.

## Where types show

- **Look at the labels** groups the class filter by identity (*plaque: all*, then each type and *type not recorded*);
  objects people split on the type read *plaque (people disagree on the type)*. Advanced on an object: *Use as the
  example picture of …*.
- **Labelled so far** counts objects per class and per type (*tau+ neuron 420 (pretangle 170 · mature tangle 250)*).
- **Download all labels** adds `identity`, `subtype`, `role` and `class` (as the page shows it) to every class label;
  `value` stays as stored.
- Detection's calls read in the same words (*tau+ neuron · mature tangle*); "Detection gives another type" compares types.

## Changing the classes (Settings → Label classes)

The list of each stain's keys, in their groups, is always shown; *Change the classes* (Advanced) edits names, keys,
descriptions, example pictures, hides classes or types, adds types and adds classes (choosing their role). Built-in
classes can be hidden and renamed but not removed, and keep their role and their key, so a project's settings never
change what the identity models learn from existing labels. A counted type's key is a number, a look-alike's a letter,
unique per stain; Del, `, 0, B (the brush), S, Z and V A O are taken. Settings saved with the old keys (identity 1–7,
then a type by its place, or Q W T R) get today's built-in keys (`OLD_KEYS`). `labels.subtype_wait_s` is no longer
used (one key picks the type) and is kept only so older settings load.

The classes are kept in `stainid.yaml`, per stain, only when they differ from the built-in ones:

```yaml
labels:
  classes:
    AT8:
      - {id: tau+ neuron, name: tau+ neuron, role: positive, key: "1", description: ..., example: 3f2a9c1b7d4e.jpg, hidden: false,
         subtypes: [{id: pretangle, name: pretangle, key: "1", description: ..., example: "", hidden: false}, ...]}
      - ...
```

Example pictures are stored with the labels, `outputs.reviews/class_examples/<stain>/<file>.jpg` (at most 480 px).

## Dataset packages

A package's class map (`dataset.class_map`, label → stainID class) may target a type: `plaque/compact`,
`tau+ neuron/pretangle`. Imported labels are stored as identity and type like any other, and the scores against
the models compare types through the same vocabulary (a rater's `plaque/compact` is detection's *compact plaque*; a
rater who marked plaques without a type is compared with every accepted plaque). The import page offers every class
and `class/type` of the stain's family.

## Code

- `stainid/labels/vocabulary.py`: built-in classes, legacy mapping, reading values (`Vocabulary.resolve`), the
  settings (`of`, `from_config`, `problems`) and the page keys (`keymap`).
- `stainid/labels/store.py`: `classes` (each class label with identity, subtype and role) and `consensus`.
- `stainid/api/routes/classes.py`: `/api/labels/classes` (read, save, back to built-in), example pictures.
- `webapp/src/labels/classKeys.ts` (key logic, tested by `tests/js/class_keys.test.ts`), `labels/classes.tsx`
  (class chooser), `labels/ClassSettings.tsx` (Settings).
