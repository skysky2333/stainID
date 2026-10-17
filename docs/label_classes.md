# Label classes: identity and type

Every object label names an **identity class** — what the object is — and, where the class has them, a **type**
(subtype). The stain's detection model learns from the identity only, exactly as before types existed; the types are
kept so that a second-stage model can learn them later (for example pretangle against mature tangle).

Each identity has a **role** in its stain's model:

- *the object the model finds* (`positive`): tau+ neuron, plaque, NeuN+ neuron;
- *not the object* (`negative`): everything else people may say it is, which teaches the model what to leave out;
- *left out of training* (`skip`): "unsure".

## The built-in classes

| Stain | Class (role) | Types |
| --- | --- | --- |
| AT8 | tau+ neuron (positive) | pretangle · mature tangle · ghost (extracellular) tangle |
| | neuritic plaque (tau crown) · thread or dystrophic neurite · not tau · artifact or unclear (negative) | |
| | astroglial tau (negative) | tufted astrocyte · thorn-shaped astrocyte |
| | unsure (left out) | |
| 6E10 | plaque (positive) | compact (dense-core) · diffuse · cored-neuritic · small (too small to type) |
| | CAA / vascular amyloid · intraneuronal staining (APP/Aβ inside neurons, which 6E10 also stains) · not a plaque · artifact or unclear (negative) | |
| | unsure (left out) | |
| NeuN | NeuN+ neuron (positive) | strongly stained · weakly stained · large pyramidal neuron (hidden until shown in Settings) |
| | not a neuron (glia, endothelium) · artifact or unclear (negative) | |
| | unsure (left out) | |

A stain stainID has no classes for (a dataset from another lab) starts from *object*, *not an object*, *artifact or
unclear* and *unsure*. Every class and type has a plain description, shown on the labelling pages, and may have an
example picture.

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

The identity models therefore learn from exactly the same positives and negatives as before (checked on the study's
label log: every stain's training truth is identical, and so are the objects that set the tangle-maturity and
plaque-morphotype thresholds). An object's identity is what most people said; its type is what most of the people
who gave one said (a label without a type does not count against one with a type).

New labels store the identity's id as `value` and the type's id as `subtype`. Ids never change: renaming a class or a
type in Settings changes only the name people see. Pages and imports may also name a class as `identity / type`,
`identity/type` or `identity: type` (by id or name), e.g. `plaque/compact`.

## Labelling with types

The person who creates a session chooses whether people are also asked for the type (*Also ask for the type*, on by
default; sessions made before this choice ask for it). On the quick check and on the field page:

- a class key (1–7) answers at once when the class has no types;
- for a class with types, a number right after it (within 1 s, `labels.subtype_wait_s`) picks the type by its place
  (tau+ neuron: 1 pretangle, 2 mature tangle, 3 ghost tangle); 0 or Enter means no type; when nothing follows, the
  class stands without a type;
- a type's own key (Q, W, T… by default; E is the field tool's exclusion brush) answers class and type in one press;
- with the mouse: click the class, then its type or *No type / not sure*.

The keys are shown on screen next to the classes and under the picture. Without types asked, only the class keys apply
and detection's confirmed calls (fields sessions, *Finished*) are stored without a type.

The annotation rounds use the same class and type keys, acting on the object under the mouse (round 2) or on screen
(round 1). Their other keys are listed in [webapp.md](webapp.md#annotation-rounds): 0 unsure, Delete not an object,
Tab / Shift+Tab for the outline size, Space for precise editing, Z undo and Enter to keep detection's call.

## Where types show

- **Look at the labels** groups the class filter by identity (*plaque: all*, then each type and *type not recorded*);
  objects people split on the type read *plaque (people disagree on the type)*. Advanced on an object: *Use as the
  example picture of …*.
- **Labelled so far** counts objects per class and per type (*tau+ neuron 420 (pretangle 170 · mature tangle 250)*).
- **Download all labels** adds `identity`, `subtype`, `role` and `class` (as the page shows it) to every class label;
  `value` stays as stored.
- Detection's calls read in the same words (*tau+ neuron · mature tangle*); "Detection gives another type" compares types.

## Changing the classes (Settings → Label classes)

The list of each stain's classes, keys and types is always shown; *Change the classes* (Advanced) edits names, keys,
descriptions, example pictures, hides classes or types, adds types and adds classes (choosing their role), and sets the
time for the type key. Built-in classes can be hidden and renamed but not removed, and keep their role, so a project's
settings never change what the identity models learn from existing labels. Keys are one letter or number, unique per
stain; V, A, O (field-page tools), E (the exclusion brush), Z (undo) and 0 (no type) are taken, and types cannot have number keys.

The classes are kept in `stainid.yaml`, per stain, only when they differ from the built-in ones:

```yaml
labels:
  subtype_wait_s: 1.0
  classes:
    AT8:
      - {id: tau+ neuron, name: tau+ neuron, role: positive, key: "1", description: ..., example: 3f2a9c1b7d4e.jpg, hidden: false,
         subtypes: [{id: pretangle, name: pretangle, key: q, description: ..., example: "", hidden: false}, ...]}
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
