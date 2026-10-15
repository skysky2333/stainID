# Adding new slides to a study

This guide is for anyone who has new TMA slides to add, whether to a new study or to one that already runs. You do
not need to know how stainID works inside: the **guided setup** in the app does the fiddly parts and asks you only to
confirm what it found. It takes about 15 minutes of your time, then the computer works on its own.

## Before you start

Have these ready:

1. **The slide scans**, one file per TMA and stain (`.vsi`, `.svs`, `.ndpi`, `.czi`, `.tif` …). Copy them into the
   study's slides folder (by default `data/slides` in the project folder). Names like `TMA LIP-8 NeuN.vsi`, with the
   TMA number and the stain in them, let stainID fill the table in by itself; with other names you choose them from
   a list.
2. **The TMA maps**: which donor is on which core. Whatever your lab keeps them in works: an Excel sheet, a table in a
   Word document, a CSV file, or even a photo or a scan of a printed map (you type that one in).
3. **The case spreadsheet** (optional): one row per donor with age, sex, PMI, APOE, CERAD, Braak and so on.

Then open stainID, pick the study on the **Projects** page (for a new study: **New study**, which opens the guided
setup) and press **Add slides** on Home.

## 1. Slides

stainID lists the scans in the slides folder, each with a small picture of the slide so you can read its label
(click it to see it larger). New scans are marked *new*. For each scan choose its **TMA** and its **stain** from the
lists:

- where the file name says them clearly, they are already filled in and marked *from the file name*; check them;
- highlighted rows still need a choice ("choose the TMA", "choose the stain"); when the file name holds several
  numbers, the row says which ones;
- a TMA the study does not have yet: choose **New TMA…**, type it (`LIP-12` or just `12`) and press **Add**; it is
  then in the list for the other slides too.

**Save and continue** becomes available once every slide has a TMA and a stain (the line beside it says how many
still need one) and no TMA has the same stain twice.

*If no scans are listed:* press *Choose another folder…* and pick the folder with the scans.

## 2. Cores

Press **Find cores** (about a minute per slide). stainID finds every core on each slide and works out how many rows
and columns of cores the slide has, even when one array is smaller than the others. You then see one picture per
slide with a circle around every core and a label such as `B-2` (column B, row 2; `A-1` is top left).

For each picture:

- every circle sits on a core and A-1 is top left → press **Looks right**;
- the circles miss cores, or there are too many or too few rows or columns → press **Fix**, enter the right number of
  rows and columns, and press **Find the cores again**;
- the slide lies upside down on the scan (A-1 is bottom right) → **Fix**, choose *turned 180°*, **Find the cores again**.

## 3. Map

Press **Add map files** and choose the map files (you can choose several at once). For each map stainID shows a grid
like the array, with what it read from every cell underneath: the donor, the brain region, the group and the
pathology. Typing slips such as extra spaces (`17 00 FR`), `OC C` for OCC or `JOG3` for JOG_003 are read correctly,
and cells such as `CBLM` (cerebellum) are recognised as control cores.

- Check that **This is the map of** shows the right TMA.
- Cells with a **yellow edge** could not be read completely: click into the cell and correct it, e.g. `1673 FR AD`.
- Press **Save as the map of TMA …**.

*A map that is only a photo:* press **Type it in** next to its TMA. You get an empty grid with a small picture of the
core at each position; type each cell (e.g. `1673 FR AD`). You can also copy cells from Excel and paste them into the
first cell; they spread over the grid.

## 4. Donors (optional)

Press **Add the case spreadsheet** and choose the file. stainID finds the column names, matches them (age, sex, PMI,
APOE …), reads section rows such as “CONTROL <65” as the group, and understands values like `~15` or `>90`. It keeps
the donors that are on your maps. Look at the preview and the notes above it, then press **Save**. If a column was
matched wrongly, open **Advanced** and change it.

No case spreadsheet? Press **Skip**.

## 5. Checks

stainID compares the maps, the donor sheet and the slides and lists anything that does not fit, in plain words, for
example:

- *Donor 2323: map says CERAD B, donor table says 0. Use the donor table?* → press the answer you want;
- *3 core positions found on the slides are not on the map* → if those positions hold no donor, press **Mark them as
  positions without a donor**; otherwise add them on the Map screen;
- *1 donor id is used more than once in the donor table* → **Keep the row on the map's TMA**.

Items marked **Needs fixing** must be fixed before the analysis can run; **Please look** items are worth a look but do
not stop it.

## 6. Run

stainID lists what is left to do and roughly how long it will take on this computer. Press **Run everything**. The
steps start in the right order by themselves; you can close the browser, and the Workflow page shows how far they
have got. When they are done, the results are on the **Results** page.

## If something goes wrong

- A step that fails says why on the Workflow page; the guided setup's screens can be opened again at any time to
  correct a slide, a map or the donor sheet, and the step run again.
- Nothing is lost by going back: saving a map replaces only that TMA's map, and saving donors only updates the donors
  in the sheet.
- Everything the guided setup does is also visible under **Advanced** on each screen (the files it writes, the full
  tables, the options of the steps) and on the [Workflow page](webapp.md#the-workflow-page).
