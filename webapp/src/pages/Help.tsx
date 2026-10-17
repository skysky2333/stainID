import { useState } from 'react'
import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { PageHead } from '../components'
import { GLOSSARY } from '../glossary'

const FAQ: [string, ReactNode][] = [
  ['What do I need to start?', <>Your slide scans (one per TMA and stain) and the TMA maps saying which donor sits at each core position, in whatever form your lab keeps
    them (Excel, Word, CSV, or a photo you type in); optionally a spreadsheet with donor information (age, sex, APOE …). Create a project, then follow
    the <Link to="/setup">Set up</Link> page.</>],
  ['How do I add new slides?', <>Copy the scans into the project's slides folder and press <b>Add slides</b> on <Link to="/">Home</Link>. Set up lists the new
    scans, finds their cores and reads their maps; then <b>Run everything</b> analyses them. The study's other slides are left as they are.</>],
  ['The circles for the cores are in the wrong places.', <>Open the <Link to="/setup/cores">Cores</Link> screen of Set up and press <b>Fix</b> under that slide:
    enter its rows and columns, or say that it lies turned or mirrored, and its cores are found again.</>],
  ['How long does a full study take?', <>Setting up (finding and exporting the cores) takes about an hour for 20 slides. Detection is the slow part, because it looks at
    every tile of every core (a core holds 15–40 tiles): a few seconds per tile with an NVIDIA GPU, a minute or more on a laptop, so a few hours to a few days
    for a large study. Everything can be stopped and continued later.</>],
  ['Is the whole core measured?', <>Yes. Each core is cut into square tiles of about 560 µm and every tile that holds tissue is analysed, so every
    measure covers the whole core. Open a core on <Link to="/cohort">Cores</Link> and switch on the tile grid to see them.</>],
  ['What runs when? Why does a step say “Waiting”?', <>You can start several steps at once; stainID runs them in the right order.
    A step waits while a step it needs is still running or waiting, and at most two heavy steps (Cellpose, detection, outlines) run at the same
    time so the computer stays usable. The <Link to="/workflow">Run</Link> page shows what is running now, how far it has got, roughly how long
    is left, and why anything is waiting.</>],
  ['Can I close the browser?', <>Yes. Steps keep running as long as the stainID server is running (the terminal window started by “Start stainID”).
    Closing that window stops running steps; starting a step again continues where it stopped.</>],
  ['A step failed or was skipped. What now?', <>The step on the <Link to="/workflow">Run</Link> page says why. Steps that were waiting for a failed step are
    marked <b>Skipped</b> (they would only fail too); run them again once the failed step works. Common causes: a missing file (the step says which earlier step makes it),
    a model that is not downloaded yet (see <Link to="/models">Models</Link>), or results made with a different model (run again with <b>Start over</b>,
    under Advanced). <b>Show log</b> has every detail.</>],
  ['Where are my files?', <>Everything lives in the project folder (see <Link to="/settings">Settings</Link>). Each step lists the files it makes, with “Show in Finder”.
    The main results table is <code>data/analysis/results_donor_region.csv</code>; download it and the others on <Link to="/results">Results</Link> (Downloads).</>],
  ['What does a measure in the results mean?', <>Open <Link to="/results">Results</Link> → <b>What the measures mean</b>: every column of the results tables with
    its unit and a plain explanation, and the column name used in the CSV file.</>],
  ['How do I know the results are right?', <>Look at the cores and what was found in them on <Link to="/cohort">Cores</Link>, and ask colleagues to check detection’s calls
    on <Link to="/labels">Labels</Link>: each stain shows how often detection agrees with people, and <Link to="/models">Models</Link> shows the cross-validated precision and
    recall on all labels.</>],
  ['The detections look wrong for my staining.', <>Label objects on your own slides (an annotation round on <Link to="/labels">Labels</Link>; 100–200 objects is a good start),
    then train on <Link to="/models">Models</Link> and switch to the new model if it scores better. Then run the detection steps again with <b>Start over</b>.</>],
  ['A step says “Made with an older model”.', <>You switched to a different model after that step ran. Its results stay as they are
    until you run the step again with <b>Start over</b> (under Advanced); then make the results tables again.</>],
  ['How do other people label for me?', <>On <Link to="/labels">Labels</Link>, press <b>New labelling session</b>, pick a stain and a round
    (<b>Fix the field</b>: whole fields, adding what detection missed; <b>Check detection’s calls</b>: object by object), then send people the <b>Link for labellers</b>
    (it ends in /annotate and names the study). They type their name and work in their own browser, with no password. Someone new first gets a short practice
    tutorial. Each person is handed a field nobody else is on, so no one does the same work twice; you follow their progress and download the labels on the
    same page.</>],
  ['How does labelling work?', <>What stainID detection found is already on the picture: objects with a class are solid in its colour, objects it rejected (or someone removed) dotted grey.
    Point at an object and press a key: <b>1–9</b> its class, <b>0</b> unsure, <b>Delete</b> not an object, <b>Tab</b> / <b>Shift+Tab</b> a bigger or smaller
    outline, a <b>right-click</b> on the object to edit the outline spot by spot (a click outside keeps it), <b>Space</b> to hide the drawings for a clean look, <b>Z</b> undo. In <b>Fix the field</b>, clicking empty tissue outlines a missed object. Nothing
    is saved until <b>Save</b> or <b>Save and continue</b>; unsaved changes come back after a reload. The keys are always shown in the side panel.</>],
  ['A labelling session says no core has detection’s candidates.', <>The annotation rounds show the candidates of the staged detection. Run the
    <b>Detect</b> steps on the <Link to="/workflow">Run</Link> page for that stain first (at least <b>Find candidates</b> and <b>Decide</b>; <b>Run what’s left</b> runs them).</>],
  ['A Detect step says some cores are out of date.', <>Something those cores were made from changed after they ran: a setting, a model, a hand-drawn
    exclusion or an earlier step. Run the step again: it redoes only the out-of-date cores and keeps the rest.</>],
  ['Who can sign in?', <>Everything except the annotation pages needs the admin password, from this computer or any other. It is <b>admin</b> until you
    change it in <Link to="/settings">Settings</Link>.</>],
  ['Is anything deleted?', <>No. <b>Start over</b> and uploads move old files aside with a date in their name; nothing is removed.
    “Remove from list” on the Projects page only takes a study off the list.</>],
  ['Can several people work at once?', <>Yes, in the same study or in different ones: each person picks a study on the <a href="/">Projects</a> page, and switching
    study in one browser changes nothing for anyone else. Steps that need a lot of computing take turns across all studies; a waiting step says which study it waits for.</>],
]

export default function Help() {
  const [filter, setFilter] = useState('')
  const words = filter.toLowerCase().split(/\s+/).filter(Boolean)
  const terms = Object.entries(GLOSSARY).filter(([k, v]) => words.every((w) => `${k} ${v}`.toLowerCase().includes(w)))
  return (
    <>
      <PageHead title="Help" subtitle="How to go from slide scans to results, answers to common questions, and what the words mean." />
      <div className="help-layout">
        <div className="grid">
          <div className="card">
            <h2>Getting started</h2>
            <ol className="guide">
              <li><b>Create a project</b> for your study: the project menu at the top left → All projects → New study. All files for the study are kept
                in its folder.</li>
              <li><b><Link to="/setup">Set up</Link></b>: slides → cores → TMA maps → donor information → checks. Each screen shows what stainID found
                by itself; you confirm or correct it.</li>
              <li><b>Get the models</b> on <Link to="/models">Models</Link>: download the published ones, and put in (or train) the NeuN, 6E10 and AT8 models.</li>
              <li><b><Link to="/workflow">Run</Link></b> the analysis and follow its progress; any step can also be run on its own.</li>
              <li><b>Check</b> what was found on <Link to="/cohort">Cores</Link>, and ask colleagues to check and correct detection’s calls on <Link to="/labels">Labels</Link>;
                their labels train better models on <Link to="/models">Models</Link>.</li>
              <li><b>Use the <Link to="/results">Results</Link></b>: every measure plotted by diagnosis group, what it means, and the tables to download.</li>
            </ol>
          </div>
          <div className="card">
            <h2>Common questions</h2>
            {FAQ.map(([q, a]) => <details key={q} className="faq"><summary>{q}</summary><p className="secondary">{a}</p></details>)}
          </div>
        </div>
        <div className="card">
          <h2>Glossary</h2>
          <input type="search" placeholder="Find a word" value={filter} onChange={(e) => setFilter(e.target.value)} style={{ width: '100%', marginBottom: 10 }} aria-label="Find a word" />
          <dl className="glossary">{terms.map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}</dl>
          {!terms.length && <p className="muted small">No word matches “{filter}”.</p>}
        </div>
      </div>
    </>
  )
}
