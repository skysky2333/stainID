import { useState } from 'react'
import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { PageHead } from '../components'
import { GLOSSARY } from '../glossary'
import { TutorialCards } from '../labels/tutorial'
import { useSession } from '../session'

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
  ['How do I know the results are right?', <>Look at the cores and what was found in them on <Link to="/cohort">Cores</Link>, and ask colleagues to fix detection on a few
    fields on <Link to="/labels">Labels</Link> (<b>Fix the field</b>; <b>Mark bad areas</b> for folds and artifacts): each stain shows how often detection agrees with people, and <Link to="/models">Models</Link> shows the cross-validated precision and
    recall on all labels. Its <b>Outline fixes</b> card compares the outlines people fixed with each of detection’s outlines, per stain: which outline to draw first.</>],
  ['The detections look wrong for my staining.', <>Label objects on your own slides (an annotation round on <Link to="/labels">Labels</Link>; 100–200 objects is a good start),
    then train on <Link to="/models">Models</Link> and switch to the new model if it scores better. Then run the detection steps again with <b>Start over</b>.</>],
  ['A step says “Made with an older model”.', <>You switched to a different model after that step ran. Its results stay as they are
    until you run the step again with <b>Start over</b> (under Advanced); then make the results tables again.</>],
  ['How do other people label for me?', <>On <Link to="/labels">Labels</Link>, press <b>New labelling session</b> and pick what people should do
    (<b>Fix the field</b>: whole fields of one stain, correcting detection and adding what it missed; <b>Mark bad areas</b>: core by core with every stain, with a brush), then send people the
    address of stainID with the session’s <b>key</b> (see “Sharing a set with a key”). They sign in on the main page, or make an account, add the set with the key and work in their own
    browser; their home offers both tutorials. Each person is handed a field nobody else is on, so no one does the same work twice; you follow their progress and download the labels on the
    same page.</>],
  ['How does labelling work?', <>Each field starts with <b>step 1</b>: the <b>acceptance threshold</b> slider at detection’s own threshold. Moving it changes
    which objects start solid and which dotted grey; <b>Start editing</b> (or Enter, or your first edit) locks it, and only <b>Reset field</b> frees it again
    (discarding the field’s unsaved changes). It is saved with the field as a label; the analysis keeps detection’s threshold.
    {' '}What stainID detection found is then on the picture: each type solid in its own colour, objects it rejected dotted grey.
    Point at an object and press one key: a <b>number</b> picks a counted type, a <b>letter</b> a named look-alike, <b>Del</b> rejects it (dotted grey,
    like detection’s own rejects; a number makes it solid again), <b>`</b> unsure (dotted amber with “?”, left out of training and evaluation).
    <b>Tab</b> switches its outline between Segment Anything’s and the stain outline (with the brush on, Tab sizes the brush), a <b>click</b> on the object to edit the outline spot by spot (a click outside or Enter
    keeps it), <b>Space</b> to hide the drawings for a clean look, <b>S</b> to split an object in two (click it, Enter; or draw a line across objects),
    {' '}<b>M</b> to merge objects (click or drag over them, Enter), <b>B</b> the brush for bad areas (a drag adds to the chosen type’s area, Shift-drag
    takes out of it, <b>Tab</b> / <b>Shift+Tab</b> a smaller or bigger brush: it keeps its size on screen, so zoom in for a finer one), <b>Z</b> undo. Splits and merges are your labels: detection’s objects stay what the
    analysis counts. In <b>Fix the field</b>, clicking empty
    tissue outlines a missed object. Nothing is saved until <b>Save</b> or <b>Save and continue</b>; unsaved changes come back after a reload. The keys
    are always shown in the side panel.</>],
  ['Which key is which type?', <>The same on every labelling page, shown in the side panel:
    <ul className="help-keys">
      <li><b>6E10</b>: plaques <b>1</b> compact (dense-core), <b>2</b> diffuse, <b>3</b> cored-neuritic, <b>4</b> small (too small to type) · not plaques
        {' '}<b>Q</b> CAA, <b>W</b> staining inside a neuron · <b>Del</b> not a plaque · <b>`</b> unsure</li>
      <li><b>AT8</b>: tau+ neurons <b>1</b> pretangle, <b>2</b> mature tangle, <b>3</b> ghost tangle · not tau+ neurons <b>Q</b> neuritic crown,
        {' '}<b>W</b> tau in an astrocyte, <b>E</b> thread · <b>Del</b> not tau · <b>`</b> unsure</li>
      <li><b>NeuN</b>: <b>1</b> neuron · <b>Del</b> not a neuron · <b>`</b> unsure (how strongly a neuron is stained is measured, not chosen)</li>
    </ul>
    Only the numbers are counted (together they are “plaque” or “tau+ neuron”); the letters name what looks like one but is not. <b>0</b> also means
    unsure and <b>Backspace</b> also rejects. With the brush on (<b>B</b>), <b>1</b>–<b>4</b> are the area types instead (fold or damage, artifact,
    no tissue, out of focus or bad staining) and the class keys rest. Classes you add in <Link to="/settings#label-classes">Settings</Link> get
    {' '}<b>5</b>–<b>9</b> (counted types) or <b>T Y U …</b> (look-alikes). Debris is <b>Del</b>, unclear is <b>`</b>, and bad areas are painted with the brush.</>],
  ['What happens to the bad areas people mark?', <>Like every label, they only teach and test models: they train and score the bad-area detection
    (<Link to="/models">Models</Link> → Bad-area detector: detection’s areas against people’s, per TMA and donor), and objects inside them are left out of
    training and scoring. The analysis always uses detection’s own areas (hatched, “Excluded from analysis” on the Cores view), so marking an area never
    changes a result or makes a step out of date. The brush (the same in <b>Fix the field</b> and <b>Mark bad areas</b>) changes each type’s area as it
    stands: a drag adds to the chosen type, Shift-drag takes out of it (detection’s part too: that is how a part is marked fine). <b>Mark bad areas</b> (a kind of session) goes core by core with all its stains; a sample of cores is enough.</>],
  ['Which keys work on the Cores view and in Mark bad areas?', <>
    <b>Cores view</b>: <b>Space</b> hides or shows everything drawn (the minimap stays), <b>←</b> / <b>→</b> the previous or next core with the same stain and
    layers, <b>↑</b> / <b>↓</b> move the picture, scroll zooms, drag moves, click an object for its details; on the minimap a click or drag moves there and a
    double-click opens that tile for annotation. <b>Mark bad areas</b>: the brush of <b>Fix the field</b>, always on — <b>1</b>–<b>4</b> the area type, a drag
    adds to that type’s area, Shift-drag takes out of it, <b>Tab</b> / <b>Shift+Tab</b> (or <b>−</b> / <b>+</b>) a smaller or bigger brush, the wheel zooms, the
    middle button pans, <b>Z</b> undoes a stroke, <b>Space</b> hides the drawings, <b>]</b> / <b>[</b> the next or previous stain, <b>Enter</b> says this stain
    is fine and moves on (on the last stain it saves and opens the next core).</>],
  ['A labelling session says no core has detection’s candidates.', <>The annotation rounds show the candidates of the staged detection. Run the
    <b>Detect</b> steps on the <Link to="/workflow">Run</Link> page for that stain first (at least <b>Find candidates</b> and <b>Decide</b>; <b>Run what’s left</b> runs them).</>],
  ['A Detect step says some cores are out of date.', <>Something those cores were made from changed after they ran: a setting, a model or an earlier
    step. (Areas people mark never make a step out of date: the analysis uses detection’s own.) Run the step again: it redoes only the out-of-date cores and keeps the rest.</>],
  ['Who can sign in?', <>Everyone signs in on the main page with their name and password and lands on their home: their labelling sets, Add a set and
    the tutorials. <b>Admins</b> label too (a session they open joins their sets, and their labels carry their own account) and also see the study’s pages
    and every labelling set on Labels; <b>annotators</b> only the labelling pages and the sets they added with a key. Anyone can make an annotator account; an admin promotes people,
    resets forgotten passwords and removes accounts in <Link to="/settings">Settings → People</Link> (there is always at least one admin). The first admin is
    <b>admin</b>, with the admin password of earlier versions (<b>admin</b> on a new server, until you change it). The tutorial needs no account.</>],
  ['How to join a labelling set (annotators)', <>Open stainID’s address and <b>sign in</b>, or press <b>New here? Make an account</b> and choose your name
    and a password (names are unique on the server). Then type the <b>key</b> you were given under <b>Add a set</b>: the set appears in <b>My labelling sets</b>
    with its progress, and <b>Start</b> or <b>Continue</b> opens it. Sign in with the same name next time, on any computer, to carry on. Forgot your password?
    Ask the person running the study: they give you a one-time password, and you choose a new one when you sign in with it.</>],
  ['Sharing a set with a key (admins)', <>Every labelling session has its own key. On <Link to="/labels">Labels</Link>, open a session’s <b>More</b>: <b>Who can
    label</b> shows the key (<b>Copy</b> it and send it with the link), the people who joined (<b>Remove</b> takes someone out) and <b>Joining on</b>. <b>New key</b>
    makes the old key stop working; people who already joined keep their access. Turn joining off once everyone is in. Only people who joined a set (and
    admins) can open it: the server checks on every request. Labels made before accounts existed keep their names; nobody can sign up under them, and in
    Settings → People you link an account to such a name so it carries on that history.</>],
  ['Can a script use the server?', <>Yes, with an API token of an account: Settings → API tokens for scripts, or <code>stainid accounts token NAME</code> on
    the server. Send it as <code>Authorization: Bearer &lt;token&gt;</code>. A token can do what its account can; revoke it there, and removing the account
    ends its tokens.</>],
  ['Is anything deleted?', <>No. <b>Start over</b> and uploads move old files aside with a date in their name; nothing is removed.
    “Remove from list” on the Projects page only takes a study off the list.</>],
  ['Can several people work at once?', <>Yes, in the same study or in different ones: each person picks a study on the <a href="/projects">Projects</a> page, and switching
    study in one browser changes nothing for anyone else. Steps that need a lot of computing take turns across all studies; a waiting step says which study it waits for.</>],
]

/** The questions an annotator asks (the rest are about running the study). */
const LABELLING_FAQ = new Set(['How does labelling work?', 'Which key is which type?', 'What happens to the bad areas people mark?',
  'Which keys work on the Cores view and in Mark bad areas?', 'Who can sign in?', 'How to join a labelling set (annotators)'])

export default function Help() {
  const { session } = useSession()
  const admin = session.admin
  const [filter, setFilter] = useState('')
  const words = filter.toLowerCase().split(/\s+/).filter(Boolean)
  const terms = Object.entries(GLOSSARY).filter(([k, v]) => words.every((w) => `${k} ${v}`.toLowerCase().includes(w)))
  return (
    <>
      <PageHead title="Help" subtitle={admin ? 'The tutorials, how to go from slide scans to results, answers to common questions, and what the words mean.'
        : 'The tutorials, answers to common questions, and what the words mean.'} />
      <div className="help-layout">
        <div className="grid">
          <div className="card">
            <h2>Labelling: the two tasks and their tutorials</h2>
            <p className="small secondary"><b>Fix the field</b>: one field of one stain at a time, with what detection found drawn on it; correct it (reject,
              change types, fix outlines, split, merge, add what it missed) and save. <b>Mark bad areas</b>: one core at a time with every stain; paint
              folds, artifacts, missing tissue and blur that should not be counted. Each tutorial practises on real images; nothing is saved.</p>
            <TutorialCards compact />
          </div>
          {admin && <div className="card">
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
          </div>}
          <div className="card">
            <h2>Common questions</h2>
            {FAQ.filter(([q]) => admin || LABELLING_FAQ.has(q)).map(([q, a]) => <details key={q} className="faq"><summary>{q}</summary><p className="secondary">{a}</p></details>)}
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
