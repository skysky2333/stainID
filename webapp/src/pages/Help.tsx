import { useState } from 'react'
import { Link } from 'react-router-dom'
import { PageHead } from '../components'
import { GLOSSARY } from '../glossary'
import { useProject } from '../project'

const FAQ: [string, React.ReactNode][] = [
  ['What do I need to start?', <>Your slide scans (one per TMA and stain), and a spreadsheet saying which donor, brain region and diagnostic group sits at each core position.
    Create a project, then follow the <Link to="/workflow">Workflow</Link> page from the top.</>],
  ['How long does a full study take?', <>Setting up (finding and exporting cores) takes about an hour for 20 slides. Detection is the slow part: roughly a minute per field
    and stain on a laptop, so a few hours to a day for a large study. Everything is resumable, so you can stop and continue later.</>],
  ['What runs when? Why does a step say “Waiting to start”?', <>You can press Run on several steps at once; stainID starts them in the right order.
    A step waits while a step it depends on (the arrows on the map at the top of the Workflow page) is still running or waiting, and at most two
    heavy steps (Cellpose, detection, outlines) run at the same time so the computer stays usable. The <Link to="/workflow">Workflow</Link> page
    shows what is running now, how far it has got, roughly how long is left, and why anything is waiting.</>],
  ['Can I close the browser?', <>Yes. Steps keep running as long as the stainID window (the terminal started by “Start stainID”) is open. Closing that window stops running steps;
    starting a step again continues where it stopped.</>],
  ['A step failed. What now?', <>The step card shows the reason. Common ones: a missing file (the card says which earlier step makes it), a model that is not downloaded yet
    (see <Link to="/models">Models</Link>), or results made with a different model (run again with “Start over” under Advanced options). “Show log” has every detail.</>],
  ['Where are my files?', <>Everything lives in the project folder shown on the Home page. Each step card lists the files it makes; “Show in Finder” opens them.
    The main results table is <code>data/analysis/results_donor_region.csv</code> (downloadable on the <Link to="/results">Results</Link> page).</>],
  ['How do I know the results are right?', <>Look at cores in <Link to="/cohort">Cohort</Link> with the detections drawn on top, and label a check set on
    <Link to="/label"> Label &amp; check</Link>: comparing your labels with the model’s answers gives its precision and recall on your data.</>],
  ['The detections look wrong for my staining.', <>Train a stain model on your own slides: on <Link to="/models">Models</Link>, create a training set, label it
    (100–200 objects is a good start), train, and switch to the new model if it scores better. Then re-run the detection step with “Start over”.</>],
  ['A step says “Made with an older model”.', <>You switched to a different model after that step ran. The existing results stay as they are
    until you run the step again with <b>Start over</b> (under Advanced options); then “Make results tables” again.</>],
  ['Is anything deleted?', <>No. “Start over” and uploads move old files aside with a date in their name; nothing is removed.</>],
]

export default function Help() {
  const { info } = useProject()
  const [filter, setFilter] = useState('')
  const terms = Object.entries(GLOSSARY).filter(([k, v]) => `${k} ${v}`.toLowerCase().includes(filter.toLowerCase()))
  return (
    <>
      <PageHead title="Help" subtitle="Getting started, common questions, and what the words mean." />
      <div className="grid grid-2" style={{ alignItems: 'start' }}>
        <div className="card">
          <h2>Getting started</h2>
          <ol className="guide">
            <li><b>Create a project</b> for your study (Settings → Open or create another project). All files for the study are kept in its folder{info ? <> — currently <code>{info.root}</code></> : null}.</li>
            <li><b>Describe your TMAs</b> in <Link to="/settings">Settings</Link>: rows and columns of cores per slide, the name prefix, and the diagnostic group codes.</li>
            <li><b>Get the models</b> on <Link to="/models">Models</Link>: download the published ones; put in (or train) the NeuN / 6E10 / AT8 models.</li>
            <li><b>Follow the Workflow</b> from the top: register slides → find cores → attach the TMA map → export cores → check quality → choose fields → calibrate → detect → results.</li>
            <li><b>Check</b> detections in <Link to="/cohort">Cohort</Link> and with a check set on <Link to="/label">Label &amp; check</Link>.</li>
            <li><b>Use the results</b>: download the main table on <Link to="/results">Results</Link>, which also explains every column.</li>
          </ol>
          <h2 style={{ marginTop: 20 }}>Common questions</h2>
          {FAQ.map(([q, a]) => <details key={q} className="faq"><summary>{q}</summary><p className="secondary">{a}</p></details>)}
        </div>
        <div className="card">
          <h2>Glossary</h2>
          <input type="text" placeholder="search…" value={filter} onChange={(e) => setFilter(e.target.value)} style={{ width: '100%', marginBottom: 10 }} />
          <dl className="glossary">{terms.map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}</dl>
        </div>
      </div>
    </>
  )
}
