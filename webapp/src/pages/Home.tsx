import { Link } from 'react-router-dom'
import type { Step, Summary } from '../api'
import { ErrorNote, HelpBox, PageHead, Stat, StatusBadge, Term } from '../components'
import { useFetch } from '../hooks'
import { useProject } from '../project'
import { JobsTable, MAIN_STAGES, stepState } from './Workflow'

const OPTIONAL = new Set(['masks'])

export default function Home() {
  const { info, groups, groupColor, groupLabel } = useProject()
  const steps = useFetch<Step[]>('/api/steps', 5000)
  const summary = useFetch<Summary>('/api/summary', 15000)
  const main = (steps.data ?? []).filter((s) => MAIN_STAGES.includes(s.stage))
  const next = main.find((s) => !OPTIONAL.has(s.id) && s.progress.state !== 'done')
  const running = main.filter((s) => ['running', 'queued'].includes(stepState(s)))
  const data = summary.data

  return (
    <>
      <PageHead title={info?.name ?? 'stainID'} subtitle={<>Project folder: <code>{info?.root}</code></>} />
      <HelpBox id="home" title="New here? Start with this">
        <ol>
          <li><b>Workflow</b> walks you through every step, from slide scans to results tables. Run the steps in order; each card says what it needs, what it makes and where it saves it.</li>
          <li><b>Cohort</b> lets you look at every core and field with the detections drawn on top — use it to check results by eye.</li>
          <li><b>Label &amp; check</b> shows you objects blind (without diagnosis or the model’s answer) so you can label them, to check accuracy or to train a model on the <b>Models</b> page.</li>
          <li><b>Results</b> has the final tables (one row per donor and brain region), what every column means, and plots by group.</li>
          <li>Unsure about a word? Hover over dotted words, or open <b>Help</b> for a glossary.</li>
        </ol>
      </HelpBox>

      {next ? (
        <div className="card next-card">
          <div className="muted small">{running.length ? 'Running now: ' + running.map((s) => s.title).join(', ') : 'Your next step'}</div>
          <h2 style={{ margin: '4px 0' }}>{next.title}</h2>
          <p className="secondary" style={{ margin: 0 }}>{next.summary}</p>
          <div className="row" style={{ marginTop: 12 }}>
            <Link className="btn primary" to={`/workflow#${next.id}`}>Go to this step</Link>
            <StatusBadge status={stepState(next)} />
          </div>
        </div>
      ) : steps.data && (
        <div className="card next-card done">
          <h2 style={{ margin: 0 }}>All steps are done</h2>
          <p className="secondary">Open <Link to="/results">Results</Link> for the tables and plots, or check detections in <Link to="/cohort">Cohort</Link>.</p>
        </div>
      )}
      <ErrorNote error={steps.error} />

      <div className="grid grid-3" style={{ marginTop: 16 }}>
        {MAIN_STAGES.map((stage) => {
          const list = main.filter((s) => s.stage === stage)
          return (
            <div key={stage} className="card">
              <h3>{stage}</h3>
              <ul className="checklist">
                {list.map((s) => (
                  <li key={s.id} className={`check-${stepState(s)}`}>
                    <Link to={`/workflow#${s.id}`}>{s.title}</Link>{OPTIONAL.has(s.id) && <span className="muted small"> (optional)</span>}
                    <span className="spacer" /><StatusBadge status={stepState(s)} />
                  </li>
                ))}
              </ul>
            </div>
          )
        })}
      </div>

      {data && data.cores > 0 && (
        <div className="grid" style={{ gap: 16, marginTop: 16 }}>
          <div className="grid grid-4">
            <Stat value={data.donors} label="donors" />
            <Stat value={data.donor_regions} label={<Term term="donor-region">donor-regions</Term>} />
            <Stat value={data.cores} label={<>cores on {data.tmas.length} <Term>TMA</Term>s</>} />
            <Stat value={Object.values(data.fields).reduce((a, b) => a + b, 0).toLocaleString()} label={<>analysis <Term>field</Term>s (all stains)</>} />
          </div>
          <div className="card">
            <h2>Donors by group</h2>
            <div style={{ display: 'grid', gap: 8 }}>
              {groups.filter((g) => data.groups[g]).map((g) => (
                <div key={g} className="row">
                  <span className="swatch" style={{ background: groupColor(g) }} />
                  <span style={{ width: 110 }}>{groupLabel(g)}</span>
                  <div className="bar-track"><div style={{ width: `${(100 * data.groups[g]) / data.donors}%`, background: groupColor(g) }} /></div>
                  <span style={{ width: 30, textAlign: 'right' }}>{data.groups[g]}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
      <div style={{ marginTop: 16 }}><JobsTable limit={5} /></div>
    </>
  )
}
