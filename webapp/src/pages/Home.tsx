import { useState } from 'react'
import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import type { Job, ScreenState, SetupOverview, Step, Summary } from '../api'
import { api } from '../api'
import { ErrorNote, HelpBox, Loaded, Loading, PageHead, Progress, Stat, StatusBadge, Term, withError } from '../components'
import { useFetch } from '../hooks'
import { MAIN_STAGES, NowRunning, OPTIONAL, STAGE, groupState, isActive, progressText, stepState } from '../pipeline'
import { useProject } from '../project'
import { Advanced, ScreenMark } from '../setup/parts'
import { JobsTable } from './Workflow'

/** Home: where the study stands, the one thing to do next, what runs now, the study in numbers and the latest runs. */
export default function Home() {
  const { info } = useProject()
  const steps = useFetch<Step[]>('/api/steps', 5000)
  const setup = useFetch<SetupOverview>('/api/setup', 10000)
  const jobs = useFetch<Job[]>('/api/jobs', 5000)
  const main = (steps.data ?? []).filter((s) => MAIN_STAGES.includes(s.stage))
  const refresh = () => { steps.reload(); setup.reload(); jobs.reload() }

  return (
    <>
      <PageHead title={info?.name ?? 'stainID'} subtitle="Where this study stands and what to do next." />
      <HelpBox id="home" title="New here? How stainID works">
        <ol>
          <li><b>Set up</b>: stainID reads your slide scans, finds the cores and takes your TMA maps and donor sheet; you confirm what it found.</li>
          <li><b>Run</b>: the analysis finds nuclei, neurons, plaques and tau in every core. It runs in the background, for hours on a big study.</li>
          <li><b>Check</b>: look at any core with the detections drawn on top, to see by eye that they are right.</li>
          <li><b>Results</b>: one table row per donor and brain region, what every column means, and plots by group.</li>
          <li><b>Improve accuracy</b> (optional): label objects without seeing the diagnosis, to measure accuracy or train better models.</li>
        </ol>
        <p style={{ marginTop: 6 }}>Unsure about a word? Hover over dotted words, or open <b>Help</b>.</p>
      </HelpBox>

      {steps.data && setup.data ? <NextStep steps={main} setup={setup.data} onChanged={refresh} />
        : steps.error || setup.error ? <ErrorNote error={steps.error ?? setup.error} />
        : <div className="card next-card"><Loading label="Looking at where the study stands…" /></div>}
      <NowRunning jobs={jobs.data ?? []} steps={main} onChanged={refresh} />
      <Loaded state={steps} label="Reading the steps…">{() => <StageOverview steps={main} />}</Loaded>
      <StudyFacts />
      <div className="home-section"><JobsTable limit={5} /></div>
      {info && (
        <Advanced title="About this project">
          <div className="small secondary">Project folder: <code>{info.root}</code></div>
        </Advanced>
      )}
    </>
  )
}

/** The one obvious next action: finish set-up, see what failed, run what is left, follow the running work, or open the results. */
function NextStep({ steps, setup, onChanged }: { steps: Step[]; setup: SetupOverview; onChanged: () => void }) {
  const [error, setError] = useState<string | null>(null)
  const [starting, setStarting] = useState(false)
  const counted = steps.filter((s) => !OPTIONAL.has(s.id))
  const done = counted.filter((s) => stepState(s) === 'done').length
  const failed = counted.filter((s) => stepState(s) === 'failed')
  const active = steps.filter((s) => isActive(s.job?.status))
  const remaining = setup.remaining
  const slides = setup.screens.find((s) => s.id === 'slides')
  const newSlides = setup.set_up && slides?.state === 'todo' ? slides.summary : ''
  const runAll = () => withError(async () => {
    setStarting(true)
    try { await api.post('/api/setup/run', {}); onChanged() } finally { setStarting(false) }
  }, setError)

  if (!setup.set_up) {
    const begun = setup.screens.some((s) => s.state === 'done')
    return (
      <div className="card next-card home-next">
        <div className="muted small">Your next step</div>
        <h2>Set up your study</h2>
        <p className="secondary">A few short screens: stainID reads your slide scans, finds the cores and takes the TMA maps and donor sheet; you only confirm what it found.</p>
        <ol className="setup-overview">
          {setup.screens.map((s, i) => <li key={s.id}><ScreenMark state={s.state} number={i + 1} /><b>{s.title}</b><span className="muted small">{s.summary}</span></li>)}
        </ol>
        <div className="row"><Link className="btn primary" to="/setup">{begun ? 'Continue the set-up' : 'Start the set-up'}</Link></div>
      </div>
    )
  }

  const card = (tone: string, label: string, title: string, text: ReactNode, actions: ReactNode) => (
    <div className={`card next-card home-next${tone}`}>
      <div className="muted small">{label}</div>
      <h2>{title}</h2>
      <p className="secondary">{text}</p>
      {counted.length > 0 && <div className="home-next-progress"><Progress done={done} total={counted.length} /></div>}
      <div className="row">{actions}</div>
      {newSlides && <p className="small secondary home-note">{newSlides}. <Link to="/setup/slides">Add them to the study</Link></p>}
      <ErrorNote error={error} />
    </div>
  )
  const tally = `${done} of ${plural(counted.length, 'step')} done`

  if (failed.length) {
    const first = failed[0]
    const more = failed.length > 1 ? ` (${plural(failed.length - 1, 'more step')} also failed.)` : ''
    return card(' attention', 'Needs your attention', `“${first.title}” stopped with an error`, `${first.job?.progress.error || 'The log says what went wrong.'}${more}`,
      <Link className="btn primary" to={`/workflow#${first.id}`}>See what went wrong</Link>)
  }
  if (remaining.length) {
    return card('', `Your next step · ${tally}`, remaining.length === 1 ? `“${remaining[0].title}” is ready to run` : `${remaining.length} steps are ready to run`,
      <>{remaining.map((s) => s.title).join(', ')}{setup.estimate.text ? `: ${setup.estimate.text} on this computer, roughly` : ''}.
        They start in the right order by themselves and carry on with the browser closed.</>,
      <>
        <button className="btn primary" disabled={starting} onClick={runAll}>{starting ? 'Starting…' : 'Run what’s left'}</button>
        <Link className="btn" to="/workflow">Look at the steps first</Link>
      </>)
  }
  if (active.length) {
    return card('', `Running · ${tally}`, 'The analysis is running',
      `${active.map((s) => s.title).join(', ')}. Nothing to do now: it carries on with the browser closed. Progress is below.`,
      <Link className="btn primary" to="/workflow">See every step</Link>)
  }
  return card(' done', `All ${plural(counted.length, 'step')} done`, 'The results are ready',
    'One row per donor and brain region, with plots by group. Check the detections by eye in the cores viewer.',
    <><Link className="btn primary" to="/results">Open the results</Link><Link className="btn" to="/cohort">Check the cores</Link></>)
}

function plural(n: number, word: string): string {
  return `${n.toLocaleString()} ${word}${n === 1 ? '' : 's'}`
}

const MARK: Record<string, ScreenState> = { done: 'done', running: 'running', queued: 'running', failed: 'attention', outdated: 'warning', partial: 'todo', todo: 'todo' }

/** The three stages side by side, each with its steps: a tick when done, how far it got otherwise. */
function StageOverview({ steps }: { steps: Step[] }) {
  return (
    <div className="grid grid-3 home-section">
      {MAIN_STAGES.map((stage) => {
        const list = steps.filter((s) => s.stage === stage)
        const done = list.filter((s) => stepState(s) === 'done').length
        return (
          <div key={stage} className="card stage-card">
            <div className="row">
              <h2 className="stage-card-title">{STAGE[stage]?.name ?? stage}</h2>
              <span className="spacer" />
              <StatusBadge status={groupState(list)} />
            </div>
            <div className="small muted">{done} of {plural(list.length, 'step')} done</div>
            <ul className="stage-steps">
              {list.map((s, i) => {
                const state = stepState(s)
                const count = state === 'running' || state === 'partial' ? progressText(s) : ''
                return (
                  <li key={s.id}>
                    <ScreenMark state={MARK[state] ?? 'todo'} number={i + 1} />
                    <span className="stage-step">
                      <span><Link to={`/workflow#${s.id}`}>{s.title}</Link>{OPTIONAL.has(s.id) && <span className="muted small"> · optional</span>}</span>
                      {count && <span className="muted small stage-count">{count}</span>}
                    </span>
                    {state !== 'done' && <StatusBadge status={state} />}
                  </li>
                )
              })}
            </ul>
          </div>
        )
      })}
    </div>
  )
}

/** The study in numbers: donors, donor-regions, cores, fields, and donors per group (once there are cores). */
function StudyFacts() {
  const summary = useFetch<Summary>('/api/summary', 60000)
  const { groups, groupColor, groupLabel } = useProject()
  if (!summary.data) return summary.error ? <ErrorNote error={summary.error} /> : null
  const data = summary.data
  if (!data.cores) return null
  const fields = Object.values(data.fields).reduce((a, b) => a + b, 0)
  return (
    <div className="card home-section">
      <h2>Your study</h2>
      <div className="grid grid-4">
        <Stat value={data.donors.toLocaleString()} label="donors" />
        <Stat value={data.donor_regions.toLocaleString()} label={<Term term="donor-region">donor-regions</Term>} />
        <Stat value={data.cores.toLocaleString()} label={<>cores on {data.tmas.length} <Term>TMA</Term>s</>} />
        <Stat value={fields.toLocaleString()} label={<>analysis <Term>field</Term>s (all stains)</>} />
      </div>
      {Object.keys(data.groups).length > 0 && (
        <div className="group-bars">
          <div className="io-title">Donors by group</div>
          {groups.filter((g) => data.groups[g]).map((g) => (
            <div key={g} className="group-bar">
              <span className="swatch" style={{ background: groupColor(g) }} />
              <span className="group-name">{groupLabel(g)}</span>
              <div className="bar-track"><div style={{ width: `${(100 * data.groups[g]) / data.donors}%`, background: groupColor(g) }} /></div>
              <span className="num">{data.groups[g]}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
