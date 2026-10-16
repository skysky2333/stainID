import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import type { Job, ResourceInfo, SetupOverview, SlideRow, SlidesInfo, Step, StepOption } from '../api'
import { api, apiUrl, runStep } from '../api'
import { CommandLineTag, Empty, ErrorNote, FilePicker, HelpBox, Loaded, OptionField, PageHead, PathLine, Progress, StatusBadge, Term, timeAgo, withError } from '../components'
import { useFetch } from '../hooks'
import { MAIN_STAGES, OPTIONAL, PipelineMap, STAGE, V2_STAGE, duration, groupState, isActive, runFacts, stepState, stopJobs } from '../pipeline'
import { useProject } from '../project'
import { Advanced } from '../setup/parts'
import { SlideTable, slideProblems } from '../slides'

export { MAIN_STAGES, stepState }

/** The Run page: every step of the analysis by stage, with one button for everything that is left. */
export default function Workflow() {
  const steps = useFetch<Step[]>('/api/steps', 3000)
  const setup = useFetch<SetupOverview>('/api/setup', 8000)
  const navigate = useNavigate()
  const { hash } = useLocation()
  const list = (steps.data ?? []).filter((s) => MAIN_STAGES.includes(s.stage))
  const v2 = (steps.data ?? []).filter((s) => s.stage === V2_STAGE)
  useEffect(() => {
    if (hash && steps.data) document.getElementById(hash.slice(1))?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [hash, steps.data === null])
  const refresh = () => { steps.reload(); setup.reload() }

  return (
    <>
      <PageHead title="Run" subtitle="Every step from slide scans to results tables. Steps run in the background: you can switch pages or close the browser.">
        <Link className="btn" to="/setup">Guided set-up</Link>
      </PageHead>
      <HelpBox id="workflow">
        <ul>
          <li><b>Run what’s left</b> starts every step that is not done yet, in the right order. Or open a step and press <b>Run</b> on it alone.</li>
          <li>Steps can be stopped and run again: they continue where they stopped and skip what is finished.</li>
          <li>Open a step to see what it needs and makes, its options and its log. Technical options are under <b>Advanced</b>.</li>
        </ul>
      </HelpBox>
      <Loaded state={steps} label="Reading where every step stands…">
        {() => (
          <>
            <RunSummary steps={list} setup={setup.data} onChanged={refresh} />
            {MAIN_STAGES.map((stage) => <StageSection key={stage} stage={stage} steps={list.filter((s) => s.stage === stage)} onChanged={refresh} />)}
            {v2.length > 0 && <StageSection stage={V2_STAGE} steps={v2} onChanged={refresh} />}
            <Advanced title="Pipeline map" hint="which step needs which">
              <PipelineMap steps={list} state={stepState} onOpen={(id) => navigate(`/workflow#${id}`)} />
            </Advanced>
          </>
        )}
      </Loaded>
      <details className="advanced" open={hash === '#jobs'}>
        <summary><span>All runs</span><span className="muted small"> · every run of every step, with its log</span></summary>
        <div className="advanced-body"><JobsTable bare /></div>
      </details>
    </>
  )
}

/** How many steps are done, what runs, and the one button for everything that is left. */
function RunSummary({ steps, setup, onChanged }: { steps: Step[]; setup: SetupOverview | null; onChanged: () => void }) {
  const [error, setError] = useState<string | null>(null)
  const [starting, setStarting] = useState(false)
  const [started, setStarted] = useState<string[]>([])
  const counted = steps.filter((s) => !OPTIONAL.has(s.id))
  const done = counted.filter((s) => stepState(s) === 'done').length
  const active = steps.filter((s) => isActive(s.job?.status))
  const failed = counted.filter((s) => stepState(s) === 'failed')
  const remaining = setup?.remaining ?? []
  const runAll = () => withError(async () => {
    setStarting(true)
    try {
      const result = await api.post<{ queued: Job[] }>('/api/setup/run', {})
      setStarted(result.queued.map((j) => j.title))
      onChanged()
    } finally { setStarting(false) }
  }, setError)
  const allDone = done === counted.length

  return (
    <div className={`card next-card run-summary${allDone ? ' done' : ''}`}>
      <div className="run-summary-main">
        <h2>{allDone ? 'Every step is done' : `${done} of ${counted.length} steps done`}</h2>
        <Progress done={done} total={counted.length} />
        <p className="small secondary">
          {active.length ? <>Running now: {active.map((s) => s.title).join(', ')}. </> : null}
          {failed.length ? <>{failed.map((s) => `“${s.title}”`).join(', ')} stopped with an error: open {failed.length > 1 ? 'them' : 'it'} below to see why. </> : null}
          {allDone ? <>The tables are on the <Link to="/results">Results</Link> page.</>
            : !remaining.length && active.length ? 'Everything that is left is running or waiting its turn.' : null}
        </p>
        {started.length > 0 && <p className="small status-finished">Started: {started.join(', ')}. They run one after another as needed.</p>}
      </div>
      {setup && !setup.set_up ? (
        <div className="run-all">
          <Link className="btn primary" to="/setup">Finish the set-up first</Link>
          <span className="small muted">The slides, their cores and the TMA maps are needed before anything can run.</span>
        </div>
      ) : remaining.length > 0 && (
        <div className="run-all">
          <button className="btn primary" disabled={starting} onClick={runAll} title={remaining.map((s) => s.title).join(', ')}>
            {starting ? 'Starting…' : `Run what’s left (${remaining.length} ${remaining.length === 1 ? 'step' : 'steps'})`}</button>
          <span className="small muted">Starts with “{remaining[0].title}”{setup?.estimate.text ? ` · ${setup.estimate.text}, roughly` : ''}</span>
        </div>
      )}
      {!setup && <span className="small muted">Working out what is left…</span>}
      <ErrorNote error={error} />
    </div>
  )
}

/** One stage (Set up, Run, Results): its steps; a stage that is all done starts folded to one line. */
function StageSection({ stage, steps, onChanged }: { stage: string; steps: Step[]; onChanged: () => void }) {
  const { hash } = useLocation()
  const state = groupState(steps)
  const done = steps.filter((s) => stepState(s) === 'done').length
  const target = steps.some((s) => hash === `#${s.id}`)
  const [open, setOpen] = useState(state !== 'done' || target)
  useEffect(() => { if (target) setOpen(true) }, [target])
  const info = STAGE[stage] ?? { name: stage, about: '' }
  return (
    <section className="stage">
      <button className="stage-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className="chevron">{open ? '▾' : '▸'}</span>
        <span className="stage-name">{info.name}</span>
        <span className="muted small stage-about">{info.about}</span>
        <span className="spacer" />
        <span className="small muted">{done} of {steps.length} done</span>
        <StatusBadge status={state} />
      </button>
      {open && steps.map((step) => <StepCard key={step.id} step={step} onChanged={onChanged} />)}
    </section>
  )
}

function initialValues(options: StepOption[], outdated: boolean): Record<string, unknown> {
  return Object.fromEntries(options.map((o) => [o.key, o.key === 'fresh' && outdated ? true : o.default]))
}

/** One step: where it stands and its main button; opened, what it needs and makes, its options and its log. */
export function StepCard({ step, onChanged }: { step: Step; onChanged: () => void }) {
  const optionsKey = JSON.stringify(step.options)
  const outdated = Boolean(step.progress.outdated)
  const defaults = useMemo(() => initialValues(JSON.parse(optionsKey) as StepOption[], outdated), [optionsKey, outdated])
  const [values, setValues] = useState<Record<string, unknown>>(defaults)
  useEffect(() => setValues(defaults), [defaults])
  const [log, setLog] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const state = stepState(step)
  const { hash } = useLocation()
  const [open, setOpen] = useState(hash === `#${step.id}`)
  useEffect(() => { if (hash === `#${step.id}`) setOpen(true) }, [hash, step.id])
  const missing = step.needs.filter((n) => !n.exists)
  const job = step.job
  const parts = step.active ?? (job && isActive(job.status) ? [job] : [])
  const active = parts.length > 0
  const basic = step.options.filter((o) => !o.advanced)
  const extra = step.options.filter((o) => o.advanced)
  const progress = active && job?.progress.total && !step.progress.total ? job.progress : step.progress
  const changed = JSON.stringify(values) !== JSON.stringify(defaults)

  const run = (options: Record<string, unknown>) => withError(async () => {
    if (options.fresh && !window.confirm(`Start “${step.title}” over?\n\nIts existing results are moved aside (nothing is deleted) and the step runs on everything again.`)) return
    setBusy(true)
    try { await runStep(step.id, options); onChanged() } finally { setBusy(false) }
  }, setError)
  const stop = () => withError(async () => { if (await stopJobs(step.title, parts)) onChanged() }, setError)
  const runLabel = state === 'done' ? 'Run again' : step.progress.state === 'partial' ? 'Continue' : 'Run'
  const blocked = missing.length > 0

  return (
    <div className={`card step-card state-${state}${open ? ' open' : ''}`} id={step.id}>
      <div className="step-row">
        <button className="step-head" onClick={() => setOpen(!open)} aria-expanded={open}>
          <span className="chevron">{open ? '▾' : '▸'}</span>
          <div>
            <h3 className="step-title">{step.title}{OPTIONAL.has(step.id) && <span className="muted small"> · optional</span>}</h3>
            <p className="step-summary">{step.summary}</p>
          </div>
        </button>
        {progress.total > 0 && (
          <div className="step-mini">
            <Progress done={progress.done} total={progress.total} />
            <span className="muted small">{progress.done.toLocaleString()} of {progress.total.toLocaleString()} {step.progress.unit}</span>
          </div>
        )}
        <StatusBadge status={state} />
        {step.command && (active ? <button className="btn small" onClick={stop}>Stop</button>
          : <button className="btn small" disabled={blocked || busy} onClick={() => run(defaults)}
            title={blocked ? `Needs first: ${missing.map((m) => m.label).join(', ')}` : step.duration ? `Typical time: ${step.duration}` : undefined}>{busy ? 'Starting…' : runLabel}</button>)}
        {!step.command && step.id === 'slides' && <Link className="btn small" to="/setup/slides">Open</Link>}
      </div>
      <StepNotes step={step} state={state} parts={parts} blocked={blocked && !active} onDetails={() => setOpen(true)} />
      <ErrorNote error={error} />
      {open && (
        <div className="step-body">
          <p className="secondary step-details">{step.details}{step.duration && <><br /><span className="muted">Typical time: {step.duration}.</span></>}</p>
          {step.progress.note && <p className="small secondary">{step.progress.note}</p>}
          {step.id === 'slides' && <p className="small">The <Link to="/setup/slides">guided set-up</Link> does this for you; the table here is the same.</p>}
          {step.id === 'layout' && <p className="small">Easiest: add the maps as Excel, Word or CSV files in the <Link to="/setup/map">guided set-up</Link>. Or fill in the CSV:</p>}
          {step.id === 'slides' && <SlidesPanel onSaved={onChanged} />}
          {step.id === 'layout' && <MapUpload onSaved={onChanged} />}
          <div className="io-grid">
            <div>
              <div className="io-title">Needs</div>
              {step.needs.length ? step.needs.map((n) => <NeedRow key={n.id} need={n} onChanged={onChanged} />) : <span className="muted small">Nothing extra.</span>}
              {step.uses.length > 0 && (
                <>
                  <div className="io-title" style={{ marginTop: 8 }}>Uses if there</div>
                  {step.uses.map((u) => <NeedRow key={u.id} need={u} onChanged={onChanged} optional />)}
                </>
              )}
            </div>
            <div>
              <div className="io-title">Makes</div>
              {step.produces.map((p) => <ResourceLine key={p.id} resource={p} />)}
              {step.view && step.progress.done > 0 && <Link className="small" to={step.view}>See the results →</Link>}
            </div>
          </div>
          {step.command && (
            <>
              {basic.length > 0 && <div className="row options">{basic.map((o) => <OptionField key={o.key} option={o} value={values[o.key]} onChange={(v) => setValues({ ...values, [o.key]: v })} />)}</div>}
              <div className="row step-actions">
                {!active && <button className="btn" disabled={blocked || busy} onClick={() => run(values)}>{changed ? `${runLabel} with these options` : runLabel}</button>}
                {changed && <button className="link-btn small" onClick={() => setValues(defaults)}>Back to the usual options</button>}
                <span className="spacer" />
                {job && <span className="muted small">Last run {timeAgo(job.created)} · <StatusBadge status={job.status} /> <CommandLineTag job={job} /></span>}
                {job && <button className="link-btn small" onClick={() => setLog(!log)}>{log ? 'Hide log' : 'Show log'}</button>}
              </div>
              {log && job && <JobLog id={job.id} job={job} />}
              <Advanced hint={extra.length ? 'more options, start over, the files and the command' : 'the files and the command'}>
                {extra.length > 0 && <div className="row options">{extra.map((o) => <OptionField key={o.key} option={o} value={values[o.key]} onChange={(v) => setValues({ ...values, [o.key]: v })} />)}</div>}
                <div>
                  <div className="io-title">Files</div>
                  {[...step.needs, ...step.uses, ...step.produces].map((r) => <PathLine key={r.id} path={r.path} exists={r.exists} label={r.label} />)}
                </div>
                <div>
                  <div className="io-title">Command</div>
                  <code className="path-box">stainid {(job?.argv ?? step.command).join(' ')}</code>
                </div>
              </Advanced>
            </>
          )}
        </div>
      )}
    </div>
  )
}

/** The lines under a step's title: why it waits, how far it runs, what went wrong, what it needs first. */
function StepNotes({ step, state, parts, blocked, onDetails }: { step: Step; state: string; parts: Job[]; blocked: boolean; onDetails: () => void }) {
  const job = step.job
  const facts = parts.length ? runFacts(parts, step, false).text : ''
  return (
    <>
      {state === 'queued' && facts && <div className="small secondary step-note">{facts}. It starts on its own.</div>}
      {state === 'running' && facts && <div className="small secondary step-note">{facts.charAt(0).toUpperCase() + facts.slice(1)}</div>}
      {job?.status === 'failed' && state !== 'done' && <ErrorNote error={`This step stopped with an error: ${job.progress.error || job.progress.last_line || 'see the log'}`} />}
      {job?.status === 'skipped' && state !== 'done' && <div className="small secondary step-note">Last time: {job.progress.error.replace(/^Not started: /, 'not started, ').replace(/^Not started /, 'not started ')}.</div>}
      {blocked && step.command && state !== 'done' && (
        <div className="small muted step-note">Needs first: {step.needs.filter((n) => !n.exists).map((m) => m.download ? `${m.label} (download on the Models page)` : m.format ? `${m.label} (yours to provide)` : m.label).join(', ')}.
          {' '}<button className="link-btn small" onClick={onDetails}>How to get it</button></div>
      )}
      {step.progress.outdated && (
        <div className="warning-note">These results were made with a different model than the one in use now. Run the step again to redo them with the
          current model: old results are moved aside, not deleted.</div>
      )}
    </>
  )
}

/** Something a step makes or reads, by name; where it is shows on hover (and under Advanced). */
function ResourceLine({ resource, optional = false }: { resource: ResourceInfo; optional?: boolean }) {
  return (
    <div className="resource-line" title={`${resource.description}\n${resource.path}`}>
      <span className={`exists-dot ${resource.exists ? 'yes' : 'no'}`} />
      <span>{resource.label}</span>
      <span className="muted small">{resource.exists ? 'there' : optional ? 'not there (optional)' : 'not there yet'}</span>
    </div>
  )
}

/** One input of a step: whether it is there, and how to get it if it is not (go to the step that makes it, download it,
see what to provide and pick / upload it). */
function NeedRow({ need, onChanged, optional = false }: { need: ResourceInfo; onChanged: () => void; optional?: boolean }) {
  const navigate = useNavigate()
  const { reload } = useProject()
  const [open, setOpen] = useState(false)
  const [picking, setPicking] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [started, setStarted] = useState(false)
  const download = () => withError(async () => { await runStep('download_models', { which: [need.download] }); setStarted(true); onChanged() }, setError)
  const choose = (path: string) => withError(async () => {
    const [section, key] = need.setting.split('.')
    await api.put('/api/project/config', { changes: { [section]: { [key]: path } } })
    setPicking(false); reload(); onChanged()
  }, setError)
  const upload = (file: File | undefined) => file && withError(async () => { await api.upload(`/api/upload/${need.template}`, file); onChanged() }, setError)
  return (
    <div className="need-row">
      <ResourceLine resource={need} optional={optional} />
      {!need.exists && !optional && (
        <div className="need-actions">
          {need.made_by && <button className="btn small" onClick={() => navigate(`/workflow#${need.made_by!.id}`)}>Go to “{need.made_by.title}”</button>}
          {need.download && <button className="btn small" disabled={started} onClick={download}>{started ? 'Downloading…' : 'Download now'}</button>}
          {need.format && <button className="btn small" onClick={() => setOpen(!open)}>{open ? 'Hide' : 'What should this be?'}</button>}
          {need.id.startsWith('model_') && !need.download && <Link className="btn small" to="/models#train">Train one on the Models page</Link>}
        </div>
      )}
      {open && (
        <div className="format-box">
          <pre>{need.format}</pre>
          <div className="row">
            {need.template && <a className="btn small" href={apiUrl(`/api/templates/${need.template}.csv`)}>Download template</a>}
            {need.template && <label className="btn small">Upload filled-in file…<input type="file" accept=".csv" hidden onChange={(e) => upload(e.target.files?.[0])} /></label>}
            {need.setting && !need.template && <button className="btn small" onClick={() => setPicking(true)}>Choose…</button>}
          </div>
        </div>
      )}
      <ErrorNote error={error} />
      {picking && <FilePicker title={`Choose the ${need.label.toLowerCase()}`} pickFiles={need.id.startsWith('model_')} show={need.id.startsWith('model_') ? 'models' : 'slides'}
        onClose={() => setPicking(false)} onPick={choose} />}
    </div>
  )
}

function SlidesPanel({ onSaved }: { onSaved: () => void }) {
  const { info: project, reload: reloadProject } = useProject()
  const info = useFetch<SlidesInfo>('/api/slides')
  const [rows, setRows] = useState<SlideRow[]>([])
  const [picking, setPicking] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState(false)
  useEffect(() => { if (info.data) setRows(info.data.rows) }, [info.data])
  const problems = slideProblems(rows, info.data?.stains ?? [])
  const chooseFolder = (path: string) => withError(async () => {
    await api.put('/api/project/config', { changes: { inputs: { slides_dir: path } } })
    setPicking(false)
    reloadProject()
    info.reload()
  }, setError)
  const save = () => withError(async () => { await api.put('/api/slides', { rows }); setSaved(true); onSaved() }, setError)

  return (
    <div className="panel">
      <Loaded state={info} label="Reading the slides folder…">
        {(data) => (
          <>
            <div className="row">
              <span className="small secondary">Slides folder:</span><code className="path-box">{data.folder}</code>
              <button className="btn small" onClick={() => setPicking(true)}>Choose folder…</button>
            </div>
            {!data.folder_exists && <p className="small muted">This folder does not exist yet — choose the folder that holds your slide scans.</p>}
            {rows.length > 0 && (
              <>
                <p className="small secondary">{rows.length} scans found. Choose the TMA and the stain of each one; values marked <i>from the file name</i> were read
                  from a file name that said them clearly.</p>
                <SlideTable info={data} rows={rows} prefix={project?.tma_prefix ?? ''} onChange={(next) => { setSaved(false); setRows(next) }} />
                <div className="row" style={{ marginTop: 8 }}>
                  <button className="btn primary" disabled={!problems.ok} onClick={save}>Save slides table</button>
                  {!problems.ok && <span className="small cell-note">{problems.text}</span>}
                  {saved && <span className="small status-finished">Saved.</span>}
                </div>
              </>
            )}
            {data.folder_exists && !rows.length && <p className="small muted">No slide scans (.vsi, .svs, .ndpi, .tif …) in this folder.</p>}
          </>
        )}
      </Loaded>
      <ErrorNote error={error} />
      {picking && <FilePicker title="Choose the folder with your slide scans" show="slides" onPick={chooseFolder} onClose={() => setPicking(false)} />}
    </div>
  )
}

function MapUpload({ onSaved }: { onSaved: () => void }) {
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState('')
  const upload = (kind: string, file: File | undefined) => file && withError(async () => {
    const result = await api.upload<{ saved: string; rows: number }>(`/api/upload/${kind}`, file)
    setMessage(`Saved ${result.rows} rows to ${result.saved}. Next: press Run below. You can give the group codes readable names in Settings.`)
    onSaved()
  }, setError)
  return (
    <div className="panel">
      <p className="small secondary">
        The <Term term="TMA">TMA</Term> map is a spreadsheet with one row per <Term>core position</Term>: <code>tma</code>, <code>core_label</code>,
        {' '}<code>donor_id</code>, <code>region</code>, <code>disease_group</code> (optional: <code>cerad</code>, <code>braak</code>, <code>sample_region_id</code>).
        Leave <code>donor_id</code> empty for orientation / control cores. Save it as CSV from Excel.
      </p>
      <div className="row">
        <a className="btn small" href={apiUrl('/api/templates/tma_layout.csv')}>Download template (all positions filled in)</a>
        <label className="btn small">Upload TMA map (CSV)…<input type="file" accept=".csv" hidden onChange={(e) => upload('tma_layout', e.target.files?.[0])} /></label>
        <label className="btn small">Upload donor information (optional)…<input type="file" accept=".csv" hidden onChange={(e) => upload('donor_metadata', e.target.files?.[0])} /></label>
        <a className="btn small" href={apiUrl('/api/templates/donor_metadata.csv')}>Donor template</a>
      </div>
      {message && <p className="small status-finished">{message}</p>}
      <ErrorNote error={error} />
    </div>
  )
}

const PAGE = 20

function jobFacts(job: Job): string {
  const { done, total } = job.progress
  const took = job.started && job.ended ? ` · took ${duration(job.ended - job.started)}` : ''
  if (job.status === 'queued') return job.waiting || 'Starts in a moment'
  if (job.status === 'failed' || job.status === 'skipped') return job.progress.error || 'See the log'
  if (job.status === 'running') return total ? `${done.toLocaleString()} of ${total.toLocaleString()} done` : 'Starting…'
  return (total ? `${done.toLocaleString()} of ${total.toLocaleString()} done` : '') + took
}

/** Every run of every step, newest first: step, status, how far it got, when; its log and command on demand. */
export function JobsTable({ limit, bare = false }: { limit?: number; bare?: boolean }) {
  const jobs = useFetch<Job[]>('/api/jobs', 3000)
  const [openLog, setOpenLog] = useState<string | null>(null)
  const [shown, setShown] = useState(limit ?? PAGE)
  const [error, setError] = useState<string | null>(null)
  const all = jobs.data ?? []
  const list = all.slice(0, shown)
  const stop = (job: Job) => withError(async () => { if (await stopJobs(job.title, [job])) jobs.reload() }, setError)
  const body = (
    <Loaded state={jobs} label="Reading the runs…">
      {() => list.length ? (
        <>
          <div className="table-wrap jobs-table">
            <table>
              <thead><tr><th>Step</th><th>Status</th><th className="jobs-progress">Progress</th><th>Started</th><th /></tr></thead>
              <tbody>
                {list.map((job) => (
                  <tr key={job.id} className={openLog === job.id ? 'selected' : ''}>
                    <td>{job.title} <CommandLineTag job={job} /></td>
                    <td><StatusBadge status={job.status} /></td>
                    <td className="jobs-progress">
                      {job.status === 'running' && <Progress done={job.progress.done} total={job.progress.total} />}
                      <div className="muted small ellipsis" title={jobFacts(job)}>{jobFacts(job)}</div>
                    </td>
                    <td className="muted small">{job.started ? timeAgo(job.started) : '—'}</td>
                    <td className="num">
                      <button className="link-btn small" onClick={() => setOpenLog(openLog === job.id ? null : job.id)}>{openLog === job.id ? 'Hide log' : 'Log'}</button>
                      {isActive(job.status) && <> · <button className="link-btn small" onClick={() => stop(job)}>Stop</button></>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {all.length > shown && !limit && <button className="link-btn small" onClick={() => setShown(shown + PAGE)}>Show {Math.min(PAGE, all.length - shown)} more (of {all.length})</button>}
          {limit && all.length > limit && <Link className="small" to="/workflow#jobs">All {all.length} runs →</Link>}
          {openLog && <JobLog id={openLog} job={all.find((j) => j.id === openLog)} />}
        </>
      ) : <Empty>No runs yet. Press <b>Run what’s left</b> on the <Link to="/workflow">Run</Link> page to start.</Empty>}
    </Loaded>
  )
  return (
    <div className={bare ? '' : 'card'} id="jobs">
      {!bare && <h2>Recent runs</h2>}
      {body}
      <ErrorNote error={error} />
    </div>
  )
}

/** A run's output (newest lines), with where it ran and its command under Advanced. */
export function JobLog({ id, job }: { id: string; job?: Job }) {
  const log = useFetch<string>(`/api/jobs/${id}/log`, isActive(job?.status) ? 3000 : undefined)
  const box = useRef<HTMLDivElement>(null)
  const follow = useRef(true)  // keep the newest line in view unless the reader scrolled up
  useEffect(() => { box.current?.scrollIntoView({ block: 'nearest', behavior: 'smooth' }) }, [id])
  useEffect(() => { if (box.current && follow.current) box.current.scrollTop = box.current.scrollHeight }, [log.data])
  const scrolled = () => { const el = box.current; if (el) follow.current = el.scrollHeight - el.scrollTop - el.clientHeight < 30 }
  return (
    <div className="job-log">
      <div className="log" ref={box} onScroll={scrolled}>{log.data === null ? (log.error ? '' : 'Reading the log…') : log.data || '(no output yet)'}</div>
      <ErrorNote error={log.error} />
      {job && (
        <Advanced title="Details of this run">
          <div className="small secondary">
            Started from {job.source === 'command line' ? 'a terminal' : 'the app'}{job.started ? ` ${timeAgo(job.started)}` : ''}{job.pid ? ` · process ${job.pid}` : ''}
            {job.returncode !== null ? ` · exit code ${job.returncode}` : ''}
          </div>
          <code className="path-box">stainid {job.argv.join(' ')}</code>
          {job.log_file && <div className="small secondary">Output file: <code>{job.log_file}</code></div>}
        </Advanced>
      )}
    </div>
  )
}
