import { useEffect, useMemo, useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import type { Job, ResourceInfo, SlideRow, SlidesInfo, Step } from '../api'
import { api, runStep } from '../api'
import { Empty, ErrorNote, FilePicker, HelpBox, OptionField, PageHead, PathLine, Progress, StatusBadge, Term, timeAgo, withError } from '../components'
import { useFetch } from '../hooks'
import { NowRunning, PipelineMap } from '../pipeline'
import { useProject } from '../project'

export const MAIN_STAGES = ['1 · Set up', '2 · Detect', '3 · Results']

export function stepState(step: Step): string {
  if (step.job && (step.job.status === 'running' || step.job.status === 'queued')) return step.job.status
  if (step.job?.status === 'failed' && step.progress.state !== 'done') return 'failed'
  if (step.progress.outdated) return 'outdated'
  return step.progress.state
}

export default function Workflow() {
  const steps = useFetch<Step[]>('/api/steps', 3000)
  const jobs = useFetch<Job[]>('/api/jobs', 3000)
  const navigate = useNavigate()
  const { hash } = useLocation()
  const list = (steps.data ?? []).filter((s) => MAIN_STAGES.includes(s.stage))
  useEffect(() => {
    if (hash && steps.data) document.getElementById(hash.slice(1))?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [hash, steps.data === null])
  const refresh = () => { steps.reload(); jobs.reload() }

  return (
    <>
      <PageHead title="Workflow" subtitle="Every step from slide scans to results tables, in order. Run them top to bottom." />
      <HelpBox id="workflow">
        <ul>
          <li>Each card says <b>what the step needs</b>, <b>what it makes</b> and <b>where the files go</b> (<span className="exists-dot yes" /> exists, <span className="exists-dot no" /> not yet). Use “Show in Finder” to open a location.</li>
          <li>Press <b>Run</b>. Steps run in the background: you can switch pages or start other steps. At most two heavy steps run at the same time; others wait in line.</li>
          <li>Steps are <b>resumable</b>: stopping or re-running a step continues where it left off. “Start over” (under Advanced) moves old results aside instead of deleting them.</li>
          <li>If a step fails, the card shows the reason in plain words; “Show log” has the full details.</li>
          <li>Keep the stainID window (terminal) open while steps run — closing it stops them.</li>
        </ul>
      </HelpBox>
      <ErrorNote error={steps.error} />
      {list.length > 0 && <PipelineMap steps={list} state={stepState} onOpen={(id) => navigate(`/workflow#${id}`)} />}
      <NowRunning jobs={jobs.data ?? []} steps={list} onChanged={refresh} />
      {MAIN_STAGES.map((stage) => (
        <section key={stage} className="stage">
          <h2 className="stage-title">{stage}</h2>
          {list.filter((s) => s.stage === stage).map((step) => <StepCard key={step.id} step={step} onChanged={refresh} />)}
        </section>
      ))}
      <JobsTable />
    </>
  )
}

export function StepCard({ step, onChanged }: { step: Step; onChanged: () => void }) {
  const optionsKey = JSON.stringify(step.options)
  const outdated = Boolean(step.progress.outdated)
  const defaults = useMemo(() => Object.fromEntries((JSON.parse(optionsKey) as Step['options']).map((o) => [o.key, o.key === 'fresh' && outdated ? true : o.default])), [optionsKey, outdated])
  const [values, setValues] = useState<Record<string, unknown>>(defaults)
  useEffect(() => setValues(defaults), [defaults])
  const [advanced, setAdvanced] = useState(false)
  const [details, setDetails] = useState(false)
  const [log, setLog] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const state = stepState(step)
  const { hash } = useLocation()
  const [open, setOpen] = useState(state !== 'done' || hash === `#${step.id}`)
  useEffect(() => { if (hash === `#${step.id}`) setOpen(true) }, [hash, step.id])
  const missing = step.needs.filter((n) => !n.exists)
  const active = state === 'running' || state === 'queued'
  const job = step.job
  const basic = step.options.filter((o) => !o.advanced)
  const extra = step.options.filter((o) => o.advanced)

  const run = () => withError(async () => { await runStep(step.id, values); onChanged() }, setError)
  const cancel = () => job && api.del(`/api/jobs/${job.id}`).then(onChanged)

  return (
    <div className={`card step-card state-${state}${open ? '' : ' collapsed'}`} id={step.id}>
      <button className="step-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <div>
          <h3 className="step-title">{step.title}</h3>
          <p className="step-summary">{step.summary}</p>
        </div>
        <span className="spacer" />
        {!open && step.progress.total > 0 && <span className="muted small">{step.progress.done.toLocaleString()} {step.progress.unit}</span>}
        <StatusBadge status={state} />
        <span className="chevron">{open ? '▾' : '▸'}</span>
      </button>
      {open && <>
      {step.progress.total > 0 && (
        <div className="step-progress">
          <Progress done={active && job ? job.progress.done : step.progress.done} total={active && job?.progress.total ? job.progress.total : step.progress.total} />
          <span className="muted small">{step.progress.done.toLocaleString()} of {step.progress.total.toLocaleString()} {step.progress.unit}{step.progress.note ? ` · ${step.progress.note}` : ''}</span>
        </div>
      )}
      {step.progress.total === 0 && step.progress.note && <p className="small secondary" style={{ margin: '4px 0' }}>{step.progress.note}</p>}
      <button className="link-btn" onClick={() => setDetails(!details)}>{details ? 'Less' : 'What does this step do?'}</button>
      {details && <p className="secondary step-details">{step.details}{step.duration && <><br /><span className="muted">Typical time: {step.duration}.</span></>}</p>}

      <div className="io-grid">
        <div>
          <div className="io-title">Needs</div>
          {step.needs.length ? step.needs.map((n) => <NeedRow key={n.id} need={n} onChanged={onChanged} />)
            : <span className="muted small">Nothing extra.</span>}
          {step.uses.length > 0 && (
            <>
              <div className="io-title" style={{ marginTop: 8 }}>Uses if available</div>
              {step.uses.map((u) => <NeedRow key={u.id} need={u} onChanged={onChanged} optional />)}
            </>
          )}
        </div>
        <div>
          <div className="io-title">Makes</div>
          {step.produces.map((p) => <div key={p.id} title={p.description}><PathLine path={p.path} exists={p.exists} label={p.label} /></div>)}
          {step.view && step.progress.done > 0 && <Link className="small" to={step.view}>See the results →</Link>}
        </div>
      </div>

      {step.progress.outdated && (
        <div className="warning-note">These results were made with a different model than the one in use now. To redo them with the current model,
          run the step again: <b>Start over</b> (under Advanced options) is already ticked, so old results are moved aside (not deleted) and redone.</div>
      )}
      {step.id === 'slides' && <SlidesPanel onSaved={onChanged} />}
      {step.id === 'layout' && <MapUpload onSaved={onChanged} />}

      {step.command && (
        <>
          {basic.length > 0 && <div className="row options">{basic.map((o) => <OptionField key={o.key} option={o} value={values[o.key]} onChange={(v) => setValues({ ...values, [o.key]: v })} />)}</div>}
          {extra.length > 0 && <button className="link-btn small" onClick={() => setAdvanced(!advanced)}>{advanced ? 'Hide advanced options' : 'Advanced options'}</button>}
          {advanced && <div className="row options">{extra.map((o) => <OptionField key={o.key} option={o} value={values[o.key]} onChange={(v) => setValues({ ...values, [o.key]: v })} />)}</div>}
          <div className="row step-actions">
            <button className="btn primary" disabled={active || missing.length > 0} onClick={run}>
              {step.progress.state === 'done' ? 'Run again' : step.progress.state === 'partial' ? 'Continue' : 'Run'}
            </button>
            {active && <button className="btn" onClick={cancel}>Stop</button>}
            {missing.length > 0 && <span className="muted small">Waiting for: {missing.map((m) => m.label).join(', ')}</span>}
            <span className="spacer" />
            {job && <span className="muted small">Last run {timeAgo(job.created)}</span>}
            {job && <button className="link-btn small" onClick={() => setLog(!log)}>{log ? 'Hide log' : 'Show log'}</button>}
          </div>
          {state === 'queued' && job?.waiting && <div className="small secondary live-line">{job.waiting}. It starts on its own.</div>}
          {state === 'running' && job?.progress.last_line && <div className="muted small live-line">{job.progress.last_line}</div>}
          {job?.status === 'failed' && <ErrorNote error={`This step stopped with an error: ${job.progress.error || job.progress.last_line || 'see the log'}`} />}
          {job?.status === 'skipped' && <div className="warning-note">{job.progress.error}</div>}
          <ErrorNote error={error} />
          {log && job && <JobLog id={job.id} />}
        </>
      )}
      </>}
    </div>
  )
}

/** One input of a step: where it is, whether it exists, and how to get it if it does not (go to the step that makes it,
download it, or see what to provide and pick / upload it). */
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
    <div title={need.description} className="need-row">
      <PathLine path={need.path} exists={need.exists} label={need.label} />
      {!need.exists && (
        <div className="need-actions">
          {need.made_by && <button className="btn small" onClick={() => navigate(`/workflow#${need.made_by!.id}`)}>Go to “{need.made_by.title}”</button>}
          {need.download && <button className="btn small primary" disabled={started} onClick={download}>{started ? 'Downloading…' : 'Download now'}</button>}
          {need.format && <button className="btn small" onClick={() => setOpen(!open)}>{open ? 'Hide' : 'What should this be?'}</button>}
          {need.id.startsWith('model_') && !need.download && <Link className="btn small" to="/models#train">Train one on the Models page</Link>}
          {optional && <span className="muted small">optional</span>}
        </div>
      )}
      {open && (
        <div className="format-box">
          <pre>{need.format}</pre>
          <div className="row">
            {need.template && <a className="btn small" href={`/api/templates/${need.template}.csv`}>Download template</a>}
            {need.template && <label className="btn small primary">Upload filled-in file…<input type="file" accept=".csv" hidden onChange={(e) => upload(e.target.files?.[0])} /></label>}
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
  const { reload: reloadProject } = useProject()
  const info = useFetch<SlidesInfo>('/api/slides')
  const [rows, setRows] = useState<SlideRow[]>([])
  const [picking, setPicking] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState(false)
  useEffect(() => { if (info.data) setRows(info.data.rows) }, [info.data])
  const edit = (i: number, key: keyof SlideRow, value: string) => { setSaved(false); setRows(rows.map((r, j) => (j === i ? { ...r, [key]: value } : r))) }
  const chooseFolder = (path: string) => withError(async () => {
    await api.put('/api/project/config', { changes: { inputs: { slides_dir: path } } })
    setPicking(false)
    reloadProject()
    info.reload()
  }, setError)
  const save = () => withError(async () => { await api.put('/api/slides', { rows }); setSaved(true); onSaved() }, setError)

  return (
    <div className="panel">
      <div className="row">
        <span className="small secondary">Slides folder:</span><code className="path-box">{info.data?.folder}</code>
        <button className="btn small" onClick={() => setPicking(true)}>Choose folder…</button>
      </div>
      {info.data && !info.data.folder_exists && <p className="small muted">This folder does not exist yet — choose the folder that holds your slide scans.</p>}
      {rows.length > 0 && (
        <>
          <p className="small secondary">{rows.length} scans found. The TMA number and stain were guessed from each file name — please check them.</p>
          <div className="table-wrap" style={{ maxHeight: 320 }}>
            <table>
              <thead><tr><th>Scan</th><th style={{ width: 110 }}>TMA</th><th style={{ width: 130 }}>Stain</th></tr></thead>
              <tbody>{rows.map((r, i) => (
                <tr key={r.slide_path}>
                  <td title={r.slide_path}>{r.slide_path.split('/').pop()}</td>
                  <td><input type="text" value={r.tma} style={{ width: 80 }} onChange={(e) => edit(i, 'tma', e.target.value)} /></td>
                  <td><select value={r.stain} onChange={(e) => edit(i, 'stain', e.target.value)}>
                    <option value="">choose…</option>{info.data?.stains.map((s) => <option key={s}>{s}</option>)}</select></td>
                </tr>
              ))}</tbody>
            </table>
          </div>
          <div className="row" style={{ marginTop: 8 }}>
            <button className="btn primary" onClick={save}>Save slides table</button>
            {saved && <span className="small status-finished">Saved.</span>}
          </div>
        </>
      )}
      {info.data?.folder_exists && !rows.length && <p className="small muted">No slide scans (.vsi, .svs, .ndpi, .tif …) in this folder.</p>}
      <ErrorNote error={error ?? info.error} />
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
        <a className="btn small" href="/api/templates/tma_layout.csv">Download template (all positions filled in)</a>
        <label className="btn small primary">Upload TMA map (CSV)…<input type="file" accept=".csv" hidden onChange={(e) => upload('tma_layout', e.target.files?.[0])} /></label>
        <label className="btn small">Upload donor information (optional)…<input type="file" accept=".csv" hidden onChange={(e) => upload('donor_metadata', e.target.files?.[0])} /></label>
        <a className="btn small" href="/api/templates/donor_metadata.csv">Donor template</a>
      </div>
      {message && <p className="small status-finished">{message}</p>}
      <ErrorNote error={error} />
    </div>
  )
}

export function JobsTable({ limit }: { limit?: number }) {
  const jobs = useFetch<Job[]>('/api/jobs', 3000)
  const [openLog, setOpenLog] = useState<string | null>(null)
  const list = (jobs.data ?? []).slice(0, limit)
  return (
    <div className="card" id="jobs">
      <h2>Jobs</h2>
      {list.length ? (
        <div className="table-wrap" style={{ maxHeight: 420 }}>
          <table>
            <thead><tr><th>Step</th><th>Status</th><th style={{ width: 240 }}>Progress</th><th>Started</th><th /></tr></thead>
            <tbody>
              {list.map((job) => (
                <tr key={job.id}>
                  <td>{job.title}<div className="muted small ellipsis" title={`stainid ${job.argv.join(' ')}`}><code>stainid {job.argv.join(' ')}</code></div></td>
                  <td><StatusBadge status={job.status} /></td>
                  <td><Progress done={job.progress.done} total={job.progress.total} />
                    <div className="muted small ellipsis">{job.status === 'failed' ? job.progress.error || job.progress.last_line : job.status === 'queued' ? job.waiting : job.progress.last_line}</div></td>
                  <td className="muted small">{job.started ? timeAgo(job.started) : '—'}</td>
                  <td className="row">
                    <button className="btn small" onClick={() => setOpenLog(openLog === job.id ? null : job.id)}>Log</button>
                    {(job.status === 'running' || job.status === 'queued') && <button className="btn small" onClick={() => api.del(`/api/jobs/${job.id}`).then(jobs.reload)}>Stop</button>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : <Empty>No jobs yet. Press Run on a step to start one.</Empty>}
      {openLog && <JobLog id={openLog} />}
    </div>
  )
}

export function JobLog({ id }: { id: string }) {
  const log = useFetch<string>(`/api/jobs/${id}/log`, 3000)
  return <div className="log">{log.data || '(no output yet)'}</div>
}
