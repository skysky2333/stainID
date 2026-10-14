import { useState } from 'react'
import { Link } from 'react-router-dom'
import type { CoresInfo, Job, SlideReport, Step } from '../api'
import { api, apiUrl, runStep } from '../api'
import { Empty, ErrorNote, Progress, withError } from '../components'
import { useFetch } from '../hooks'
import { StepCard } from '../pages/Workflow'
import { Advanced, Found, ScreenFoot } from './parts'
import type { ScreenProps } from './Setup'

const active = (j: Job) => j.status === 'running' || j.status === 'queued'

function fitText(report: SlideReport['report']): [string, string] {
  if (report.score === undefined) return ['', '']
  if (report.score >= 0.85) return ['fits well', 'status-done']
  if (report.score >= 0.7) return ['please check', 'status-outdated']
  return ['poor fit: please check', 'status-failed']
}

export default function CoresScreen({ next, back, onChanged }: ScreenProps) {
  const cores = useFetch<CoresInfo>('/api/setup/cores', 5000)
  const jobs = useFetch<Job[]>('/api/jobs', 3000)
  const steps = useFetch<Step[]>('/api/steps', 5000)
  const [error, setError] = useState<string | null>(null)
  const finding = (jobs.data ?? []).filter((j) => j.workflow === 'dearray' && active(j))
  const slides = cores.data?.slides ?? []
  const waiting = slides.filter((s) => !s.found && !s.error)
  const failed = slides.filter((s) => s.error)
  const found = slides.filter((s) => s.found && !s.error)
  const checked = found.filter((s) => s.checked)
  const allJob = finding.find((j) => !j.options.slide)
  const refresh = () => { cores.reload(); jobs.reload(); onChanged() }
  const findAll = () => withError(async () => { await runStep('dearray', {}); refresh() }, setError)
  const dearray = steps.data?.find((s) => s.id === 'dearray')

  if (!cores.data) return <ErrorNote error={cores.error} />
  if (!slides.length) return <div className="screen"><Empty>Register the slide scans first (the Slides step).</Empty><ScreenFoot back={back} /></div>

  return (
    <div className="screen">
      {(waiting.length > 0 || allJob) && (
        <div className="card next-card">
          {allJob ? (
            <>
              <h2 style={{ margin: 0 }}>Finding the cores…</h2>
              <p className="small secondary">{allJob.status === 'queued' ? allJob.waiting || 'Starts in a moment.' : allJob.progress.last_line || 'Starting…'}</p>
              <Progress done={allJob.progress.done} total={allJob.progress.total || waiting.length} />
              <p className="small muted">About a minute per slide. You can leave this page; it keeps running.</p>
            </>
          ) : (
            <>
              <h2 style={{ margin: 0 }}>Find the cores on {waiting.length} {waiting.length === 1 ? 'slide' : 'slides'}</h2>
              <p className="secondary">stainID reads a small picture of each slide, works out how many rows and columns of cores it holds, and centres each core on its tissue. About a minute per slide.</p>
              <button className="btn primary" onClick={findAll}>Find cores</button>
            </>
          )}
        </div>
      )}
      {found.length > 0 && (
        <Found tone={failed.length ? 'warn' : 'good'}>
          Cores found on <b>{found.length} of {slides.length} slides</b>.{' '}
          {checked.length < found.length ? <>Look at each picture: every circle should sit on a core, and the labels (A-1 top left) should match your map. Press <b>Looks right</b> or <b>Fix</b>.</>
            : <>You checked all of them.</>}
        </Found>
      )}
      <ErrorNote error={error ?? cores.error} />
      <div className="slide-grid">
        {[...failed, ...found, ...waiting].map((s) => (
          <SlideCard key={s.slide_path} slide={s} info={cores.data!} running={finding.find((j) => String(j.options.slide ?? '') === s.name)} onChanged={refresh} />
        ))}
      </div>

      <Advanced hint="the Find cores step, its options and log">
        <p className="small secondary">A slide without its own rows and columns gets the grid size that fits best: the project's {cores.data.rows} × {cores.data.columns} unless another size
          fits clearly better. The size it chose is written to the slides table, where it can be changed (Fix). The pictures are in <code>qc/grids</code>; the same pictures are on
          the <Link to="/setup/grids">core grid check</Link> page.</p>
        {dearray && <StepCard step={dearray} onChanged={refresh} />}
      </Advanced>
      <ScreenFoot back={back} next={next} nextLabel="Next: who is on each core" />
    </div>
  )
}

function SlideCard({ slide, info, running, onChanged }: { slide: SlideReport; info: CoresInfo; running?: Job; onChanged: () => void }) {
  const [fixing, setFixing] = useState(Boolean(slide.error))
  const [rows, setRows] = useState(slide.rows || String(slide.report.rows ?? info.rows))
  const [columns, setColumns] = useState(slide.columns || String(slide.report.columns ?? info.columns))
  const [orientation, setOrientation] = useState(slide.orientation)
  const [error, setError] = useState<string | null>(null)
  const [fit, tone] = fitText(slide.report)
  const report = slide.report
  const mark = (checked: boolean) => withError(async () => { await api.post('/api/setup/cores/checked', { slide_path: slide.slide_path, checked }); onChanged() }, setError)
  const findAgain = (automatic: boolean) => withError(async () => {
    await api.put('/api/setup/cores/grid', automatic ? { slide_path: slide.slide_path, orientation } : { slide_path: slide.slide_path, rows: Number(rows), columns: Number(columns), orientation })
    setFixing(false); onChanged()
  }, setError)
  const size = slide.rows ? `${slide.rows} × ${slide.columns}` : report.rows ? `${report.rows} × ${report.columns}` : ''

  return (
    <div className={`card slide-card${slide.checked ? ' checked' : ''}${slide.error ? ' failed' : ''}`}>
      <div className="row">
        <b>TMA {slide.tma} · {slide.stain}</b><span className="muted small ellipsis" title={slide.name}>{slide.name}</span>
        <span className="spacer" />{slide.checked && <span className="badge status-done">✓ looks right</span>}
      </div>
      {slide.image ? (
        <a href={apiUrl(`/api/grids/${encodeURIComponent(slide.image)}?t=${report.time ?? ''}`)} target="_blank" rel="noreferrer" title="Open the picture full size">
          <img className="slide-shot" src={apiUrl(`/api/grids/${encodeURIComponent(slide.image)}?t=${report.time ?? ''}`)} alt={`Core grid on ${slide.name}`} loading="lazy" />
        </a>
      ) : <div className="slide-shot empty-shot muted small">{slide.error ? 'No cores found' : running ? 'Finding cores…' : 'Not looked at yet'}</div>}
      {slide.error && <div className="error-note small">{slide.error}</div>}
      {size && (
        <div className="small secondary">
          {size} grid{slide.grid_source === 'auto' ? ' (found automatically)' : slide.grid_source === 'manual' ? ' (set by you)' : ''}
          {report.positions_with_tissue !== undefined && ` · ${report.positions_with_tissue} of ${report.positions} positions hold tissue`}
          {slide.orientation && ` · ${info.orientations[slide.orientation]}`}
          {fit && <> · <span className={tone}>{fit}</span></>}
        </div>
      )}
      {running ? <div className="small secondary live-line">{running.status === 'queued' ? running.waiting || 'Waiting to start…' : running.progress.last_line || 'Finding cores again…'}</div> : slide.found && (
        <div className="row">
          {!slide.checked ? <button className="btn small primary" onClick={() => mark(true)}>Looks right</button>
            : <button className="link-btn small" onClick={() => mark(false)}>Not checked</button>}
          <button className="btn small" onClick={() => setFixing(!fixing)}>{fixing ? 'Cancel' : 'Fix…'}</button>
        </div>
      )}
      {fixing && !running && (
        <div className="panel">
          <span className="small secondary">How many rows and columns of cores does this slide have, and does it lie the right way round (A-1 top left)?</span>
          <div className="row">
            <label className="field">Rows<input type="number" min={1} max={26} value={rows} style={{ width: 70 }} onChange={(e) => setRows(e.target.value)} /></label>
            <label className="field">Columns<input type="number" min={1} max={26} value={columns} style={{ width: 70 }} onChange={(e) => setColumns(e.target.value)} /></label>
            <label className="field">Slide lies<select value={orientation} onChange={(e) => setOrientation(e.target.value)}>
              {Object.entries(info.orientations).map(([k, v]) => <option key={k} value={k}>{v}</option>)}</select></label>
          </div>
          <div className="row">
            <button className="btn small primary" disabled={!Number(rows) || !Number(columns)} onClick={() => findAgain(false)}>Find the cores again</button>
            <button className="link-btn small" onClick={() => findAgain(true)}>Find the size automatically</button>
          </div>
        </div>
      )}
      <ErrorNote error={error} />
    </div>
  )
}
