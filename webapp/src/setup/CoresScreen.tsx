import { useState } from 'react'
import { Link } from 'react-router-dom'
import type { CoresInfo, Job, SlideReport, Step } from '../api'
import { api, apiUrl, runStep } from '../api'
import { Empty, ErrorNote, Loaded, Progress, withError } from '../components'
import { useFetch } from '../hooks'
import { StepCard } from '../pages/Workflow'
import { Advanced, Found, plural, ScreenData, ScreenFoot } from './parts'
import type { ScreenProps } from './Setup'

const active = (j: Job) => j.status === 'running' || j.status === 'queued'

function fitText(report: SlideReport['report']): [string, string] {
  if (report.score === undefined) return ['', '']
  if (report.score >= 0.85) return ['fits well', 'status-done']
  if (report.score >= 0.7) return ['please check', 'status-outdated']
  return ['poor fit: please check', 'status-failed']
}

export default function CoresScreen(props: ScreenProps) {
  const cores = useFetch<CoresInfo>('/api/setup/cores', 5000)
  return <ScreenData state={cores} label="Looking at the slides…">{(data) => <Cores {...props} info={data} reload={cores.reload} />}</ScreenData>
}

function Cores({ next, back, nextLabel, go, onChanged, info, reload }: ScreenProps & { info: CoresInfo; reload: () => void }) {
  const jobs = useFetch<Job[]>('/api/jobs', 3000)
  const [error, setError] = useState<string | null>(null)
  const [show, setShow] = useState<'all' | 'todo'>('all')
  const finding = (jobs.data ?? []).filter((j) => j.workflow === 'dearray' && active(j))
  const slides = [...info.slides].sort((a, b) => Number(a.tma) - Number(b.tma) || a.tma.localeCompare(b.tma) || a.stain.localeCompare(b.stain))
  const waiting = slides.filter((s) => !s.found && !s.error)
  const failed = slides.filter((s) => s.error)
  const found = slides.filter((s) => s.found && !s.error)
  const checked = found.filter((s) => s.checked)
  const allJob = finding.find((j) => !j.options.slide)
  const refresh = () => { reload(); jobs.reload(); onChanged() }
  const findAll = () => withError(async () => { await runStep('dearray', {}); refresh() }, setError)
  const shown = [...failed, ...found, ...waiting].filter((s) => show === 'all' || !s.checked)

  if (!slides.length) {
    return (
      <div className="screen">
        <Empty><p>No slides are registered yet: tell stainID which scan is which on the Slides step first.</p>
          <button className="btn primary" onClick={() => go('slides')}>Go to Slides</button></Empty>
        <ScreenFoot back={back} />
      </div>
    )
  }
  return (
    <div className="screen">
      {(waiting.length > 0 || allJob) && (
        <div className="card next-card">
          {allJob ? (
            <>
              <h2>Finding the cores…</h2>
              <Progress done={allJob.progress.done} total={allJob.progress.total || waiting.length} />
              <p className="small secondary">{allJob.status === 'queued' ? allJob.waiting || 'Starts in a moment.'
                : allJob.progress.total ? `${allJob.progress.done} of ${plural(allJob.progress.total, 'slide')} done` : 'Starting…'}
                {' '}· about a minute per slide. You can leave this page; it keeps running.</p>
            </>
          ) : (
            <>
              <h2>Find the cores on {plural(waiting.length, 'slide')}</h2>
              <p className="secondary">stainID reads a small picture of each slide, works out how many rows and columns of cores it holds, and centres each core on its tissue. About a minute per slide.</p>
              <div><button className="btn primary" onClick={findAll}>Find cores</button></div>
            </>
          )}
        </div>
      )}
      {found.length > 0 && (
        <div className="card check-summary">
          <div className="row">
            <b>{checked.length === found.length ? `You checked all ${plural(found.length, 'slide')}` : `${checked.length} of ${plural(found.length, 'slide')} checked`}</b>
            <span className="spacer" />
            <div className="segmented" role="group" aria-label="Which slides to show">
              <button className={show === 'all' ? 'on' : ''} aria-pressed={show === 'all'} onClick={() => setShow('all')}>All ({slides.length})</button>
              <button className={show === 'todo' ? 'on' : ''} aria-pressed={show === 'todo'} onClick={() => setShow('todo')}>Still to check ({slides.length - checked.length})</button>
            </div>
          </div>
          <Progress done={checked.length} total={found.length} />
          <Found tone={failed.length ? 'warn' : checked.length === found.length ? 'good' : 'plain'}>
            {failed.length > 0 && <><b>No cores were found on {plural(failed.length, 'slide')}</b>: open <b>Fix</b> on {failed.length === 1 ? 'it' : 'them'} and give the number of rows and columns. </>}
            {checked.length < found.length ? <>Look at each picture: every circle should sit on a core, and the labels (A-1 at the top left) should match your map. Then press <b>Looks right</b>, or <b>Fix</b> when it does not.</>
              : <>Every slide with cores looks right. Go on to the map.</>}
          </Found>
        </div>
      )}
      {failed.length > 0 && !found.length && !allJob && (
        <Found tone="warn">
          <b>No cores were found on {failed.length === slides.length ? 'any slide' : plural(failed.length, 'slide')}.</b> Open <b>Fix</b> on a slide and give its rows and columns,
          or check that the scans open in your slide viewer.{' '}
          <button className="btn small" onClick={findAll}>Try all slides again</button>
        </Found>
      )}
      <ErrorNote error={error} />
      {shown.length ? (
        <div className="slide-grid">
          {shown.map((s) => (
            <SlideCard key={s.slide_path} slide={s} info={info} running={finding.find((j) => String(j.options.slide ?? '') === s.name)} onChanged={refresh} />
          ))}
        </div>
      ) : <Empty><p>Every slide is checked.</p><button className="btn" onClick={() => setShow('all')}>Show all slides</button></Empty>}

      <Advanced hint="the Find cores step, its options and log">
        <p className="small secondary">A slide without its own rows and columns gets the grid size that fits best: the project's {info.rows} × {info.columns} unless another size
          fits clearly better. The size it chose is written to the slides table, where it can be changed (Fix). The pictures are in <code>qc/grids</code>; the same pictures are on
          the <Link to="/setup/grids">core grid check</Link> page.</p>
        <FindCoresStep onChanged={refresh} />
      </Advanced>
      <ScreenFoot back={back} next={next} nextLabel={nextLabel} quiet={waiting.length > 0 || Boolean(allJob) || checked.length < found.length} />
    </div>
  )
}

/** The Find cores step as on the Workflow page (options, log); only loaded when Advanced is opened. */
function FindCoresStep({ onChanged }: { onChanged: () => void }) {
  const steps = useFetch<Step[]>('/api/steps', 10000)
  return (
    <Loaded state={steps} label="Loading the step…">
      {(list) => { const step = list.find((s) => s.id === 'dearray'); return step ? <StepCard step={step} onChanged={() => { steps.reload(); onChanged() }} /> : null }}
    </Loaded>
  )
}

function SlideCard({ slide, info, running, onChanged }: { slide: SlideReport; info: CoresInfo; running?: Job; onChanged: () => void }) {
  const [fixing, setFixing] = useState(Boolean(slide.error))
  const [rows, setRows] = useState(slide.rows || String(slide.report.rows ?? info.rows))
  const [columns, setColumns] = useState(slide.columns || String(slide.report.columns ?? info.columns))
  const [orientation, setOrientation] = useState(slide.orientation)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [fit, tone] = fitText(slide.report)
  const report = slide.report
  const act = (action: () => Promise<unknown>) => withError(async () => { setBusy(true); try { await action(); onChanged() } finally { setBusy(false) } }, setError)
  const mark = (checked: boolean) => act(() => api.post('/api/setup/cores/checked', { slide_path: slide.slide_path, checked }))
  const findAgain = (automatic: boolean) => act(async () => {
    await api.put('/api/setup/cores/grid', automatic ? { slide_path: slide.slide_path, orientation } : { slide_path: slide.slide_path, rows: Number(rows), columns: Number(columns), orientation })
    setFixing(false)
  })
  const size = slide.rows ? `${slide.rows} × ${slide.columns}` : report.rows ? `${report.rows} × ${report.columns}` : ''
  const picture = (width: number) => apiUrl(`/api/grids/${encodeURIComponent(slide.image)}?t=${report.time ?? ''}${width ? `&width=${width}` : ''}`)

  return (
    <div className={`card slide-card${slide.checked ? ' checked' : ''}${slide.error ? ' failed' : ''}`}>
      <div className="row slide-card-head">
        <b>TMA {slide.tma} · {slide.stain}</b>
        <span className="spacer" />{slide.checked && <span className="badge status-done">✓ Looks right</span>}
      </div>
      <span className="muted small ellipsis" title={slide.slide_path}>{slide.name}</span>
      {slide.image ? (
        <a href={picture(0)} target="_blank" rel="noreferrer" title="Open the picture full size">
          <img className="slide-shot" src={picture(720)} alt={`Core grid on ${slide.name}`} loading="lazy" />
        </a>
      ) : <div className="slide-shot empty-shot muted small">{slide.error ? 'No cores found' : running ? 'Finding cores…' : 'Not looked at yet'}</div>}
      {slide.error && <div className="error-note small">Could not find the cores: {slide.error.split(slide.slide_path).join(slide.name)}.
        {' '}Give the number of rows and columns under <b>Fix</b>, or check that the scan opens.</div>}
      {size && (
        <div className="small secondary">
          {size} grid{slide.grid_source === 'auto' ? ' (found automatically)' : slide.grid_source === 'manual' ? ' (set by you)' : ''}
          {report.positions_with_tissue !== undefined && ` · ${report.positions_with_tissue} of ${report.positions} positions hold tissue`}
          {slide.orientation && ` · ${info.orientations[slide.orientation]}`}
          {fit && <> · <span className={tone}>{fit}</span></>}
        </div>
      )}
      {running ? <div className="small secondary live-line">{running.status === 'queued' ? running.waiting || 'Waiting to start…' : 'Finding the cores again…'}</div> : (slide.found || slide.error) && (
        <div className="row">
          {slide.found && (!slide.checked ? <button className="btn small outline-accent" disabled={busy} onClick={() => mark(true)}>Looks right</button>
            : <button className="link-btn small" disabled={busy} onClick={() => mark(false)}>Undo “looks right”</button>)}
          <button className="btn small" onClick={() => setFixing(!fixing)}>{fixing ? 'Cancel' : 'Fix…'}</button>
        </div>
      )}
      {fixing && !running && (
        <div className="panel">
          <span className="small secondary">How many rows and columns of cores does this slide have, and does it lie the right way round (A-1 at the top left)?</span>
          <div className="row">
            <label className="field">Rows<input type="number" min={1} max={26} value={rows} style={{ width: 70 }} onChange={(e) => setRows(e.target.value)} /></label>
            <label className="field">Columns<input type="number" min={1} max={26} value={columns} style={{ width: 70 }} onChange={(e) => setColumns(e.target.value)} /></label>
            <label className="field">Slide lies<select value={orientation} onChange={(e) => setOrientation(e.target.value)}>
              {Object.entries(info.orientations).map(([k, v]) => <option key={k} value={k}>{v}</option>)}</select></label>
          </div>
          <div className="row">
            <button className="btn small outline-accent" disabled={busy || !Number(rows) || !Number(columns)} onClick={() => findAgain(false)}>{busy ? 'Starting…' : 'Find the cores again'}</button>
            <button className="link-btn small" disabled={busy} onClick={() => findAgain(true)}>Find the size automatically</button>
          </div>
        </div>
      )}
      <ErrorNote error={error} />
    </div>
  )
}
