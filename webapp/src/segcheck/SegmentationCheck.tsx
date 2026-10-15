/* "How well does detection outline objects?" on the Models page: the segmentation check per stain (people outline every
   object of a few fields; the AI's candidates and counted objects are compared with them), how far the outlining is,
   the scores in plain words, and every figure and file under Advanced. */
import { useState } from 'react'
import type { Job } from '../api'
import { api, appPath, downloadUrl, runStep } from '../api'
import { Empty, ErrorNote, Loaded, Modal, Progress, Term, fmt, timeAgo, withError } from '../components'
import { useFetch } from '../hooks'
import { objectsOf, percent } from '../labelling'
import { useProject } from '../project'
import { Advanced } from '../setup/parts'
import type { CheckGroup, CheckOverview, CheckSession, Outcome, SizeBin, StainScores } from './types'

const OUTCOMES: { key: Outcome; label: string; hint: string }[] = [
  { key: 'counted', label: 'found and counted', hint: 'the AI counted it, with an outline that fits (IoU ≥ 0.5)' },
  { key: 'rejected', label: 'proposed, not counted', hint: 'the AI outlined it well but its model decided it was not one' },
  { key: 'poor outline', label: 'poor outline', hint: 'the AI looked at it but its outline does not fit' },
  { key: 'split', label: 'cut into pieces', hint: 'the AI saw several objects where people see one' },
  { key: 'merged', label: 'merged with a neighbour', hint: 'the AI saw one object where people see two or more' },
  { key: 'missed', label: 'never proposed', hint: 'the AI did not look at it at all' },
]
const active = (job: Job | undefined) => !!job && (job.status === 'running' || job.status === 'queued')

export function SegmentationCheck() {
  const overview = useFetch<CheckOverview>('/api/segmentation', 15000)
  const jobs = useFetch<Job[]>('/api/jobs', 5000)
  const [settingUp, setSettingUp] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const job = (jobs.data ?? []).find((j) => j.workflow === 'score_segmentation')
  const running = active(job)
  const score = () => withError(async () => { await runStep('score_segmentation', { stain: 'all' }); jobs.reload() }, setError)
  return (
    <section className="card segcheck">
      <div className="row"><h2 style={{ margin: 0 }}>How well does detection outline objects?</h2><span className="spacer" />
        {overview.data && overview.data.checks.length > 0 && <button className="btn small" disabled={running || !overview.data.checks.some((c) => c.complete)}
          title={overview.data.checks.some((c) => c.complete) ? 'Compares every complete field with what the AI found' : 'No field is marked complete yet'}
          onClick={score}>{running ? 'Scoring…' : 'Score now'}</button>}</div>
      <p className="secondary small" style={{ marginTop: 4 }}>The model scores above are measured only on the objects the AI proposed. Here people outline
        <i> every</i> object in a few fields, so the check also counts objects the AI never proposed, cut into pieces or merged with a neighbour.</p>
      <Loaded state={overview} label="Loading the segmentation check…">
        {(data) => data.checks.length ? <>
          {running && job && <JobLine job={job} />}
          {!running && job?.status === 'failed' && <ErrorNote error={`Scoring stopped: ${job.progress.error || job.progress.last_line || 'see the log on the Run page'}`} />}
          <div className="segcheck-stains">
            {data.stains.map((stain) => <StainCheck key={stain} stain={stain} checks={data.checks.filter((c) => c.stain === stain)}
              scores={data.report?.stains.find((s) => s.stain === stain)} onSetUp={() => setSettingUp(true)} />)}
          </div>
          {data.report && <p className="muted small">Scored {timeAgo(data.report.made)}{data.report.failed.length > 0 && ` · ${data.report.failed.length} fields could not be scored`}.
            {' '}Fields marked complete since then count after <b>Score now</b>.</p>}
          <Details data={data} />
        </> : (
          <Empty>
            <p style={{ marginTop: 0 }}>No segmentation check yet. stainID picks about {data.default_fields} fields per stain, spread over TMAs, groups, regions and
              how much stain they hold; people outline every object in them, blind to the AI. That takes about 5–15 minutes per field.</p>
            <button className="btn primary" onClick={() => setSettingUp(true)}>Set up the check…</button>
          </Empty>
        )}
      </Loaded>
      <ErrorNote error={error} />
      {settingUp && overview.data && <SetUp data={overview.data} onClose={() => setSettingUp(false)} onDone={() => { overview.reload(); jobs.reload() }} />}
    </section>
  )
}

function JobLine({ job }: { job: Job }) {
  return (
    <div className="training-now">
      <div className="row"><span className="pulse" /><b>{job.status === 'queued' ? 'Waiting to start…' : job.options.proposals_only
        ? 'Finding the AI’s candidates in the check’s fields…' : 'Scoring…'}</b></div>
      {job.progress.total > 0 && <><Progress done={job.progress.done} total={job.progress.total} />
        <span className="muted small">{job.progress.done.toLocaleString()} of {job.progress.total.toLocaleString()} fields · 10–30 s each, once per field</span></>}
    </div>
  )
}

function StainCheck({ stain, checks, scores, onSetUp }: { stain: string; checks: CheckSession[]; scores?: StainScores; onSetUp: () => void }) {
  const fields = checks.reduce((n, c) => n + c.fields, 0)
  const complete = checks.reduce((n, c) => n + c.complete, 0)
  const words = objectsOf(stain)
  if (!checks.length) {
    return <div className="segcheck-stain"><h3>{stain}</h3><p className="small muted">No check for {stain} yet.</p>
      <button className="btn small" onClick={onSetUp}>Set up…</button></div>
  }
  return (
    <div className="segcheck-stain">
      <div className="row"><h3 style={{ margin: 0 }}>{stain}</h3><span className="spacer" />
        {checks.map((c) => <a key={c.name} className="small" href={appPath(`/annotate/${c.name}`)} target="_blank" rel="noreferrer"
          title={c.title}>Open the outlining page ↗</a>)}</div>
      <Progress done={complete} total={fields} />
      <div className="small secondary">{complete.toLocaleString()} of {fields.toLocaleString()} fields outlined completely
        {checks.some((c) => !c.open) && <span className="muted"> · closed to labellers (open it on Labels)</span>}</div>
      {scores && scores.objects > 0 ? <Scores s={scores} words={words} /> : (
        <p className="small muted">{complete ? 'Press Score now to compare the complete fields with what the AI found.'
          : `Scores appear once people have outlined fields completely. Send them the link on the Labels page.`}</p>
      )}
    </div>
  )
}

function Scores({ s, words }: { s: StainScores; words: string }) {
  const o = s.outcomes!
  return <>
    <div className="segcheck-numbers">
      <div><b>{percent(s.detection_recall)}</b><span className="small secondary">of real {words} found and counted</span></div>
      <div><b>{percent(s.proposal_recall)}</b><span className="small secondary">proposed with a fitting outline</span></div>
      <div><b>{percent(s.detection_precision)}</b><span className="small secondary">of what it counts is right</span></div>
    </div>
    <OutcomeBar outcomes={o} />
    <p className="small secondary" style={{ margin: 0 }}>Of {s.objects.toLocaleString()} {words} outlined in {s.fields_scored} fields: {o.missed.toLocaleString()} never proposed,
      {' '}{o.split.toLocaleString()} cut into pieces, {o.merged.toLocaleString()} merged with a neighbour, {o['poor outline'].toLocaleString()} with a poor outline,
      {' '}{o.rejected.toLocaleString()} proposed but not counted.</p>
  </>
}

function OutcomeBar({ outcomes }: { outcomes: Record<Outcome, number> }) {
  const total = OUTCOMES.reduce((n, k) => n + (outcomes[k.key] ?? 0), 0)
  if (!total) return null
  return (
    <div>
      <div className="outcome-bar" role="img" aria-label={OUTCOMES.map((k) => `${k.label} ${outcomes[k.key]}`).join(', ')}>
        {OUTCOMES.map((k) => outcomes[k.key] > 0 && <span key={k.key} className={`outcome-${k.key.replace(' ', '-')}`}
          style={{ width: `${(100 * outcomes[k.key]) / total}%` }} title={`${k.label}: ${outcomes[k.key]} (${k.hint})`} />)}
      </div>
      <div className="outcome-legend small">
        {OUTCOMES.map((k) => <span key={k.key} title={k.hint}><span className={`swatch outcome-${k.key.replace(' ', '-')}`} />{k.label} {outcomes[k.key].toLocaleString()}</span>)}
      </div>
    </div>
  )
}

/* ---------- every figure ---------- */

function Details({ data }: { data: CheckOverview }) {
  const report = data.report
  const { tmaName } = useProject()
  return (
    <Advanced hint="all figures by TMA, group, density and size; IoU 0.5 and 0.3; the files">
      {report ? report.stains.filter((s) => s.objects > 0).map((s) => (
        <div key={s.stain} className="segcheck-details">
          <h3>{s.stain}</h3>
          <FigureTable rows={[{ key: 'All fields', fields: s.fields_scored, ...(s as Required<StainScores>) }]} first="" />
          <FigureTable rows={s.by_tma.map((g) => ({ ...g, key: tmaName(g.key) }))} first="TMA" />
          <FigureTable rows={s.by_group} first="Group" />
          <FigureTable rows={s.by_density} first="Stain density" />
          <SizeTable bins={s.by_size} />
          <IouHistogram proposal={s.proposal_iou_histogram} detection={s.detection_iou_histogram} />
          <p className="muted small">Median outline fit of proposed objects: IoU {fmt(s.proposal_iou_median, 2)}, boundary IoU {fmt(s.boundary_iou_median, 2)}
            {s.outline_iou_median != null && <>; counted objects’ Segment Anything outlines: IoU {fmt(s.outline_iou_median, 2)}</>}.
            {s.not_found_again > 0 && <> {s.not_found_again} counted objects matched no candidate found again (the candidates may be out of date: run the check with
              “Find the candidates again”).</>}
            {' '}Labelled by {s.people.join(', ')}.</p>
        </div>
      )) : <p className="muted small">No scores yet.</p>}
      <p className="muted small"><Term>IoU</Term> (intersection over union) compares two outlines: 1 = the same, 0.5 = they share half of their combined area.
        An object is found when one candidate matches it one to one at IoU ≥ {report?.iou ?? 0.5} (also shown at {report?.iou_low ?? 0.3}). Candidates are everything
        the detection step’s segmentation proposes; counted objects are those its model accepted. Objects marked “unsure” are left out, and only objects whose middle
        lies inside a field’s square count.</p>
      <div className="row">{data.files.map((f) => f.exists
        ? <a key={f.path} className="btn small" href={downloadUrl(f.path)} title={f.path}>{f.label}</a>
        : <span key={f.path} className="muted small" title={f.path}>{f.label}: not made yet</span>)}</div>
      {report && report.failed.length > 0 && <details className="small"><summary>Fields that could not be scored ({report.failed.length})</summary>
        <ul>{report.failed.map((f) => <li key={f}><code>{f}</code></li>)}</ul></details>}
    </Advanced>
  )
}

function FigureTable({ rows, first }: { rows: CheckGroup[]; first: string }) {
  if (!rows.length || (first && rows.length < 2 && first !== 'TMA')) return null
  return (
    <div className="table-scroll">
      <table className="compact segcheck-table">
        <thead><tr><th>{first}</th><th className="num">Fields</th><th className="num">Objects</th><th className="num">Proposed</th><th className="num">Found and counted</th>
          <th className="num">Right</th><th className="num">At IoU 0.3: proposed · counted · right</th>
          {OUTCOMES.slice(2).map((k) => <th key={k.key} className="num">{k.label}</th>)}<th className="num">Counted</th></tr></thead>
        <tbody>{rows.map((r) => (
          <tr key={r.key}><td>{r.key}</td><td className="num">{r.fields}</td><td className="num">{r.objects.toLocaleString()}</td>
            <td className="num">{percent(r.proposal_recall)}</td><td className="num">{percent(r.detection_recall)}</td><td className="num">{percent(r.detection_precision)}</td>
            <td className="num">{percent(r.proposal_recall_low)} · {percent(r.detection_recall_low)} · {percent(r.detection_precision_low)}</td>
            {OUTCOMES.slice(2).map((k) => <td key={k.key} className="num">{(r.outcomes?.[k.key] ?? 0).toLocaleString()}</td>)}
            <td className="num">{r.counted.toLocaleString()}</td></tr>
        ))}</tbody>
      </table>
    </div>
  )
}

function SizeTable({ bins }: { bins: SizeBin[] }) {
  return (
    <div className="table-scroll">
      <table className="compact segcheck-table">
        <thead><tr><th>Object size (diameter)</th><th className="num">Objects</th><th className="num">Proposed</th><th className="num">Found and counted</th>
          {OUTCOMES.slice(1).map((k) => <th key={k.key} className="num">{k.label}</th>)}</tr></thead>
        <tbody>{bins.map((b) => (
          <tr key={b.key}><td>{b.key}</td><td className="num">{b.objects.toLocaleString()}</td><td className="num">{percent(b.proposal_recall)}</td>
            <td className="num">{percent(b.detection_recall)}</td>{OUTCOMES.slice(1).map((k) => <td key={k.key} className="num">{b.outcomes[k.key].toLocaleString()}</td>)}</tr>
        ))}</tbody>
      </table>
    </div>
  )
}

/** How well the best candidate fits each object: objects per IoU tenth, proposals and counted objects side by side. */
function IouHistogram({ proposal, detection }: { proposal: number[]; detection: number[] }) {
  const top = Math.max(1, ...proposal, ...detection)
  return (
    <div className="iou-histogram" role="img" aria-label="Objects by how well the best candidate fits them">
      {proposal.map((n, i) => (
        <div key={i} className="iou-bin" title={`IoU ${(i / 10).toFixed(1)}–${((i + 1) / 10).toFixed(1)}: ${n} by the best proposal, ${detection[i]} by the best counted object`}>
          <div className="iou-bars"><span className="iou-proposal" style={{ height: `${(100 * n) / top}%` }} /><span className="iou-detection" style={{ height: `${(100 * detection[i]) / top}%` }} /></div>
          <span className="muted">{(i / 10).toFixed(1)}</span>
        </div>
      ))}
      <div className="outcome-legend small"><span><span className="swatch iou-proposal" />best proposal</span><span><span className="swatch iou-detection" />best counted object</span>
        <span className="muted">objects by IoU (0 = not overlapped)</span></div>
    </div>
  )
}

/* ---------- setting it up ---------- */

function SetUp({ data, onClose, onDone }: { data: CheckOverview; onClose: () => void; onDone: () => void }) {
  const { tmaName } = useProject()
  const missing = data.stains.filter((s) => !data.checks.some((c) => c.stain === s))
  const [stains, setStains] = useState<string[]>(missing)
  const [n, setN] = useState(data.default_fields)
  const allTmas = [...new Set(data.stains.flatMap((s) => data.tmas[s]?.all ?? []))].sort((a, b) => a.length - b.length || a.localeCompare(b))
  const [tmas, setTmas] = useState<string[]>([...new Set(data.stains.flatMap((s) => data.tmas[s]?.default ?? []))])
  const [open, setOpen] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const toggle = (list: string[], value: string) => (list.includes(value) ? list.filter((v) => v !== value) : [...list, value])
  const create = async () => {
    setBusy(true)
    const made = await withError(() => api.post('/api/segmentation/sessions', { stains, n_fields: n, tmas, open }), setError)
    setBusy(false)
    if (made) { onDone(); onClose() }
  }
  return (
    <Modal title="Set up the segmentation check" onClose={onClose}>
      <p className="secondary" style={{ marginTop: 0 }}>One session per stain, “Segmentation check – <i>stain</i>”. People outline <b>every</b> object in each field, without
        seeing what the AI found, and mark the field complete. Fields are ready ones, spread over TMAs, diagnosis groups, regions and sparse, medium and dense
        staining. Meanwhile stainID finds the AI’s candidates in those fields in the background.</p>
      <fieldset className="field"><legend>Stains</legend>
        <div className="row">{data.stains.map((s) => {
          const exists = !missing.includes(s)
          return <label key={s} className="layer-toggle" title={exists ? 'This stain has a check already' : undefined}>
            <input type="checkbox" disabled={exists} checked={!exists && stains.includes(s)} onChange={() => setStains(toggle(stains, s))} />{s}{exists && ' (set up)'}</label>
        })}</div>
      </fieldset>
      <label className="field">Fields per stain<input type="number" min={1} max={200} value={n} style={{ width: 90 }} onChange={(e) => setN(Math.max(1, Number(e.target.value)))} /></label>
      <label className="layer-toggle"><input type="checkbox" checked={open} onChange={(e) => setOpen(e.target.checked)} />Open to labellers right away</label>
      <Advanced hint="which TMAs">
        <fieldset className="field"><legend>Fields come from these TMAs</legend>
          <div className="row">{allTmas.map((t) => <label key={t} className="layer-toggle">
            <input type="checkbox" checked={tmas.includes(t)} onChange={() => setTmas(toggle(tmas, t))} />{tmaName(t)}</label>)}</div>
        </fieldset>
      </Advanced>
      <ErrorNote error={error} />
      <div className="row"><span className="spacer" />
        <button className="btn" onClick={onClose}>Cancel</button>
        <button className="btn primary" disabled={busy || !stains.length || !tmas.length} onClick={create}>
          {busy ? 'Choosing fields…' : `Create ${stains.length} session${stains.length === 1 ? '' : 's'}`}</button>
      </div>
    </Modal>
  )
}
