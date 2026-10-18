/* The segmentation check on the Models page, shown only where there is something to score: polygons a dataset's raters
   drew (imported sessions, the public-dataset projects) and outline sessions of older projects ("Outline every object",
   no longer made), compared with detection's candidates and counted objects; the scores in plain words, and every figure
   and file under Advanced. In a study, outline accuracy comes from the outlines people fix in "Fix the field"
   (OutlineFixes). */
import { useState } from 'react'
import type { Job } from '../api'
import { downloadUrl, runStep } from '../api'
import { ErrorNote, Loaded, Progress, Term, fmt, timeAgo, withError } from '../components'
import { useFetch } from '../hooks'
import { objectsOf, percent } from '../labelling'
import { useProject } from '../project'
import { Advanced } from '../setup/parts'
import type { CheckGroup, CheckOverview, CheckSession, Outcome, RaterScores, SizeBin, StainScores } from './types'

const OUTCOMES: { key: Outcome; label: string; hint: string }[] = [
  { key: 'counted', label: 'found and counted', hint: 'detection counted it, with an outline that fits (IoU ≥ 0.5)' },
  { key: 'rejected', label: 'proposed, not counted', hint: 'detection outlined it well but its model decided it was not one' },
  { key: 'poor outline', label: 'poor outline', hint: 'detection looked at it but its outline does not fit' },
  { key: 'split', label: 'cut into pieces', hint: 'detection saw several objects where people see one' },
  { key: 'merged', label: 'merged with a neighbour', hint: 'detection saw one object where people see two or more' },
  { key: 'missed', label: 'never proposed', hint: 'detection did not look at it at all' },
]
const active = (job: Job | undefined) => !!job && (job.status === 'running' || job.status === 'queued')

export function SegmentationCheck() {
  const overview = useFetch<CheckOverview>('/api/segmentation', 15000)
  const jobs = useFetch<Job[]>('/api/jobs', 5000)
  const [error, setError] = useState<string | null>(null)
  const job = (jobs.data ?? []).find((j) => j.workflow === 'score_segmentation')
  const running = active(job)
  const score = () => withError(async () => { await runStep('score_segmentation', { stain: 'all' }); jobs.reload() }, setError)
  const scorable = !!overview.data && (overview.data.checks.some((c) => c.complete > 0) || overview.data.imported_sessions > 0)
  if (overview.data && !overview.data.checks.length && !overview.data.imported_sessions) return null  // nothing to score
  return (
    <section className="card segcheck">
      <div className="row"><h2 style={{ margin: 0 }}>How well does detection outline objects?</h2><span className="spacer" />
        {overview.data && (overview.data.checks.length > 0 || overview.data.imported_sessions > 0) && <button className="btn small" disabled={running || !scorable}
          title={scorable ? 'Compares every complete field with what detection found' : 'No field is marked complete yet'}
          onClick={score}>{running ? 'Scoring…' : 'Score now'}</button>}</div>
      <p className="secondary small" style={{ marginTop: 4 }}>The model scores above are measured only on the objects detection proposed. Here every object
        was outlined in a few fields (by a dataset’s raters, or in an older outline session), so the check also counts objects detection never proposed, cut
        into pieces or merged with a neighbour.</p>
      <Loaded state={overview} label="Loading the segmentation check…">
        {(data) => data.checks.length || data.imported_sessions ? <>
          {running && job && <JobLine job={job} />}
          {!running && job?.status === 'failed' && <ErrorNote error={`Scoring stopped: ${job.progress.error || job.progress.last_line || 'see the log on the Run page'}`} />}
          <div className="segcheck-stains">
            {data.stains.map((stain) => <StainCheck key={stain} stain={stain} checks={data.checks.filter((c) => c.stain === stain)}
              scores={data.report?.stains.find((s) => s.stain === stain)} />)}
          </div>
          {data.imported_sessions > 0 && <Imported raters={data.report?.imported ?? []} scored={!!data.report} />}
          {data.report && <p className="muted small">Scored {timeAgo(data.report.made)}{data.report.failed.length > 0 && ` · ${data.report.failed.length} fields could not be scored`}.
            {' '}Fields marked complete since then count after <b>Score now</b>.</p>}
          <Details data={data} />
        </> : null}
      </Loaded>
      <ErrorNote error={error} />
    </section>
  )
}

/** The polygons of a dataset package's raters, each scored like a check (only fields where a rater drew polygons). */
function Imported({ raters, scored }: { raters: RaterScores[]; scored: boolean }) {
  return (
    <div>
      <h3 style={{ margin: '6px 0' }}>Outlines from the dataset’s raters</h3>
      {raters.length ? <div className="segcheck-stains">{raters.map((r) => (
        <div key={r.sessions[0]?.name ?? r.title} className="segcheck-stain">
          <div className="row"><h3 style={{ margin: 0 }}>{r.title}</h3></div>
          <div className="small muted">{r.stain} · {r.fields_scored.toLocaleString()} images or fields with polygons</div>
          {r.objects > 0 ? <Scores s={r} words={objectsOf(r.stain)} /> : <p className="small muted">No polygon of the stain’s classes.</p>}
        </div>
      ))}</div> : <p className="small muted">{scored ? 'None of the imported raters drew polygons (boxes and points cannot measure outlines).'
        : 'Press Score now to compare the polygons the dataset’s raters drew with what detection found.'}</p>}
    </div>
  )
}

function JobLine({ job }: { job: Job }) {
  return (
    <div className="training-now">
      <div className="row"><span className="pulse" /><b>{job.status === 'queued' ? 'Waiting to start…' : job.options.proposals_only
        ? 'Finding detection’s candidates in the check’s fields…' : 'Scoring…'}</b></div>
      {job.progress.total > 0 && <><Progress done={job.progress.done} total={job.progress.total} />
        <span className="muted small">{job.progress.done.toLocaleString()} of {job.progress.total.toLocaleString()} fields · 10–30 s each, once per field</span></>}
    </div>
  )
}

function StainCheck({ stain, checks, scores }: { stain: string; checks: CheckSession[]; scores?: StainScores }) {
  const fields = checks.reduce((n, c) => n + c.fields, 0)
  const complete = checks.reduce((n, c) => n + c.complete, 0)
  const words = objectsOf(stain)
  if (!checks.length) return null
  return (
    <div className="segcheck-stain">
      <div className="row"><h3 style={{ margin: 0 }}>{stain}</h3><span className="spacer" />
        <span className="small muted">an older outline session</span></div>
      <Progress done={complete} total={fields} />
      <div className="small secondary">{complete.toLocaleString()} of {fields.toLocaleString()} fields outlined completely
</div>
      {scores && scores.objects > 0 ? <Scores s={scores} words={words} />
        : scores && scores.fields_scored > 0 ? <p className="small secondary">The {scores.fields_scored} fields outlined so far hold no {words}; detection counted
          {' '}{(scores.counted ?? 0).toLocaleString()} there{scores.counted ? ', all wrong' : ''}.</p>
        : <p className="small muted">{complete ? 'Press Score now to compare the complete fields with what detection found.'
          : 'No field of it was outlined completely.'}</p>}
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
      {report ? [...report.stains, ...(report.imported ?? [])].filter((s) => s.fields_scored > 0).map((s) => (
        <div key={'title' in s ? `${s.title}` : s.stain} className="segcheck-details">
          <h3>{'title' in s ? `${s.title} (${s.stain})` : s.stain}</h3>
          <FigureTable sections={[
            ['', [{ key: 'All fields', fields: s.fields_scored, ...(s as Required<StainScores>) }]],
            ...(s.objects ? [
              [data.tma_study ? 'By TMA' : 'By batch', s.by_tma.map((g) => ({ ...g, key: data.tma_study ? tmaName(g.key) : g.key }))],
              ['By group', s.by_group], ['By stain density', s.by_density]] as [string, CheckGroup[]][] : [])]} />
          {s.objects > 0 && <>
            <SizeTable bins={s.by_size} />
            <IouHistogram proposal={s.proposal_iou_histogram} detection={s.detection_iou_histogram} />
          </>}
          <p className="muted small">{s.objects > 0 && <>Median outline fit of proposed objects: IoU {fmt(s.proposal_iou_median, 2)}, boundary IoU {fmt(s.boundary_iou_median, 2)}
            {s.outline_iou_median != null && <>; counted objects’ Segment Anything outlines: IoU {fmt(s.outline_iou_median, 2)}</>}.</>}
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

/** Every figure in one table: all fields, then a section per grouping (sections with a single row are left out). */
function FigureTable({ sections }: { sections: [string, CheckGroup[]][] }) {
  const columns = 7 + OUTCOMES.length - 1
  return (
    <div className="table-scroll">
      <table className="compact segcheck-table">
        <thead><tr><th /><th className="num">Fields</th><th className="num">Objects</th><th className="num" title="proposed with a fitting outline">Proposed</th>
          <th className="num">Found and counted</th><th className="num" title="of what detection counted">Right</th>
          <th className="num" title="proposed · found and counted · right, at IoU 0.3">At IoU 0.3</th>
          {OUTCOMES.slice(1).map((k) => <th key={k.key} className="num" title={k.hint}>{k.label}</th>)}<th className="num">Detection counted</th></tr></thead>
        {sections.filter(([title, rows]) => rows.length && (!title || rows.length > 1)).map(([title, rows]) => (
          <tbody key={title || 'all'}>
            {title && <tr className="section-row"><th colSpan={columns}>{title}</th></tr>}
            {rows.map((r) => (
              <tr key={r.key}><td>{r.key}</td><td className="num">{r.fields}</td><td className="num">{r.objects.toLocaleString()}</td>
                <td className="num">{percent(r.proposal_recall)}</td><td className="num">{percent(r.detection_recall)}</td><td className="num">{percent(r.detection_precision)}</td>
                <td className="num">{percent(r.proposal_recall_low)} · {percent(r.detection_recall_low)} · {percent(r.detection_precision_low)}</td>
                {OUTCOMES.slice(1).map((k) => <td key={k.key} className="num">{(r.outcomes?.[k.key] ?? 0).toLocaleString()}</td>)}
                <td className="num">{r.counted.toLocaleString()}</td></tr>
            ))}
          </tbody>
        ))}
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
