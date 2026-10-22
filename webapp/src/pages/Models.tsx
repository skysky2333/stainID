import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import type { BadAreas, Detection, Job, LabelSet, LabelsOverview, Metrics, ModelInfo, TrainedModel } from '../api'
import { api, runStep } from '../api'
import { Empty, ErrorNote, HelpBox, Loaded, Modal, PageHead, PathLine, Progress, Term, fmt, timeAgo, withError } from '../components'
import { useFetch } from '../hooks'
import { ConfirmDialog, DetectionLine, QualityBadge, QualityLine, objectsOf, quality } from '../labelling'
import { useProject } from '../project'
import { ScoreOnLabels } from '../scoring'
import { Advanced } from '../setup/parts'
import { OutlineFixes } from '../segcheck/OutlineFixes'
import { SegmentationCheck } from '../segcheck/SegmentationCheck'
import { SimpleTraining, mappingText } from '../labels/SimpleSettings'

const TITLES: Record<string, string> = {
  neun: 'NeuN model', amyloid: '6E10 model', tau: 'AT8 model', cellpose: 'Cellpose-SAM', sam: 'Segment Anything',
  huggingface_home: 'Phikon', plaque_cnn_dir: 'Published plaque CNNs',
}
const PURPOSE: Record<string, string> = {
  neun: 'Decides which objects on NeuN slides are neurons.', amyloid: 'Decides which 6E10 deposits are plaques, and whether compact or diffuse.',
  tau: 'Decides which AT8 objects are tau+ neurons (tangles and pretangles).',
}
const DETECT_STEP: Record<string, string> = { NeuN: 'Detect NeuN neurons', '6E10': 'Detect plaques and tau', AT8: 'Detect plaques and tau' }
const DETAILS: Record<string, string> = {
  n_features: 'measurements per object', training_labels: 'training labels', training_n: 'training labels', training_positive_n: 'positive labels',
  probability_threshold: 'counts an object at probability ≥', context_threshold: 'counts an object at probability ≥', identity_threshold: 'counts a plaque at probability ≥',
  morphotype_threshold: 'compact if core stain ≥ × slide threshold', morphotype_minimum_diameter_um: 'compact / diffuse from diameter (µm)',
  nms_radius_um: 'merges detections closer than (µm)',
}
const f1 = (m: Metrics) => (m.precision + m.recall ? (2 * m.precision * m.recall) / (m.precision + m.recall) : 0)
const day = (seconds: number) => (Date.now() / 1000 - seconds < 86400 ? timeAgo(seconds)
  : new Date(seconds * 1000).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' }))
const activeJob = (job: Job | null | undefined) => !!job && (job.status === 'running' || job.status === 'queued')

export default function Models() {
  const models = useFetch<ModelInfo[]>('/api/models')
  const jobs = useFetch<Job[]>('/api/jobs', 4000)
  const latest = (workflow: string, stain?: string) => (jobs.data ?? []).find((j) => j.workflow === workflow && (!stain || j.options.stain === stain)) ?? null
  const trainingStain = (jobs.data ?? []).find((j) => j.workflow === 'train' && activeJob(j))?.options.stain as string | undefined
  const training = !!trainingStain
  const trained = useFetch<TrainedModel[]>('/api/models/trained', training ? 4000 : undefined)
  const labels = useFetch<LabelsOverview>('/api/labels/overview', 30000)
  const download = latest('download_models')
  const { reload: reloadTrained } = trained
  const wasTraining = useRef(false)
  useEffect(() => {  // a training just ended: show the new model at once
    if (wasTraining.current && !training) reloadTrained()
    wasTraining.current = training
  }, [training, reloadTrained])

  return (
    <>
      <PageHead title="Models" subtitle="The models that decide what counts as a neuron, a plaque or a tau+ neuron — how good each one is, and how to train a better one." />
      <HelpBox id="models">
        <ul>
          <li><b>Stain models</b> (NeuN, 6E10, AT8) learn what a real neuron, plaque or tau+ neuron looks like with <i>your</i> staining, from the objects
            people labelled on <Link to="/labels">Labels</Link>. To improve one: label more, press <b>Train a new model</b>, then switch to it if it scores better.</li>
          <li>Each score is measured fairly: the model is tested on whole TMAs it never saw while learning. “Finds 90% of real neurons” means it misses 1 in 10;
            “85% of what it counts is right” means 15 in 100 of its calls are wrong.</li>
          <li><b>Published models</b> (Cellpose-SAM, Segment Anything, Phikon) are the same for every study; download them once.</li>
        </ul>
        <p className="muted small">Switching models does not change results you already have: run the detection step again with “Start over” to redo them.</p>
      </HelpBox>
      <Loaded state={models} label="Loading models…">
        {(list) => <>
          <h2 className="stage-title">Stain models</h2>
          <div className="stain-models">
            {list.filter((m) => m.source === 'train').map((m) => (
              <StainModel key={m.key} info={m} versions={(trained.data ?? []).filter((t) => t.stain === m.stain)} versionsLoading={!trained.data && !trained.error}
                job={latest('train', m.stain ?? '')} otherTraining={training && trainingStain !== m.stain}
                labelled={labels.data?.stains.find((s) => s.stain === m.stain)?.objects_labelled ?? null}
                detection={labels.data?.stains.find((s) => s.stain === m.stain)?.detection}
                onChanged={() => { jobs.reload(); trained.reload(); models.reload() }} />
            ))}
          </div>
          <ErrorNote error={trained.error} />
          <ScoreOnLabels />
          <SimpleTraining />
          <h2 className="stage-title">Outlines</h2>
          <OutlineFixes />
          <SegmentationCheck />
          <h2 className="stage-title">Bad-area detector</h2>
          <BadAreaDetector />
          {list.some((m) => m.stain === 'AT8') && <><h2 className="stage-title">Thread detector</h2><ThreadDetector /></>}
          <h2 className="stage-title">Published models</h2>
          <PublishedModels models={list.filter((m) => m.source !== 'train')} job={download} onChanged={() => { jobs.reload(); models.reload() }} />
        </>}
      </Loaded>
    </>
  )
}

/* ---------- one stain ---------- */

function StainModel({ info, versions, versionsLoading, job, otherTraining, labelled, detection, onChanged }: {
  info: ModelInfo; versions: TrainedModel[]; versionsLoading: boolean; job: Job | null; otherTraining: boolean; labelled: number | null
  detection?: Detection; onChanged: () => void
}) {
  const { reload: reloadProject } = useProject()
  const stain = info.stain ?? ''
  const [asking, setAsking] = useState<'train' | TrainedModel | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [all, setAll] = useState(false)
  const inUse = versions.find((v) => v.active)
  const newest = versions[0]
  const fresh = newest && !newest.active && (!inUse || newest.created > inUse.created) ? newest : undefined
  const freshBetter = fresh && (!inUse || f1(fresh.new_model) > f1(inUse.new_model) + 0.005)
  const others = versions.filter((v) => !v.active && v !== fresh)
  const training = activeJob(job)
  const failed = job?.status === 'failed' && (!newest || (job.ended ?? 0) > newest.created) ? job : null

  const train = (sessions: string[] | null) => withError(async () => { await runStep('train', { stain, session: sessions ?? '' }); onChanged() }, setError)
  const use = (v: TrainedModel) => withError(async () => { await api.post('/api/models/activate', { path: v.path }); onChanged(); reloadProject() }, setError)

  return (
    <section className={`card stain-model${info.exists ? '' : ' missing'}`}>
      <div className="row"><h2 style={{ margin: 0 }}>{TITLES[info.key]}</h2><span className="spacer" />
        {info.exists ? <QualityBadge m={inUse?.new_model} /> : <span className="badge status-failed">No model yet</span>}</div>
      <p className="secondary small" style={{ marginTop: 4 }}>{PURPOSE[info.key]}</p>

      <div className="in-use">
        {!info.exists ? <p>There is no {stain} model yet. Label some {objectsOf(stain)} on <Link to="/labels">Labels</Link>, then train one.</p>
          : versionsLoading ? <p className="muted small">Loading its test results…</p>
          : inUse ? <>
            <div className="small muted">In use · trained {day(inUse.created)} from {inUse.new_model.n.toLocaleString()} labelled objects{trainedOn(inUse)}{addedNote(inUse)}</div>
            <div className="quality-line"><QualityLine m={inUse.new_model} stain={stain} d={inUse.person_added?.detection} /></div>
          </> : <p className="small secondary">The model in use was not trained in this project, so it has no test score here. Train one to compare.</p>}
        {info.exists && detection && detection.real > 0 && (
          <div className="small detection-line" title="Every real object people labelled counts, also the ones they added where detection circled nothing">
            Checked on all labels so far (it learned from some of them): <DetectionLine d={detection} stain={stain} /></div>)}
      </div>

      {fresh && !training && (
        <div className={`fresh-model ${freshBetter ? 'better' : ''}`}>
          <div className="row"><b>New model, trained {day(fresh.created)}</b><QualityBadge m={fresh.new_model} /></div>
          {fresh.sessions && <div className="small muted">Learned from {fresh.new_model.n.toLocaleString()} objects{trainedOn(fresh)}{addedNote(fresh)}</div>}
          <div className="small"><QualityLine m={fresh.new_model} stain={stain} d={fresh.person_added?.detection} /></div>
          <div className="small secondary">{inUse ? (freshBetter ? 'Scores better than the model in use.' : 'Does not score better than the model in use — keeping the current one is fine.') : 'Not in use yet.'}</div>
          <button className={`btn small${freshBetter ? ' primary' : ''}`} onClick={() => setAsking(fresh)}>Use the new model</button>
        </div>
      )}

      {training && job && (
        <div className="training-now">
          <div className="row"><span className="pulse" /><b>{job.status === 'queued' ? 'Waiting to start training…' : 'Training a new model…'}</b></div>
          <Progress done={job.progress.done} total={job.progress.total || 3} />
          <span className="muted small">Usually under a minute. You can leave this page; the result appears here.</span>
        </div>
      )}
      {failed && <ErrorNote error={`Training stopped: ${failed.progress.error || failed.progress.last_line || 'see the log on the Run page'}`} />}
      <ErrorNote error={error} />

      <div className="row stain-model-actions">
        <button className={`btn small${info.exists ? '' : ' primary'}`} disabled={training || otherTraining} onClick={() => setAsking('train')}
          title={otherTraining ? 'Another model is training; wait for it to finish' : 'Learns from the labelling sessions you choose'}>Train a new model…</button>
        {labelled != null && <span className="muted small">{labelled.toLocaleString()} {stain} objects labelled so far</span>}
      </div>

      {others.length > 0 && (
        <details className="versions">
          <summary className="small">Earlier versions ({others.length})</summary>
          <table className="compact versions-table"><tbody>
            {(all ? others : others.slice(0, 5)).map((v) => (
              <tr key={v.path}>
                <td className="small">{day(v.created)}<div className="muted">{v.new_model.n.toLocaleString()} labels{trainedOn(v)}</div></td>
                <td><QualityBadge m={v.new_model} /></td>
                <td className="small wrap-cell"><QualityLine m={v.new_model} stain={stain} d={v.person_added?.detection} /></td>
                <td className="num"><button className="btn small" onClick={() => setAsking(v)}>Use</button></td>
              </tr>
            ))}
          </tbody></table>
          {others.length > 5 && <button className="link-btn small" onClick={() => setAll(!all)}>{all ? 'Show fewer' : `Show all ${others.length}`}</button>}
        </details>
      )}

      <Advanced hint="file, settings, all test figures">
        <PathLine path={info.path} exists={info.exists} wrap />
        {info.bundle && (
          <table className="compact"><tbody>
            {Object.entries(info.bundle).filter(([k]) => k in DETAILS).map(([k, v]) => (
              <tr key={k}><td className="secondary">{DETAILS[k]}</td><td className="num">{typeof v === 'number' ? +v.toPrecision(3) : String(v)}</td></tr>
            ))}
          </tbody></table>
        )}
        {versions.length > 0 && <VersionTable versions={versions} />}
      </Advanced>

      {asking === 'train' && <TrainDialog stain={stain} onClose={() => setAsking(null)} onStart={train} />}
      {asking && asking !== 'train' && (
        <ConfirmDialog title={`Use the ${stain} model trained ${day(asking.created)}?`} confirm="Switch model" onClose={() => setAsking(null)} onConfirm={() => use(asking)}>
          {quality(asking.new_model) && <p>Its test score: <b>{quality(asking.new_model)!.word}</b> — <QualityLine m={asking.new_model} stain={stain} d={asking.person_added?.detection} />.
            {inUse && <> The model in use: <QualityLine m={inUse.new_model} stain={stain} d={inUse.person_added?.detection} />.</>}</p>}
          <p>New detections use it from now on. Results you already have stay as they are; to redo them with this model, run “{DETECT_STEP[stain]}” again
            on the <Link to="/workflow">Run</Link> page with “Start over”. You can switch back at any time.</p>
        </ConfirmDialog>
      )}
    </section>
  )
}

/** ", including 12 that people added": the objects detection never proposed that a model learned from (measured where people added them). */
function addedNote(model: TrainedModel): string {
  const added = model.person_added
  if (!added || !(added.objects || added.could_not_be_measured)) return ''
  const unmeasured = added.could_not_be_measured ? ` (${added.could_not_be_measured.toLocaleString()} more could not be measured)` : ''
  return `, including ${added.objects.toLocaleString()} that people added where detection proposed nothing${unmeasured}`
}

/** " · from Round 1 and AI reference labels": the sessions a model learned from (models trained before sessions were recorded say nothing). */
function trainedOn(model: TrainedModel): string {
  const titles = (model.sessions ?? []).map((s) => s.title)
  return titles.length ? ` · from ${titles.length > 1 ? `${titles.slice(0, -1).join(', ')} and ${titles[titles.length - 1]}` : titles[0]}` : ''
}

/** Which sessions a new model learns from: every session with labels of this stain is listed with its label counts per class;
 *  all are chosen at first. */
function TrainDialog({ stain, onClose, onStart }: { stain: string; onClose: () => void; onStart: (sessions: string[] | null) => Promise<unknown> }) {
  const sets = useFetch<LabelSet[]>(`/api/labels/sets?stain=${encodeURIComponent(stain)}`)
  const [left, setLeft] = useState<string[]>([])  // sessions left out
  const [busy, setBusy] = useState(false)
  const all = sets.data ?? []
  const chosen = all.filter((s) => !left.includes(s.name))
  const objects = chosen.reduce((n, s) => n + s.objects, 0)
  const start = async () => {
    setBusy(true)
    try { await onStart(left.length ? chosen.map((s) => s.name) : null) } finally { setBusy(false) }
    onClose()
  }
  return (
    <Modal title={`Train a new ${stain} model`} onClose={onClose} wide>
      <div className="train-dialog">
        <p style={{ margin: 0 }}>Choose the labels it learns from. It is tested on whole TMAs it did not see, usually takes under a minute and uses this computer’s
          processors. The model in use does not change: you choose afterwards whether to switch.</p>
        <Loaded state={sets} label="Loading the sessions…">
          {(list) => list.length ? (
            <div className="train-sets">{list.map((s) => (
              <label key={s.name} className={`choice-card${left.includes(s.name) ? '' : ' chosen'}`}>
                <input type="checkbox" checked={!left.includes(s.name)}
                  onChange={() => setLeft(left.includes(s.name) ? left.filter((n) => n !== s.name) : [...left, s.name])} />
                <span><span className="row" style={{ gap: 6 }}><b>{s.title}</b>{s.reference && <span className="badge">read only</span>}
                  {s.simple && <span className="badge" title="Simple classes: they train through the mapping under “Simple labels in training”">simple</span>}</span>
                  <span className="small secondary">{s.objects.toLocaleString()} objects on {s.fields.toLocaleString()} fields · labelled by {s.people.map((p) => p.replace(/^import: /, '')).join(', ')}</span>
                  <span className="small muted">{Object.entries(s.classes).map(([c, n]) => `${c} ${n.toLocaleString()}`).join(' · ')}</span></span>
              </label>
            ))}</div>
          ) : <Empty>No {stain} objects are labelled yet. Start a labelling session on <Link to="/labels">Labels</Link> first.</Empty>}
        </Loaded>
        <div className="row"><span className="spacer" />
          <span className="small secondary">{chosen.length} of {all.length} session{all.length === 1 ? '' : 's'} · {objects.toLocaleString()} objects</span>
          <button className="btn" onClick={onClose}>Cancel</button>
          <button className="btn primary" disabled={busy || !chosen.length} onClick={start}>{busy ? 'Starting…' : 'Start training'}</button>
        </div>
      </div>
    </Modal>
  )
}

function VersionTable({ versions }: { versions: TrainedModel[] }) {
  return (
    <>
      <p className="muted small" style={{ margin: 0 }}>
        Each model is tested with <Term>leave-one-TMA-out</Term> checks on the labels of its day. <Term>AUC</Term>: overall ranking quality (1 = perfect);
        <Term> precision</Term>: of the objects counted, how many are right; <Term>recall</Term>: of the real objects, how many are found.
        “Model in use then” is the model that was in use, scored on the same labels (it may have learned from some of them).
      </p>
      <div className="table-scroll">
        <table className="compact">
          <thead><tr><th>Trained</th><th className="num">Labels</th><th>Tested by</th><th>This model</th><th>Model in use then</th><th className="num">Threshold</th><th /></tr></thead>
          <tbody>{versions.map((t) => (
            <tr key={t.path}>
              <td className="small">{day(t.created)}<div className="muted ellipsis" title={Object.keys(t.sources ?? {}).join(', ')}>
                {t.sessions ? t.sessions.map((s) => s.title).join(', ') : t.sources ? `${Object.keys(t.sources).length} label sources` : (t.sets ?? []).join(', ')}</div>
                {mappingText(t) && <div className="muted small simple-used">{mappingText(t)}</div>}</td>
              <td className="num">{t.new_model.n.toLocaleString()}<div className="muted small">{t.new_model.positives.toLocaleString()} positive</div>
                {t.person_added && t.person_added.objects > 0 && <div className="muted small">{t.person_added.objects.toLocaleString()} added by people</div>}</td>
              <td className="small">{t.validation}</td>
              <td><MetricLine m={t.new_model} /></td>
              <td>{t.current_model ? <MetricLine m={t.current_model} /> : <span className="muted small">none</span>}</td>
              <td className="num">{fmt(t.threshold, 2)}</td>
              <td>{t.active && <span className="badge status-done">in use</span>}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
    </>
  )
}

function MetricLine({ m }: { m: Metrics }) {
  return <span className="small">AUC <b>{fmt(m.auc, 2)}</b> · precision <b>{fmt(m.precision, 2)}</b> · recall <b>{fmt(m.recall, 2)}</b></span>
}

/* ---------- published models ---------- */

function PublishedModels({ models, job, onChanged }: { models: ModelInfo[]; job: Job | null; onChanged: () => void }) {
  const [error, setError] = useState<string | null>(null)
  const downloading = activeJob(job) ? (job!.options.which as string[] | undefined) ?? [] : []
  const download = (key: string) => withError(async () => { await runStep('download_models', { which: [key] }); onChanged() }, setError)
  const wasDownloading = useRef(false)
  useEffect(() => {  // a download just ended: show the model as ready
    if (wasDownloading.current && !downloading.length) onChanged()
    wasDownloading.current = downloading.length > 0
  }, [downloading.length, onChanged])
  return (
    <section className="card">
      <div className="table-scroll">
        <table className="published-table">
          <thead><tr><th>Model</th><th>What it does</th><th>Status</th><th /></tr></thead>
          <tbody>{models.map((m) => (
            <tr key={m.key}>
              <td><b>{TITLES[m.key] ?? m.key}</b></td>
              <td className="small secondary wrap-cell">{m.role}</td>
              <td>{m.exists ? <span className="badge status-done">Ready</span> : <span className={`badge ${m.source === 'manual' ? '' : 'status-failed'}`}>{m.source === 'manual' ? 'Optional' : 'Missing'}</span>}</td>
              <td className="num">
                {!m.exists && m.source === 'download' && (downloading.includes(m.key)
                  ? <span className="small secondary"><span className="pulse" />{job?.status === 'queued' ? 'Waiting…' : 'Downloading…'}</span>
                  : <button className="btn small" disabled={downloading.length > 0} onClick={() => download(m.key)}>Download</button>)}
              </td>
            </tr>
          ))}</tbody>
        </table>
      </div>
      <ErrorNote error={error} />
      {job?.status === 'failed' && models.some((m) => !m.exists && m.source === 'download') && <ErrorNote error={`The download stopped: ${job.progress.error || job.progress.last_line || 'see the log on the Run page'}`} />}
      <Advanced hint="where the files are">
        {models.map((m) => <div key={m.key}><PathLine path={m.path} exists={m.exists} label={TITLES[m.key] ?? m.key} wrap />
          {m.size_mb != null && <span className="muted small"> {m.size_mb.toLocaleString()} MB · {m.modified}</span>}</div>)}
        <p className="muted small" style={{ margin: 0 }}>Published plaque CNNs are optional: copy the published weights (Wong et al. 2022; Tang et al. 2019) into their folder.</p>
      </Advanced>
    </section>
  )
}

/* ---------- the bad-area detector ---------- */

/** The bad-area detector (today detection's rules: the automatic excluded areas the analysis uses) against the areas
 *  people marked in "Mark bad areas": per stain, type and TMA or donor, the area people added and cleared and the
 *  pixel F1. A trained bad-area model is scored here too once there is one. */
function BadAreaDetector() {
  const state = useFetch<BadAreas>('/api/models/bad-areas')
  const [by, setBy] = useState<'tma' | 'donor'>('tma')
  return (
    <section className="card">
      <div className="row"><h2 style={{ margin: 0 }}>Bad-area detector</h2><span className="spacer" />
        <span className="badge">rules</span></div>
      <p className="small secondary">Finds folds, artifacts, empty glass and blurred parts, which the analysis leaves out. Here its areas are compared with the areas people
        marked in <b>Mark bad areas</b> (<Link to="/labels">Labels</Link>): how much people added and cleared, and how well the two agree pixel by pixel (F1; 100% = the same).</p>
      <Loaded state={state} label="Comparing detection’s areas with people’s…">
        {(d) => {
          const rows = by === 'tma' ? d.by_tma : d.by_donor
          return <>
            {d.model == null && <p className="small muted">Today the detector is a set of rules. A trained bad-area model comes later: it will learn from these areas and be scored here the same way.</p>}
            {!d.images ? <Empty><p style={{ margin: 0 }}>No core image has been checked in “Mark bad areas” yet. Start a session on <Link to="/labels">Labels</Link>.</p></Empty> : <>
              <div className="row small">
                <span className="secondary">{d.images.toLocaleString()} stain images checked by people.</span><span className="spacer" />
                <div className="segmented">
                  <button type="button" className={`btn small${by === 'tma' ? ' chosen' : ''}`} onClick={() => setBy('tma')}>Per TMA</button>
                  <button type="button" className={`btn small${by === 'donor' ? ' chosen' : ''}`} onClick={() => setBy('donor')}>Per donor</button>
                </div>
              </div>
              <div className="table-wrap" style={{ marginTop: 8 }}><table className="compact small areas-table">
                <thead><tr><th>Stain</th><th>{by === 'tma' ? 'TMA' : 'Donor'}</th><th>Type</th><th>Images</th>
                  <th title="Area people marked bad that detection did not">Added by people (mm²)</th><th title="Area detection left out that people marked fine">Cleared by people (mm²)</th><th>F1</th></tr></thead>
                <tbody>{rows.map((r) => (
                  <tr key={`${r.stain}-${r.tma ?? r.donor_id}-${r.type}`}><td>{r.stain}</td><td>{(by === 'tma' ? r.tma : r.donor_id) ?? '—'}</td><td>{r.type_name}</td>
                    <td>{r.images.toLocaleString()}</td><td>{fmt(r.added_mm2)}</td><td>{fmt(r.cleared_mm2)}</td>
                    <td>{r.f1 == null ? '—' : `${Math.round(100 * r.f1)}%`}</td></tr>
                ))}{!rows.length && <tr><td colSpan={7} className="muted">Nothing to compare yet: detection has not marked bad areas on the checked images (the v2 detection’s proposals make them).</td></tr>}</tbody>
              </table></div>
            </>}
          </>
        }}
      </Loaded>
    </section>
  )
}

/* ---------- the thread detector ---------- */

interface ThreadGroup { tma?: string; slide?: string; fields: number; labels: number; median: number; q25: number; q75: number; length_ratio: number | null; factors: number[] }
interface ThreadSummary {
  fields: number; people: number; by_tma: ThreadGroup[]; by_slide: ThreadGroup[]; length_ratio?: number | null
  suggested: number | null; per_tma: { tma: string; suggested: number; fields: number }[]
}

const LOW = 0.5
const HIGH = 2
const factorText = (f: number) => (Math.abs(f - 1) < 1e-6 ? 'detection’s' : `${f.toFixed(2)}×`)
const ratioText = (r: number | null | undefined) => (r == null ? '—' : `${Math.round(100 * r)}%`)

/** People's factors on the slider's log scale (0.5–2), detection's (1) marked. */
function FactorStrip({ factors }: { factors: number[] }) {
  const x = (f: number) => 4 + (152 * (Math.log(f) - Math.log(LOW))) / (Math.log(HIGH) - Math.log(LOW))
  return (
    <svg width="160" height="16" viewBox="0 0 160 16" aria-label={`people’s thread thresholds: ${factors.map(factorText).join(', ')}`}>
      <line x1={4} x2={156} y1={8} y2={8} stroke="var(--border)" />
      <line x1={x(1)} x2={x(1)} y1={1} y2={15} stroke="var(--warning)" strokeWidth={2} />
      {factors.map((f, i) => <circle key={i} cx={x(f)} cy={8} r={3} fill="var(--accent)" fillOpacity={0.55} />)}
    </svg>
  )
}

/** The AT8 thread detector against people: the thread threshold people set in threads mode (H in "Fix the field"),
 *  relative to detection's, per TMA or slide; the thread length it gives against the current; a suggested setting.
 *  It only tunes: the analysis keeps detection's threshold until a new one is validated and the pipeline run again. */
function ThreadDetector() {
  const state = useFetch<ThreadSummary>('/api/models/thread-detector')
  const [by, setBy] = useState<'tma' | 'slide'>('tma')
  return (
    <section className="card">
      <div className="row"><h2 style={{ margin: 0 }}>Thread detector</h2><span className="spacer" /><span className="badge">AT8 · tuning only</span></div>
      <p className="small secondary">Finds the neuropil thread network that the analysis measures by length. In AT8 <b>Fix the field</b> people press <b>H</b> and move one
        slider until the drawn threads match the threads they see; here are their thresholds against detection’s own (1 = detection’s, marked).</p>
      <Loaded state={state} label="Reading people’s thread thresholds…">
        {(d) => !d.fields ? <Empty><p style={{ margin: 0 }}>No thread threshold set yet. In an AT8 “Fix the field” session, press <b>H</b> on a field, move the slider and press Enter.</p></Empty> : <>
          <p className="small"><b>{d.fields.toLocaleString()}</b> field{d.fields === 1 ? '' : 's'} set by <b>{d.people}</b> {d.people === 1 ? 'person' : 'people'}.
            {' '}Suggested setting for AT8: <b>{d.suggested == null ? '—' : factorText(d.suggested)}</b>
            {d.suggested != null && Math.abs(d.suggested - 1) > 1e-6 ? ' detection’s threshold' : ''} · thread length {ratioText(d.length_ratio)} of the current.
            {d.per_tma.length > 0 && <> Per TMA, where they differ consistently: {d.per_tma.map((t) => `TMA ${t.tma} ${factorText(t.suggested)} (${t.fields} fields)`).join(' · ')}.</>}</p>
          <p className="small muted">Nothing changes in the analysis: a new setting is validated first, then the pipeline runs again with it.</p>
          <div className="row small"><span className="spacer" />
            <div className="segmented">
              <button type="button" className={`btn small${by === 'tma' ? ' chosen' : ''}`} onClick={() => setBy('tma')}>Per TMA</button>
              <button type="button" className={`btn small${by === 'slide' ? ' chosen' : ''}`} onClick={() => setBy('slide')}>Per slide</button>
            </div>
          </div>
          <div className="table-wrap" style={{ marginTop: 8 }}><table className="compact small">
            <thead><tr><th>{by === 'tma' ? 'TMA' : 'Slide'}</th><th>Fields</th><th title="The middle of people’s thresholds, and the middle half">Median (middle half)</th>
              <th title="0.5× to 2× of detection’s; the mark is detection’s">People’s thresholds</th><th title="The thread length people’s thresholds give, against detection’s on the same fields">Thread length vs current</th></tr></thead>
            <tbody>{(by === 'tma' ? d.by_tma : d.by_slide).map((g) => (
              <tr key={String(g.tma ?? g.slide)}><td>{(by === 'tma' ? g.tma : g.slide) || '—'}</td><td>{g.fields.toLocaleString()}</td>
                <td>{factorText(g.median)} ({g.q25.toFixed(2)}–{g.q75.toFixed(2)})</td><td><FactorStrip factors={g.factors} /></td><td>{ratioText(g.length_ratio)}</td></tr>
            ))}</tbody>
          </table></div>
        </>}
      </Loaded>
    </section>
  )
}
