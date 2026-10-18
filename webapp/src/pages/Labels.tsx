import { useState } from 'react'
import { Link } from 'react-router-dom'
import type { AiView, AreasOverview, ClassCount, LabelsOverview, Labeller, ReferenceLabels, RoundProgress, SessionInfo, StainOverview, Vocabulary } from '../api'
import { api, apiUrl, appPath, enc } from '../api'
import { Empty, ErrorNote, HelpBox, Loaded, Modal, PageHead, Progress, fmt, timeAgo, withError } from '../components'
import { useFetch } from '../hooks'
import { AI_VIEW_TEXT, ConfirmDialog, CopyButton, DetectionLine, KIND_TEXT, TOOL_TEXT, objectsOf, percent } from '../labelling'
import { SessionAccessCard } from '../labels/access'
import { SessionLabelsView } from '../labels/examine'
import { FIELD_TEXT, FieldChoice, SessionFields } from '../labels/fields'
import { AREAS_TITLE } from '../labels/markareas'
import { NewRoundSession } from '../labels/newround'
import { ROUND_TITLE } from '../labels/rounds'
import { byKey } from '../labels/classKeys'
import { Advanced } from '../setup/parts'

interface Defaults {
  classes: Record<string, string[]>; vocabularies: Record<string, Vocabulary>; tags: string[]; tools: string[]; ai_views: AiView[]; rules: Record<string, string>
}
interface Share { links: string[]; network: boolean }

export default function Labels() {
  const overview = useFetch<LabelsOverview>('/api/labels/overview', 10000)
  const share = useFetch<Share>('/api/labels/share')
  const [creating, setCreating] = useState(false)
  const [older, setOlder] = useState(false)  // one of the kinds of session made before the annotation rounds
  const [notice, setNotice] = useState<string | null>(null)
  const linkFor = (name = '') => {
    const base = share.data?.network && share.data.links[0] ? share.data.links[0] : `${location.origin}${appPath('/annotate')}`
    return name ? `${base}/${name}` : base
  }

  return (
    <>
      <PageHead title="Labels" subtitle="Ask colleagues to fix detection on a few fields and mark bad areas on some cores; their labels train and test the models, never the analysis.">
        <a className="btn" href={apiUrl('/api/labels/export.csv')}>Download all labels</a>
        <a className="btn" href={appPath('/annotate/tutorial?preview=1')} target="_blank" rel="noreferrer"
          title="See exactly what annotators see first: the 4-minute practice on a field of this project (nothing is saved)">Try the tutorial</a>
        <a className="btn" href={appPath('/annotate/tutorial?kind=areas&preview=1')} target="_blank" rel="noreferrer"
          title="The short practice of “Mark bad areas” on a core of this project (nothing is saved)">Bad-areas tutorial</a>
        <button className="btn primary" onClick={() => setCreating(true)}>New labelling session</button>
      </PageHead>
      <HelpBox id="labels">
        <p style={{ marginTop: 0 }}>{FIELD_TEXT}</p>
        <ol className="plain-steps">
          <li><b>Start a session</b>: pick a stain and what people do. <b>Fix the field</b> (start here): whole fields with every object detection
            considered; people point at an object and press a key to change its call or outline, and click to add what it missed. <b>Mark bad
            areas</b>: core by core with a brush, for folds, artifacts, glass and blur (a sample of cores is enough).</li>
          <li><b>Try the tutorial</b> to see what annotators see first (nothing is saved).</li>
          <li><b>Send the link and the set’s key</b> (a session → More → Who can label). People sign in or make an account, add the set with the key and label
            in their browser. Each person gets their own field, and their work is kept when they press Save.</li>
          <li><b>Watch progress</b> here, field by field, then <Link to="/models">train a new model</Link> from the sessions you choose and compare it with the one in use.</li>
        </ol>
      </HelpBox>
      {notice && <div className="found found-good labels-notice">{notice}<button className="link-btn" onClick={() => setNotice(null)}>Dismiss</button></div>}
      <Loaded state={overview} label="Loading labelling sessions…">
        {(data) => <>
          <Sessions sessions={data.sessions} reference={data.reference} linkFor={linkFor} onChanged={overview.reload} onNew={() => setCreating(true)} />
          {data.sessions.some((s) => s.kind === 'areas') && <AreasSoFar />}
          <ShareCard share={share.data} error={share.error} link={linkFor()} anyOpen={data.sessions.some((s) => s.open)} />
          <LabelledSoFar stains={data.stains} onChanged={overview.reload} />
          {creating && (older
            ? <NewSession stains={data.stains.map((s) => s.stain)} onClose={() => { setCreating(false); setOlder(false) }}
              onCreated={(s) => { overview.reload(); setNotice(s.open
                ? `“${s.title}” is open. Fields are prepared in the background (10–30 s each) and appear to people as they become ready. Copy its link below and send it.`
                : `“${s.title}” is ready but closed. Open it when you want people to start.`) }} />
            : <NewRoundSession stains={data.stains.map((s) => s.stain)} onClose={() => setCreating(false)} onOlder={() => setOlder(true)}
              onCreated={(s) => {
                overview.reload()
                const d = s as unknown as { fields?: string[] | number; fields_asked?: number; fields_remaining?: number; field_seed?: number
                  fields_from_checked_areas?: number; cores?: string[]; second?: string[] }
                const n = Array.isArray(d.fields) ? d.fields.length : d.fields ?? 0
                const fromAreas = d.fields_from_checked_areas ? ` ${n.toLocaleString()} fields: ${d.fields_from_checked_areas.toLocaleString()} from cores with checked areas.` : ''
                const drawn = s.kind === 'areas' ? ` It has ${(d.cores?.length ?? 0).toLocaleString()} cores${d.second?.length ? `, ${d.second.length.toLocaleString()} of them also for a second person` : ''}.`
                  : d.fields_asked == null || d.fields_remaining == null ? fromAreas
                  : (d.fields_asked < 0 ? ` It has all ${n.toLocaleString()} remaining fields (in a random order, seed ${d.field_seed}).`
                    : d.fields_asked > d.fields_remaining ? ` Only ${d.fields_remaining.toLocaleString()} fields remained, so it has all of them (seed ${d.field_seed}).`
                      : ` ${n.toLocaleString()} fields were drawn at random from the ${d.fields_remaining.toLocaleString()} remaining (seed ${d.field_seed}).`) + fromAreas
                setNotice((s.open ? `“${s.title}” is open: copy its link below and send it. Pictures and outlines are made ahead in the background.`
                  : `“${s.title}” is ready but closed. Open it when you want people to start.`) + drawn)
              }} />)}
        </>}
      </Loaded>
    </>
  )
}

/* ---------- sessions ---------- */

/** The annotation rounds ("Fix the field", "Mark bad areas"; old "Check detection's calls" ones as history), shown before the older kinds. */
const isRound = (s: SessionInfo) => !!s.round || s.kind === 'areas'

function Sessions({ sessions, reference, linkFor, onChanged, onNew }: {
  sessions: SessionInfo[]; reference: ReferenceLabels | null; linkFor: (name: string) => string; onChanged: () => void; onNew: () => void
}) {
  const ordered = [...sessions].sort((a, b) => b.created - a.created)  // newest first; stays put when one is opened or closed
  return (
    <section className="card labels-section">
      <h2>Labelling sessions</h2>
      {ordered.length || reference ? <div className="session-list">
        {ordered.filter((s) => isRound(s)).map((s) => <SessionRow key={s.name} s={s} link={linkFor(s.name)} onChanged={onChanged} />)}
        {ordered.some((s) => !isRound(s)) && (ordered.some((s) => isRound(s))
          ? <details className="advanced older-sessions">
            <summary><span>Older kinds of session</span><span className="muted small"> · {ordered.filter((s) => !isRound(s)).length} (older segmentation checks, whole-field reviews, quick checks)
              {ordered.some((s) => !isRound(s) && s.open) ? ` · ${ordered.filter((s) => !isRound(s) && s.open).length} open` : ''}</span></summary>
            <div className="advanced-body session-list">{ordered.filter((s) => !isRound(s)).map((s) => <SessionRow key={s.name} s={s} link={linkFor(s.name)} onChanged={onChanged} />)}</div>
          </details>
          : ordered.map((s) => <SessionRow key={s.name} s={s} link={linkFor(s.name)} onChanged={onChanged} />))}
        {reference && <ReferenceRow r={reference} />}
      </div>
        : <Empty>
          <p style={{ marginTop: 0 }}>No labelling sessions yet. A session asks people to check a set of fields (or objects) for one stain.</p>
          <button className="btn" onClick={onNew}>Start the first session</button>
        </Empty>}
      {!ordered.length && reference && <div className="row" style={{ marginTop: 10 }}><button className="btn" onClick={onNew}>Start the first session</button></div>}
    </section>
  )
}

/** "Mark bad areas" across its sessions: stain images checked per stain, cores with every stain checked, and how well
 *  two people agree on the cores checked twice, with the disagreements to settle in the Cores view. */
function AreasSoFar() {
  const state = useFetch<AreasOverview>('/api/labels/areas', 30000)
  return (
    <section className="card labels-section">
      <h2>{AREAS_TITLE}: checked so far</h2>
      <Loaded state={state} label="Loading the bad areas checked…">
        {(a) => <>
          <div className="session-progress">
            <Progress done={a.cores_checked} total={a.cores} />
            <span className="small secondary">{a.cores_checked.toLocaleString()} of {a.cores.toLocaleString()} cores have every stain checked. A sample is enough: the areas
              people mark teach and test the bad-area detection (see <Link to="/models">Models</Link>); the analysis uses detection’s own areas.</span>
          </div>
          <div className="table-wrap" style={{ marginTop: 8 }}><table className="compact small areas-table">
            <thead><tr><th>Stain</th><th>Images checked</th><th>Checked by two people</th></tr></thead>
            <tbody>{a.stains.map((s) => (
              <tr key={s.stain}><td>{s.stain}</td><td>{s.checked.toLocaleString()} of {s.images.toLocaleString()}</td><td>{s.double_checked.toLocaleString()}</td></tr>
            ))}</tbody>
          </table></div>
          <p className="small secondary" style={{ marginBottom: 4 }}>{a.agreement.pairs
            ? <>Agreement: on {a.agreement.agree.toLocaleString()} of {a.agreement.pairs.toLocaleString()} images checked by two people they marked much the same
              {a.agreement.mean_iou != null ? ` (mean overlap ${percent(a.agreement.mean_iou)})` : ''}.</>
            : 'Agreement: no image has been checked by two people yet (about 1 core in 10 goes to a second person).'}</p>
          {a.agreement.disagreements.length > 0 && <>
            <p className="small muted" style={{ margin: '0 0 4px' }}>Where they disagree — open the core to settle it:</p>
            <ul className="small" style={{ margin: 0, paddingLeft: 18 }}>{a.agreement.disagreements.slice(0, 30).map((d) => (
              <li key={`${d.core_id}-${d.stain}`}><Link to={`/cores/${enc(d.core_id)}?stain=${enc(d.stain)}`}>{d.core_id} · {d.stain}</Link>
                <span className="muted"> · {d.people.join(' and ')} · overlap {percent(d.iou)}</span></li>
            ))}</ul>
            {a.agreement.disagreements.length > 30 && <p className="small muted">and {(a.agreement.disagreements.length - 30).toLocaleString()} more</p>}
          </>}
        </>}
      </Loaded>
    </section>
  )
}

/** The labels made before labelling sessions existed, shown as one read-only session. */
function ReferenceRow({ r }: { r: ReferenceLabels }) {
  const [examining, setExamining] = useState(false)
  return (
    <div className="session-row reference-row">
      <div className="session-main">
        <div className="session-title"><b>{r.title}</b> <span className="badge">read only</span></div>
        <div className="muted small">AI reference labels, brought in from earlier label sets ({r.sets.map((s) => s.replace(/^import: /, '')).join(', ')})</div>
        <div className="small secondary">{r.labels.toLocaleString()} objects labelled on {r.fields.toLocaleString()} fields ·{' '}
          {Object.entries(r.stains).map(([s, n]) => `${s} ${n.toLocaleString()}`).join(' · ')}</div>
      </div>
      <div className="session-actions"><button className="btn small" onClick={() => setExamining(true)}>Look at the labels</button></div>
      {examining && <SessionLabelsView name={r.name} title={r.title} onClose={() => setExamining(false)} />}
    </div>
  )
}

function sessionState(s: SessionInfo): { text: string; className: string } {
  if (s.round_progress && s.round_progress.total > 0 && s.round_progress.decided >= s.round_progress.total) return { text: 'Done', className: 'status-done' }
  if (!s.round && s.kind !== 'queue' && s.fields && (s.fields_covered ?? 0) >= s.fields) return { text: 'Done', className: 'status-done' }
  return s.open ? { text: 'Open', className: 'status-running' } : { text: 'Closed', className: '' }
}

function SessionRow({ s, link, onChanged }: { s: SessionInfo; link: string; onChanged: () => void }) {
  const [open, setOpen] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const state = sessionState(s)
  const toggle = async () => {
    setBusy(true)
    await withError(() => api.post(`/api/labels/sessions/${s.name}/open`, { open: !s.open }), setError)
    setBusy(false)
    onChanged()
  }
  return (
    <div className={`session-row${s.open ? ' is-open' : ''}`}>
      <div className="session-main">
        <div className="session-title">
          <b>{s.title}</b> {s.read_only ? <span className="badge" title="Imported from a dataset package; the labels stay as they were">read only</span> : <span className={`badge ${state.className}`}>{state.text}</span>}
          {s.labelling_now > 0 && <span className="badge status-running"><span className="pulse" />{s.labelling_now} labelling now</span>}
        </div>
        <div className="muted small">{s.kind === 'areas' ? `Every stain · ${AREAS_TITLE}${s.order === 'random' ? ' · random order' : s.order ? ' · likely problems first' : ''}`
          : `${s.stain} · ${s.round ? `Round ${s.round} · ${ROUND_TITLE[s.round]}` : KIND_TEXT[s.kind]}`}
          {(s.labels_needed ?? s.per_field) > 1 ? ` · ${s.labels_needed ?? s.per_field} people per ${s.kind === 'queue' ? 'object' : s.kind === 'areas' ? 'core' : 'field'}` : ''}</div>
        <SessionProgress s={s} />
        {s.kind === 'areas' ? <CorePeople progress={s.round_progress} labellers={s.labellers} />
          : <People people={s.labellers} fields={s.kind !== 'queue' ? s.fields ?? 0 : null} />}
      </div>
      <div className="session-actions">
        {s.open && !s.read_only && <CopyButton text={link} />}
        {!s.read_only && <button className="btn small" disabled={busy} onClick={toggle} title={s.open ? 'People can no longer open it; nothing is lost' : 'People with the link can start labelling'}>
          {busy ? '…' : s.open ? 'Close' : 'Open to labellers'}</button>}
        <button className="btn small" onClick={() => setOpen(!open)} aria-expanded={open}>{open ? 'Less' : 'More'}</button>
      </div>
      <ErrorNote error={error} />
      {open && <SessionMore s={s} onChanged={onChanged} onError={setError} />}
    </div>
  )
}

function SessionProgress({ s }: { s: SessionInfo }) {
  if (s.kind === 'areas' && s.round_progress) {
    const p = s.round_progress
    return (
      <div className="session-progress">
        <Progress done={p.decided} total={p.total} />
        <span className="small secondary">{p.decided.toLocaleString()} of {p.total.toLocaleString()} cores checked (by anyone)
          {p.images_checked ? ` · ${p.images_checked.toLocaleString()} stain images` : ''}{p.open_now ? ` · ${p.open_now.toLocaleString()} in progress` : ''}
          {p.second_checks ? <span className="muted"> · {p.second_checks.toLocaleString()} cores also go to a second person</span> : null}</span>
      </div>
    )
  }
  if (s.round_progress) {
    const p = s.round_progress
    return (
      <div className="session-progress">
        <Progress done={p.decided} total={p.total} />
        <span className="small secondary">{p.decided.toLocaleString()} of {p.total.toLocaleString()} fields done (by anyone){p.open_now ? ` · ${p.open_now.toLocaleString()} in progress` : ''}
          {s.round === 2 && s.fields_from_checked_areas != null && <span className="muted"> · {p.total.toLocaleString()} fields: {s.fields_from_checked_areas.toLocaleString()} from cores with checked areas</span>}
          {s.pictures && s.pictures.ready < s.pictures.total && <span className="muted"> · pictures ready: {s.pictures.ready.toLocaleString()} of {s.pictures.total.toLocaleString()}</span>}</span>
      </div>
    )
  }
  if (s.kind === 'queue') {
    const pictures = s.pictures
    return <div className="small secondary session-progress">{s.labels.toLocaleString()} objects labelled so far · open-ended
      {pictures && pictures.ready < pictures.total && <span className="muted"> · pictures ready: {pictures.ready.toLocaleString()} of {pictures.total.toLocaleString()}</span>}</div>
  }
  const total = s.fields ?? 0
  const preparing = total - (s.fields_prepared ?? total)
  return (
    <div className="session-progress">
      <Progress done={s.fields_covered ?? 0} total={total} />
      <span className="small secondary">{(s.fields_covered ?? 0).toLocaleString()} of {total.toLocaleString()} fields done
        {preparing > 0 && <span className="muted"> · {preparing} still being prepared</span>}</span>
    </div>
  )
}

/** "Mark bad areas" goes core by core: each person with the cores they checked. */
function CorePeople({ progress, labellers }: { progress?: RoundProgress; labellers: Labeller[] }) {
  const people = progress?.people ?? []
  if (!people.length) return <div className="muted small">Nobody has checked a core yet.</div>
  const last = new Map(labellers.map((p) => [p.person, p.last_active]))
  return (
    <div className="people-chips">
      {people.map((p) => (
        <span key={p.person} className="person-chip"><b>{p.person}</b> {p.decided.toLocaleString()} of {progress!.total.toLocaleString()} cores
          {last.get(p.person) ? <span className="muted"> · {timeAgo(last.get(p.person)!)}</span> : null}</span>
      ))}
    </div>
  )
}

function People({ people, fields }: { people: Labeller[]; fields: number | null }) {
  if (!people.length) return <div className="muted small">Nobody has labelled yet.</div>
  return (
    <div className="people-chips">
      {people.map((p) => (
        <span key={p.person} className="person-chip" title={`last saved ${timeAgo(p.last_active)}`}>
          <b>{p.person}</b> {fields != null ? `${p.fields_done} of ${fields} fields` : `${p.labels.toLocaleString()} labels`}
          <span className="muted"> · {timeAgo(p.last_active)}</span>
        </span>
      ))}
    </div>
  )
}

function SessionMore({ s, onChanged, onError }: { s: SessionInfo; onChanged: () => void; onError: (e: string | null) => void }) {
  const defaults = useFetch<Defaults>(s.kind === 'queue' ? '/api/labels/defaults' : null)
  const [confirm, setConfirm] = useState<'delete' | null>(null)
  const [adding, setAdding] = useState(false)
  const [examining, setExamining] = useState(false)
  const prepareMore = async () => {  // a queue draws objects from prepared fields: prepare 10 more
    setAdding(true)
    await withError(() => api.post(`/api/labels/sessions/${s.name}/fields`, { n: 10 }), onError)
    setAdding(false)
    onChanged()
  }
  return (
    <div className="session-more">
      <div className="row">
        <a className="btn small" href={appPath(`/annotate/${s.name}`)} target="_blank" rel="noreferrer">Open the labelling page</a>
        <button className="btn small" onClick={() => setExamining(true)} disabled={!s.labels}
          title={s.labels ? 'Every labelled object, field by field' : 'Nobody has labelled anything in this session yet'}>Look at the labels</button>
        <a className="btn small" href={apiUrl(`/api/labels/export.csv?session=${s.name}`)}>Download this session’s labels</a>
        {s.kind === 'queue' && !s.round && <button className="btn small" disabled={adding} onClick={prepareMore} title="Prepares 10 more fields the queue can draw objects from">
          {adding ? 'Preparing…' : 'Prepare 10 more fields'}</button>}
        <span className="spacer" />
        <button className="btn small danger-outline" onClick={() => setConfirm('delete')}>Delete session</button>
      </div>
      {s.instructions && <p className="small secondary"><b>Note for labellers:</b> {s.instructions}</p>}
      {!s.read_only && <SessionAccessCard name={s.name} />}
      {s.kind === 'areas' ? <p className="small secondary">{typeof s.cores === 'number' ? s.cores.toLocaleString() : (s.cores?.length ?? 0).toLocaleString()} cores, each with
        all its stains; a core is handed to one person at a time{s.round_progress?.second_checks ? `, and ${s.round_progress.second_checks.toLocaleString()} also go to a second person` : ''}.
        {' '}To check other cores, make a new session.</p>
        : s.kind !== 'queue' && <SessionFields s={s} onChanged={onChanged} />}
      {examining && <SessionLabelsView name={s.name} title={s.title} onClose={() => setExamining(false)} />}
      <Advanced hint="settings and per-person counts">
        <table className="compact"><tbody>
          <tr><td className="secondary">Session ID</td><td><code>{s.name}</code></td></tr>
          <tr><td className="secondary">Created</td><td>{timeAgo(s.created)}</td></tr>
          <tr><td className="secondary">What people see of detection</td><td>{AI_VIEW_TEXT[s.ai_view]}</td></tr>
          {s.round && <tr><td className="secondary">How generous</td><td>{s.level == null ? 'every candidate' : `proposed at ${s.level} × the stain threshold or stronger`}</td></tr>}
          {s.kind !== 'queue' ? <>
            <tr><td className="secondary">Tasks</td><td>{s.tools.map((t) => TOOL_TEXT[t] ?? t).join(' · ')}</td></tr>
            <tr><td className="secondary">People per field</td><td>{s.per_field}</td></tr>
          </> : <tr><td className="secondary">Which objects</td><td>{defaults.data?.rules[s.rule] ?? s.rule}</td></tr>}
          <tr><td className="secondary">Labels saved</td><td>{s.labels.toLocaleString()}</td></tr>
        </tbody></table>
        {s.labellers.length > 0 && (
          <div className="table-scroll">
            <table className="compact">
              <thead><tr><th>Name</th><th className="num">Fields finished</th><th className="num">Objects changed or added</th><th className="num">Detection calls confirmed</th>
                <th className="num">Outlines</th><th className="num">Tags</th><th>Last saved</th></tr></thead>
              <tbody>{s.labellers.map((p) => (
                <tr key={p.person}><td>{p.person}</td><td className="num">{p.fields_done}</td><td className="num">{p.changed}</td><td className="num">{p.confirmed}</td>
                  <td className="num">{p.outlines}</td><td className="num">{p.tags}</td><td>{timeAgo(p.last_active)}</td></tr>
              ))}</tbody>
            </table>
          </div>
        )}
      </Advanced>
      {confirm === 'delete' && (
        <ConfirmDialog title={`Delete “${s.title}”?`} confirm="Delete session" danger onClose={() => setConfirm(null)}
          onConfirm={() => withError(async () => { await api.del(`/api/labels/sessions/${s.name}`); onChanged() }, onError)}>
          <p>The session closes for everyone and disappears from this list. <b>Labels people made are kept</b> and still count for training;
            the session’s settings are moved aside, not erased.</p>
        </ConfirmDialog>
      )}
    </div>
  )
}

/* ---------- link ---------- */

function ShareCard({ share, error, link, anyOpen }: { share: Share | null; error: string | null; link: string; anyOpen: boolean }) {
  return (
    <section className="card labels-section">
      <div className="row"><h2 style={{ margin: 0 }}>Link for labellers</h2><span className="spacer" />{share && <CopyButton text={link} />}</div>
      <p className="small secondary">People sign in at this link (or make an account), then add a set with its key (a session → More → Who can label);
        they see only the open sets they joined{anyOpen ? '' : ' (none is open right now)'}. The rest of stainID is for admins only.</p>
      {share ? <code className="path-box share-link">{link}</code> : !error && <p className="muted small">Finding this computer’s address…</p>}
      <ErrorNote error={error} />
      {share && !share.network && (
        <div className="warning-note small">Right now only this computer can open the link. To let colleagues on your network in, restart stainID with
          {' '}<code>stainid serve</code> (without <code>--host 127.0.0.1</code>).</div>
      )}
      {share && share.network && share.links.length > 1 && (
        <Advanced title="Other addresses of this computer" hint="if the link above does not work for someone">
          {share.links.slice(1).map((l) => <div key={l} className="row"><code className="path-box">{l}</code><CopyButton text={l} label="Copy" /></div>)}
          <p className="muted small">For people outside your network, forward this computer’s port or use a tunnel (e.g. <code>cloudflared tunnel --url http://localhost:8765</code>).
            On macOS, allow incoming connections for Python if asked.</p>
        </Advanced>
      )}
    </section>
  )
}

/* ---------- what has been labelled ---------- */

function LabelledSoFar({ stains, onChanged }: { stains: StainOverview[]; onChanged: () => void }) {
  const total = stains.reduce((a, s) => a + s.objects_labelled, 0)
  return (
    <section className="card labels-section">
      <div className="row"><h2 style={{ margin: 0 }}>Labelled so far</h2><span className="spacer" />
        {total > 0 && <Link className="btn small" to="/models">Train a model from these labels →</Link>}</div>
      <div className="table-scroll">
        <table className="labelled-table">
          <thead><tr><th>Stain</th><th className="num">Objects labelled</th><th>Fields with labels</th><th>What people said</th><th>How detection does on them</th></tr></thead>
          <tbody>{stains.map((s) => (
            <tr key={s.stain}>
              <td><b>{s.stain}</b></td>
              <td className="num">{s.objects_labelled.toLocaleString()}</td>
              <td><div className="labelled-fields"><Progress done={s.fields_with_labels} total={s.fields} />
                <span className="small secondary">{s.fields_with_labels.toLocaleString()} of {s.fields.toLocaleString()}
                  {s.fields_reviewed > 0 && <span className="muted"> · {s.fields_reviewed.toLocaleString()} fully checked</span>}</span></div></td>
              <td className="small secondary wrap-cell">{s.classes ? <ClassCounts classes={s.classes} />
                : Object.entries(s.values).map(([k, v]) => `${k} ${v.toLocaleString()}`).join(' · ') || '—'}</td>
              <td className="small wrap-cell">{s.detection ? <DetectionLine d={s.detection} stain={s.stain} /> : '—'}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
      <p className="muted small">“Never proposed”: real objects people labelled where detection had circled nothing (added with “Add a missed object”) — its blind spots.
        “Proposed but rejected”: detection looked at the object and said no.</p>
      <Advanced hint="agreement, detection against people, outlines, tags, sources, preparing fields">
        <div className="grid grid-3">{stains.map((s) => <StainDetails key={s.stain} s={s} onChanged={onChanged} />)}</div>
      </Advanced>
    </section>
  )
}

/** "tau+ neuron 431 (pretangle 175 · mature tangle 250 · no type 6) · thread 12 · not tau 1,100": objects per class
 *  (most agreed), with their types; classes nobody used are left out. */
function ClassCounts({ classes }: { classes: ClassCount[] }) {
  const used = classes.filter((c) => c.objects > 0)
  if (!used.length) return <>—</>
  return (
    <span className="class-counts">{used.map((c) => {
      const typed = c.subtypes.filter((t) => t.objects > 0)
      return (
        <span key={c.id} className="class-count"><b className={c.role === 'positive' ? '' : 'secondary'}>{c.name}</b> {c.objects.toLocaleString()}
          {typed.length > 0 && <span className="muted"> ({[...typed.map((t) => `${t.name} ${t.objects.toLocaleString()}`),
            ...(c.untyped ? [`no type ${c.untyped.toLocaleString()}`] : [])].join(' · ')})</span>}</span>
      )
    })}</span>
  )
}

function StainDetails({ s, onChanged }: { s: StainOverview; onChanged: () => void }) {
  const [error, setError] = useState<string | null>(null)
  const [asked, setAsked] = useState(false)
  const prepare = () => withError(async () => { await api.post('/api/labels/prepare', { stain: s.stain, fields: 20 }); setAsked(true); onChanged() }, setError)
  const row = (label: string, value: string) => <tr><td className="secondary">{label}</td><td className="num">{value}</td></tr>
  return (
    <div className="stain-details">
      <h3>{s.stain}</h3>
      <table className="compact"><tbody>
        {row('fields fully checked (every object)', fmt(s.fields_reviewed))}
        {row('fields prepared for labelling', fmt(s.fields_prepared))}
        {row(`${objectsOf(s.stain)} people added where detection proposed nothing`, fmt(s.objects_added))}
        {s.added && s.added.objects > 0 && row('of those added in sessions: measured · could not be measured · waiting',
          `${fmt(s.added.measured)} · ${fmt(s.added.could_not_be_measured)} · ${fmt(s.added.waiting)}`)}
        {s.detection && row(`real ${objectsOf(s.stain)}: found · never proposed · rejected`,
          `${fmt(s.detection.found)} · ${fmt(s.detection.never_proposed)} · ${fmt(s.detection.rejected)}`)}
        {s.detection && s.detection.not_checked > 0 && row('real objects on fields not prepared (not counted)', fmt(s.detection.not_checked))}
        {row('outlines (edited by people)', `${fmt(s.outlines)} (${fmt(s.outlines_edited)})`)}
        {row(`people agree (${fmt(s.objects_with_several_labels)} objects with ≥2 labels)`, percent(s.agreement))}
        {row(`Detection agrees with people (${fmt(s.ai.n)} objects)`, percent(s.ai.agreement))}
        {row('Detection precision · recall, among proposed objects only', `${fmt(s.ai.precision, 2)} · ${fmt(s.ai.recall, 2)}`)}
        {Object.entries(s.tags).map(([k, v]) => <tr key={k}><td className="secondary">fields tagged “{k}”</td><td className="num">{fmt(v)}</td></tr>)}
      </tbody></table>
      <p className="muted small">The detection figures include objects the current model was trained on; the cross-validated estimate is on Models.
        The older recall counts only objects detection proposed, so it cannot see what detection never looked at.
        Sources: {s.people.length ? s.people.join(', ') : 'none yet'}.</p>
      <button className="btn small" disabled={asked} onClick={prepare} title="Finds the candidate objects in 20 more fields in the background, so queues have more to draw from">
        {asked ? 'Preparing 20 more fields…' : 'Prepare 20 more fields'}</button>
      <ErrorNote error={error} />
    </div>
  )
}

/* ---------- new session ---------- */

/** The classes people will choose from, and whether they are also asked for each object's type (pretangle or mature tangle…). */
function SessionClasses({ vocabulary, typesAsked, onTypesAsked }: { vocabulary: Vocabulary; typesAsked: boolean; onTypesAsked: (on: boolean) => void }) {
  const shown = byKey(vocabulary, { direct: {} })  // in the order of the keys, unsure last
  const typed = shown.filter((c) => c.subtypes.some((s) => !s.hidden))
  return (
    <div className="session-classes">
      <p className="muted small" style={{ margin: 0 }}>People choose between: {shown.map((c) => c.name).join(' · ')}.
        {' '}<Link to="/settings#label-classes">Change the classes</Link></p>
      {typed.length > 0 && <label className="layer-toggle">
        <input type="checkbox" checked={typesAsked} onChange={(e) => onTypesAsked(e.target.checked)} />
        Also ask for the type: {typed.map((c) => `${c.name} (${c.subtypes.filter((s) => !s.hidden).map((s) => s.name).join(', ')})`).join('; ')}
      </label>}
    </div>
  )
}

type Kind = 'fields' | 'queue'

function NewSession({ stains, onClose, onCreated }: { stains: string[]; onClose: () => void; onCreated: (s: SessionInfo) => void }) {
  const defaults = useFetch<Defaults>('/api/labels/defaults')
  const [form, setForm] = useState({ stain: stains[0] ?? 'NeuN', kind: 'fields' as Kind, n_fields: 20, title: '', instructions: '', open: true,
    name: '', per_field: 1, tools: ['class', 'point', 'outline', 'tag'], ai_view: 'full' as AiView, rule: 'uncertain', tile_ids: '', subtypes: true })
  const [mode, setMode] = useState<'auto' | 'pick'>('auto')
  const [chosen, setChosen] = useState<string[]>([])
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const set = <K extends keyof typeof form>(key: K, value: (typeof form)[K]) => setForm((f) => ({ ...f, [key]: value }))
  const placeholder = `${form.stain} ${form.kind === 'fields' ? 'review' : 'quick check'} – ${new Date().toLocaleDateString(undefined, { day: 'numeric', month: 'short' })}`
  const pasted = form.tile_ids.split(/[\s,]+/).filter(Boolean)
  const ids = mode === 'pick' ? chosen : pasted
  const create = async () => {
    setBusy(true)
    const body = { ...form, title: form.title.trim() || placeholder, tile_ids: form.kind !== 'queue' && ids.length ? ids : null,
      tools: form.kind === 'queue' ? ['class'] : form.tools }
    const created = await withError(() => api.post<SessionInfo>('/api/labels/sessions', body), setError)
    setBusy(false)
    if (created) { onCreated(created); onClose() }
  }
  const choice = (kind: Kind, title: string, text: string) => (
    <label className={`choice-card${form.kind === kind ? ' chosen' : ''}`}>
      <input type="radio" name="kind" checked={form.kind === kind} onChange={() => setForm((f) => ({ ...f, kind }))} />
      <span><b>{title}</b><span className="small secondary">{text}</span></span>
    </label>
  )
  const nothingChosen = form.kind !== 'queue' && mode === 'pick' && !chosen.length
  return (
    <Modal title="New session of an older kind" onClose={onClose} wide>
      <div className="new-session">
        <fieldset className="field"><legend>Stain</legend>
          <div className="segmented">{stains.map((s) => (
            <button key={s} type="button" className={`btn${form.stain === s ? ' chosen' : ''}`} aria-pressed={form.stain === s}
              onClick={() => { set('stain', s); setChosen([]) }}>{s}</button>
          ))}</div>
        </fieldset>
        <fieldset className="field"><legend>What should people do?</legend>
          <div className="choice-grid">
            {choice('fields', 'Whole-field review (current label pool)', 'People see a whole field with detection’s calls filled in, correct the wrong ones and add what it missed. Gives complete labels for every object.')}
            {choice('queue', 'Quick check, one object at a time (current label pool)', 'People see one object and pick what it is with a key press. Fast; aims at the objects detection is unsure about.')}
          </div>
        </fieldset>
        {form.kind !== 'queue' && <FieldChoice stain={form.stain} mode={mode} n={form.n_fields} chosen={chosen} onMode={setMode}
          onN={(n) => set('n_fields', n)} onChosen={setChosen} />}
        <label className="field">Title people see<input type="text" value={form.title} placeholder={placeholder} onChange={(e) => set('title', e.target.value)} /></label>
        <label className="field">Note for labellers (optional)<textarea rows={2} value={form.instructions} placeholder="e.g. Count only neurons with a visible nucleus." onChange={(e) => set('instructions', e.target.value)} /></label>
        {defaults.data?.vocabularies[form.stain] && <SessionClasses vocabulary={defaults.data.vocabularies[form.stain]} typesAsked={form.subtypes}
          onTypesAsked={(on) => set('subtypes', on)} />}
        <label className="layer-toggle"><input type="checkbox" checked={form.open} onChange={(e) => set('open', e.target.checked)} />Open to labellers right away</label>
        <Advanced hint="session ID, people per field, what people see of detection, tasks, field IDs">
          <div className="row" style={{ alignItems: 'flex-end' }}>
            <label className="field">Session ID (in links and files)<input type="text" value={form.name} placeholder="made from the title"
              onChange={(e) => set('name', e.target.value.replace(/[^A-Za-z0-9_-]/g, '_'))} /></label>
            {form.kind === 'fields' && <label className="field">People per field<input type="number" min={1} value={form.per_field} style={{ width: 80 }}
              onChange={(e) => set('per_field', Math.max(1, Number(e.target.value)))} /></label>}
            <label className="field">What people see of detection<select value={form.ai_view} onChange={(e) => set('ai_view', e.target.value as AiView)}>
              {(defaults.data?.ai_views ?? ['full', 'positions', 'none'] as AiView[]).map((v) => <option key={v} value={v}>{AI_VIEW_TEXT[v]}</option>)}</select></label>
            {form.kind === 'queue' && <label className="field">Which objects<select value={form.rule} onChange={(e) => set('rule', e.target.value)}>
              {Object.entries(defaults.data?.rules ?? { uncertain: 'where detection is unsure' }).map(([k, v]) => <option key={k} value={k}>{v}</option>)}</select></label>}
          </div>
          {form.kind === 'fields' && <>
            <div className="row">{(defaults.data?.tools ?? Object.keys(TOOL_TEXT)).map((t) => (
              <label key={t} className="layer-toggle"><input type="checkbox" checked={form.tools.includes(t)}
                onChange={() => set('tools', form.tools.includes(t) ? form.tools.filter((x) => x !== t) : [...form.tools, t])} />{TOOL_TEXT[t]}</label>
            ))}</div>
            {mode === 'auto' && <label className="field">Or paste field IDs (replaces “How many fields”)
              <textarea rows={2} value={form.tile_ids} placeholder="e.g. LIP-1_B-2_NeuN_W0102, LIP-3_C-4_NeuN_W0201" onChange={(e) => set('tile_ids', e.target.value)} /></label>}
            {defaults.data && <p className="muted small" style={{ margin: 0 }}>Field tags people can set: {defaults.data.tags.join(' · ')}.</p>}
          </>}
          <p className="muted small" style={{ margin: 0 }}>Fields are prepared in the background (10–30 s each) and appear to people as they become ready; they are shown in
            shuffled order. Only fields of cores that are exported, calibrated and detected can be chosen.</p>
        </Advanced>
        <ErrorNote error={error ?? defaults.error} />
        <div className="row"><span className="spacer" />
          {nothingChosen && <span className="small muted">Click squares on the cores above to choose fields.</span>}
          <button className="btn" onClick={onClose}>Cancel</button>
          <button className="btn primary" disabled={busy || nothingChosen || (form.kind === 'fields' && !form.tools.length)} onClick={create}>
            {busy ? 'Creating…' : form.kind !== 'queue' && mode === 'pick' && chosen.length ? `Create session with ${chosen.length} field${chosen.length === 1 ? '' : 's'}` : 'Create session'}</button>
        </div>
      </div>
    </Modal>
  )
}
