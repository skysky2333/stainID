import { useState } from 'react'
import { Link } from 'react-router-dom'
import type { AiView, LabelsOverview, Labeller, SessionInfo, StainOverview } from '../api'
import { api, apiUrl, appPath } from '../api'
import { Empty, ErrorNote, HelpBox, Loaded, Modal, PageHead, Progress, fmt, timeAgo, withError } from '../components'
import { useFetch } from '../hooks'
import { AI_VIEW_TEXT, ConfirmDialog, CopyButton, KIND_TEXT, TOOL_TEXT, objectsOf, percent } from '../labelling'
import { Advanced } from '../setup/parts'

interface Defaults { classes: Record<string, string[]>; tags: string[]; tools: string[]; ai_views: AiView[]; rules: Record<string, string> }
interface Share { links: string[]; network: boolean }

export default function Labels() {
  const overview = useFetch<LabelsOverview>('/api/labels/overview', 10000)
  const share = useFetch<Share>('/api/labels/share')
  const [creating, setCreating] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const linkFor = (name = '') => {
    const base = share.data?.network && share.data.links[0] ? share.data.links[0] : `${location.origin}${appPath('/annotate')}`
    return name ? `${base}/${name}` : base
  }

  return (
    <>
      <PageHead title="Labels" subtitle="Ask colleagues to check the AI’s calls on a few fields; their labels then train better models.">
        <a className="btn" href={apiUrl('/api/labels/export.csv')}>Download all labels</a>
        <button className="btn primary" onClick={() => setCreating(true)}>New labelling session</button>
      </PageHead>
      <HelpBox id="labels">
        <ol className="plain-steps">
          <li><b>Start a session</b>: pick a stain and how many fields. stainID chooses fields nobody has checked yet, spread over TMAs and groups.</li>
          <li><b>Send the link</b>. People type their name and label in their browser — no account needed. Everything is saved as they go.</li>
          <li><b>Watch progress</b> here, then <Link to="/models">train a new model</Link> from all labels and compare it with the one in use.</li>
        </ol>
      </HelpBox>
      {notice && <div className="found found-good labels-notice">{notice}<button className="link-btn" onClick={() => setNotice(null)}>Dismiss</button></div>}
      <Loaded state={overview} label="Loading labelling sessions…">
        {(data) => <>
          <Sessions sessions={data.sessions} linkFor={linkFor} onChanged={overview.reload} onNew={() => setCreating(true)} />
          <ShareCard share={share.data} error={share.error} link={linkFor()} anyOpen={data.sessions.some((s) => s.open)} />
          <LabelledSoFar stains={data.stains} onChanged={overview.reload} />
          {creating && <NewSession stains={data.stains.map((s) => s.stain)} onClose={() => setCreating(false)}
            onCreated={(s) => { overview.reload(); setNotice(s.open
              ? `“${s.title}” is open. Fields are prepared in the background (10–30 s each) and appear to people as they become ready. Copy its link below and send it.`
              : `“${s.title}” is ready but closed. Open it when you want people to start.`) }} />}
        </>}
      </Loaded>
    </>
  )
}

/* ---------- sessions ---------- */

function Sessions({ sessions, linkFor, onChanged, onNew }: { sessions: SessionInfo[]; linkFor: (name: string) => string; onChanged: () => void; onNew: () => void }) {
  const ordered = [...sessions].sort((a, b) => b.created - a.created)  // newest first; stays put when one is opened or closed
  return (
    <section className="card labels-section">
      <h2>Labelling sessions</h2>
      {ordered.length ? <div className="session-list">{ordered.map((s) => <SessionRow key={s.name} s={s} link={linkFor(s.name)} onChanged={onChanged} />)}</div>
        : <Empty>
          <p style={{ marginTop: 0 }}>No labelling sessions yet. A session asks people to check a set of fields (or objects) for one stain.</p>
          <button className="btn" onClick={onNew}>Start the first session</button>
        </Empty>}
    </section>
  )
}

function sessionState(s: SessionInfo): { text: string; className: string } {
  if (s.kind === 'fields' && s.fields && (s.fields_covered ?? 0) >= s.fields) return { text: 'Done', className: 'status-done' }
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
          <b>{s.title}</b> <span className={`badge ${state.className}`}>{state.text}</span>
          {s.labelling_now > 0 && <span className="badge status-running"><span className="pulse" />{s.labelling_now} labelling now</span>}
        </div>
        <div className="muted small">{s.stain} · {KIND_TEXT[s.kind]}{s.kind === 'fields' && s.per_field > 1 ? ` · ${s.per_field} people per field` : ''}</div>
        <SessionProgress s={s} />
        <People people={s.labellers} fields={s.kind === 'fields' ? s.fields ?? 0 : null} />
      </div>
      <div className="session-actions">
        {s.open && <CopyButton text={link} />}
        <button className="btn small" disabled={busy} onClick={toggle} title={s.open ? 'People can no longer open it; nothing is lost' : 'People with the link can start labelling'}>
          {busy ? '…' : s.open ? 'Close' : 'Open to labellers'}</button>
        <button className="btn small" onClick={() => setOpen(!open)} aria-expanded={open}>{open ? 'Less' : 'More'}</button>
      </div>
      <ErrorNote error={error} />
      {open && <SessionMore s={s} onChanged={onChanged} onError={setError} />}
    </div>
  )
}

function SessionProgress({ s }: { s: SessionInfo }) {
  if (s.kind !== 'fields') return <div className="small secondary session-progress">{s.labels.toLocaleString()} objects labelled so far · open-ended</div>
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
  const addFields = async () => {
    setAdding(true)
    await withError(() => api.post(`/api/labels/sessions/${s.name}/fields`, { n: 10 }), onError)
    setAdding(false)
    onChanged()
  }
  return (
    <div className="session-more">
      <div className="row">
        <a className="btn small" href={appPath(`/annotate/${s.name}`)} target="_blank" rel="noreferrer">Open the labelling page</a>
        <a className="btn small" href={apiUrl(`/api/labels/export.csv?session=${s.name}`)}>Download this session’s labels</a>
        <button className="btn small" disabled={adding} onClick={addFields}
          title={s.kind === 'fields' ? 'Adds 10 fields nobody has checked yet' : 'Prepares 10 more fields the queue can draw objects from'}>
          {adding ? 'Adding…' : '+ 10 more fields'}</button>
        <span className="spacer" />
        <button className="btn small danger-outline" onClick={() => setConfirm('delete')}>Delete session</button>
      </div>
      {s.instructions && <p className="small secondary"><b>Note for labellers:</b> {s.instructions}</p>}
      <Advanced hint="settings and per-person counts">
        <table className="compact"><tbody>
          <tr><td className="secondary">Session ID</td><td><code>{s.name}</code></td></tr>
          <tr><td className="secondary">Created</td><td>{timeAgo(s.created)}</td></tr>
          <tr><td className="secondary">What people see of the AI</td><td>{AI_VIEW_TEXT[s.ai_view]}</td></tr>
          {s.kind === 'fields' ? <>
            <tr><td className="secondary">Tasks</td><td>{s.tools.map((t) => TOOL_TEXT[t] ?? t).join(' · ')}</td></tr>
            <tr><td className="secondary">People per field</td><td>{s.per_field}</td></tr>
          </> : <tr><td className="secondary">Which objects</td><td>{defaults.data?.rules[s.rule] ?? s.rule}</td></tr>}
          <tr><td className="secondary">Labels saved</td><td>{s.labels.toLocaleString()}</td></tr>
        </tbody></table>
        {s.labellers.length > 0 && (
          <div className="table-scroll">
            <table className="compact">
              <thead><tr><th>Name</th><th className="num">Fields finished</th><th className="num">Objects changed or added</th><th className="num">AI calls confirmed</th>
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
      <p className="small secondary">Anyone with this link sees the open sessions{anyOpen ? '' : ' (none is open right now)'}, types their name and starts.
        The rest of stainID stays behind your admin password.</p>
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
          <thead><tr><th>Stain</th><th className="num">Objects labelled</th><th>Fields with labels</th><th>What people said</th></tr></thead>
          <tbody>{stains.map((s) => (
            <tr key={s.stain}>
              <td><b>{s.stain}</b></td>
              <td className="num">{s.objects_labelled.toLocaleString()}</td>
              <td><div className="labelled-fields"><Progress done={s.fields_with_labels} total={s.fields} />
                <span className="small secondary">{s.fields_with_labels.toLocaleString()} of {s.fields.toLocaleString()}
                  {s.fields_reviewed > 0 && <span className="muted"> · {s.fields_reviewed.toLocaleString()} fully checked</span>}</span></div></td>
              <td className="small secondary wrap-cell">{Object.entries(s.values).map(([k, v]) => `${k} ${v.toLocaleString()}`).join(' · ') || '—'}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
      <Advanced hint="agreement, AI against people, outlines, tags, sources, preparing fields">
        <div className="grid grid-3">{stains.map((s) => <StainDetails key={s.stain} s={s} onChanged={onChanged} />)}</div>
      </Advanced>
    </section>
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
        {row(`${objectsOf(s.stain)} people added (AI missed)`, fmt(s.objects_added))}
        {row('outlines (edited by people)', `${fmt(s.outlines)} (${fmt(s.outlines_edited)})`)}
        {row(`people agree (${fmt(s.objects_with_several_labels)} objects with ≥2 labels)`, percent(s.agreement))}
        {row(`AI agrees with people (${fmt(s.ai.n)} objects)`, percent(s.ai.agreement))}
        {row('AI precision · recall', `${fmt(s.ai.precision, 2)} · ${fmt(s.ai.recall, 2)}`)}
        {Object.entries(s.tags).map(([k, v]) => <tr key={k}><td className="secondary">fields tagged “{k}”</td><td className="num">{fmt(v)}</td></tr>)}
      </tbody></table>
      <p className="muted small">The AI figures include objects the current model was trained on; the cross-validated estimate is on Models.
        Sources: {s.people.length ? s.people.join(', ') : 'none yet'}.</p>
      <button className="btn small" disabled={asked} onClick={prepare} title="Finds the candidate objects in 20 more fields in the background, so queues have more to draw from">
        {asked ? 'Preparing 20 more fields…' : 'Prepare 20 more fields'}</button>
      <ErrorNote error={error} />
    </div>
  )
}

/* ---------- new session ---------- */

type Kind = 'fields' | 'queue'

function NewSession({ stains, onClose, onCreated }: { stains: string[]; onClose: () => void; onCreated: (s: SessionInfo) => void }) {
  const defaults = useFetch<Defaults>('/api/labels/defaults')
  const [form, setForm] = useState({ stain: stains[0] ?? 'NeuN', kind: 'fields' as Kind, n_fields: 20, title: '', instructions: '', open: true,
    name: '', per_field: 1, tools: ['class', 'point', 'outline', 'tag'], ai_view: 'full' as AiView, rule: 'uncertain', tile_ids: '' })
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const set = <K extends keyof typeof form>(key: K, value: (typeof form)[K]) => setForm((f) => ({ ...f, [key]: value }))
  const placeholder = `${form.stain} ${form.kind === 'fields' ? 'review' : 'quick check'} – ${new Date().toLocaleDateString(undefined, { day: 'numeric', month: 'short' })}`
  const ids = form.tile_ids.split(/[\s,]+/).filter(Boolean)
  const create = async () => {
    setBusy(true)
    const body = { ...form, title: form.title.trim() || placeholder, tile_ids: form.kind === 'fields' && ids.length ? ids : null,
      tools: form.kind === 'queue' ? ['class'] : form.tools }
    const created = await withError(() => api.post<SessionInfo>('/api/labels/sessions', body), setError)
    setBusy(false)
    if (created) { onCreated(created); onClose() }
  }
  const choice = (kind: Kind, title: string, text: string) => (
    <label className={`choice-card${form.kind === kind ? ' chosen' : ''}`}>
      <input type="radio" name="kind" checked={form.kind === kind} onChange={() => set('kind', kind)} />
      <span><b>{title}</b><span className="small secondary">{text}</span></span>
    </label>
  )
  return (
    <Modal title="New labelling session" onClose={onClose} wide>
      <div className="new-session">
        <fieldset className="field"><legend>Stain</legend>
          <div className="segmented">{stains.map((s) => (
            <button key={s} type="button" className={`btn${form.stain === s ? ' chosen' : ''}`} aria-pressed={form.stain === s} onClick={() => set('stain', s)}>{s}</button>
          ))}</div>
        </fieldset>
        <fieldset className="field"><legend>What should people do?</legend>
          <div className="choice-grid">
            {choice('fields', 'Check whole fields (recommended)', 'People see a whole field with the AI’s calls filled in, correct the wrong ones and add what it missed. Gives complete labels for every object.')}
            {choice('queue', 'Quick check, one object at a time', 'People see one object and pick what it is with a key press. Fast; aims at the objects the AI is unsure about.')}
          </div>
        </fieldset>
        <div className="row" style={{ alignItems: 'flex-end' }}>
          {form.kind === 'fields' && !ids.length && <label className="field">How many fields<input type="number" min={1} max={500} value={form.n_fields} style={{ width: 90 }}
            onChange={(e) => set('n_fields', Math.max(1, Number(e.target.value)))} /></label>}
          <label className="field" style={{ flex: 1, minWidth: 220 }}>Title people see<input type="text" value={form.title} placeholder={placeholder} onChange={(e) => set('title', e.target.value)} /></label>
        </div>
        <label className="field">Note for labellers (optional)<textarea rows={2} value={form.instructions} placeholder="e.g. Count only neurons with a visible nucleus." onChange={(e) => set('instructions', e.target.value)} /></label>
        {defaults.data && <p className="muted small" style={{ margin: 0 }}>People choose between: {defaults.data.classes[form.stain]?.join(' · ')}.</p>}
        <label className="layer-toggle"><input type="checkbox" checked={form.open} onChange={(e) => set('open', e.target.checked)} />Open to labellers right away</label>
        <Advanced hint="session ID, people per field, what people see of the AI, tasks, exact fields">
          <div className="row" style={{ alignItems: 'flex-end' }}>
            <label className="field">Session ID (in links and files)<input type="text" value={form.name} placeholder="made from the title"
              onChange={(e) => set('name', e.target.value.replace(/[^A-Za-z0-9_-]/g, '_'))} /></label>
            {form.kind === 'fields' && <label className="field">People per field<input type="number" min={1} value={form.per_field} style={{ width: 80 }}
              onChange={(e) => set('per_field', Math.max(1, Number(e.target.value)))} /></label>}
            <label className="field">What people see of the AI<select value={form.ai_view} onChange={(e) => set('ai_view', e.target.value as AiView)}>
              {(defaults.data?.ai_views ?? ['full', 'positions', 'none'] as AiView[]).map((v) => <option key={v} value={v}>{AI_VIEW_TEXT[v]}</option>)}</select></label>
            {form.kind === 'queue' && <label className="field">Which objects<select value={form.rule} onChange={(e) => set('rule', e.target.value)}>
              {Object.entries(defaults.data?.rules ?? { uncertain: 'where the AI is unsure' }).map(([k, v]) => <option key={k} value={k}>{v}</option>)}</select></label>}
          </div>
          {form.kind === 'fields' && <>
            <div className="row">{(defaults.data?.tools ?? Object.keys(TOOL_TEXT)).map((t) => (
              <label key={t} className="layer-toggle"><input type="checkbox" checked={form.tools.includes(t)}
                onChange={() => set('tools', form.tools.includes(t) ? form.tools.filter((x) => x !== t) : [...form.tools, t])} />{TOOL_TEXT[t]}</label>
            ))}</div>
            <label className="field">Or exactly these fields (field IDs; replaces “How many fields”)
              <textarea rows={2} value={form.tile_ids} placeholder="e.g. LIP-1_B-2_NeuN_S01, LIP-3_C-4_NeuN_S02" onChange={(e) => set('tile_ids', e.target.value)} /></label>
            {defaults.data && <p className="muted small" style={{ margin: 0 }}>Field tags people can set: {defaults.data.tags.join(' · ')}.</p>}
          </>}
          <p className="muted small" style={{ margin: 0 }}>Fields are picked among those nobody has checked, spread over TMAs and groups, in shuffled order. They are prepared
            in the background (10–30 s each) and appear to people as they become ready.</p>
        </Advanced>
        <ErrorNote error={error ?? defaults.error} />
        <div className="row"><span className="spacer" />
          <button className="btn" onClick={onClose}>Cancel</button>
          <button className="btn primary" disabled={busy || (form.kind === 'fields' && !form.tools.length)} onClick={create}>{busy ? 'Creating…' : 'Create session'}</button>
        </div>
      </div>
    </Modal>
  )
}
