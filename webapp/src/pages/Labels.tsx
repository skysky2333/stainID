import { useState } from 'react'
import { Link } from 'react-router-dom'
import type { AiView, LabelsOverview, SessionInfo, StainOverview } from '../api'
import { api } from '../api'
import { Empty, ErrorNote, HelpBox, PageHead, Progress, fmt, withError } from '../components'
import { useFetch } from '../hooks'
import { useSession } from '../session'

export const AI_VIEW_TEXT: Record<AiView, string> = {
  full: 'AI calls and outlines pre-filled',
  positions: 'blind: candidates shown, AI calls hidden',
  none: 'no AI shown',
}
const TOOL_TEXT: Record<string, string> = {
  class: 'correct the class of each object', point: 'add objects the AI missed', outline: 'fix or draw outlines', tag: 'tag the whole field',
}
interface Defaults { classes: Record<string, string[]>; tags: string[]; tools: string[]; ai_views: AiView[]; rules: Record<string, string> }

export default function Labels() {
  const overview = useFetch<LabelsOverview>('/api/labels/overview', 10000)
  const [error, setError] = useState<string | null>(null)
  const stains = overview.data?.stains ?? []
  const total = stains.reduce((a, s) => ({ fields: a.fields + s.fields, reviewed: a.reviewed + s.fields_reviewed, objects: a.objects + s.objects_labelled }),
    { fields: 0, reviewed: 0, objects: 0 })

  return (
    <>
      <PageHead title="Labels" subtitle="What people have said about the objects and fields, and the sessions in which they say it." >
        <a className="btn" href="/api/labels/export.csv">Download all labels (CSV)</a>
      </PageHead>
      <HelpBox id="labels">
        <ul>
          <li>Every object the pipeline considers (accepted or rejected) can carry labels. The AI's call is shown next to people's labels; an object's label is what most people said.</li>
          <li><b>Review sessions</b> show whole fields with the AI's calls and outlines already filled in: people correct what is wrong, add what was missed and press Finished,
            which counts the calls they left alone as confirmed. <b>Queues</b> show one object at a time, picked by a rule (where the AI is unsure, where people disagree, …).</li>
          <li>Start a session and send the link below; people register a name and label in their browser. Train new models from all labels on the <Link to="/models">Models</Link> page.</li>
        </ul>
      </HelpBox>
      <ErrorNote error={overview.error ?? error} />
      {overview.data && <>
        <div className="card" style={{ marginBottom: 16 }}>
          <div className="row"><h2 style={{ margin: 0 }}>Overall</h2><span className="spacer" />
            <span className="muted small">{total.objects.toLocaleString()} objects labelled · {total.reviewed} of {total.fields} fields fully reviewed</span></div>
          <div style={{ marginTop: 8 }}><Progress done={total.reviewed} total={total.fields} /></div>
        </div>
        <div className="grid grid-3">{stains.map((s) => <StainCard key={s.stain} s={s} onError={setError} onChanged={overview.reload} />)}</div>
        <ShareLink />
        <Sessions sessions={overview.data.sessions} onChanged={overview.reload} onError={setError} />
        <NewSession onCreated={overview.reload} />
      </>}
    </>
  )
}

function StainCard({ s, onChanged, onError }: { s: StainOverview; onChanged: () => void; onError: (e: string | null) => void }) {
  const prepare = () => withError(async () => { await api.post('/api/labels/prepare', { stain: s.stain, fields: 20 }); onChanged() }, onError)
  const pct = (v: number | null) => (v == null ? '—' : `${Math.round(100 * v)}%`)
  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>{s.stain}</h3>
      <div className="small secondary">Fields fully reviewed</div>
      <Progress done={s.fields_reviewed} total={s.fields} />
      <div className="muted small">{s.fields_reviewed} of {s.fields} · {s.fields_with_labels} with some labels · {s.fields_prepared} prepared
        <button className="link-btn" style={{ marginLeft: 6 }} onClick={prepare}>prepare 20 more</button></div>
      <table className="compact" style={{ marginTop: 10 }}><tbody>
        <tr><td className="secondary">objects labelled</td><td className="num"><b>{s.objects_labelled.toLocaleString()}</b></td></tr>
        {Object.entries(s.values).map(([k, v]) => <tr key={k}><td className="muted" style={{ paddingLeft: 16 }}>{k}</td><td className="num">{v}</td></tr>)}
        <tr><td className="secondary">objects people added (AI missed)</td><td className="num">{s.objects_added}</td></tr>
        <tr><td className="secondary">outlines (edited by people)</td><td className="num">{s.outlines} ({s.outlines_edited})</td></tr>
        <tr><td className="secondary">people agree (objects with ≥2 labels)</td><td className="num">{pct(s.agreement)} <span className="muted">({s.objects_with_several_labels})</span></td></tr>
        <tr><td className="secondary">AI agrees with people</td><td className="num">{pct(s.ai.agreement)} <span className="muted">({s.ai.n})</span></td></tr>
        <tr><td className="secondary">AI precision · recall</td><td className="num">{fmt(s.ai.precision, 2)} · {fmt(s.ai.recall, 2)}</td></tr>
        {Object.entries(s.tags).map(([k, v]) => <tr key={k}><td className="secondary">fields tagged “{k}”</td><td className="num">{v}</td></tr>)}
      </tbody></table>
      <p className="muted small">AI figures include objects the current model was trained on, so they flatter it; the fair estimate is the cross-validated one on Models.
        Sources: {s.people.length ? s.people.join(', ') : 'none yet'}.</p>
    </div>
  )
}

function ShareLink() {
  const { session } = useSession()
  const share = useFetch<{ links: string[]; network: boolean }>('/api/labels/share')
  return (
    <div className="card" style={{ margin: '16px 0' }}>
      <h2>Link for people who label</h2>
      {share.data && (share.data.network ? <>
        <p className="muted small">People on your network open this address, register a name and see the sessions you have started. The rest of stainID stays behind your admin password.</p>
        <div className="grid" style={{ gap: 6 }}>
          {share.data.links.map((link) => (
            <div key={link} className="row"><code className="path-box">{link}</code><button className="btn small" onClick={() => navigator.clipboard?.writeText(link)}>Copy</button></div>
          ))}
        </div>
        <p className="muted small">For people elsewhere, forward this computer's port or use a tunnel (e.g. <code>cloudflared tunnel --url http://localhost:8765</code>). On macOS, allow incoming connections for Python if asked.</p>
      </> : <p className="small">stainID only listens on this computer. Restart it with <code>stainid serve</code> (without <code>--host 127.0.0.1</code>) so others can reach the labelling pages.</p>)}
      {session.default_password && <p className="small" style={{ color: 'var(--warning)' }}>Your admin password is still the default. Change it in Settings before sharing the link.</p>}
      <ErrorNote error={share.error} />
    </div>
  )
}

function Sessions({ sessions, onChanged, onError }: { sessions: SessionInfo[]; onChanged: () => void; onError: (e: string | null) => void }) {
  const [open, setOpen] = useState<string | null>(null)
  const act = (fn: () => Promise<unknown>) => withError(async () => { await fn(); onChanged() }, onError)
  const remove = (s: SessionInfo) => {
    if (window.confirm(`Delete the session “${s.title}”? It closes to people and is moved aside; its labels stay in the label store.`)) act(() => api.del(`/api/labels/sessions/${s.name}`))
  }
  return (
    <div className="card">
      <h2>Sessions</h2>
      {sessions.length ? (
        <table>
          <thead><tr><th>Session</th><th>Status</th><th>Stain</th><th>Kind</th><th style={{ width: 170 }}>Progress</th><th className="num">People</th><th className="num">Labels</th><th /></tr></thead>
          <tbody>
            {sessions.map((s) => (
              <tr key={s.name}>
                <td>{s.title}<div className="muted small">{s.name} · {AI_VIEW_TEXT[s.ai_view]}</div></td>
                <td>{s.open ? <span className="badge status-running">running</span> : <span className="badge">stopped</span>}</td>
                <td>{s.stain}</td>
                <td className="small">{s.kind === 'fields' ? `review fields · ${s.per_field} per field` : `queue · ${s.rule}`}</td>
                <td>{s.kind === 'fields' ? <><Progress done={s.fields_covered ?? 0} total={s.fields ?? 0} />
                  <span className="muted small">{s.fields_covered} of {s.fields} done{s.fields_prepared! < s.fields! ? ` · ${s.fields_prepared} prepared` : ''}</span></> : <span className="muted small">open-ended</span>}</td>
                <td className="num">{s.people}</td>
                <td className="num">{s.labels}</td>
                <td className="row" style={{ flexWrap: 'nowrap' }}>
                  {s.open ? <button className="btn small" onClick={() => act(() => api.post(`/api/labels/sessions/${s.name}/open`, { open: false }))}>Stop</button>
                    : <button className="btn small primary" onClick={() => act(() => api.post(`/api/labels/sessions/${s.name}/open`, { open: true }))}>Start</button>}
                  <button className="btn small" onClick={() => setOpen(open === s.name ? null : s.name)}>{open === s.name ? 'Hide' : 'People'}</button>
                  <button className="btn small" title={s.kind === 'fields' ? 'Add 10 more unreviewed fields' : 'Prepare 10 more fields for the queue'}
                    onClick={() => act(() => api.post(`/api/labels/sessions/${s.name}/fields`, { n: 10 }))}>+10 fields</button>
                  <a className="btn small" href={`/annotate/${s.name}`} target="_blank" rel="noreferrer">Label</a>
                  <a className="btn small" href={`/api/labels/export.csv?session=${s.name}`}>CSV</a>
                  <button className="btn small" onClick={() => remove(s)}>Delete</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : <Empty>No sessions yet — create one below.</Empty>}
      {open && <People key={open} name={open} />}
    </div>
  )
}

function People({ name }: { name: string }) {
  const people = useFetch<{ person: string; fields_done: number; changed: number; confirmed: number; outlines: number; tags: number }[]>(`/api/labels/sessions/${name}/people`, 15000)
  return (
    <div style={{ marginTop: 12 }}>
      <h3>People in {name}</h3>
      <ErrorNote error={people.error} />
      {people.data && (people.data.length ? (
        <table className="compact">
          <thead><tr><th>name</th><th className="num">fields finished</th><th className="num">objects changed or added</th><th className="num">AI calls confirmed</th><th className="num">outlines</th><th className="num">tags</th></tr></thead>
          <tbody>{people.data.map((p) => (
            <tr key={p.person}><td>{p.person}</td><td className="num">{p.fields_done}</td><td className="num">{p.changed}</td><td className="num">{p.confirmed}</td><td className="num">{p.outlines}</td><td className="num">{p.tags}</td></tr>
          ))}</tbody>
        </table>
      ) : <p className="muted small">Nobody has labelled in this session yet.</p>)}
    </div>
  )
}

function NewSession({ onCreated }: { onCreated: () => void }) {
  const defaults = useFetch<Defaults>('/api/labels/defaults')
  const [form, setForm] = useState({ name: '', title: '', stain: 'NeuN', kind: 'fields' as 'fields' | 'queue', n_fields: 20, tile_ids: '',
    per_field: 1, tools: ['class', 'point', 'outline', 'tag'], ai_view: 'full' as AiView, rule: 'uncertain', instructions: '' })
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const set = (key: string, value: unknown) => setForm((f) => ({ ...f, [key]: value }))
  const create = async () => {
    setBusy(true)
    const ids = form.tile_ids.split(/[\s,]+/).filter(Boolean)
    const body = { ...form, tile_ids: ids.length ? ids : null, tools: form.kind === 'queue' ? ['class'] : form.tools }
    const result = await withError(() => api.post('/api/labels/sessions', body), setError)
    setBusy(false)
    if (result) { onCreated(); setForm((f) => ({ ...f, name: '', title: '', tile_ids: '' })) }
  }
  return (
    <div className="card" style={{ marginTop: 16 }}>
      <h2>New session</h2>
      <div className="tabs">
        <button className={`tab${form.kind === 'fields' ? ' active' : ''}`} onClick={() => set('kind', 'fields')}>Review whole fields</button>
        <button className={`tab${form.kind === 'queue' ? ' active' : ''}`} onClick={() => set('kind', 'queue')}>Quick object queue</button>
      </div>
      <p className="muted small">{form.kind === 'fields'
        ? 'People see whole fields with the AI calls pre-filled, correct them, add missed objects, fix outlines and tag the field. Gives complete labels for every object in the field.'
        : 'People see one object at a time and pick its class with the keyboard. Objects come from prepared fields and are chosen by the rule, so the queue follows new data and new labels.'}</p>
      <div className="row" style={{ alignItems: 'flex-end' }}>
        <label className="field">Name<input type="text" value={form.name} placeholder="e.g. neun_review_1" onChange={(e) => set('name', e.target.value.replace(/[^A-Za-z0-9_-]/g, '_'))} /></label>
        <label className="field" style={{ flex: 1 }}>Title shown to people<input type="text" value={form.title} onChange={(e) => set('title', e.target.value)} /></label>
        <label className="field">Stain<select value={form.stain} onChange={(e) => set('stain', e.target.value)}>{Object.keys(defaults.data?.classes ?? {}).map((s) => <option key={s}>{s}</option>)}</select></label>
        {form.kind === 'fields' ? <>
          <label className="field">Fields<input type="number" min={1} value={form.n_fields} style={{ width: 80 }} onChange={(e) => set('n_fields', Number(e.target.value))} /></label>
          <label className="field">People per field<input type="number" min={1} value={form.per_field} style={{ width: 80 }} onChange={(e) => set('per_field', Number(e.target.value))} /></label>
        </> : <label className="field">Which objects<select value={form.rule} onChange={(e) => set('rule', e.target.value)}>
          {Object.entries(defaults.data?.rules ?? {}).map(([k, v]) => <option key={k} value={k}>{v}</option>)}</select></label>}
        <label className="field">What people see of the AI<select value={form.ai_view} onChange={(e) => set('ai_view', e.target.value)}>
          {(defaults.data?.ai_views ?? []).map((v) => <option key={v} value={v}>{AI_VIEW_TEXT[v]}</option>)}</select></label>
      </div>
      {form.kind === 'fields' && <div className="row" style={{ marginTop: 10 }}>
        {(defaults.data?.tools ?? []).map((t) => (
          <label key={t} className="layer-toggle"><input type="checkbox" checked={form.tools.includes(t)}
            onChange={() => set('tools', form.tools.includes(t) ? form.tools.filter((x) => x !== t) : [...form.tools, t])} />{TOOL_TEXT[t]}</label>
        ))}
      </div>}
      {defaults.data && <p className="muted small">Classes for {form.stain}: {defaults.data.classes[form.stain].join(' · ')}. Field tags: {defaults.data.tags.join(' · ')}.</p>}
      <label className="field" style={{ marginTop: 6 }}>Instructions shown to people (optional)<textarea rows={2} value={form.instructions} onChange={(e) => set('instructions', e.target.value)} /></label>
      {form.kind === 'fields' && <label className="field" style={{ marginTop: 6 }}>Or exactly these fields (field IDs, optional)
        <textarea rows={2} value={form.tile_ids} placeholder="e.g. LIP-1_B-2_NeuN_S01, LIP-3_C-4_NeuN_S02" onChange={(e) => set('tile_ids', e.target.value)} /></label>}
      <p className="muted small">New sessions start stopped. Fields are picked among those nobody has reviewed, spread over TMAs and groups, in shuffled order; they are prepared in the
        background (10–30 s each) and appear to people as they become ready.</p>
      <button className="btn primary" disabled={!form.name || busy || (form.kind === 'fields' && !form.tools.length)} onClick={create}>{busy ? 'Creating…' : 'Create session'}</button>
      <ErrorNote error={error ?? defaults.error} />
    </div>
  )
}
