import { useEffect, useState } from 'react'
import type { AiView, AnnotationListItem, AnnotationSummary } from '../api'
import { api } from '../api'
import { Empty, ErrorNote, HelpBox, PageHead, Progress, withError } from '../components'
import { useFetch } from '../hooks'
import { useSession } from '../session'

export const AI_VIEW_TEXT: Record<AiView, string> = {
  full: 'AI detections, classes and overlays shown',
  positions: 'AI positions only (blind to what the AI decided)',
  none: 'no AI shown',
}
export const TOOL_TEXT: Record<string, string> = {
  object: 'label the AI detections', point: 'mark objects the AI missed', polygon: 'draw outlines', tag: 'tag the whole field',
}

export default function AnnotationSessions() {
  const { session } = useSession()
  const list = useFetch<AnnotationListItem[]>('/api/annotate', 10000)
  const share = useFetch<{ links: string[]; network: boolean }>('/api/annotate-share')
  const [open, setOpen] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const sessions = list.data ?? []
  const running = sessions.filter((s) => s.open)

  const setRunning = async (name: string, value: boolean) => {
    if (await withError(() => api.post(`/api/annotate/${name}/open`, { open: value }), setError)) list.reload()
  }
  const remove = async (s: AnnotationListItem) => {
    if (!window.confirm(`Delete the session “${s.title}”? It is closed to annotators and moved aside (its annotations are kept in the project folder).`)) return
    if (await withError(() => api.del(`/api/annotate/${s.name}`), setError)) { setOpen(null); list.reload() }
  }

  return (
    <>
      <PageHead title="Annotation sessions" subtitle="Let other people annotate fields, with or without seeing the AI, and collect their annotations here." />
      <HelpBox id="annotation-sessions">
        <ol>
          <li><b>Create a session</b> below: which fields, which labels and tools, and what people see of the AI.</li>
          <li><b>Start</b> it. Annotators open the link below, type their name and annotate; each person works in their own layer, so nobody overwrites anyone else.</li>
          <li>Follow progress and agreement under <b>Results</b>, download everything as CSV, and <b>Stop</b> the session when you have enough. Stopped sessions are closed to annotators but keep all annotations.</li>
        </ol>
      </HelpBox>
      <div className="card" style={{ marginBottom: 16 }}>
        <h2>Link for annotators</h2>
        {share.data && (share.data.network ? <>
          <p className="muted small">People on the same network open this address (it lists the sessions you have started). The rest of stainID stays behind your admin password.</p>
          <div className="grid" style={{ gap: 6 }}>
            {share.data.links.map((link) => (
              <div key={link} className="row"><code className="path-box">{link}</code>
                <button className="btn small" onClick={() => navigator.clipboard?.writeText(link)}>Copy</button></div>
            ))}
          </div>
          <p className="muted small">For people outside your network, forward this computer's port or use a tunnel (e.g. <code>cloudflared tunnel --url http://localhost:8765</code>). On macOS, allow incoming connections for Python if asked.</p>
        </> : <p className="small">stainID only listens on this computer. Restart it with <code>stainid serve</code> (without <code>--host 127.0.0.1</code>) so others can reach the annotation pages.</p>)}
        {session.default_password && <p className="small" style={{ color: 'var(--warning)' }}>Your admin password is still the default. Change it in Settings before sharing the link.</p>}
        <p className="muted small">{running.length ? `${running.length} session${running.length > 1 ? 's' : ''} running.` : 'No session is running, so annotators will see an empty list.'}</p>
        <ErrorNote error={share.error} />
      </div>
      <ErrorNote error={list.error ?? error} />
      <div className="card">
        <h2>Sessions</h2>
        {sessions.length ? (
          <table>
            <thead><tr><th>Session</th><th>Status</th><th>Stain</th><th>What people do · see</th><th style={{ width: 180 }}>Fields covered</th><th className="num">People</th><th /></tr></thead>
            <tbody>
              {sessions.map((s) => (
                <tr key={s.name}>
                  <td>{s.title}<div className="muted small">{s.name} · {s.per_field} annotator{s.per_field > 1 ? 's' : ''} per field</div></td>
                  <td>{s.open ? <span className="badge status-running">running</span> : <span className="badge">stopped</span>}</td>
                  <td>{s.stain}</td>
                  <td className="small secondary wrap" style={{ maxWidth: 280 }}>{s.tools.map((t) => TOOL_TEXT[t]).join(', ')} · {AI_VIEW_TEXT[s.ai_view]}</td>
                  <td><Progress done={s.fields_covered} total={s.fields} /><span className="muted small">{s.fields_covered} of {s.fields}</span></td>
                  <td className="num">{s.annotators}</td>
                  <td className="row" style={{ flexWrap: 'nowrap' }}>
                    {s.open ? <button className="btn small" onClick={() => setRunning(s.name, false)}>Stop</button>
                      : <button className="btn small primary" onClick={() => setRunning(s.name, true)}>Start</button>}
                    <button className="btn small" onClick={() => setOpen(open === s.name ? null : s.name)}>{open === s.name ? 'Hide results' : 'Results'}</button>
                    <a className="btn small" href={`/annotate/${s.name}`} target="_blank" rel="noreferrer">Annotate</a>
                    <button className="btn small" onClick={() => remove(s)}>Delete</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <Empty>No sessions yet — create one below.</Empty>}
      </div>
      {open && <Results key={open} name={open} />}
      <NewSession onCreated={list.reload} />
    </>
  )
}

function Results({ name }: { name: string }) {
  const summary = useFetch<AnnotationSummary>(`/api/annotate/${name}/summary`, 15000)
  const data = summary.data
  const labels = Array.from(new Set((data?.ai_vs_human ?? []).map((r) => r.label)))
  const classes = Array.from(new Set((data?.ai_vs_human ?? []).map((r) => r.ai_class)))
  const count = (ai: string, label: string) => data?.ai_vs_human.find((r) => r.ai_class === ai && r.label === label)?.n ?? 0
  return (
    <div className="card" style={{ marginTop: 16 }}>
      <div className="row"><h2 style={{ margin: 0 }}>Results · {name}</h2><span className="spacer" />
        <button className="btn small" onClick={summary.reload}>Refresh</button>
        <a className="btn small primary" href={`/api/annotate/${name}/export.csv`}>Download all annotations (CSV)</a>
      </div>
      <ErrorNote error={summary.error} />
      {data && (
        <div className="grid grid-2" style={{ marginTop: 12 }}>
          <div>
            <h3>Annotators</h3>
            <p className="muted small">{data.fields_covered} of {data.fields} fields have all the annotators they need.</p>
            {data.annotators.length ? (
              <table className="compact">
                <thead><tr><th>name</th><th className="num">fields done</th><th className="num">AI objects labelled</th><th className="num">points</th><th className="num">outlines</th><th className="num">tags</th></tr></thead>
                <tbody>{data.annotators.map((a) => (
                  <tr key={a.annotator}><td>{a.annotator}</td><td className="num">{a.fields_done}</td><td className="num">{a.object}</td><td className="num">{a.point}</td><td className="num">{a.polygon}</td><td className="num">{a.tag}</td></tr>
                ))}</tbody>
              </table>
            ) : <p className="muted small">Nobody has annotated yet.</p>}
            {data.agreement.length > 0 && <>
              <h3 style={{ marginTop: 14 }}>Agreement on the same AI objects</h3>
              <table className="compact">
                <thead><tr><th>between</th><th className="num">objects</th><th className="num">same label</th></tr></thead>
                <tbody>{data.agreement.map((a) => (
                  <tr key={`${a.a}-${a.b}`}><td>{a.a} · {a.b}</td><td className="num">{a.objects}</td><td className="num">{(100 * a.agreement).toFixed(0)}%</td></tr>
                ))}</tbody>
              </table>
            </>}
          </div>
          <div>
            <h3>What the AI called it vs what people said</h3>
            {classes.length ? (
              <table className="compact">
                <thead><tr><th>AI class</th>{labels.map((l) => <th key={l} className="num">{l}</th>)}</tr></thead>
                <tbody>{classes.map((c) => (
                  <tr key={c}><td>{c.replaceAll('_', ' ')}</td>{labels.map((l) => <td key={l} className="num">{count(c, l) || ''}</td>)}</tr>
                ))}</tbody>
              </table>
            ) : <p className="muted small">No AI objects labelled yet.</p>}
          </div>
        </div>
      )}
    </div>
  )
}

function NewSession({ onCreated }: { onCreated: () => void }) {
  const defaults = useFetch<{ labels: Record<string, string[]>; tags: string[]; tools: string[]; ai_views: AiView[] }>('/api/annotate-defaults')
  const [form, setForm] = useState({ name: '', title: '', stain: 'NeuN', n_fields: 20, per_group: true, tile_ids: '', labels: '', tags: '',
    tools: ['object', 'point', 'polygon', 'tag'], ai_view: 'full' as AiView, per_field: 1, instructions: '' })
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const set = (key: string, value: unknown) => setForm((f) => ({ ...f, [key]: value }))

  useEffect(() => {
    if (defaults.data) setForm((f) => ({ ...f, labels: defaults.data!.labels[f.stain].join('\n'), tags: f.tags || defaults.data!.tags.join('\n') }))
  }, [defaults.data, form.stain])

  const lines = (text: string) => text.split('\n').map((l) => l.trim()).filter(Boolean)
  const create = async () => {
    setBusy(true)
    const tileIds = form.tile_ids.split(/[\s,]+/).filter(Boolean)
    const result = await withError(() => api.post<{ name: string }>('/api/annotate', {
      ...form, labels: lines(form.labels), tags: lines(form.tags), tile_ids: tileIds.length ? tileIds : null,
    }), setError)
    setBusy(false)
    if (result) { onCreated(); setForm((f) => ({ ...f, name: '', title: '', tile_ids: '' })) }
  }

  return (
    <div className="card" style={{ marginTop: 16 }}>
      <h2>New annotation session</h2>
      <div className="row" style={{ alignItems: 'flex-end' }}>
        <label className="field">Name<input type="text" value={form.name} placeholder="e.g. neun_fields_1" onChange={(e) => set('name', e.target.value.replace(/[^A-Za-z0-9_-]/g, '_'))} /></label>
        <label className="field" style={{ flex: 1 }}>Title shown to annotators<input type="text" value={form.title} onChange={(e) => set('title', e.target.value)} /></label>
        <label className="field">Stain<select value={form.stain} onChange={(e) => set('stain', e.target.value)}>{Object.keys(defaults.data?.labels ?? {}).map((s) => <option key={s}>{s}</option>)}</select></label>
        <label className="field">Fields<input type="number" min={1} value={form.n_fields} style={{ width: 80 }} onChange={(e) => set('n_fields', Number(e.target.value))} /></label>
        <label className="layer-toggle"><input type="checkbox" checked={form.per_group} onChange={() => set('per_group', !form.per_group)} />same number per group</label>
        <label className="field">Annotators per field<input type="number" min={1} value={form.per_field} style={{ width: 80 }} onChange={(e) => set('per_field', Number(e.target.value))} /></label>
      </div>
      <p className="muted small">New sessions start stopped: press Start when you want people to annotate. Fields are drawn at random (in shuffled order, so the diagnosis cannot be guessed from the order). 1 annotator per field shares the fields out; 2 or more gives independent repeat annotations.</p>
      <div className="grid grid-3" style={{ marginTop: 10 }}>
        <div>
          <div className="small secondary" style={{ marginBottom: 4 }}>What people can do</div>
          {(defaults.data?.tools ?? []).map((t) => (
            <label key={t} className="layer-toggle"><input type="checkbox" checked={form.tools.includes(t)}
              onChange={() => set('tools', form.tools.includes(t) ? form.tools.filter((x) => x !== t) : [...form.tools, t])} />{TOOL_TEXT[t]}</label>
          ))}
          <label className="field" style={{ marginTop: 8 }}>What people see of the AI
            <select value={form.ai_view} onChange={(e) => set('ai_view', e.target.value)}>{(defaults.data?.ai_views ?? []).map((v) => <option key={v} value={v}>{AI_VIEW_TEXT[v]}</option>)}</select>
          </label>
        </div>
        <label className="field">Labels for objects, points and outlines (one per line)<textarea rows={6} value={form.labels} onChange={(e) => set('labels', e.target.value)} /></label>
        <label className="field">Field tags (one per line)<textarea rows={6} value={form.tags} onChange={(e) => set('tags', e.target.value)} /></label>
      </div>
      <label className="field" style={{ marginTop: 10 }}>Instructions shown to annotators (optional)<textarea rows={2} value={form.instructions} onChange={(e) => set('instructions', e.target.value)} /></label>
      <label className="field" style={{ marginTop: 10 }}>Or annotate these exact fields (field IDs, optional)<textarea rows={2} value={form.tile_ids} placeholder="e.g. LIP-1_B-2_NeuN_S01, LIP-3_C-4_NeuN_S02" onChange={(e) => set('tile_ids', e.target.value)} /></label>
      <div className="row" style={{ marginTop: 10 }}>
        <button className="btn primary" disabled={!form.name || !form.tools.length || busy} onClick={create}>{busy ? 'Creating…' : 'Create session'}</button>
      </div>
      <ErrorNote error={error ?? defaults.error} />
    </div>
  )
}
