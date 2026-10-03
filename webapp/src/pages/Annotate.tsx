import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import type { AiView, AnnotationListItem, AnnotationSummary } from '../api'
import { api } from '../api'
import { Empty, ErrorNote, HelpBox, PageHead, Progress, withError } from '../components'
import { useFetch } from '../hooks'
import { SignIn, SignedIn, useSession } from '../session'

export const AI_VIEW_TEXT: Record<AiView, string> = {
  full: 'AI detections, classes and overlays shown',
  positions: 'AI positions only (blind to what the AI decided)',
  none: 'no AI shown',
}
const TOOL_TEXT: Record<string, string> = {
  object: 'label the AI detections', point: 'mark objects the AI missed', polygon: 'draw outlines', tag: 'tag the whole field',
}

export default function Annotate() {
  const { session } = useSession()
  const admin = session.mode === 'full'
  const list = useFetch<AnnotationListItem[]>(admin || session.annotator ? '/api/annotate' : null)
  const [open, setOpen] = useState<string | null>(null)

  return (
    <>
      <PageHead title="Annotate together" subtitle={admin ? 'Several people annotate whole fields, each in their own layer, and can see what the AI found.' : `Project ${session.project}`}>
        <SignedIn />
      </PageHead>
      {admin && (
        <HelpBox id="annotate">
          <ul>
            <li>Create an <b>annotation project</b> below: a set of fields, the labels people may use, and what they see of the AI.</li>
            <li>Each person signs in with their name; every annotation is saved under that name, so nobody overwrites anyone else.
              With 1 annotator per field the fields are shared out; with 2 or more, each field is annotated independently by several people (for agreement).</li>
            <li>To let others in, run <code>stainid serve --annotate --access-code &lt;choose a code&gt;</code> in the project folder. It starts a second,
              annotation-only server on port 8766 that people on your network can open (it prints the address). Running steps, settings and files stay private.</li>
            <li>For people outside your network, put the annotation server behind a tunnel, e.g. <code>cloudflared tunnel --url http://localhost:8766</code>.</li>
          </ul>
        </HelpBox>
      )}
      {!session.annotator && <div style={{ marginBottom: 16 }}><SignIn /></div>}
      <ErrorNote error={list.error} />
      {list.data && (
        <div className="card">
          <h2>Annotation projects</h2>
          {list.data.length ? (
            <table>
              <thead><tr><th>Project</th><th>Stain</th><th>Tools</th><th>AI</th><th style={{ width: 200 }}>Fields covered</th>{session.annotator && <th>Done by you</th>}<th /></tr></thead>
              <tbody>
                {list.data.map((p) => (
                  <tr key={p.name}>
                    <td>{p.title}<div className="muted small">{p.name} · {p.per_field} annotator{p.per_field > 1 ? 's' : ''} per field</div></td>
                    <td>{p.stain}</td>
                    <td className="small secondary wrap">{p.tools.map((t) => TOOL_TEXT[t]).join(', ')}</td>
                    <td className="small secondary wrap">{AI_VIEW_TEXT[p.ai_view]}</td>
                    <td><Progress done={p.fields_covered} total={p.fields} /><span className="muted small">{p.fields_covered} of {p.fields}</span></td>
                    {session.annotator && <td className="num">{p.done_by_me}</td>}
                    <td className="row" style={{ flexWrap: 'nowrap' }}>
                      {session.annotator && <Link className="btn small primary" to={`/annotate/${p.name}`}>{p.done_by_me ? 'Continue' : 'Start'}</Link>}
                      {admin && <button className="btn small" onClick={() => setOpen(open === p.name ? null : p.name)}>{open === p.name ? 'Hide results' : 'Results'}</button>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <Empty>{admin ? 'No annotation projects yet — create one below.' : 'No annotation projects yet.'}</Empty>}
        </div>
      )}
      {admin && open && <Results key={open} name={open} />}
      {admin && <NewProject onCreated={list.reload} />}
    </>
  )
}

function Results({ name }: { name: string }) {
  const summary = useFetch<AnnotationSummary>(`/api/annotate/${name}/summary`)
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

function NewProject({ onCreated }: { onCreated: () => void }) {
  const navigate = useNavigate()
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
    if (result) { onCreated(); navigate(`/annotate/${result.name}`) }
  }

  return (
    <div className="card" style={{ marginTop: 16 }}>
      <h2>New annotation project</h2>
      <div className="row" style={{ alignItems: 'flex-end' }}>
        <label className="field">Name<input type="text" value={form.name} placeholder="e.g. neun_fields_1" onChange={(e) => set('name', e.target.value.replace(/[^A-Za-z0-9_-]/g, '_'))} /></label>
        <label className="field" style={{ flex: 1 }}>Title shown to annotators<input type="text" value={form.title} onChange={(e) => set('title', e.target.value)} /></label>
        <label className="field">Stain<select value={form.stain} onChange={(e) => set('stain', e.target.value)}>{Object.keys(defaults.data?.labels ?? {}).map((s) => <option key={s}>{s}</option>)}</select></label>
        <label className="field">Fields<input type="number" min={1} value={form.n_fields} style={{ width: 80 }} onChange={(e) => set('n_fields', Number(e.target.value))} /></label>
        <label className="layer-toggle"><input type="checkbox" checked={form.per_group} onChange={() => set('per_group', !form.per_group)} />same number per group</label>
        <label className="field">Annotators per field<input type="number" min={1} value={form.per_field} style={{ width: 80 }} onChange={(e) => set('per_field', Number(e.target.value))} /></label>
      </div>
      <p className="muted small">Fields are drawn at random (in shuffled order, so the diagnosis cannot be guessed from the order). 1 annotator per field shares the fields out; 2 or more gives independent repeat annotations.</p>
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
        <button className="btn primary" disabled={!form.name || !form.tools.length || busy} onClick={create}>{busy ? 'Creating…' : 'Create annotation project'}</button>
      </div>
      <ErrorNote error={error ?? defaults.error} />
    </div>
  )
}
