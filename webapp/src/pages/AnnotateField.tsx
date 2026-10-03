import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import type { AnnotationDetail, FieldObject, FieldWork, Mark } from '../api'
import { api, enc } from '../api'
import { Empty, ErrorNote, PageHead, Progress, fmt } from '../components'
import { useFetch, useStored } from '../hooks'
import { SignIn, SignedIn, useSession } from '../session'
import type { ImagePoint } from '../viewer'
import { CLASS_STYLE, RASTER_LAYERS, ZoomViewer } from '../viewer'
import { AI_VIEW_TEXT } from './Annotate'

const COLORS = ['#ff4fd8', '#ff8c1a', '#facc15', '#4ade80', '#a78bfa', '#f87171', '#e2e8f0', '#22d3ee', '#2dd4bf']
type Tool = 'object' | 'point' | 'polygon'
const TOOLS: { id: Tool; key: string; name: string; hint: string }[] = [
  { id: 'object', key: 'v', name: 'Label AI objects', hint: 'Click a circled AI object to give it the chosen label; click again with the same label to remove it.' },
  { id: 'point', key: 'p', name: 'Mark missed objects', hint: 'Click on an object the AI missed to put a point with the chosen label on it. Click a point to select it, then press a number to relabel or Delete to remove it.' },
  { id: 'polygon', key: 'o', name: 'Draw outlines', hint: 'Click around the object to place corners; click the first corner or press Enter to close. Backspace removes the last corner.' },
]

const newId = () => `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`
const objectId = (o: { x: number; y: number }) => `obj:${Math.round(o.x)}:${Math.round(o.y)}`
const typing = (e: KeyboardEvent) => ['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement).tagName)

function inside(polygon: number[][], x: number, y: number) {
  let hit = false
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i]
    const [xj, yj] = polygon[j]
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) hit = !hit
  }
  return hit
}

export default function AnnotateFieldPage() {
  const { session } = useSession()
  const { name = '', tile } = useParams()
  if (!session.annotator) return <><PageHead title="Annotate" /><SignIn /></>
  return tile ? <Workspace key={tile} name={name} tile={tile} /> : <NextField name={name} />
}

function NextField({ name }: { name: string }) {
  const navigate = useNavigate()
  const next = useFetch<{ tile_id: string | null }>(`/api/annotate/${name}/next`)
  useEffect(() => {
    if (next.data?.tile_id) navigate(`/annotate/${name}/${enc(next.data.tile_id)}`, { replace: true })
  }, [next.data, name, navigate])
  return (
    <>
      <PageHead title="Annotate" subtitle={<Link to="/annotate">← All annotation projects</Link>}><SignedIn /></PageHead>
      <ErrorNote error={next.error} />
      {next.data && !next.data.tile_id && <Empty>You have finished every field in this project. Thank you! You can still reopen a field from the list on any field page.</Empty>}
    </>
  )
}

function Workspace({ name, tile }: { name: string; tile: string }) {
  const navigate = useNavigate()
  const detail = useFetch<AnnotationDetail>(`/api/annotate/${name}`)
  const work = useFetch<FieldWork>(`/api/annotate/${name}/fields/${enc(tile)}`)
  const meta = detail.data?.meta
  const [marks, setMarks] = useState<Record<string, Mark>>({})
  const [tool, setTool] = useState<Tool | null>(null)
  const [labelIndex, setLabelIndex] = useState(0)
  const [selected, setSelected] = useState<string | null>(null)
  const [draft, setDraft] = useState<number[][]>([])
  const [hover, setHover] = useState<number[] | null>(null)
  const [note, setNote] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [layers, setLayers] = useStored<Record<string, boolean>>('stainid.annotate.layers', { ai: true, rejected: false, outlines: false, mine: true, box: true })

  useEffect(() => {
    if (!work.data) return
    setMarks(Object.fromEntries(work.data.mine.map((m) => [m.id, m])))
    setNote(work.data.mine.find((m) => m.id === 'note')?.note ?? '')
  }, [work.data])

  const objects = useMemo(() => work.data?.ai.objects ?? [], [work.data])
  const tools = useMemo(() => TOOLS.filter((t) => meta?.tools.includes(t.id) && (t.id !== 'object' || objects.length > 0)), [meta, objects])
  const active = tools.find((t) => t.id === tool) ?? tools[0]
  const labels = useMemo(() => meta?.labels ?? [], [meta])
  const label = labels[labelIndex] ?? labels[0]
  const color = (l: string) => COLORS[Math.max(0, labels.indexOf(l)) % COLORS.length]
  const full = meta?.ai_view === 'full'
  const shown = objects.filter((o) => layers.rejected || !full || o.model_class !== 'rejected')
  const hasRejected = full && objects.some((o) => o.model_class === 'rejected')
  const drawn = Object.values(marks).filter((m) => ['object', 'point', 'polygon'].includes(m.type))
  const done = !!marks.done
  const fields = detail.data?.fields ?? []
  const position = fields.findIndex((f) => f.tile_id === tile)

  const save = useCallback(async (records: Partial<Mark>[]) => {
    setSaving(true)
    try {
      const saved = await api.post<Mark[]>(`/api/annotate/${name}/fields/${enc(tile)}`, { records })
      setMarks((current) => {
        const next = { ...current }
        for (const r of saved) {
          if (r.deleted) delete next[r.id]
          else next[r.id] = r
        }
        return next
      })
      setError(null)
      return true
    } catch (e) {
      setError((e as Error).message)
      return false
    } finally {
      setSaving(false)
    }
  }, [name, tile])

  const remove = useCallback((m: Mark) => {
    setSelected(null)
    save([{ id: m.id, type: m.type, points: m.points, deleted: true }])
  }, [save])

  const closeDraft = useCallback(() => {
    if (draft.length < 3) return
    save([{ id: newId(), type: 'polygon', label, points: draft.map((p) => p.map(Math.round)) }])
    setDraft([])
    setHover(null)
  }, [draft, label, save])

  const nearestObject = (p: ImagePoint) => {
    let best: FieldObject | null = null
    let bestDistance = Infinity
    for (const o of shown) {
      const d = Math.hypot(o.x - p.x, o.y - p.y)
      if (d < Math.max(8, o.radius_px) + 4 + 8 * p.unit && d < bestDistance) { best = o; bestDistance = d }
    }
    return best
  }
  const hitMine = (p: ImagePoint) => drawn.find((m) => m.type === 'point' && Math.hypot(m.points[0][0] - p.x, m.points[0][1] - p.y) < 10 * p.unit)
    ?? drawn.find((m) => m.type === 'polygon' && inside(m.points, p.x, p.y))

  const click = (p: ImagePoint) => {
    if (!active || !label) return
    if (active.id === 'polygon' && draft.length) {
      if (draft.length >= 3 && Math.hypot(draft[0][0] - p.x, draft[0][1] - p.y) < 10 * p.unit) closeDraft()
      else setDraft([...draft, [p.x, p.y]])
      return
    }
    if (active.id === 'object') {
      const o = nearestObject(p)
      if (o) {
        const id = objectId(o)
        setSelected(null)
        if (marks[id]?.label === label) save([{ id, type: 'object', points: [[o.x, o.y]], deleted: true }])
        else save([{ id, type: 'object', label, points: [[o.x, o.y]] }])
        return
      }
    }
    const mine = hitMine(p)
    if (mine) { setSelected(mine.id); return }
    setSelected(null)
    if (active.id === 'point') {
      save([{ id: newId(), type: 'point', label, points: [[Math.round(p.x), Math.round(p.y)]] }])
    }
    if (active.id === 'polygon') setDraft([[p.x, p.y]])
  }

  const finish = async () => {
    if (await save([{ id: 'done', type: 'done' }])) navigate(`/annotate/${name}`)
  }
  const skip = () => {
    const later = [...fields.slice(position + 1), ...fields.slice(0, position)].find((f) => !f.done_by_me)
    if (later) navigate(`/annotate/${name}/${enc(later.tile_id)}`)
  }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (typing(e) || e.metaKey || e.ctrlKey) return
      const n = Number(e.key)
      if (n >= 1 && n <= Math.min(9, labels.length)) {
        setLabelIndex(n - 1)
        const m = selected ? marks[selected] : undefined
        if (m && ['object', 'point', 'polygon'].includes(m.type) && m.label !== labels[n - 1]) save([{ ...m, label: labels[n - 1] }])
        return
      }
      const chosen = tools.find((t) => t.key === e.key.toLowerCase())
      if (chosen) { setTool(chosen.id); setDraft([]); return }
      if (e.key === 'Escape') { setDraft([]); setSelected(null) }
      if (e.key === 'Enter') closeDraft()
      if (e.key === 'Backspace' || e.key === 'Delete') {
        e.preventDefault()
        if (draft.length) setDraft(draft.slice(0, -1))
        else if (selected && marks[selected]) remove(marks[selected])
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [labels, selected, marks, save, tools, draft, closeDraft, remove])

  const selectedMark = selected ? marks[selected] : undefined
  const selectedObject = selected?.startsWith('obj:') ? objects.find((o) => objectId(o) === selected) : undefined
  const counts = (l: string) => drawn.filter((m) => m.label === l).length
  const toggle = (key: string) => setLayers({ ...layers, [key]: !layers[key] })

  return (
    <>
      <PageHead title={meta?.title ?? name} subtitle={<span><Link to="/annotate">← All annotation projects</Link> · {meta?.stain} · field {position + 1} of {fields.length}
        {meta && ` · ${AI_VIEW_TEXT[meta.ai_view]}`}</span>}><SignedIn /></PageHead>
      <ErrorNote error={detail.error ?? work.error ?? error} />
      {meta?.instructions && <div className="instructions">{meta.instructions}</div>}
      {work.data && meta && (
        <div className="grid" style={{ gridTemplateColumns: 'minmax(0, 1fr) 330px', alignItems: 'start' }}>
          <div>
            <ZoomViewer src={`/api/annotate/${name}/fields/${enc(tile)}/image.jpg`}
              layers={work.data.ai.layers.filter((l) => layers[l]).map((l) => `/api/annotate/${name}/fields/${enc(tile)}/layer/${l}.png`)}
              title={<span className="row"><h3 style={{ margin: 0 }}>{tile}</h3>{done && <span className="badge status-done">finished</span>}{saving && <span className="muted small">saving…</span>}</span>}
              cursor={active?.id === 'object' ? 'pointer' : 'crosshair'} onClick={click} onMove={(p) => { if (draft.length) setHover([p.x, p.y]) }}>
              {(_zoom, _size, unit) => <>
                {layers.box && <rect x={work.data!.box.x} y={work.data!.box.y} width={work.data!.box.width} height={work.data!.box.height} fill="none" stroke="#ffd21f" strokeWidth={2 * unit} strokeDasharray={`${12 * unit} ${8 * unit}`} />}
                {layers.outlines && work.data!.ai.outlines.map((o) => (
                  <polygon key={o.label} points={o.points.map((p) => p.join(',')).join(' ')} fill="none" stroke="#ffffff" strokeOpacity={0.8} strokeWidth={1.5 * unit} />
                ))}
                {layers.ai && shown.map((o) => {
                  const id = objectId(o)
                  const mine = layers.mine ? marks[id] : undefined
                  const stroke = mine ? color(mine.label) : full ? CLASS_STYLE[o.model_class]?.color ?? '#fff' : '#ffffff'
                  return (
                    <circle key={id} cx={o.x} cy={o.y} r={Math.max(8, o.radius_px) + 4} fill={mine ? stroke : 'none'} fillOpacity={0.22} stroke={stroke}
                      strokeWidth={(id === selected ? 3.5 : 2) * unit} strokeDasharray={mine ? undefined : `${5 * unit} ${3 * unit}`}>
                      <title>{[full && `AI: ${CLASS_STYLE[o.model_class]?.label ?? o.model_class} (p ${fmt(o.probability, 2)})`, mine && `you: ${mine.label}`].filter(Boolean).join(' · ') || 'AI detection'}</title>
                    </circle>
                  )
                })}
                {layers.mine && drawn.filter((m) => m.type === 'polygon').map((m) => (
                  <polygon key={m.id} points={m.points.map((p) => p.join(',')).join(' ')} fill={color(m.label)} fillOpacity={0.2} stroke={m.id === selected ? '#fff' : color(m.label)}
                    strokeWidth={(m.id === selected ? 3 : 2) * unit} />
                ))}
                {layers.mine && drawn.filter((m) => m.type === 'point').map((m) => (
                  <circle key={m.id} cx={m.points[0][0]} cy={m.points[0][1]} r={6 * unit} fill={color(m.label)} stroke={m.id === selected ? '#fff' : '#000'} strokeWidth={(m.id === selected ? 2.5 : 1.2) * unit} />
                ))}
                {draft.length > 0 && <>
                  <polyline points={[...draft, ...(hover ? [hover] : [])].map((p) => p.join(',')).join(' ')} fill="none" stroke={color(label)} strokeWidth={2 * unit} />
                  {draft.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r={(i === 0 ? 6 : 3.5) * unit} fill={i === 0 ? '#fff' : color(label)} stroke="#000" strokeWidth={unit} />)}
                </>}
              </>}
            </ZoomViewer>
            <p className="muted small">Scroll to zoom, drag to pan. The dashed yellow square is the field to annotate; the margin is context. Keys: <span className="kbd">1</span>–<span className="kbd">{Math.min(9, labels.length)}</span> label,
              {tools.map((t) => <span key={t.id}> <span className="kbd">{t.key.toUpperCase()}</span> {t.name.toLowerCase()},</span>)} <span className="kbd">Delete</span> remove selected, <span className="kbd">Esc</span> cancel.</p>
          </div>
          <div className="grid">
            <div className="card">
              <h3>Tool</h3>
              <div className="grid" style={{ gap: 6 }}>
                {tools.map((t) => (
                  <button key={t.id} className={`btn${active?.id === t.id ? ' primary' : ''}`} style={{ textAlign: 'left' }} onClick={() => { setTool(t.id); setDraft([]) }}>
                    <span className="kbd">{t.key.toUpperCase()}</span> {t.name}
                  </button>
                ))}
              </div>
              {active && <p className="muted small">{active.hint}</p>}
              {tools.length > 0 && <>
                <h3 style={{ marginTop: 12 }}>Label</h3>
                <div className="grid" style={{ gap: 4 }}>
                  {labels.map((l, i) => (
                    <button key={l} className={`btn small${l === label ? ' primary' : ''}`} style={{ textAlign: 'left', display: 'flex', gap: 6, alignItems: 'center' }}
                      onClick={() => { setLabelIndex(i); if (selectedMark && selectedMark.label !== l && selectedMark.type !== 'tag') save([{ ...selectedMark, label: l }]) }}>
                      {i < 9 && <span className="kbd">{i + 1}</span>}<span className="swatch" style={{ background: color(l) }} />{l}<span className="spacer" /><span className="muted">{counts(l) || ''}</span>
                    </button>
                  ))}
                </div>
              </>}
              {!tools.length && <p className="muted small">This project only asks for field tags (below).</p>}
            </div>
            {(selectedMark || selectedObject) && (
              <div className="card">
                <h3>Selected</h3>
                {selectedObject && full && <p className="small">AI: {CLASS_STYLE[selectedObject.model_class]?.label ?? selectedObject.model_class} · p {fmt(selectedObject.probability, 2)} · {fmt(selectedObject.area_um2, 3)} µm²</p>}
                {selectedObject && !full && <p className="small muted">AI detection</p>}
                {selectedMark ? <>
                  <p className="small">Your label: <b>{selectedMark.label}</b> ({selectedMark.type === 'object' ? 'AI object' : selectedMark.type})</p>
                  <p className="muted small">Press a number or click a label to change it.</p>
                  <button className="btn small" onClick={() => remove(selectedMark)}>Remove</button>
                </> : <p className="muted small">Not labelled by you yet.</p>}
              </div>
            )}
            <div className="card">
              <h3>This field</h3>
              {meta.tools.includes('tag') && meta.tags.map((t) => (
                <label key={t} className="layer-toggle"><input type="checkbox" checked={!!marks[`tag:${t}`]}
                  onChange={() => save([{ id: `tag:${t}`, type: 'tag', label: t, deleted: !!marks[`tag:${t}`] }])} />{t}</label>
              ))}
              <label className="field" style={{ marginTop: 8 }}>Note (optional)
                <textarea rows={2} value={note} onChange={(e) => setNote(e.target.value)}
                  onBlur={() => { if (note !== (marks.note?.note ?? '')) save([{ id: 'note', type: 'note', note, deleted: !note }]) }} />
              </label>
              <p className="muted small">{drawn.length} annotation{drawn.length === 1 ? '' : 's'} by you on this field. Everything is saved as you go.</p>
              <div className="row">
                {done
                  ? <button className="btn" onClick={() => save([{ id: 'done', type: 'done', deleted: true }])}>Reopen field</button>
                  : <button className="btn primary" disabled={draft.length > 0} onClick={finish}>Finished — next field</button>}
                <button className="btn" onClick={skip}>Skip for now</button>
              </div>
            </div>
            <div className="card">
              <h3>Show</h3>
              {objects.length > 0 && <label className="layer-toggle"><input type="checkbox" checked={!!layers.ai} onChange={() => toggle('ai')} />AI detections ({shown.length})</label>}
              {full && layers.ai && Array.from(new Set(shown.map((o) => o.model_class))).map((c) => (
                <div key={c} className="layer-toggle" style={{ paddingLeft: 22 }}><span className="swatch" style={{ background: CLASS_STYLE[c]?.color ?? '#999' }} />{CLASS_STYLE[c]?.label ?? c}</div>
              ))}
              {hasRejected && <label className="layer-toggle" style={{ paddingLeft: 22 }}><input type="checkbox" checked={!!layers.rejected} onChange={() => toggle('rejected')} />candidates the AI rejected</label>}
              {work.data.ai.outlines.length > 0 && <label className="layer-toggle"><input type="checkbox" checked={!!layers.outlines} onChange={() => toggle('outlines')} />AI outlines ({work.data.ai.outlines.length})</label>}
              {work.data.ai.layers.map((l) => (
                <label key={l} className="layer-toggle"><input type="checkbox" checked={!!layers[l]} onChange={() => toggle(l)} /><span className="swatch" style={{ background: RASTER_LAYERS[l].color }} />{RASTER_LAYERS[l].label}</label>
              ))}
              <label className="layer-toggle"><input type="checkbox" checked={!!layers.mine} onChange={() => toggle('mine')} />my annotations</label>
              <label className="layer-toggle"><input type="checkbox" checked={!!layers.box} onChange={() => toggle('box')} />field boundary</label>
            </div>
            <div className="card">
              <h3>Fields</h3>
              <Progress done={fields.filter((f) => f.done_by_me).length} total={fields.length} />
              <p className="muted small">{fields.filter((f) => f.done_by_me).length} of {fields.length} finished by you. Green = finished by you; a dot = someone else has it open.</p>
              <div className="row" style={{ gap: 4 }}>
                {fields.map((f, i) => (
                  <Link key={f.tile_id} to={`/annotate/${name}/${enc(f.tile_id)}`} title={`${f.tile_id} · finished by ${f.annotators_done} annotator(s)`}
                    className="btn small" style={{ minWidth: 34, textAlign: 'center', position: 'relative',
                      background: f.tile_id === tile ? 'var(--accent)' : f.done_by_me ? 'color-mix(in srgb, var(--good) 25%, var(--surface-2))' : undefined,
                      color: f.tile_id === tile ? '#fff' : undefined }}>
                    {i + 1}{f.open_by_others > 0 && <span style={{ position: 'absolute', top: 2, right: 3, width: 6, height: 6, borderRadius: 3, background: 'var(--warning)' }} />}
                  </Link>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
