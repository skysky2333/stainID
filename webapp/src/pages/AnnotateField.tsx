import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import type { FieldObjectLite, FieldWork, Label, QueueItem, SessionDetail } from '../api'
import { api, apiUrl, enc } from '../api'
import { Empty, ErrorNote, PageHead, Progress, fmt } from '../components'
import { useFetch, useStored } from '../hooks'
import { Register, useSession } from '../session'
import type { ImagePoint } from '../viewer'
import { RASTER_LAYERS, ZoomViewer } from '../viewer'
import { AI_VIEW_TEXT } from './Labels'

const COLORS = ['#ff4fd8', '#ff8c1a', '#facc15', '#4ade80', '#a78bfa', '#f87171', '#e2e8f0', '#22d3ee', '#2dd4bf']
type Tool = 'label' | 'add' | 'outline'
const TOOLS: { id: Tool; needs: string; key: string; name: string; hint: string }[] = [
  { id: 'label', needs: 'class', key: 'v', name: 'Label objects', hint: 'Click an object to give it the chosen class. Dashed = the AI’s call, not yet checked by you; solid = your decision. Click again with the same class to go back to the AI’s call.' },
  { id: 'add', needs: 'point', key: 'a', name: 'Add missed objects', hint: 'Click on an object the AI did not circle to add it with the chosen class. Click one of your added objects to select it; Delete removes it.' },
  { id: 'outline', needs: 'outline', key: 'o', name: 'Fix outlines', hint: 'Click an object to select it, then drag the corners of its outline or press Redraw. Clicking empty tissue starts a new outline (a new object): click its corners, then the first corner or Enter.' },
]

const rkey = (x: number, y: number) => `${Math.round(x)}:${Math.round(y)}`
const typing = (e: KeyboardEvent) => ['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement).tagName)
const NEGATIVE = /^not a|artifact|^unsure/

function inside(polygon: number[][], x: number, y: number) {
  let hit = false
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i]
    const [xj, yj] = polygon[j]
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) hit = !hit
  }
  return hit
}

export default function AnnotatePage() {
  const { session } = useSession()
  const { name = '', tile } = useParams()
  const detail = useFetch<SessionDetail>(session.annotator ? `/api/annotate/${name}` : null)
  if (!session.annotator) return <><PageHead title="Label tissue images" /><Register /></>
  if (detail.error) return <><PageHead title="Label" subtitle={<Link to="/annotate">← All sessions</Link>} /><ErrorNote error={detail.error} /></>
  if (!detail.data) return null
  if (detail.data.meta.kind === 'queue') return <Queue detail={detail.data} />
  return tile ? <Workspace key={tile} detail={detail.data} reload={detail.reload} tile={tile} /> : <NextField name={name} />
}

function NextField({ name }: { name: string }) {
  const navigate = useNavigate()
  const next = useFetch<{ tile_id: string | null; waiting: number }>(`/api/annotate/${name}/next`)
  useEffect(() => {
    if (next.data?.tile_id) navigate(`/annotate/${name}/${enc(next.data.tile_id)}`, { replace: true })
  }, [next.data, name, navigate])
  return (
    <>
      <PageHead title="Label" subtitle={<Link to="/annotate">← All sessions</Link>} />
      <ErrorNote error={next.error} />
      {next.data && !next.data.tile_id && (next.data.waiting
        ? <Empty>{next.data.waiting} more fields are still being prepared. <button className="link-btn" onClick={next.reload}>Check again</button></Empty>
        : <Empty>You have finished every field in this session. Thank you!</Empty>)}
    </>
  )
}

function Workspace({ detail, reload, tile }: { detail: SessionDetail; reload: () => void; tile: string }) {
  const navigate = useNavigate()
  const meta = detail.meta
  const name = meta.name
  const work = useFetch<FieldWork>(`/api/annotate/${name}/fields/${enc(tile)}`)
  const [mine, setMine] = useState<Record<string, Label>>({})
  const [tool, setTool] = useState<Tool | null>(null)
  const [labelIndex, setLabelIndex] = useState(0)
  const [selected, setSelected] = useState<string | null>(null)
  const [draft, setDraft] = useState<number[][]>([])
  const [redrawing, setRedrawing] = useState(false)
  const [hover, setHover] = useState<number[] | null>(null)
  const [dragging, setDragging] = useState<{ index: number; points: number[][] } | null>(null)
  const [note, setNote] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [layers, setLayers] = useStored<Record<string, boolean>>('stainid.label.layers', { negatives: true, outlines: true, mine: true, box: true })

  useEffect(() => {
    if (!work.data) return
    setMine(Object.fromEntries(work.data.mine.map((m) => [m.id, m])))
    setNote(work.data.mine.find((m) => m.task === 'note')?.value ?? '')
  }, [work.data])

  const classes = meta.classes
  const label = classes[labelIndex] ?? classes[0]
  const color = (v: string | undefined) => (v ? COLORS[Math.max(0, classes.indexOf(v)) % COLORS.length] : '#ffffff')
  const full = meta.ai_view === 'full'
  const tools = useMemo(() => TOOLS.filter((t) => meta.tools.includes(t.needs)), [meta.tools])
  const active = tools.find((t) => t.id === tool) ?? tools[0]
  const objects = useMemo(() => work.data?.objects ?? [], [work.data])
  const objectKeys = useMemo(() => new Set(objects.map((o) => rkey(o.x, o.y))), [objects])
  const aiOutlines = useMemo(() => Object.fromEntries((work.data?.outlines ?? []).map((o) => [rkey(o.x, o.y), o.points])), [work.data])
  const myClass = useCallback((x: number, y: number) => mine[`class:${tile}:${rkey(x, y)}`], [mine, tile])
  const myOutline = useCallback((x: number, y: number) => mine[`outline:${tile}:${rkey(x, y)}`], [mine, tile])
  const value = (o: FieldObjectLite) => myClass(o.x, o.y)?.value ?? (full ? o.ai_value : undefined)
  const added = Object.values(mine).filter((m) => m.task === 'class' && !objectKeys.has(rkey(m.x!, m.y!)))
  const done = !!mine[`done:${tile}`]
  const fields = detail.fields
  const position = fields.findIndex((f) => f.tile_id === tile)
  const all = [...objects.map((o) => ({ x: o.x, y: o.y, r: o.radius_px })), ...added.map((m) => ({ x: m.x!, y: m.y!, r: 8 }))]
  const selectedObject = selected ? all.find((o) => rkey(o.x, o.y) === selected) : undefined
  const outlineOf = (x: number, y: number) => myOutline(x, y)?.points ?? aiOutlines[rkey(x, y)]

  const save = useCallback(async (records: Partial<Label>[]) => {
    setSaving(true)
    try {
      const saved = await api.post<Label[]>(`/api/annotate/${name}/labels`, { records: records.map((r) => ({ tile_id: tile, ...r })) })
      setMine((current) => {
        const next = { ...current }
        for (const r of saved) {
          if (r.deleted) delete next[r.id]
          else next[r.id] = r
        }
        return next
      })
      setError(null)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setSaving(false)
    }
  }, [name, tile])

  const closeDraft = useCallback(() => {
    if (draft.length < 3) return
    const points = draft.map((p) => p.map(Math.round))
    const centre = { x: Math.round(points.reduce((a, p) => a + p[0], 0) / points.length), y: Math.round(points.reduce((a, p) => a + p[1], 0) / points.length) }
    const target = selectedObject ?? centre
    const records: Partial<Label>[] = [{ task: 'outline', x: target.x, y: target.y, points }]
    if (!selectedObject) records.push({ task: 'class', x: target.x, y: target.y, value: label })
    save(records)
    setSelected(rkey(target.x, target.y))
    setDraft([])
    setHover(null)
  }, [draft, selectedObject, label, save])

  const hit = (p: ImagePoint) => {
    let best: { x: number; y: number; r: number } | null = null
    let distance = Infinity
    for (const o of all) {
      const d = Math.hypot(o.x - p.x, o.y - p.y)
      if (d < Math.max(8, o.r) + 4 + 6 * p.unit && d < distance) { best = o; distance = d }
    }
    if (best) return best
    return all.find((o) => { const poly = outlineOf(o.x, o.y); return poly && inside(poly, p.x, p.y) }) ?? null
  }

  const click = (p: ImagePoint) => {
    if (!active) return
    if (draft.length || redrawing) {
      if (draft.length >= 3 && Math.hypot(draft[0][0] - p.x, draft[0][1] - p.y) < 10 * p.unit) closeDraft()
      else setDraft([...draft, [p.x, p.y]])
      setRedrawing(false)
      return
    }
    const o = hit(p)
    if (active.id === 'label') {
      if (!o) return
      setSelected(rkey(o.x, o.y))
      if (myClass(o.x, o.y)?.value === label) save([{ task: 'class', x: o.x, y: o.y, deleted: true }])
      else save([{ task: 'class', x: o.x, y: o.y, value: label }])
      return
    }
    if (active.id === 'add') {
      if (o) { setSelected(rkey(o.x, o.y)); return }
      save([{ task: 'class', x: Math.round(p.x), y: Math.round(p.y), value: label }])
      setSelected(rkey(p.x, p.y))
      return
    }
    if (o) { setSelected(rkey(o.x, o.y)); return }
    setSelected(null)
    setDraft([[p.x, p.y]])
  }

  const move = (p: ImagePoint) => {
    if (dragging) setDragging({ ...dragging, points: dragging.points.map((q, i) => (i === dragging.index ? [p.x, p.y] : q)) })
    else if (draft.length) setHover([p.x, p.y])
  }

  useEffect(() => {
    if (!dragging || !selectedObject) return
    const finish = () => {
      save([{ task: 'outline', x: selectedObject.x, y: selectedObject.y, points: dragging.points.map((q) => q.map(Math.round)) }])
      setDragging(null)
    }
    window.addEventListener('mouseup', finish)
    return () => window.removeEventListener('mouseup', finish)
  }, [dragging, selectedObject, save])

  const removeSelected = useCallback(() => {
    if (!selectedObject) return
    const { x, y } = selectedObject
    const records: Partial<Label>[] = []
    if (myOutline(x, y)) records.push({ task: 'outline', x, y, deleted: true })
    if (!objectKeys.has(rkey(x, y)) && myClass(x, y)) records.push({ task: 'class', x, y, deleted: true })
    if (records.length) save(records)
    setSelected(null)
  }, [selectedObject, myOutline, myClass, objectKeys, save])

  const finishField = async () => {
    setSaving(true)
    try {
      await api.post(`/api/annotate/${name}/fields/${enc(tile)}/finish`, {})
      reload()
      navigate(`/annotate/${name}`)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setSaving(false)
    }
  }
  const skip = () => {
    const later = [...fields.slice(position + 1), ...fields.slice(0, position)].find((f) => !f.done_by_me && f.prepared)
    if (later) navigate(`/annotate/${name}/${enc(later.tile_id)}`)
  }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (typing(e) || e.metaKey || e.ctrlKey) return
      const n = Number(e.key)
      if (n >= 1 && n <= Math.min(9, classes.length)) { setLabelIndex(n - 1); return }
      const chosen = tools.find((t) => t.key === e.key.toLowerCase())
      if (chosen) { setTool(chosen.id); setDraft([]); return }
      if (e.key === 'Escape') { setDraft([]); setRedrawing(false); setSelected(null) }
      if (e.key === 'Enter') closeDraft()
      if (e.key === 'Backspace' || e.key === 'Delete') {
        e.preventDefault()
        if (draft.length) setDraft(draft.slice(0, -1))
        else removeSelected()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [classes, tools, draft, closeDraft, removeSelected])

  const counts: Record<string, number> = {}
  for (const o of objects) { const v = value(o); if (v) counts[v] = (counts[v] ?? 0) + 1 }
  for (const m of added) counts[m.value] = (counts[m.value] ?? 0) + 1
  const changed = Object.values(mine).filter((m) => m.task === 'class' && !m.confirmed).length
  const toggle = (key: string) => setLayers({ ...layers, [key]: !layers[key] })
  const selectedAi = selected ? objects.find((o) => rkey(o.x, o.y) === selected) : undefined
  const selectedValue = selectedObject ? (myClass(selectedObject.x, selectedObject.y)?.value ?? selectedAi?.ai_value) : undefined
  const editing = active?.id === 'outline' && selectedObject && !draft.length ? (dragging?.points ?? outlineOf(selectedObject.x, selectedObject.y)) : undefined

  return (
    <>
      <PageHead title={meta.title} subtitle={<span><Link to="/annotate">← All sessions</Link> · {meta.stain} · field {position + 1} of {fields.length} · {AI_VIEW_TEXT[meta.ai_view]}</span>} />
      <ErrorNote error={work.error ?? error} />
      {meta.instructions && <div className="instructions">{meta.instructions}</div>}
      {work.data && (
        <div className="grid" style={{ gridTemplateColumns: 'minmax(0, 1fr) 340px', alignItems: 'start' }}>
          <div>
            <ZoomViewer src={apiUrl(`/api/annotate/${name}/fields/${enc(tile)}/image.jpg`)}
              layers={work.data.layers.filter((l) => layers[l]).map((l) => apiUrl(`/api/annotate/${name}/fields/${enc(tile)}/layer/${l}.png`))}
              title={<span className="row"><h3 style={{ margin: 0 }}>{tile}</h3>{done && <span className="badge status-done">finished</span>}{saving && <span className="muted small">saving…</span>}</span>}
              cursor={active?.id === 'label' ? 'pointer' : 'crosshair'} onClick={click} onMove={move}>
              {(_zoom, _size, unit) => <>
                {layers.box && <rect x={work.data!.box.x} y={work.data!.box.y} width={work.data!.box.width} height={work.data!.box.height} fill="none" stroke="#ffd21f" strokeWidth={2 * unit} strokeDasharray={`${12 * unit} ${8 * unit}`} />}
                {layers.outlines && all.map((o) => {
                  const k = rkey(o.x, o.y)
                  const v = myClass(o.x, o.y)?.value ?? objects.find((q) => rkey(q.x, q.y) === k)?.ai_value
                  const poly = k === selected && editing ? undefined : outlineOf(o.x, o.y)
                  if (!poly || (v && NEGATIVE.test(v) && !myOutline(o.x, o.y))) return null
                  return <polygon key={`o${k}`} points={poly.map((p) => p.join(',')).join(' ')} fill="none" stroke={myOutline(o.x, o.y) ? color(v) : '#ffffff'}
                    strokeOpacity={0.9} strokeWidth={(myOutline(o.x, o.y) ? 2 : 1.2) * unit} />
                })}
                {objects.map((o) => {
                  const v = value(o)
                  const decided = !!myClass(o.x, o.y)
                  const negative = !v || NEGATIVE.test(v)
                  if (negative && !decided && !layers.negatives) return null
                  const k = rkey(o.x, o.y)
                  return (
                    <circle key={k} cx={o.x} cy={o.y} r={Math.max(8, o.radius_px) + 4} fill={decided ? color(v) : 'none'} fillOpacity={0.18}
                      stroke={v ? color(v) : '#ffffff'} strokeOpacity={negative && !decided ? 0.55 : 1}
                      strokeWidth={(k === selected ? 3.5 : negative && !decided ? 1.2 : 2) * unit} strokeDasharray={decided ? undefined : `${5 * unit} ${3 * unit}`}>
                      <title>{[full && o.ai_value && `AI: ${o.ai_value} (p ${fmt(o.ai_probability, 2)})`, decided && `you: ${v}`].filter(Boolean).join(' · ') || 'candidate'}</title>
                    </circle>
                  )
                })}
                {layers.mine && added.map((m) => (
                  <circle key={m.id} cx={m.x!} cy={m.y!} r={6 * unit} fill={color(m.value)} stroke={rkey(m.x!, m.y!) === selected ? '#fff' : '#000'} strokeWidth={(rkey(m.x!, m.y!) === selected ? 2.5 : 1.2) * unit} />
                ))}
                {editing && <>
                  <polygon points={editing.map((p) => p.join(',')).join(' ')} fill={color(selectedValue)} fillOpacity={0.15} stroke="#fff" strokeWidth={2 * unit} />
                  {editing.map((p, i) => (
                    <circle key={i} cx={p[0]} cy={p[1]} r={5 * unit} fill="#fff" stroke="#000" strokeWidth={unit} style={{ cursor: 'move' }}
                      onMouseDown={(e) => { e.stopPropagation(); setDragging({ index: i, points: editing }) }} />
                  ))}
                </>}
                {draft.length > 0 && <>
                  <polyline points={[...draft, ...(hover ? [hover] : [])].map((p) => p.join(',')).join(' ')} fill="none" stroke={color(selectedValue ?? label)} strokeWidth={2 * unit} />
                  {draft.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r={(i === 0 ? 6 : 3.5) * unit} fill={i === 0 ? '#fff' : color(label)} stroke="#000" strokeWidth={unit} />)}
                </>}
              </>}
            </ZoomViewer>
            <p className="muted small">Scroll to zoom, drag to pan. Keys: <span className="kbd">1</span>–<span className="kbd">{Math.min(9, classes.length)}</span> choose class,
              {tools.map((t) => <span key={t.id}> <span className="kbd">{t.key.toUpperCase()}</span> {t.name.toLowerCase()},</span>)} <span className="kbd">Delete</span> remove, <span className="kbd">Esc</span> cancel.</p>
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
              {active && <p className="muted small">{redrawing ? 'Click the corners of the new outline, then the first corner or Enter.' : active.hint}</p>}
              <h3 style={{ marginTop: 12 }}>Class</h3>
              <div className="grid" style={{ gap: 4 }}>
                {classes.map((c, i) => (
                  <button key={c} className={`btn small${c === label ? ' primary' : ''}`} style={{ textAlign: 'left', display: 'flex', gap: 6, alignItems: 'center' }} onClick={() => setLabelIndex(i)}>
                    {i < 9 && <span className="kbd">{i + 1}</span>}<span className="swatch" style={{ background: color(c) }} />{c}<span className="spacer" /><span className="muted">{counts[c] ?? ''}</span>
                  </button>
                ))}
              </div>
              <p className="muted small">{meta.class_instructions}</p>
            </div>
            {selectedObject && (
              <div className="card">
                <h3>Selected object</h3>
                {selectedAi?.ai_value && <p className="small">AI: {selectedAi.ai_value} · p {fmt(selectedAi.ai_probability, 2)}</p>}
                <p className="small">{myClass(selectedObject.x, selectedObject.y) ? <>Your class: <b>{myClass(selectedObject.x, selectedObject.y)!.value}</b></> : 'Not changed by you'}
                  {!objectKeys.has(selected!) && ' (added by you)'}</p>
                <div className="row">
                  {myClass(selectedObject.x, selectedObject.y) && objectKeys.has(selected!) &&
                    <button className="btn small" onClick={() => save([{ task: 'class', x: selectedObject.x, y: selectedObject.y, deleted: true }])}>Back to AI call</button>}
                  {meta.tools.includes('outline') && active?.id !== 'outline' && <button className="btn small" onClick={() => setTool('outline')}>Edit outline</button>}
                  {meta.tools.includes('outline') && active?.id === 'outline' && <button className="btn small" onClick={() => { setDraft([]); setRedrawing(true) }}>Redraw outline</button>}
                  {myOutline(selectedObject.x, selectedObject.y) && <button className="btn small" onClick={() => save([{ task: 'outline', x: selectedObject.x, y: selectedObject.y, deleted: true }])}>Outline back to AI</button>}
                  {!objectKeys.has(selected!) && <button className="btn small" onClick={removeSelected}>Remove object</button>}
                </div>
              </div>
            )}
            <div className="card">
              <h3>This field</h3>
              {meta.tools.includes('tag') && meta.tags.map((t) => (
                <label key={t} className="layer-toggle"><input type="checkbox" checked={!!mine[`tag:${tile}:${t}`]}
                  onChange={() => save([{ task: 'tag', value: t, deleted: !!mine[`tag:${tile}:${t}`] }])} />{t}</label>
              ))}
              <label className="field" style={{ marginTop: 8 }}>Note (optional)
                <textarea rows={2} value={note} onChange={(e) => setNote(e.target.value)}
                  onBlur={() => { if (note !== (Object.values(mine).find((m) => m.task === 'note')?.value ?? '')) save([{ task: 'note', value: note, deleted: !note }]) }} />
              </label>
              <p className="muted small">{changed} change{changed === 1 ? '' : 's'} by you. {full ? 'Finished counts every AI call you left as it is as checked by you.' : 'Objects you did not label stay unlabelled.'}</p>
              <div className="row">
                {done
                  ? <button className="btn" onClick={() => save([{ task: 'done', deleted: true }])}>Reopen field</button>
                  : <button className="btn primary" disabled={draft.length > 0 || saving} onClick={finishField}>Finished — next field</button>}
                <button className="btn" onClick={skip}>Skip for now</button>
              </div>
            </div>
            <div className="card">
              <h3>Show</h3>
              <label className="layer-toggle"><input type="checkbox" checked={!!layers.negatives} onChange={() => toggle('negatives')} />candidates that are not objects</label>
              {work.data.outlines.length > 0 && <label className="layer-toggle"><input type="checkbox" checked={!!layers.outlines} onChange={() => toggle('outlines')} />outlines</label>}
              {work.data.layers.map((l) => (
                <label key={l} className="layer-toggle"><input type="checkbox" checked={!!layers[l]} onChange={() => toggle(l)} /><span className="swatch" style={{ background: RASTER_LAYERS[l].color }} />{RASTER_LAYERS[l].label}</label>
              ))}
              <label className="layer-toggle"><input type="checkbox" checked={!!layers.mine} onChange={() => toggle('mine')} />objects I added</label>
              <label className="layer-toggle"><input type="checkbox" checked={!!layers.box} onChange={() => toggle('box')} />field boundary</label>
            </div>
            <div className="card">
              <h3>Fields</h3>
              <Progress done={fields.filter((f) => f.done_by_me).length} total={fields.length} />
              <p className="muted small">{fields.filter((f) => f.done_by_me).length} of {fields.length} finished by you. Green = finished; dot = someone else has it open; faded = still being prepared.</p>
              <div className="row" style={{ gap: 4 }}>
                {fields.map((f, i) => (
                  <Link key={f.tile_id} to={f.prepared ? `/annotate/${name}/${enc(f.tile_id)}` : '#'} title={`${f.tile_id} · finished by ${f.annotators_done}`}
                    className="btn small" style={{ minWidth: 34, textAlign: 'center', position: 'relative', opacity: f.prepared ? 1 : 0.4,
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

function Queue({ detail }: { detail: SessionDetail }) {
  const meta = detail.meta
  const next = useFetch<{ object: QueueItem | null }>(`/api/annotate/${meta.name}/next`)
  const [history, setHistory] = useState<QueueItem[]>([])
  const [error, setError] = useState<string | null>(null)
  const item = next.data?.object
  const classes = meta.classes

  const choose = useCallback(async (value: string) => {
    if (!item) return
    try {
      await api.post(`/api/annotate/${meta.name}/labels`, { records: [{ task: 'class', tile_id: item.tile_id, x: item.x, y: item.y, value }] })
      setHistory((h) => [item, ...h].slice(0, 20))
      next.reload()
    } catch (e) {
      setError((e as Error).message)
    }
  }, [item, meta.name, next])

  const undo = async () => {
    const last = history[0]
    if (!last) return
    await api.post(`/api/annotate/${meta.name}/labels`, { records: [{ task: 'class', tile_id: last.tile_id, x: last.x, y: last.y, deleted: true }] })
    setHistory((h) => h.slice(1))
    next.reload()
  }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (typing(e) || e.metaKey || e.ctrlKey) return
      const n = Number(e.key)
      if (n >= 1 && n <= Math.min(9, classes.length)) choose(classes[n - 1])
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [classes, choose])

  return (
    <>
      <PageHead title={meta.title} subtitle={<span><Link to="/annotate">← All sessions</Link> · {meta.stain} · one object at a time · {AI_VIEW_TEXT[meta.ai_view]}</span>} />
      <ErrorNote error={next.error ?? error} />
      {meta.instructions && <div className="instructions">{meta.instructions}</div>}
      {next.data && !item && <Empty>Nothing left for you in this queue right now. Thank you!</Empty>}
      {item && (
        <div className="grid" style={{ gridTemplateColumns: 'minmax(300px, 520px) 1fr', alignItems: 'start' }}>
          <div className="card">
            <div style={{ position: 'relative' }}>
              <img src={apiUrl(`/api/annotate/${meta.name}/object.jpg?tile_id=${enc(item.tile_id)}&x=${item.x}&y=${item.y}`)} alt="object" style={{ width: '100%', display: 'block', borderRadius: 8 }} />
              <svg viewBox="0 0 100 100" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
                <path d="M50 42 V46 M50 54 V58 M42 50 H46 M54 50 H58" stroke="#00e5ff" strokeWidth={0.8} />
              </svg>
            </div>
            <p className="muted small">{item.remaining} objects left for you · {item.labels_so_far} label{item.labels_so_far === 1 ? '' : 's'} by others so far</p>
          </div>
          <div className="card">
            <h3>What is under the cross?</h3>
            {meta.ai_view === 'full' && <p className="small">AI: {item.ai_value} · p {fmt(item.ai_probability, 2)}</p>}
            <div className="grid" style={{ gap: 6 }}>
              {classes.map((c, i) => (
                <button key={c} className="btn" style={{ textAlign: 'left' }} onClick={() => choose(c)}><span className="kbd">{i + 1}</span> {c}</button>
              ))}
            </div>
            <p className="muted small">{meta.class_instructions}</p>
            <button className="btn small" disabled={!history.length} onClick={undo}>Undo last</button>
          </div>
        </div>
      )}
    </>
  )
}
