import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import type { FieldObjectLite, FieldWork, Label, QueueItem, SessionDetail } from '../api'
import { api, apiUrl, enc } from '../api'
import { Empty, ErrorNote, Loaded, Loading, PageHead, Progress, fmt } from '../components'
import { useFetch, useStored } from '../hooks'
import { AI_VIEW_FOR_LABELLERS } from '../labelling'
import { FieldThumb } from '../labels/fields'
import { Register, useSession } from '../session'
import type { ImagePoint } from '../viewer'
import { RASTER_LAYERS, ZoomViewer } from '../viewer'

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
  if (!session.annotator) return <><PageHead title="Help check tissue images" subtitle="Type your name to start." /><Register /></>
  if (detail.error) return <SessionProblem error={detail.error} retry={detail.reload} />
  if (!detail.data) return <Loading label="Opening the session…" />
  if (detail.data.meta.kind === 'queue') return <Queue detail={detail.data} />
  return tile ? <Workspace key={tile} detail={detail.data} reload={detail.reload} tile={tile} /> : <SessionOverview detail={detail.data} reload={detail.reload} />
}

function SessionProblem({ error, retry }: { error: string; retry: () => void }) {
  const closed = /not open/.test(error)
  return (
    <>
      <PageHead title={closed ? 'This session is closed' : 'This session could not be opened'} subtitle={<Link to="/annotate">← All sessions</Link>} />
      <Empty>
        <p style={{ marginTop: 0 }}>{closed ? 'The person who runs it has closed it for now. Your work so far is saved.' : error.replace(/^\d{3}: /, '')}</p>
        <div className="row" style={{ justifyContent: 'center' }}><button className="btn" onClick={retry}>Try again</button><Link className="btn" to="/annotate">All sessions</Link></div>
      </Empty>
    </>
  )
}

type MyFieldState = 'done' | 'todo' | 'preparing' | 'taken'
const MY_STATE: Record<MyFieldState, { text: string; className: string }> = {
  done: { text: 'Finished by you', className: 'status-done' }, todo: { text: 'To do', className: 'status-todo' },
  preparing: { text: 'Being prepared', className: 'status-queued' }, taken: { text: 'Someone else has it open', className: 'status-running' },
}
const myState = (f: SessionDetail['fields'][number]): MyFieldState => (f.done_by_me ? 'done' : !f.prepared ? 'preparing' : f.open_by_others ? 'taken' : 'todo')

/** A session's start page for the person labelling: how far they are, the next field to do, and every field with what is
 *  done and what is left (they can open any prepared field). */
function SessionOverview({ detail, reload }: { detail: SessionDetail; reload: () => void }) {
  const navigate = useNavigate()
  const { meta, fields } = detail
  const name = meta.name
  const [finding, setFinding] = useState(false)
  const [nothing, setNothing] = useState(false)
  const done = fields.filter((f) => f.done_by_me).length
  const preparing = fields.filter((f) => !f.prepared).length
  const left = fields.length - done
  useEffect(() => {  // fields still being prepared: look again by itself
    if (!preparing) return
    const timer = setInterval(reload, 10000)
    return () => clearInterval(timer)
  }, [preparing, reload])
  const next = async () => {
    setFinding(true)
    const found = await api.get<{ tile_id: string | null }>(`/api/annotate/${name}/next`).catch(() => ({ tile_id: null }))
    setFinding(false)
    if (found.tile_id) navigate(`/annotate/${name}/${enc(found.tile_id)}`)
    else setNothing(true)
  }
  return (
    <div className="annotate-overview">
      <PageHead title={meta.title} subtitle={<Link to="/annotate">← All sessions</Link>}>
        {left > 0 && <button className="btn primary big" disabled={finding || left === preparing} onClick={next}>{done ? 'Continue with the next field →' : 'Start with the first field →'}</button>}
      </PageHead>
      {meta.instructions && <div className="instructions">{meta.instructions}</div>}
      <div className="card annotate-progress-card">
        <div className="row"><b>{done} of {fields.length} fields finished by you</b><span className="spacer" />
          <span className="small secondary">{left ? `${left} left` : 'All done — thank you!'}{preparing ? ` · ${preparing} still being prepared (this page checks again by itself)` : ''}</span></div>
        <Progress done={done} total={fields.length} />
        <p className="small muted" style={{ margin: 0 }}>A field is one square of tissue (about 560 µm across). {meta.stain} · {AI_VIEW_FOR_LABELLERS[meta.ai_view]}.</p>
        {nothing && <p className="small secondary" style={{ margin: 0 }}>Every field still to do is open by someone else right now or is still being prepared; try again in a minute.</p>}
      </div>
      <div className="field-grid annotate-fields">{fields.map((f, i) => {
        const state = myState(f)
        const body = <>
          <FieldThumb src={apiUrl(`/api/annotate/${name}/fields/${enc(f.tile_id)}/thumb.jpg`)} alt={`Field ${i + 1}`} />
          <span className="row small"><b>Field {i + 1}</b><span className="spacer" /><span className={`badge ${MY_STATE[state].className}`}>{MY_STATE[state].text}</span></span>
        </>
        return f.prepared
          ? <Link key={f.tile_id} className="field-card clickable" to={`/annotate/${name}/${enc(f.tile_id)}`} title={state === 'done' ? 'Open again to check or change' : 'Open this field'}>{body}</Link>
          : <div key={f.tile_id} className="field-card is-waiting">{body}</div>
      })}</div>
    </div>
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
      const next = await api.get<{ tile_id: string | null }>(`/api/annotate/${name}/next`).catch(() => ({ tile_id: null }))
      navigate(next.tile_id ? `/annotate/${name}/${enc(next.tile_id)}` : `/annotate/${name}`)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setSaving(false)
    }
  }
  const later = [...fields.slice(position + 1), ...fields.slice(0, position)].find((f) => !f.done_by_me && f.prepared && f.tile_id !== tile)
  const skip = () => { if (later) navigate(`/annotate/${name}/${enc(later.tile_id)}`) }
  const reopen = async () => { await save([{ task: 'done', deleted: true }]); reload() }

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
  const finishedByMe = fields.filter((f) => f.done_by_me).length

  return (
    <div className="annotate-page">
      <div className="annotate-top">
        <div className="annotate-heading">
          <span className="row small"><Link to="/annotate">← All sessions</Link><Link to={`/annotate/${name}`}>All fields of this session</Link></span>
          <h1>{meta.title}</h1>
          <div className="muted small">{meta.stain} · {AI_VIEW_FOR_LABELLERS[meta.ai_view]}</div>
        </div>
        <div className="annotate-progress">
          <span className="small secondary">Field <b>{position + 1}</b> of {fields.length} · {finishedByMe} finished by you</span>
          <Progress done={finishedByMe} total={fields.length} />
        </div>
        <div className="annotate-actions">
          <span className="muted small save-state" aria-live="polite">{saving ? 'Saving…' : work.data ? 'All changes saved' : ''}</span>
          <button className="btn" onClick={skip} disabled={!later} title={later ? 'Come back to this field later' : 'No other field is waiting for you'}>Skip for now</button>
          {done
            ? <button className="btn" onClick={reopen} disabled={saving}>Reopen this field</button>
            : <button className="btn primary big" disabled={draft.length > 0 || saving || !work.data} onClick={finishField}
              title={draft.length ? 'Finish or cancel (Esc) the outline you are drawing first' : undefined}>Finished — next field →</button>}
        </div>
      </div>
      {meta.instructions && <div className="instructions">{meta.instructions}</div>}
      <ErrorNote error={error} />
      <Loaded state={work} label="Opening the field…">
        {(data) => (
          <div className="annotate-workspace">
            <div className="annotate-image">
              <ZoomViewer src={apiUrl(`/api/annotate/${name}/fields/${enc(tile)}/image.jpg`)}
                layers={data.layers.filter((l) => layers[l]).map((l) => apiUrl(`/api/annotate/${name}/fields/${enc(tile)}/layer/${l}.png`))}
                title={<span className="row"><span className="muted small">{tile}</span>{done && <span className="badge status-done">finished</span>}</span>}
                cursor={active?.id === 'label' ? 'pointer' : 'crosshair'} onClick={click} onMove={move}>
                {(_zoom, _size, unit) => <>
                  {layers.box && <rect x={data.box.x} y={data.box.y} width={data.box.width} height={data.box.height} fill="none" stroke="#ffd21f" strokeWidth={2 * unit} strokeDasharray={`${12 * unit} ${8 * unit}`} />}
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
              <KeyHelp classes={classes.length} tools={tools} />
            </div>
            <aside className="annotate-side">
              <div className="card">
                <h3>1 · What to do</h3>
                <div className="tool-list">
                  {tools.map((t) => (
                    <button key={t.id} className={`btn tool-btn${active?.id === t.id ? ' chosen' : ''}`} aria-pressed={active?.id === t.id} onClick={() => { setTool(t.id); setDraft([]) }}>
                      <span className="kbd">{t.key.toUpperCase()}</span> {t.name}
                    </button>
                  ))}
                </div>
                {active && <p className="small secondary tool-hint">{redrawing ? 'Click the corners of the new outline, then the first corner or press Enter.' : active.hint}</p>}
                <h3 style={{ marginTop: 14 }}>2 · Class</h3>
                <div className="class-list">
                  {classes.map((c, i) => (
                    <button key={c} className={`btn class-btn${c === label ? ' chosen' : ''}`} aria-pressed={c === label} onClick={() => setLabelIndex(i)}>
                      {i < 9 && <span className="kbd">{i + 1}</span>}<span className="swatch" style={{ background: color(c) }} />{c}<span className="spacer" /><span className="muted small">{counts[c] ?? ''}</span>
                    </button>
                  ))}
                </div>
                <p className="muted small">{meta.class_instructions}</p>
              </div>
              {selectedObject && (
                <div className="card">
                  <h3>Selected object</h3>
                  {selectedAi?.ai_value && <p className="small">The AI’s guess: <b>{selectedAi.ai_value}</b></p>}
                  <p className="small">{myClass(selectedObject.x, selectedObject.y) ? <>Your class: <b>{myClass(selectedObject.x, selectedObject.y)!.value}</b></> : 'Not changed by you'}
                    {!objectKeys.has(selected!) && ' (added by you)'}</p>
                  <div className="row">
                    {myClass(selectedObject.x, selectedObject.y) && objectKeys.has(selected!) &&
                      <button className="btn small" onClick={() => save([{ task: 'class', x: selectedObject.x, y: selectedObject.y, deleted: true }])}>Back to the AI’s guess</button>}
                    {meta.tools.includes('outline') && active?.id !== 'outline' && <button className="btn small" onClick={() => setTool('outline')}>Edit outline</button>}
                    {meta.tools.includes('outline') && active?.id === 'outline' && <button className="btn small" onClick={() => { setDraft([]); setRedrawing(true) }}>Redraw outline</button>}
                    {myOutline(selectedObject.x, selectedObject.y) && <button className="btn small" onClick={() => save([{ task: 'outline', x: selectedObject.x, y: selectedObject.y, deleted: true }])}>Outline back to the AI’s</button>}
                    {!objectKeys.has(selected!) && <button className="btn small" onClick={removeSelected}>Remove object</button>}
                  </div>
                </div>
              )}
              <div className="card">
                <h3>3 · This field</h3>
                {meta.tools.includes('tag') && <div className="tag-list">{meta.tags.map((t) => (
                  <label key={t} className="layer-toggle"><input type="checkbox" checked={!!mine[`tag:${tile}:${t}`]}
                    onChange={() => save([{ task: 'tag', value: t, deleted: !!mine[`tag:${tile}:${t}`] }])} />{t}</label>
                ))}</div>}
                <label className="field" style={{ marginTop: 8 }}>Note (optional)
                  <textarea rows={2} value={note} onChange={(e) => setNote(e.target.value)}
                    onBlur={() => { if (note !== (Object.values(mine).find((m) => m.task === 'note')?.value ?? '')) save([{ task: 'note', value: note, deleted: !note }]) }} />
                </label>
                <p className="muted small">{changed} change{changed === 1 ? '' : 's'} by you in this field.
                  {' '}{full ? 'When you press Finished, every AI guess you left as it is counts as checked by you.' : 'Objects you did not label stay unlabelled.'}</p>
              </div>
              <details className="card side-details">
                <summary>Display options</summary>
                <label className="layer-toggle"><input type="checkbox" checked={!!layers.negatives} onChange={() => toggle('negatives')} />candidates that are not objects</label>
                {data.outlines.length > 0 && <label className="layer-toggle"><input type="checkbox" checked={!!layers.outlines} onChange={() => toggle('outlines')} />outlines</label>}
                {data.layers.map((l) => (
                  <label key={l} className="layer-toggle"><input type="checkbox" checked={!!layers[l]} onChange={() => toggle(l)} /><span className="swatch" style={{ background: RASTER_LAYERS[l].color }} />{RASTER_LAYERS[l].label}</label>
                ))}
                <label className="layer-toggle"><input type="checkbox" checked={!!layers.mine} onChange={() => toggle('mine')} />objects I added</label>
                <label className="layer-toggle"><input type="checkbox" checked={!!layers.box} onChange={() => toggle('box')} />field boundary (yellow)</label>
              </details>
              <details className="card side-details">
                <summary>Go to another field</summary>
                <p className="muted small">Green = finished by you; dot = someone else has it open; faded = still being prepared.</p>
                <div className="field-jump">
                  {fields.map((f, i) => {
                    const chip = `field-chip${f.tile_id === tile ? ' current' : f.done_by_me ? ' done' : ''}${f.prepared ? '' : ' waiting'}`
                    const inner = <>{i + 1}{f.open_by_others > 0 && <span className="busy-dot" />}</>
                    return f.prepared
                      ? <Link key={f.tile_id} to={`/annotate/${name}/${enc(f.tile_id)}`} title={`${f.tile_id} · finished by ${f.annotators_done}`} className={chip}>{inner}</Link>
                      : <span key={f.tile_id} title={`${f.tile_id} · still being prepared`} className={chip}>{inner}</span>
                  })}
                </div>
              </details>
            </aside>
          </div>
        )}
      </Loaded>
    </div>
  )
}

/** The keys of the field page, always on screen under the image. */
function KeyHelp({ classes, tools }: { classes: number; tools: { id: Tool; key: string; name: string }[] }) {
  return (
    <div className="key-help small secondary">
      <span>Scroll to zoom, drag to move</span>
      <span><span className="kbd">1</span>–<span className="kbd">{Math.min(9, classes)}</span> class</span>
      {tools.map((t) => <span key={t.id}><span className="kbd">{t.key.toUpperCase()}</span> {t.name.toLowerCase()}</span>)}
      <span><span className="kbd">Delete</span> remove</span>
      <span><span className="kbd">Enter</span> close outline</span>
      <span><span className="kbd">Esc</span> cancel</span>
    </div>
  )
}

function Queue({ detail }: { detail: SessionDetail }) {
  const meta = detail.meta
  const next = useFetch<{ object: QueueItem | null }>(`/api/annotate/${meta.name}/next`)
  const [history, setHistory] = useState<QueueItem[]>([])
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const item = next.data?.object
  const classes = meta.classes
  const { reload } = next

  const send = useCallback(async (record: Record<string, unknown>, after: () => void) => {
    setBusy(true)
    try {
      await api.post(`/api/annotate/${meta.name}/labels`, { records: [record] })
      after()
      setError(null)
      reload()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }, [meta.name, reload])

  const choose = useCallback((value: string) => {
    if (!item || busy) return
    send({ task: 'class', tile_id: item.tile_id, x: item.x, y: item.y, value }, () => setHistory((h) => [item, ...h].slice(0, 50)))
  }, [item, busy, send])

  const undo = useCallback(() => {
    const last = history[0]
    if (!last || busy) return
    send({ task: 'class', tile_id: last.tile_id, x: last.x, y: last.y, deleted: true }, () => setHistory((h) => h.slice(1)))
  }, [history, busy, send])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (typing(e) || e.metaKey || e.ctrlKey) return
      const n = Number(e.key)
      if (n >= 1 && n <= Math.min(9, classes.length)) choose(classes[n - 1])
      else if (e.key === 'Backspace' || e.key.toLowerCase() === 'z') { e.preventDefault(); undo() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [classes, choose, undo])

  return (
    <div className="annotate-page">
      <div className="annotate-top">
        <div className="annotate-heading">
          <Link to="/annotate" className="small">← All sessions</Link>
          <h1>{meta.title}</h1>
          <div className="muted small">{meta.stain} · one object at a time{meta.ai_view === 'full' ? ' · the AI’s guess is shown' : ' · decide for yourself'}</div>
        </div>
        <div className="annotate-progress">
          <span className="small secondary">{history.length ? `${history.length} labelled by you this visit` : 'Press a number key to answer'}
            {item ? ` · ${item.remaining.toLocaleString()} left` : ''}</span>
        </div>
      </div>
      {meta.instructions && <div className="instructions">{meta.instructions}</div>}
      <ErrorNote error={error} />
      <Loaded state={next} label="Finding the next object…">
        {(n) => n.object ? (
          <div className="queue-layout">
            <div className="card queue-image">
              <div className="queue-crop">
                <img key={`${n.object.tile_id}:${n.object.x}:${n.object.y}`} src={apiUrl(`/api/annotate/${meta.name}/object.jpg?tile_id=${enc(n.object.tile_id)}&x=${n.object.x}&y=${n.object.y}`)} alt="the object to label" />
                <svg viewBox="0 0 100 100" aria-hidden="true">
                  <path d="M50 40 V46 M50 54 V60 M40 50 H46 M54 50 H60" stroke="#00e5ff" strokeWidth={0.9} />
                </svg>
                {busy && <div className="queue-busy"><Loading inline label="Saving…" /></div>}
              </div>
              <p className="muted small">{n.object.labels_so_far ? `${n.object.labels_so_far} other ${n.object.labels_so_far === 1 ? 'person has' : 'people have'} labelled this one` : 'Nobody has labelled this one yet'}</p>
            </div>
            <div className="card queue-choices">
              <h2>What is under the cross?</h2>
              {meta.ai_view === 'full' && <p className="small secondary" title={`probability ${fmt(n.object.ai_probability, 2)}`}>The AI’s guess: <b>{n.object.ai_value}</b></p>}
              <div className="class-list">
                {classes.map((c, i) => (
                  <button key={c} className="btn class-btn big" disabled={busy} onClick={() => choose(c)}>{i < 9 && <span className="kbd">{i + 1}</span>} {c}</button>
                ))}
              </div>
              <p className="muted small">{meta.class_instructions}</p>
              <div className="row">
                <button className="btn small" disabled={!history.length || busy} onClick={undo}><span className="kbd">Z</span> Undo last</button>
                <span className="muted small">Keys: <span className="kbd">1</span>–<span className="kbd">{Math.min(9, classes.length)}</span> answer, <span className="kbd">Z</span> undo</span>
              </div>
            </div>
          </div>
        ) : (
          <Empty><p style={{ marginTop: 0 }}><b>Nothing left for you in this queue right now. Thank you!</b></p>
            <div className="row" style={{ justifyContent: 'center' }}><button className="btn" onClick={reload}>Check again</button><Link className="btn" to="/annotate">All sessions</Link></div></Empty>
        )}
      </Loaded>
    </div>
  )
}
