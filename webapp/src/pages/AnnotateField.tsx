import { useCallback, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import type { AddedState, FieldObjectLite, FieldWork, Label, SessionDetail } from '../api'
import { api, apiUrl, enc } from '../api'
import { Empty, ErrorNote, Loaded, Loading, PageHead, Progress, fmt } from '../components'
import { PALETTE } from '../palette'
import { useFetch, useStored } from '../hooks'
import { AI_VIEW_FOR_LABELLERS } from '../labelling'
import type { ClassPick } from '../labels/classKeys'
import { classText, resolveClass, roleOf } from '../labels/classKeys'
import { ClassChooser, ClassKeyHint, classColour, useClassKeys } from '../labels/classes'
import { QuickCheck } from '../labels/quickcheck'
import { FixField } from '../labels/fixfield'
import { MarkAreas } from '../labels/markareas'
import { MatterCorePage, MatterSession } from '../labels/matter'
import { SessionFieldMap } from '../labels/sessionfields'
import { ClosedNote, ROUND_TITLE, RoundProgressLine, useFitScreen } from '../labels/rounds'
import { HomePin } from '../labels/access'
import type { RoundDetail } from '../api'
import { Register, useSession } from '../session'
import type { ImagePoint } from '../viewer'
import { RASTER_LAYERS, ZoomViewer } from '../viewer'

type Tool = 'label' | 'add' | 'outline'
const TOOLS: { id: Tool; needs: string; key: string; name: string; hint: string }[] = [
  { id: 'label', needs: 'class', key: 'v', name: 'Label objects', hint: 'Click an object to give it the chosen class. Dashed = detection’s call, not yet checked by you; solid = your decision. Click again with the same class to go back to detection’s call.' },
  { id: 'add', needs: 'point', key: 'a', name: 'Add a missed object', hint: 'Click an object detection did not circle to add it with the chosen class. It shows as a diamond. Click one of your diamonds to select it; Delete removes it.' },
  { id: 'outline', needs: 'outline', key: 'o', name: 'Fix outlines', hint: 'Click an object to select it, then drag the corners of its outline or press Redraw. Clicking empty tissue starts a new outline (a new object): click its corners, then the first corner or Enter.' },
]

const rkey = (x: number, y: number) => `${Math.round(x)}:${Math.round(y)}`
const typing = (e: KeyboardEvent) => ['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement).tagName)
const samePick = (a: ClassPick | undefined, b: ClassPick | undefined) => !!a && !!b && a.identity === b.identity && (a.subtype ?? null) === (b.subtype ?? null)

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
  if (!session.annotator) return <><PageHead title="Help check tissue images" subtitle="Sign in to start, then add this set with the key you were given." /><Register /></>
  if (detail.error) return <SessionProblem error={detail.error} retry={detail.reload} />
  if (!detail.data) return <Loading label="Opening the session…" />
  const round = (detail.data.meta as Partial<RoundDetail>).round
  if (round === 'areas' || detail.data.meta.kind === 'areas') return <MarkAreas detail={detail.data} />
  if (detail.data.meta.kind === 'matter') return tile ? <MatterCorePage key={tile} detail={detail.data} core={tile} /> : <MatterSession detail={detail.data} />
  if (round === 1) return <><PageHead title={detail.data.meta.title} subtitle="Check detection’s calls" />
    <Empty><p style={{ marginTop: 0 }}>This kind of session is no longer used: its labels stay in the label log for training and evaluation.
      Ask for a “Fix the field” session instead.</p><Link className="btn" to="/">My labelling sets</Link></Empty></>
  if (round === 2 && tile) return <FixField key={tile} detail={detail.data} reload={detail.reload} tile={tile} />
  if (detail.data.meta.kind === 'queue') return <QuickCheck detail={detail.data} />
  if (detail.data.meta.kind === 'outline') return <><PageHead title={detail.data.meta.title} subtitle="Outline every object" />
    <Empty><p style={{ marginTop: 0 }}>This kind of session is no longer used: its outlines stay in the label log and still count in the segmentation
      check’s scores. Outlines are now fixed in “Fix the field”.</p><Link className="btn" to="/">My labelling sets</Link></Empty></>
  return tile ? <Workspace key={tile} detail={detail.data} reload={detail.reload} tile={tile} /> : <SessionOverview detail={detail.data} reload={detail.reload} />
}

function SessionProblem({ error, retry }: { error: string; retry: () => void }) {
  const closed = /not open/.test(error)
  return (
    <>
      <PageHead title={closed ? 'This session is closed' : 'This session could not be opened'} subtitle={<Link to="/">← My labelling sets</Link>} />
      <Empty>
        <p style={{ marginTop: 0 }}>{closed ? 'The person who runs it has closed it for now. Your work so far is saved.' : error.replace(/^\d{3}: /, '')}</p>
        <div className="row" style={{ justifyContent: 'center' }}><button className="btn" onClick={retry}>Try again</button><Link className="btn" to="/">My labelling sets</Link></div>
      </Empty>
    </>
  )
}

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
  const live = (meta as Partial<RoundDetail>).round === 2
  useEffect(() => {  // fields still being prepared, or others working on the round: look again by itself
    if (!preparing && !live) return
    const timer = setInterval(reload, live ? 5000 : 10000)
    return () => clearInterval(timer)
  }, [preparing, live, reload])
  const next = async () => {
    setFinding(true)
    const found = await api.get<{ tile_id: string | null }>(`/api/annotate/${name}/next`).catch(() => ({ tile_id: null }))
    setFinding(false)
    if (found.tile_id) navigate(`/annotate/${name}/${enc(found.tile_id)}`)
    else setNothing(true)
  }
  return (
    <div className="annotate-overview">
      <PageHead title={meta.title} subtitle={<Link to="/">← My labelling sets</Link>}>
        <HomePin name={name} />
        {left > 0 && meta.open !== false && <button className="btn primary big" disabled={finding || left === preparing} onClick={next}>{done ? 'Continue with the next field →' : 'Start with the first field →'}</button>}
      </PageHead>
      {meta.open === false && <ClosedNote unit="session" />}
      {meta.instructions && <div className="instructions">{meta.instructions}</div>}
      {(meta as Partial<RoundDetail>).round === 2 && <div className="card annotate-progress-card">
        <div className="row"><b>Round 2 · {ROUND_TITLE[2]}</b><span className="spacer" /><span className="small muted">everyone’s work, live</span></div>
        <RoundProgressLine progress={(meta as Partial<RoundDetail>).progress ?? null} me={detail.annotator} />
      </div>}
      <div className="card annotate-progress-card">
        <div className="row"><b>{done} of {fields.length} fields finished by you</b><span className="spacer" />
          <span className="small secondary">{left ? `${left} left` : 'All done — thank you!'}{preparing ? ` · ${preparing} still being prepared (this page checks again by itself)` : ''}</span></div>
        <Progress done={done} total={fields.length} />
        <p className="small muted" style={{ margin: 0 }}>A field is one square of tissue (about 560 µm across). {meta.stain} · {AI_VIEW_FOR_LABELLERS[meta.ai_view]}.</p>
        {nothing && <p className="small secondary" style={{ margin: 0 }}>Every field still to do is being checked by someone else right now or is still being prepared; try again in a minute.</p>}
      </div>
      <SessionFieldMap detail={detail} />
    </div>
  )
}

function Workspace({ detail, reload, tile }: { detail: SessionDetail; reload: () => void; tile: string }) {
  const navigate = useNavigate()
  const meta = detail.meta
  const name = meta.name
  const work = useFetch<FieldWork>(`/api/annotate/${name}/fields/${enc(tile)}`)
  useFitScreen()
  const [mine, setMine] = useState<Record<string, Label>>({})
  const [tool, setTool] = useState<Tool | null>(null)
  const { vocabulary, keys, subtypes: typesAsked } = meta
  const [label, setLabel] = useState<ClassPick>(() => ({ identity: vocabulary.classes.find((c) => !c.hidden)?.id ?? '', subtype: null }))
  const [selected, setSelected] = useState<string | null>(null)
  const [draft, setDraft] = useState<number[][]>([])
  const [redrawing, setRedrawing] = useState(false)
  const [hover, setHover] = useState<number[] | null>(null)
  const [dragging, setDragging] = useState<{ index: number; points: number[][] } | null>(null)
  const [note, setNote] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [layers, setLayers] = useStored<Record<string, boolean>>('stainid.label.layers', { negatives: true, outlines: true, mine: true, box: true })
  const [addedStates, setAddedStates] = useState<AddedState[]>([])

  useEffect(() => {
    if (!work.data) return
    setMine(Object.fromEntries(work.data.mine.map((m) => [m.id, m])))
    setNote(work.data.mine.find((m) => m.task === 'note')?.value ?? '')
    setAddedStates(work.data.added ?? [])
  }, [work.data])

  const classKeys = useClassKeys(keys, setLabel)
  const color = (v: ClassPick | undefined) => (v ? classColour(vocabulary, v.identity, v.subtype) : '#ffffff')  // the same colour on every page
  const negativePick = (v: ClassPick | undefined) => !v || roleOf(vocabulary, v.identity) !== 'positive'
  const pickOf = (m: Label | undefined) => (m ? resolveClass(vocabulary, m.value, m.subtype) : undefined)
  const aiPick = (o: FieldObjectLite | undefined) => (o?.ai_value ? resolveClass(vocabulary, o.ai_value) : undefined)
  const text = (v: ClassPick | undefined) => (v ? classText(vocabulary, v, typesAsked) : '')
  const asLabel = (v: ClassPick) => ({ value: v.identity, subtype: typesAsked ? v.subtype ?? '' : '' })
  const full = meta.ai_view === 'full'
  const tools = useMemo(() => TOOLS.filter((t) => meta.tools.includes(t.needs)), [meta.tools])
  const active = tools.find((t) => t.id === tool) ?? tools[0]
  const objects = useMemo(() => work.data?.objects ?? [], [work.data])
  const objectKeys = useMemo(() => new Set(objects.map((o) => rkey(o.x, o.y))), [objects])
  const aiOutlines = useMemo(() => Object.fromEntries((work.data?.outlines ?? []).map((o) => [rkey(o.x, o.y), o.points])), [work.data])
  const myClass = useCallback((x: number, y: number) => mine[`class:${tile}:${rkey(x, y)}`], [mine, tile])
  const myOutline = useCallback((x: number, y: number) => mine[`outline:${tile}:${rkey(x, y)}`], [mine, tile])
  const value = (o: FieldObjectLite) => pickOf(myClass(o.x, o.y)) ?? (full ? aiPick(o) : undefined)
  const added = Object.values(mine).filter((m) => m.task === 'class' && !objectKeys.has(rkey(m.x!, m.y!)))
  const done = !!mine[`done:${tile}`]
  const fields = detail.fields
  const position = fields.findIndex((f) => f.tile_id === tile)
  const all = [...objects.map((o) => ({ x: o.x, y: o.y, r: o.radius_px })), ...added.map((m) => ({ x: m.x!, y: m.y!, r: 8 }))]
  const selectedObject = selected ? all.find((o) => rkey(o.x, o.y) === selected) : undefined
  const addedState = (x: number, y: number) => addedStates.find((a) => rkey(a.x, a.y) === rkey(x, y))?.state ?? 'measuring'
  const measuring = added.some((m) => addedState(m.x!, m.y!) === 'measuring')
  useEffect(() => {  // follow objects people added while the server measures them (a few seconds each)
    if (!measuring) return
    const timer = setTimeout(() => {
      api.get<AddedState[]>(`/api/annotate/${name}/fields/${enc(tile)}/added`).then(setAddedStates).catch(() => undefined)
    }, 3000)
    return () => clearTimeout(timer)
  }, [measuring, addedStates, name, tile])
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
    if (!selectedObject) records.push({ task: 'class', x: target.x, y: target.y, ...asLabel(label) })
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
      if (samePick(pickOf(myClass(o.x, o.y)), { ...label, subtype: typesAsked ? label.subtype : null })) save([{ task: 'class', x: o.x, y: o.y, deleted: true }])
      else save([{ task: 'class', x: o.x, y: o.y, ...asLabel(label) }])
      return
    }
    if (active.id === 'add') {
      if (o) { setSelected(rkey(o.x, o.y)); return }
      save([{ task: 'class', x: Math.round(p.x), y: Math.round(p.y), ...asLabel(label) }])
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
      if (e.key === 'Enter' && draft.length) { closeDraft(); return }
      if (e.key === 'Backspace' || e.key === 'Delete') {  // on this page Delete takes away the selected mark (the reject is a click)
        e.preventDefault()
        if (draft.length) setDraft(draft.slice(0, -1))
        else removeSelected()
        return
      }
      if (!e.repeat && classKeys.press(e.key)) return
      const chosen = tools.find((t) => t.key === e.key.toLowerCase())
      if (chosen) { setTool(chosen.id); setDraft([]); return }
      if (e.key === 'Escape') { setDraft([]); setRedrawing(false); setSelected(null) }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [classKeys, tools, draft, closeDraft, removeSelected])

  const counts: Record<string, number> = {}  // objects of each class (identity) in this field, as they stand now
  for (const o of objects) { const v = value(o); if (v) counts[v.identity] = (counts[v.identity] ?? 0) + 1 }
  for (const m of added) { const v = pickOf(m)!; counts[v.identity] = (counts[v.identity] ?? 0) + 1 }
  const changed = Object.values(mine).filter((m) => m.task === 'class' && !m.confirmed).length
  const toggle = (key: string) => setLayers({ ...layers, [key]: !layers[key] })
  const selectedAi = selected ? objects.find((o) => rkey(o.x, o.y) === selected) : undefined
  const selectedValue = selectedObject ? (pickOf(myClass(selectedObject.x, selectedObject.y)) ?? aiPick(selectedAi)) : undefined
  const editing = active?.id === 'outline' && selectedObject && !draft.length ? (dragging?.points ?? outlineOf(selectedObject.x, selectedObject.y)) : undefined
  const finishedByMe = fields.filter((f) => f.done_by_me).length

  return (
    <div className="annotate-page fit-page">
      <div className="annotate-top">
        <div className="annotate-heading">
          <span className="row small"><Link to="/">← My labelling sets</Link><Link to={`/annotate/${name}`}>All fields of this session</Link></span>
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
              <ZoomViewer fit src={apiUrl(`/api/annotate/${name}/fields/${enc(tile)}/image.jpg`)}
                layers={data.layers.filter((l) => layers[l]).map((l) => apiUrl(`/api/annotate/${name}/fields/${enc(tile)}/layer/${l}.png`))}
                title={<span className="row"><span className="muted small">{tile}</span>{done && <span className="badge status-done">finished</span>}</span>}
                cursor={active?.id === 'label' ? 'pointer' : 'crosshair'} onClick={click} onMove={move}>
                {(_zoom, _size, unit) => <>
                  {layers.box && <rect x={data.box.x} y={data.box.y} width={data.box.width} height={data.box.height} fill="none" stroke={PALETTE.markers.box} strokeWidth={2 * unit} strokeDasharray={`${12 * unit} ${8 * unit}`} />}
                  {layers.outlines && all.map((o) => {
                    const k = rkey(o.x, o.y)
                    const v = pickOf(myClass(o.x, o.y)) ?? aiPick(objects.find((q) => rkey(q.x, q.y) === k))
                    const poly = k === selected && editing ? undefined : outlineOf(o.x, o.y)
                    if (!poly || (v && negativePick(v) && !myOutline(o.x, o.y))) return null
                    return <polygon key={`o${k}`} points={poly.map((p) => p.join(',')).join(' ')} fill="none" stroke={myOutline(o.x, o.y) ? color(v) : '#ffffff'}
                      strokeOpacity={0.9} strokeWidth={(myOutline(o.x, o.y) ? 2 : 1.2) * unit} />
                  })}
                  {objects.map((o) => {
                    const v = value(o)
                    const decided = !!myClass(o.x, o.y)
                    const negative = negativePick(v)
                    if (negative && !decided && !layers.negatives) return null
                    const k = rkey(o.x, o.y)
                    return (
                      <circle key={k} cx={o.x} cy={o.y} r={Math.max(8, o.radius_px) + 4} fill={decided ? color(v) : 'none'} fillOpacity={0.18}
                        stroke={v ? color(v) : '#ffffff'} strokeOpacity={negative && !decided ? 0.55 : 1}
                        strokeWidth={(k === selected ? 3.5 : negative && !decided ? 1.2 : 2) * unit} strokeDasharray={decided ? undefined : `${5 * unit} ${3 * unit}`}>
                        <title>{[full && o.ai_value && `detection: ${text(aiPick(o))} (p ${fmt(o.ai_probability, 2)})`, decided && `you: ${text(v)}`].filter(Boolean).join(' · ') || 'candidate'}</title>
                      </circle>
                    )
                  })}
                  {layers.mine && added.map((m) => {
                    const chosen = rkey(m.x!, m.y!) === selected
                    const d = 9 * unit
                    return (
                      <polygon key={m.id} points={`${m.x!},${m.y! - d} ${m.x! + d},${m.y!} ${m.x!},${m.y! + d} ${m.x! - d},${m.y!}`} fill={color(pickOf(m))}
                        stroke={chosen ? '#fff' : '#000'} strokeWidth={(chosen ? 2.5 : 1.2) * unit}>
                        <title>{`added by you: ${text(pickOf(m))}`}</title>
                      </polygon>
                    )
                  })}
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
              <KeyHelp classes={<ClassKeyHint vocabulary={vocabulary} keys={keys} />} tools={tools} />
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
                {tools.some((t) => t.id === 'add') && <MissedCallout adding={active?.id === 'add'} count={added.length} onAdd={() => { setTool('add'); setDraft([]) }} />}
                <h3 style={{ marginTop: 14 }}>2 · Class</h3>
                <ClassChooser vocabulary={vocabulary} keys={keys} active={label} counts={counts} onPick={(identity, subtype) => setLabel({ identity, subtype })} />
              </div>
              {selectedObject && (
                <div className="card">
                  <h3>Selected object</h3>
                  {selectedAi?.ai_value && <p className="small">Detection’s call: <b>{text(aiPick(selectedAi))}</b></p>}
                  <p className="small">{myClass(selectedObject.x, selectedObject.y) ? <>Your class: <b>{text(pickOf(myClass(selectedObject.x, selectedObject.y)))}</b></> : 'Not changed by you'}
                    {!objectKeys.has(selected!) && ' (added by you)'}</p>
                  {!objectKeys.has(selected!) && <AddedNote state={addedState(selectedObject.x, selectedObject.y)}
                    reason={addedStates.find((a) => rkey(a.x, a.y) === selected)?.reason ?? ''} />}
                  <div className="row">
                    {myClass(selectedObject.x, selectedObject.y) && objectKeys.has(selected!) &&
                      <button className="btn small" onClick={() => save([{ task: 'class', x: selectedObject.x, y: selectedObject.y, deleted: true }])}>Back to detection’s call</button>}
                    {meta.tools.includes('outline') && active?.id !== 'outline' && <button className="btn small" onClick={() => setTool('outline')}>Edit outline</button>}
                    {meta.tools.includes('outline') && active?.id === 'outline' && <button className="btn small" onClick={() => { setDraft([]); setRedrawing(true) }}>Redraw outline</button>}
                    {myOutline(selectedObject.x, selectedObject.y) && <button className="btn small" onClick={() => save([{ task: 'outline', x: selectedObject.x, y: selectedObject.y, deleted: true }])}>Outline back to detection’s</button>}
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
                  {' '}{full ? 'When you press Finished, every call of detection you left as it is counts as checked by you.' : 'Objects you did not label stay unlabelled.'}</p>
              </div>
              <details className="card side-details">
                <summary>Display options</summary>
                <label className="layer-toggle"><input type="checkbox" checked={!!layers.negatives} onChange={() => toggle('negatives')} />candidates that are not objects</label>
                {data.outlines.length > 0 && <label className="layer-toggle"><input type="checkbox" checked={!!layers.outlines} onChange={() => toggle('outlines')} />outlines</label>}
                {data.layers.map((l) => (
                  <label key={l} className="layer-toggle"><input type="checkbox" checked={!!layers[l]} onChange={() => toggle(l)} /><span className="swatch" style={{ background: RASTER_LAYERS[l].color }} />{RASTER_LAYERS[l].label}</label>
                ))}
                <label className="layer-toggle"><input type="checkbox" checked={!!layers.mine} onChange={() => toggle('mine')} />objects I added (diamonds)</label>
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

/** "Did detection miss something?": the add tool, explained, always in view while a field is open. */
function MissedCallout({ adding, count, onAdd }: { adding: boolean; count: number; onAdd: () => void }) {
  return (
    <div className="missed-callout small">
      <b>Did detection miss something?</b> {adding ? 'Click it on the picture.' : <>Press <span className="kbd">A</span> or
        {' '}<button className="link-btn" onClick={onAdd}>Add a missed object</button>, then click it.</>}
      {' '}Objects you add are how we find detection’s blind spots — and they teach the next model.
      {count > 0 && <span className="muted"> You added {count} in this field.</span>}
    </div>
  )
}

const ADDED_TEXT: Record<AddedState['state'], string> = {
  measuring: 'Being measured so it can teach the model…',
  measured: 'Measured — it will help train the next model.',
  could_not_be_measured: 'Could not be measured, so it cannot train the model; your label still counts as an object detection missed.',
  proposed: 'Detection had proposed this object too: your mark sits next to its centre, so it counts as that object.',
}

/** What became of an object the person added: detection never proposed it, and whether it could be measured. */
function AddedNote({ state, reason }: { state: AddedState['state']; reason: string }) {
  return (
    <p className={`small added-note${state === 'could_not_be_measured' ? ' warn' : ''}`}>
      {state !== 'proposed' && <><b>Detection never proposed this object.</b> </>}{ADDED_TEXT[state]}
      {state === 'could_not_be_measured' && reason && <span className="muted"> ({reason})</span>}
    </p>
  )
}

/** The keys of the field page, always on screen under the image. */
function KeyHelp({ classes, tools }: { classes: ReactNode; tools: { id: Tool; key: string; name: string }[] }) {
  return (
    <div className="key-help small secondary">
      <span>Scroll to zoom, drag to move</span>
      {classes}
      {tools.map((t) => <span key={t.id}><span className="kbd">{t.key.toUpperCase()}</span> {t.name.toLowerCase()}</span>)}
      <span><span className="kbd">Delete</span> remove</span>
      <span><span className="kbd">Enter</span> close outline</span>
      <span><span className="kbd">Esc</span> cancel</span>
    </div>
  )
}
