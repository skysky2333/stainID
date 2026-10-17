/* "Outline every object": the page where a person outlines every object of a field for the segmentation check (made
   from the Models page; an older kind of session, kept for that check). Click an object and Segment Anything outlines
   it; correct with more clicks, other sizes, dragging corners or drawing by hand; mark the field complete only when every
   object in the square is outlined. What detection found shows from the start (R hides it). The page takes the window
   exactly, with the same side panel as the field tool: the object, the class keys (0 unsure), the keys, the field. */
import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import type { SessionDetail } from '../api'
import { api, apiUrl, enc } from '../api'
import { ErrorNote, Loaded, Modal, Progress, fmt } from '../components'
import { useFetch, useStored } from '../hooks'
import { useFitScreen } from '../labels/rounds'
import { CLASS_COLOURS, classColour } from '../classPalette'
import type { Vocabulary } from '../labels/classKeys'
import { objectsOf } from '../labelling'
import type { ImagePoint } from '../viewer'
import { ZoomViewer } from '../viewer'
import type { OutlineClass, OutlineField, OutlineObject, OutlineReference, SamOptions, SamStatus } from './types'

/* drawn on the tissue picture, so the same in light and dark mode */
const UNSURE_COLOR = '#e2e8f0'
const REFERENCE_COLOR = '#22d3ee'
const BOX_COLOR = '#ffd21f'
const MAX_HANDLES = 240
const MIN_HANDLE_SPACING = 9  // screen pixels between corners: closer, and the corners are only dragged after zooming in

type Tool = 'click' | 'draw'
/** The clicks behind an object outlined in this visit, and the sizes Segment Anything offered for them. */
interface Prompt { clicks: number[][]; options: number[][][]; chosen: number }

const typing = (e: KeyboardEvent) => ['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement).tagName)
const newRef = () => Math.random().toString(36).slice(2, 10)
const svgPoints = (points: number[][]) => points.map((p) => p.join(',')).join(' ')
const flatPoints = (flat: number[]) => Array.from({ length: flat.length >> 1 }, (_, i) => `${flat[2 * i]},${flat[2 * i + 1]}`).join(' ')

function area(points: number[][]) {
  let sum = 0
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) sum += points[j][0] * points[i][1] - points[i][0] * points[j][1]
  return Math.abs(sum) / 2
}

/** The mean distance between neighbouring corners of a polygon (image pixels). */
function spacing(points: number[][]) {
  let sum = 0
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) sum += Math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1])
  return sum / Math.max(1, points.length)
}

function centre(points: number[][]) {
  return [points.reduce((a, p) => a + p[0], 0) / points.length, points.reduce((a, p) => a + p[1], 0) / points.length]
}

function inside(polygon: number[][], x: number, y: number) {
  let hit = false
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i]
    const [xj, yj] = polygon[j]
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) hit = !hit
  }
  return hit
}

export function OutlineWorkspace({ detail, reload, tile }: { detail: SessionDetail; reload: () => void; tile: string }) {
  const meta = detail.meta
  const work = useFetch<OutlineField>(`/api/annotate/${meta.name}/outline/${enc(tile)}`)
  useFitScreen()
  return (
    <Loaded state={work} label="Opening the field…">
      {(data) => <Outliner key={tile} detail={detail} reload={reload} tile={tile} field={data} />}
    </Loaded>
  )
}

function Outliner({ detail, reload, tile, field }: { detail: SessionDetail; reload: () => void; tile: string; field: OutlineField }) {
  const navigate = useNavigate()
  const { meta, fields } = detail
  const name = meta.name
  const choices = (meta as typeof meta & { outline_classes?: OutlineClass[] }).outline_classes ?? []
  const classes = useMemo(() => choices.map((c) => c.id), [choices])
  const classOf = (id: string) => choices.find((c) => c.id === id)
  const nameOf = (id: string) => classOf(id)?.name ?? id
  const scored = (id: string) => classOf(id)?.scored ?? true
  const words = objectsOf(meta.stain)
  const one = words.replace(/s$/, '')  // one plaque, one tau+ neuron
  const [objects, setObjects] = useState<Record<string, OutlineObject>>(() => Object.fromEntries(field.objects.map((o) => [o.ref, o])))
  const [prompts, setPrompts] = useState<Record<string, Prompt>>({})
  const [pending, setPending] = useState<Record<string, number[]>>({})
  const [selected, setSelected] = useState<string | null>(null)
  const [samReady, setSamReady] = useState(field.segment_anything)
  const [tool, setTool] = useState<Tool>(field.segment_anything ? 'click' : 'draw')
  const [prepared, setPrepared] = useState(field.segment_ready !== false)
  const preparing = useFetch<SamStatus>(samReady && !prepared ? `/api/annotate/${name}/outline/${enc(tile)}/segment` : null, 1500)
  useEffect(() => { if (preparing.data && preparing.data.state !== 'preparing') setPrepared(true) }, [preparing.data])
  const [labelIndex, setLabelIndex] = useState(0)
  const [draft, setDraft] = useState<number[][]>([])
  const [redrawing, setRedrawing] = useState<string | null>(null)
  const [hover, setHover] = useState<number[] | null>(null)
  const [dragging, setDragging] = useState<{ index: number; points: number[][] } | null>(null)
  const [saving, setSaving] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [complete, setComplete] = useState(field.complete)
  const [confirming, setConfirming] = useState(false)
  const [showCounted, setShowCounted] = useState(true)  // what detection counted shows from the start (R hides it)
  const [help, setHelp] = useStored<boolean>('stainid.outline.help', false)
  const [showCandidates, setShowCandidates] = useState(false)
  const reference = useFetch<OutlineReference>(showCounted || showCandidates ? `/api/annotate/${name}/outline/${enc(tile)}/reference` : null)
  const label = classes[labelIndex] ?? classes[0]
  const vocabularyOf = (meta as typeof meta & { vocabulary?: Vocabulary }).vocabulary
  const color = (value: string) => (vocabularyOf?.classes.some((c) => c.id === value) ? classColour(vocabularyOf, value)  // the same colour on every page
    : !scored(value) ? UNSURE_COLOR : CLASS_COLOURS[Math.max(0, classes.indexOf(value)) % CLASS_COLOURS.length])
  /** The key of a class: 1, 2, … for the classes that count, 0 for "unsure" (as on every labelling page). */
  const keyFor = (value: string) => {
    if (!scored(value)) return '0'
    const i = classes.filter(scored).indexOf(value)
    return i >= 0 && i < 9 ? String(i + 1) : ''
  }
  const px = field.pixel_size_um
  const box = field.box
  const position = fields.findIndex((f) => f.tile_id === tile)
  const finishedByMe = fields.filter((f) => f.done_by_me).length

  const save = useCallback(async (records: { ref: string; value?: string; points?: number[][]; deleted?: boolean }[]) => {
    setObjects((current) => {
      const next = { ...current }
      for (const r of records) {
        if (r.deleted) delete next[r.ref]
        else next[r.ref] = { ref: r.ref, value: r.value ?? current[r.ref]?.value ?? label, points: r.points ?? current[r.ref]?.points ?? [] }
      }
      return next
    })
    setSaving((n) => n + 1)
    try {
      await api.post(`/api/annotate/${name}/outline/${enc(tile)}/objects`, {
        records: records.map((r) => (r.deleted ? { ref: r.ref, deleted: true } : { ref: r.ref, value: r.value, points: r.points })),
      })
      setError(null)
    } catch (e) {
      setError(`Not saved: ${(e as Error).message}`)
    } finally {
      setSaving((n) => n - 1)
    }
  }, [name, tile, label])

  /** Ask Segment Anything for the outline of one object from its clicks, and save the outline it chooses. */
  const segment = useCallback(async (ref: string, clicks: number[][], value: string) => {
    setPending((p) => ({ ...p, [ref]: clicks[clicks.length - 1] }))
    try {
      const found = await api.post<SamOptions>(`/api/annotate/${name}/outline/${enc(tile)}/segment`, { points: clicks })
      if (!found.options.length) { setError('Segment Anything found no object there: click closer to its middle, or draw it by hand (D).'); return }
      setPrompts((p) => ({ ...p, [ref]: { clicks, options: found.options.map((o) => o.points), chosen: found.chosen } }))
      await save([{ ref, value, points: found.options[found.chosen].points }])
    } catch (e) {
      const message = (e as Error).message
      if (message.startsWith('409')) { setSamReady(false); setTool('draw') }
      setError(message)
    } finally {
      setPending((p) => { const next = { ...p }; delete next[ref]; return next })
    }
  }, [name, tile, save])

  const list = useMemo(() => Object.values(objects), [objects])
  const hit = (p: ImagePoint) => list.filter((o) => inside(o.points, p.x, p.y)).sort((a, b) => area(a.points) - area(b.points))[0] ?? null
  const chosen = selected ? objects[selected] : undefined

  const closeDraft = useCallback(() => {
    if (draft.length < 3) return
    const points = draft.map((p) => p.map(Math.round))
    const ref = redrawing ?? newRef()
    save([{ ref, value: redrawing ? objects[redrawing]?.value : label, points }])
    setPrompts((p) => { const next = { ...p }; delete next[ref]; return next })
    setSelected(ref)
    setDraft([])
    setHover(null)
    setRedrawing(null)
  }, [draft, redrawing, objects, label, save])

  const click = (p: ImagePoint, event: React.MouseEvent) => {
    if (complete) return
    if (draft.length || redrawing) {
      if (draft.length >= 3 && Math.hypot(draft[0][0] - p.x, draft[0][1] - p.y) < 10 * p.unit) closeDraft()
      else setDraft([...draft, [p.x, p.y]])
      return
    }
    const refine = event.shiftKey || event.altKey || event.button === 2
    if (refine && chosen && samReady) {  // more clicks on the selected object: on it (shift), not on it (right-click or alt)
      const before = prompts[chosen.ref]?.clicks ?? [[...centre(chosen.points).map(Math.round), 1]]
      segment(chosen.ref, [...before, [Math.round(p.x), Math.round(p.y), event.shiftKey ? 1 : 0]], chosen.value)
      return
    }
    const target = hit(p)
    if (target) { setSelected(target.ref); return }
    if (tool === 'click' && samReady) {
      const ref = newRef()
      setSelected(ref)
      segment(ref, [[Math.round(p.x), Math.round(p.y), 1]], label)
      return
    }
    if (tool === 'draw') { setSelected(null); setDraft([[p.x, p.y]]); return }
    setSelected(null)
  }

  const move = (p: ImagePoint) => {
    if (dragging) setDragging({ ...dragging, points: dragging.points.map((q, i) => (i === dragging.index ? [p.x, p.y] : q)) })
    else if (draft.length) setHover([p.x, p.y])
  }

  useEffect(() => {
    if (!dragging || !chosen) return
    const finish = () => {
      save([{ ref: chosen.ref, value: chosen.value, points: dragging.points.map((q) => q.map(Math.round)) }])
      setDragging(null)
    }
    window.addEventListener('mouseup', finish)
    return () => window.removeEventListener('mouseup', finish)
  }, [dragging, chosen, save])

  const remove = useCallback((ref: string) => { save([{ ref, deleted: true }]); setSelected(null) }, [save])
  const otherSize = useCallback((step: number) => {
    const prompt = chosen && prompts[chosen.ref]
    if (!prompt || prompt.options.length < 2) return
    const next = (prompt.chosen + step + prompt.options.length) % prompt.options.length
    setPrompts((p) => ({ ...p, [chosen.ref]: { ...prompt, chosen: next } }))
    save([{ ref: chosen.ref, value: chosen.value, points: prompt.options[next] }])
  }, [chosen, prompts, save])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (typing(e) || e.metaKey || e.ctrlKey || confirming) return
      const picked = classes.findIndex((c) => keyFor(c) === e.key)
      if (picked >= 0) {
        setLabelIndex(picked)
        if (chosen && chosen.value !== classes[picked]) save([{ ref: chosen.ref, value: classes[picked], points: chosen.points }])
        return
      }
      const key = e.key.toLowerCase()
      if (key === 's' && samReady) { setTool('click'); setDraft([]) }
      else if (key === 'd') { setTool('draw'); setDraft([]) }
      else if (key === 'r') setShowCounted((v) => !v)
      else if (e.key === 'Tab' && chosen) { e.preventDefault(); otherSize(e.shiftKey ? -1 : 1) }
      else if (e.key === 'Escape') { setDraft([]); setRedrawing(null); setSelected(null) }
      else if (e.key === 'Enter') closeDraft()
      else if (e.key === 'Backspace' || e.key === 'Delete') {
        e.preventDefault()
        if (draft.length) setDraft(draft.slice(0, -1))
        else if (chosen) remove(chosen.ref)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [classes, chosen, draft, samReady, confirming, closeDraft, remove, otherSize, save])  // eslint-disable-line react-hooks/exhaustive-deps

  const counts: Record<string, number> = {}
  for (const o of list) counts[o.value] = (counts[o.value] ?? 0) + 1
  const inBox = list.filter((o) => { const [x, y] = centre(o.points); return x >= box.x && x < box.x + box.width && y >= box.y && y < box.y + box.height }).length
  const editing = chosen && !draft.length && !redrawing ? (dragging?.points ?? chosen.points) : undefined
  const prompt = chosen ? prompts[chosen.ref] : undefined
  const busy = saving > 0 || Object.keys(pending).length > 0
  const later = [...fields.slice(position + 1), ...fields.slice(0, position)].find((f) => !f.done_by_me && f.tile_id !== tile)

  const markComplete = async (done: boolean) => {
    setSaving((n) => n + 1)
    try {
      await api.post(`/api/annotate/${name}/outline/${enc(tile)}/complete`, { complete: done })
      setComplete(done)
      reload()
      if (done) {
        const next = await api.get<{ tile_id: string | null }>(`/api/annotate/${name}/next`).catch(() => ({ tile_id: null }))
        navigate(next.tile_id && next.tile_id !== tile ? `/annotate/${name}/${enc(next.tile_id)}` : `/annotate/${name}`)
      }
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setSaving((n) => n - 1)
    }
  }

  return (
    <div className="annotate-page outline-page fit-page">
      <div className="annotate-top">
        <div className="annotate-heading">
          <span className="row small"><Link to="/annotate">← All sessions</Link><Link to={`/annotate/${name}`}>All fields of this session</Link></span>
          <h1>{meta.title}</h1>
          <div className="muted small">{meta.stain} · outline every {one} in the yellow square · field {position + 1} of {fields.length}</div>
        </div>
        <div className="annotate-progress">
          <span className="small secondary"><b>{finishedByMe}</b> of {fields.length} fields complete</span>
          <Progress done={finishedByMe} total={fields.length} />
        </div>
        <div className="annotate-actions">
          <button className="btn" onClick={() => later && navigate(`/annotate/${name}/${enc(later.tile_id)}`)} disabled={!later}
            title={later ? 'Come back to this field later' : 'No other field is waiting for you'}>Skip for now</button>
        </div>
      </div>
      {meta.instructions && <div className="instructions">{meta.instructions}</div>}
      <ErrorNote error={error} />
      <div className="annotate-workspace">
        <div className="annotate-image" onContextMenu={(e) => e.preventDefault()}>
          <ZoomViewer fit src={apiUrl(`/api/annotate/${name}/fields/${enc(tile)}/image.jpg`)}
            title={<span className="row"><span className="muted small">{tile}</span>{complete && <span className="badge status-done">complete</span>}
              {Object.keys(pending).length > 0 && <span className="small muted">Finding the outline…</span>}</span>}
            cursor={complete ? 'default' : 'crosshair'} onClick={click} onMove={move}>
            {(_zoom, size, rawUnit) => { const unit = Number.isFinite(rawUnit) && rawUnit > 0 ? rawUnit : 1; return <>
              <path d={`M0 0H${size.w}V${size.h}H0Z M${box.x} ${box.y}v${box.height}h${box.width}v${-box.height}Z`} fill="#000" fillOpacity={0.28} fillRule="evenodd" />
              <rect x={box.x} y={box.y} width={box.width} height={box.height} fill="none" stroke={BOX_COLOR} strokeWidth={2 * unit} strokeDasharray={`${12 * unit} ${8 * unit}`} />
              {showCandidates && reference.data?.candidates?.map((ring, i) => (
                <polygon key={`c${i}`} points={flatPoints(ring)} fill="none" stroke="#d0d4da" strokeWidth={2 * unit} strokeDasharray={`${5 * unit} ${4 * unit}`} />
              ))}
              {showCounted && reference.data && <>
                {reference.data.outlines.map((ring, i) => <g key={`s${i}`}>
                  <polygon points={flatPoints(ring)} fill="none" stroke="#000" strokeOpacity={0.55} strokeWidth={3.8 * unit} />
                  <polygon points={flatPoints(ring)} fill="none" stroke={REFERENCE_COLOR} strokeWidth={2 * unit} /></g>)}
                {reference.data.counted.map((c, i) => <circle key={`d${i}`} cx={c.x} cy={c.y} r={Math.max(8, c.r) + 4} fill="none" stroke={REFERENCE_COLOR}
                  strokeWidth={2 * unit} strokeDasharray={`${5 * unit} ${4 * unit}`} />)}
              </>}
              {list.map((o) => o.ref === selected && editing ? null : (
                <g key={o.ref}>
                  <polygon points={svgPoints(o.points)} fill="none" stroke="#000" strokeOpacity={0.6} strokeWidth={4.2 * unit} />
                  <polygon points={svgPoints(o.points)} fill={color(o.value)} fillOpacity={0.12} stroke={color(o.value)} strokeWidth={2.4 * unit}>
                    <title>{nameOf(o.value)}</title>
                  </polygon>
                </g>
              ))}
              {editing && <>
                <polygon points={svgPoints(editing)} fill="none" stroke="#000" strokeOpacity={0.6} strokeWidth={5 * unit} />
                <polygon points={svgPoints(editing)} fill={color(chosen!.value)} fillOpacity={0.25} stroke="#fff" strokeWidth={3 * unit} />
                {editing.length <= MAX_HANDLES && !complete && spacing(editing) / unit >= MIN_HANDLE_SPACING && editing.map((p, i) => (
                  <circle key={i} cx={p[0]} cy={p[1]} r={3.5 * unit} fill="#fff" stroke="#000" strokeWidth={0.8 * unit} style={{ cursor: 'move' }}
                    onMouseDown={(e) => { if (e.button !== 0) return; e.stopPropagation(); setDragging({ index: i, points: editing }) }} />
                ))}
                {prompt?.clicks.map((c, i) => <circle key={`k${i}`} cx={c[0]} cy={c[1]} r={4 * unit} fill={c[2] ? '#4ade80' : '#f87171'} stroke="#000" strokeWidth={unit} />)}
              </>}
              {Object.entries(pending).map(([ref, p]) => <circle key={`p${ref}`} className="sam-pending" cx={p[0]} cy={p[1]} r={10 * unit} fill="none" stroke="#fff" strokeWidth={2 * unit} />)}
              {draft.length > 0 && <>
                <polyline points={svgPoints([...draft, ...(hover ? [hover] : [])])} fill="none" stroke={color(label)} strokeWidth={2.4 * unit} />
                {draft.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r={(i === 0 ? 6 : 3.5) * unit} fill={i === 0 ? '#fff' : color(label)} stroke="#000" strokeWidth={unit} />)}
              </>}
            </> }}
          </ZoomViewer>
        </div>
        <aside className="annotate-side round-side" aria-label="The object, the keys and this field">
          {chosen ? (
            <div className="card side-block round-target" style={{ borderLeftColor: color(chosen.value) }}>
              <div className="row target-head"><span className="look-swatch" style={{ borderColor: color(chosen.value), background: color(chosen.value) }} />
                <b className="ellipsis">{nameOf(chosen.value)}</b></div>
              <dl className="target-facts small">
                <dt>Area</dt><dd>{fmt(area(chosen.points) * px * px, 3)} µm²</dd>
                {prompt && prompt.options.length > 1 && <><dt>Size</dt><dd>{prompt.chosen + 1}/{prompt.options.length}</dd></>}
              </dl>
              <div className="row" style={{ marginTop: 8 }}>
                {prompt && prompt.options.length > 1 && <button className="btn small" onClick={() => otherSize(1)}>Other size <span className="kbd">Tab</span></button>}
                <button className="btn small" onClick={() => { setDraft([]); setRedrawing(chosen.ref) }}>Redraw by hand</button>
                <button className="btn small danger-outline" onClick={() => remove(chosen.ref)}>Delete</button>
              </div>
            </div>
          ) : (
            <div className="card side-block round-target empty">
              <p className="small muted" style={{ margin: 0 }}>{samReady ? `Click the middle of a ${one}: its outline appears and is selected here.` : 'Click the corners of an outline, then the first corner or Enter.'}</p>
            </div>
          )}
          <div className="card side-block">
            <h3>Classes</h3>
            <ul className="class-legend small">
              {classes.map((c, i) => [c, i] as const).sort((a, b) => Number(!scored(a[0])) - Number(!scored(b[0]))).map(([c, i]) => (  // unsure (0) last
                <li key={c}>
                  <button className={`link-btn class-pick${c === label ? ' chosen' : ''}`} aria-pressed={c === label} title={classOf(c)?.description}
                    onClick={() => { setLabelIndex(i); if (chosen && chosen.value !== c) save([{ ref: chosen.ref, value: c, points: chosen.points }]) }}>
                    <span className="kbd key-slot">{keyFor(c) || '·'}</span><span className="look-swatch" style={{ borderColor: color(c), background: color(c) }} />
                    <span className="ellipsis">{nameOf(c)}</span></button>
                  <span className="spacer" /><span className="muted">{counts[c] ?? ''}</span>
                </li>
              ))}
            </ul>
          </div>
          <div className="card side-block">
            <h3>Keys</h3>
            <ul className="key-list small">
              {samReady && <li><span className={`kbd${tool === 'click' ? ' on' : ''}`}>S</span> click to outline{tool === 'click' ? ' (on)' : ''}</li>}
              <li><span className={`kbd${tool === 'draw' ? ' on' : ''}`}>D</span> draw by hand{tool === 'draw' ? ' (on)' : ''}</li>
              {samReady && <li><span className="kbd">Tab</span> other size · <span className="kbd">⇧ click</span> add a spot · <span className="kbd">right-click</span> leave one out</li>}
              <li><span className="kbd">drag</span> a dot to move the outline</li>
              <li><span className="kbd">Del</span> delete the selected one · <span className="kbd">Esc</span> cancel</li>
              <li><span className="kbd">R</span> {showCounted ? 'hide' : 'show'} what detection counted</li>
            </ul>
            {samReady && !prepared && tool === 'click' && <p className="muted small sam-preparing" aria-live="polite">Preparing outlines…</p>}
            {!samReady && <p className="small warning-note">Click-to-outline is not available here, so draw each outline by hand.</p>}
            <ul className="look-legend small secondary">
              <li><span className="look-swatch" style={{ borderColor: color(classes[0] ?? ''), background: color(classes[0] ?? '') }} />solid, in the class colour: your outlines</li>
              <li><span className="look-swatch" style={{ borderColor: REFERENCE_COLOR, background: REFERENCE_COLOR }} />what detection counted</li>
            </ul>
          </div>
          <div className="card side-block">
            <h3>This field</h3>
            <p className="small field-counts"><b>{inBox.toLocaleString()}</b> {inBox === 1 ? one : words} outlined in the square
              {list.length > inBox && <span className="muted"> ({list.length - inBox} more in the margin)</span>}</p>
            <label className="layer-toggle small"><input type="checkbox" checked={showCounted} onChange={() => setShowCounted(!showCounted)} />
              <span className="swatch" style={{ background: REFERENCE_COLOR }} />show what detection counted (R)</label>
            <label className="layer-toggle small"><input type="checkbox" checked={showCandidates} onChange={() => setShowCandidates(!showCandidates)} />
              everything detection considered (dotted grey)</label>
            {(showCounted || showCandidates) && !reference.data && !reference.error && <p className="muted small">Loading what detection found…</p>}
            {showCandidates && reference.data && !reference.data.candidates && <p className="muted small">Not found yet for this field; try again later.</p>}
            <ErrorNote error={reference.error} />
          </div>
          <div className="card side-block side-save">
            <span className="small muted save-state" aria-live="polite">{Object.keys(pending).length ? 'Finding the outline…' : saving ? 'Saving…' : 'All changes saved'}</span>
            <div className="row side-save-buttons">
              {complete
                ? <button className="btn" onClick={() => markComplete(false)} disabled={saving > 0}>Reopen this field</button>
                : <button className="btn primary big" disabled={busy || draft.length > 0} onClick={() => setConfirming(true)}
                  title={draft.length ? 'Finish or cancel (Esc) the outline you are drawing first' : undefined}>Mark field complete…</button>}
            </div>
          </div>
          <div className="side-foot small">
            <button className={`btn small help-toggle${help ? ' chosen' : ''}`} aria-expanded={help} onClick={() => setHelp(!help)} title="How this page works">?</button>
            <span className="spacer" /><span className="muted">{tile}</span>
          </div>
          {help && (
            <div className="card side-block round-help">
              <p className="small secondary">{tool === 'click' && samReady
                ? `Click the middle of a ${one}: its outline appears. Not right? Tab tries another size, Shift+click adds a spot that belongs to it, right-click a spot that does not.`
                : 'Click the corners of the outline, then the first corner or press Enter.'}</p>
              <p className="small secondary">Outline every {one} whose middle lies inside the yellow square, also faint, small and touching ones — each one separately.
                Objects reaching over the edge are outlined whole. When none are left, press <b>Mark field complete</b>.</p>
              <p className="small muted">{classOf(label)?.description} {choices.some((c) => !c.scored)
                && `Use “${choices.filter((c) => !c.scored).map((c) => c.name).join('” or “')}” (0) for something you cannot decide: it is left out of the scores.`}</p>
              <p className="small muted">Scroll to zoom; drag (or hold the middle button) to move the picture.</p>
              <details className="side-details compact">
                <summary className="small">Go to another field</summary>
                <p className="muted small">Green = complete.</p>
                <div className="field-jump">
                  {fields.map((f, i) => (
                    <Link key={f.tile_id} to={`/annotate/${name}/${enc(f.tile_id)}`} title={f.tile_id}
                      className={`field-chip${f.tile_id === tile ? ' current' : f.done_by_me ? ' done' : ''}`}>{i + 1}{f.open_by_others > 0 && <span className="busy-dot" />}</Link>
                  ))}
                </div>
              </details>
            </div>
          )}
        </aside>
      </div>
      {confirming && (
        <Modal title="Is every object outlined?" onClose={() => setConfirming(false)}>
          <p>{inBox ? <>You outlined <b>{inBox.toLocaleString()}</b> {inBox === 1 ? one : words} in the yellow square.</>
            : <>You outlined <b>no {words}</b> in this field. Mark it complete only if it really has none.</>}</p>
          <ul className="small secondary">
            <li>Faint and small ones are outlined too.</li>
            <li>Touching ones are outlined each on their own, and one large object as one outline.</li>
            <li>Nothing in the square is left without an outline.</li>
          </ul>
          <div className="row"><span className="spacer" />
            <button className="btn" onClick={() => setConfirming(false)}>Not yet</button>
            <button className="btn primary" autoFocus onClick={() => { setConfirming(false); markComplete(true) }}>Yes, every object is outlined</button>
          </div>
        </Modal>
      )}
    </div>
  )
}
