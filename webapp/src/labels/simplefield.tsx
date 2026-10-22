/* "Fix the field (simple)": a field worked in passes, one simple class at a time (labels/simple.py on the server).
   Pick a class (its button, or its number key): the candidates in it light up — those detection put there are lit
   already, the person only fixes them. A click on a candidate puts it in the class or, when it is in it, takes it out;
   a click on empty tissue adds a missed object to the class (one click: Segment Anything outlines it). A candidate has
   one class: the last click wins (it moves). Members of the other classes show faint in their own colour; what is in
   no class is dotted grey ("not an object"). Then the next class. AT8 "neurite" is a magic wand over the thread
   segments detection found: a click takes a segment and the ones near it (one slider: how far), Shift+click takes them
   out; the thread lines show only in that pass. There is no outline editing: outlines stay detection's.
   Nothing is sent before Save; Save and continue finishes the field, and every candidate left as it started is saved
   in the class it started in, those in no class as "not an object". Locks, the read-only finished field with Redo /
   Add an extra annotation, the draft in this browser and the progress are those of the detailed mode (fixfield.tsx). */
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import type { Label, RoundDetail, RoundField, RoundObject, RoundProgress, RoundSamOptions, RoundSaved, SessionDetail, SimpleCheck, SimpleClassDef } from '../api'
import { api, apiUrl, enc, pendingText } from '../api'
import { ErrorNote, Loading } from '../components'
import { INACTIVE_GREY } from '../classPalette'
import { PALETTE } from '../palette'
import { useFetch, useStored } from '../hooks'
import type { ImagePoint } from '../viewer'
import { RASTER_LAYERS, ZoomViewer } from '../viewer'
import { HALO_PX, LIT_PX, OUTLINE_PX, lineScale } from '../viewkeys'
import type { RoundEvent } from './rounds'
import { BackToSets, RoundProgressLine, SaveButtons, SaveState, changesText, draftKey, polygonPoints, readDraft, useLeaveGuard, useLive, useSaveKeys,
  writeDraft } from './rounds'
import type { ThreadLadder } from './threads'
import { nearestSegment, segmentsOf, wandGroup } from './wand'

type Meta = SessionDetail['meta'] & RoundDetail
type Mine = Label & { candidate?: string | null }
type Rec = Record<string, unknown>
const NONE = 'none'
const rkey = (x: number, y: number) => `${Math.round(x)}:${Math.round(y)}`
const BOX_COLOR = PALETTE.markers.box
/** The keys that only look: they work on a finished field opened to look at. */
const VIEW_KEYS = new Set([' ', 'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', '+', '=', '-', '_', '−'])

/** One object on the page: a candidate of the store, or one this person added. */
interface Shown { key: string; x: number; y: number; r: number; candidate?: RoundObject; added?: Mine }
/** An unsaved change of one label: what to send on Save, and how it looks on the page meanwhile (null: taken away). */
interface Change { mark: Rec | null; label: Mine | null }
/** One step back with Z: the changes and the wand's segments as they were. */
interface Undo { changes: Record<string, Change>; segments: number[] }
interface Hooks { onEvent?: (event: RoundEvent) => void; onFinished?: () => void; coach?: ReactNode }

function inside(polygon: number[][], x: number, y: number) {
  let hit = false
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i]
    const [xj, yj] = polygon[j]
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) hit = !hit
  }
  return hit
}
function polygonArea(points: number[][]) {
  let sum = 0
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) sum += points[j][0] * points[i][1] - points[i][0] * points[j][1]
  return Math.abs(sum) / 2
}
const finishedText = (finished: { person: string; time: number }[]) => finished.map((f) => `${f.person} on ${new Date(f.time * 1000)
  .toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })}`).join(', ')

/* ---------- the page ---------- */

export function SimpleFixer({ detail, reload, tile, field, onEvent, onFinished, coach, onOpen }: { detail: SessionDetail; reload: () => void; tile: string
  field: RoundField; onOpen: (how: 'edit' | 'own') => void } & Hooks) {
  const meta = detail.meta as Meta
  const name = meta.name
  const practice = !!(meta as { practice?: boolean }).practice
  const vocab = meta.simple!
  const classes = vocab.classes
  const objectClasses = classes.filter((c) => c.kind === 'objects')
  const wandClass = classes.find((c) => c.kind === 'wand' && meta.stain === 'AT8') ?? null
  const passes = wandClass ? [...objectClasses, wandClass] : objectClasses
  const byId = useMemo(() => new Map(classes.map((c) => [c.id, c])), [classes])
  const readOnly = !!field.read_only
  const tell = (event: RoundEvent) => onEvent?.(event)
  const navigate = useNavigate()
  useEffect(() => {  // looking only: no key edits anything
    if (!readOnly) return
    const stop = (e: KeyboardEvent) => {
      if (VIEW_KEYS.has(e.key) || ['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement).tagName)) return
      e.stopImmediatePropagation()
    }
    window.addEventListener('keydown', stop, true)
    return () => window.removeEventListener('keydown', stop, true)
  }, [readOnly])

  const draftAt = practice ? '' : draftKey(detail.annotator, name, tile)
  const kept = readOnly ? null : readDraft<{ changes: Record<string, Change>; segments?: number[] }>(draftAt)
  const [saved, setSaved] = useState<Record<string, Mine>>(() => Object.fromEntries(field.mine.map((m) => [m.id, m])))
  const [changes, setChanges] = useState<Record<string, Change>>(() => kept?.changes ?? {})
  const [restored, setRestored] = useState(() => Object.keys(kept?.changes ?? {}).length)
  const savedArea = Object.values(saved).find((m) => m.task === 'simple_area' && m.value === wandClass?.id)
  const [segmentsOn, setSegmentsOn] = useState<number[]>(() => kept?.segments ?? savedArea?.meta?.segments ?? [])
  const [segmentsSaved, setSegmentsSaved] = useState<number[]>(() => savedArea?.meta?.segments ?? [])
  const [active, setActive] = useState<string>(objectClasses[0]?.id ?? '')
  const [reach, setReach] = useStored<number>('stainid.simple.wand.reach', vocab.wand.start)
  const [hovered, setHovered] = useState<string | null>(null)
  const [hidden, setHidden] = useState(false)
  const [undos, setUndos] = useState<Undo[]>([])
  const [saving, setSaving] = useState(false)
  const [pending, setPending] = useState<number[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [checked, setChecked] = useState<Record<string, SimpleCheck>>(field.simple?.checked ?? {})
  const [openBy, setOpenBy] = useState<string[]>(field.open_by)
  const [doneBy, setDoneBy] = useState<string[]>(field.done_by)
  const [progress, setProgress] = useState<RoundProgress | null>(meta.progress)
  const [menu, setMenu] = useState<{ key: string; x: number; y: number } | null>(null)
  const [layers, setLayers] = useStored<Record<string, boolean>>('stainid.simple.layers', { box: true })
  const start = field.simple?.start ?? {}
  const sam = meta.segment_anything && field.segment_anything
  const wandOn = !!wandClass && active === wandClass.id
  const ladder = useFetch<ThreadLadder>(wandClass && !practice ? `/api/annotate/${name}/fields/${enc(tile)}/threads` : null)
  const lines = ladder.data?.status === 'done' && ladder.data.steps.length ? ladder.data.steps[ladder.data.detection].lines : null
  const segments = useMemo(() => (lines ? segmentsOf(lines) : []), [lines])
  const px = field.pixel_size_um

  useLive(name, { tile_id: tile, view: readOnly, own: !!field.own }, (live) => {
    if (live.simple_checked) setChecked(live.simple_checked)
    if (live.open_by) setOpenBy(live.open_by)
    if (live.done_by) setDoneBy(live.done_by)
    if (live.progress) setProgress(live.progress)
  })

  const mine = useMemo(() => {  // this person's labels as the page shows them: saved, with the unsaved changes on top
    const out = { ...saved }
    for (const [id, c] of Object.entries(changes)) { if (c.label) out[id] = c.label; else delete out[id] }
    return out
  }, [saved, changes])
  const myCalls = useMemo(() => {  // by candidate (a change after a save wins) and, for objects people added, by position
    const out = new Map<string, Mine>()
    for (const m of [...Object.values(saved), ...Object.values(changes).map((c) => c.label)]) {
      if (!m || m.task !== 'simple') continue
      out.set(m.candidate ? m.candidate : `at:${rkey(m.x!, m.y!)}`, m)
    }
    for (const [id, c] of Object.entries(changes)) {  // taken away before a save
      const before = saved[id]
      if (!c.label && before?.task === 'simple') out.delete(before.candidate ? before.candidate : `at:${rkey(before.x!, before.y!)}`)
    }
    return out
  }, [saved, changes])
  const segmentsDirty = segmentsOn.length !== segmentsSaved.length || segmentsOn.some((s) => !segmentsSaved.includes(s))
  const unsaved = Object.keys(changes).filter((id) => !id.startsWith('outline:')).length + (segmentsDirty ? 1 : 0)
  const discarding = useRef(false)
  useEffect(() => { if (!discarding.current) writeDraft(draftAt, unsaved ? { changes, segments: segmentsOn } : null) }, [draftAt, changes, segmentsOn, unsaved])

  const own = useMemo(() => field.objects.filter((o) => o.own), [field.objects])
  const added = [...myCalls.values()].filter((m) => !m.candidate && m.value !== NONE && m.x != null)
  const shown: Shown[] = [
    ...own.map((o) => ({ key: o.candidate_id, x: o.x, y: o.y, r: o.radius_px, candidate: o })),
    ...added.map((m) => ({ key: `add:${rkey(m.x!, m.y!)}`, x: m.x!, y: m.y!, r: 8, added: m })),
  ]
  const byKey = new Map(shown.map((s) => [s.key, s]))
  const target = hovered ? byKey.get(hovered) : undefined

  /** Where an object stands: this person's class, else what this session's people said, else where detection put it. */
  const callOf = (s: Shown): { value: string; by: 'me' | 'others' | 'ai'; who?: string[] } => {
    if (s.added) return { value: s.added.value, by: 'me' }
    const id = s.candidate!.candidate_id
    const my = myCalls.get(id)
    if (my) return { value: my.value, by: 'me' }
    if (checked[id]) return { value: checked[id].value, by: 'others', who: checked[id].by }
    return { value: start[id] ?? NONE, by: 'ai' }
  }
  const outlineOf = (s: Shown): number[][] | null => {
    if (s.candidate) return s.candidate.masks[s.candidate.mask] ?? s.candidate.masks.find(Boolean) ?? null
    return mine[`outline:${tile}:${rkey(s.x, s.y)}`]?.points ?? null
  }
  const hit = (p: ImagePoint): Shown | null => {
    let best: Shown | null = null
    let distance = Infinity
    for (const s of shown) {
      const d = Math.hypot(s.x - p.x, s.y - p.y)
      if (d < Math.max(5, Math.min(s.r, 12)) + 3 * p.unit && d < distance) { best = s; distance = d }
    }
    if (best) return best
    const under = shown.filter((s) => { const poly = outlineOf(s); return poly && inside(poly, p.x, p.y) })
    return under.sort((a, b) => polygonArea(outlineOf(a)!) - polygonArea(outlineOf(b)!))[0] ?? null
  }

  /** Puts changes on the page (nothing is sent before Save); each call is one step Z takes back. */
  const remember = () => setUndos((u) => [{ changes, segments: segmentsOn }, ...u].slice(0, 200))
  const put = (records: Rec[]) => {
    remember()
    const next = { ...changes }
    for (const r of records) {
      const x = Number(r.x), y = Number(r.y)
      const id = `${r.task}:${tile}:${rkey(x, y)}`
      if (r.deleted) { if (saved[id]) next[id] = { mark: r, label: null }; else delete next[id]; continue }
      next[id] = { mark: r, label: { id, source: detail.annotator ?? '', stain: meta.stain, task: r.task as Mine['task'], tile_id: tile, x, y,
        value: String(r.value ?? ''), subtype: null, points: (r.points as number[][]) ?? null, session: name, confirmed: false, time: Date.now() / 1000,
        candidate: (r.candidate as string) ?? null } }
    }
    setChanges(next)
  }
  /** This object goes into this class ("none": into no class; an object this person added is taken away then). */
  const giveClass = (s: Shown, value: string) => {
    if (s.added) {
      if (value === NONE) put([{ task: 'simple', x: s.x, y: s.y, deleted: true }, { task: 'outline', x: s.x, y: s.y, deleted: true }])
      else put([{ task: 'simple', x: s.x, y: s.y, value }])
      return
    }
    const o = s.candidate!
    put([{ task: 'simple', candidate: o.candidate_id, x: o.x, y: o.y, value }])
  }
  const toggle = (s: Shown) => {
    const now = callOf(s).value
    giveClass(s, now === active ? NONE : active)
    tell(now === active ? 'unpick' : 'pick')
  }
  /** One click on empty tissue: Segment Anything outlines it, an object of the class now active. */
  const addAt = async (x: number, y: number) => {
    const at = [Math.round(x), Math.round(y)]
    if (!sam) { put([{ task: 'simple', x: at[0], y: at[1], value: active }]); tell('added'); return }
    setPending(at)
    try {
      const found = await api.post<RoundSamOptions>(`/api/annotate/${name}/fields/${enc(tile)}/segment`, { points: [[at[0], at[1], 1]] })
      const best = found.options[found.chosen] ?? found.options[0]
      if (!best) { setError('Segment Anything found no object there: click closer to its middle.'); return }
      put([{ task: 'simple', x: at[0], y: at[1], value: active }, { task: 'outline', x: at[0], y: at[1], points: best.points.map((q) => q.map(Math.round)) }])
      setHovered(`add:${rkey(at[0], at[1])}`)
      setError(null)
      tell('added')
    } catch (e) {
      setError(`${(e as Error).message} Try again in a moment.`)
    } finally {
      setPending(null)
    }
  }
  const wandClick = (p: ImagePoint, shift: boolean) => {
    if (!segments.length) { setError(ladder.pending ? 'The threads of this field are still being drawn: try again in a moment.' : 'No threads were found on this field.'); return }
    const at = nearestSegment(segments, p.x, p.y, 8 * p.unit)
    if (at < 0) { setError('Click right on a thread (the thin white lines) to choose it.'); return }
    setError(null)
    const group = wandGroup(segments, at, reach / px)
    remember()
    setSegmentsOn((on) => (shift ? on.filter((i) => !group.includes(i)) : [...new Set([...on, ...group])]))
    tell('wand')
  }
  const click = (p: ImagePoint, event: React.MouseEvent) => {
    if (event.button !== 0 || readOnly) return
    setMenu(null)
    if (wandOn) { wandClick(p, event.shiftKey); return }
    if (!active) return
    const on = hit(p)
    if (on) { if (!on.candidate?.excluded) toggle(on); return }
    if (!pending) void addAt(p.x, p.y)
  }
  const move = (p: ImagePoint) => {
    const now = wandOn ? null : hit(p)?.key ?? null
    if (now && now !== hovered) tell('hover')
    setHovered(now)
  }
  const undo = () => {
    const last = undos[0]
    if (!last) return
    setUndos((u) => u.slice(1))
    setChanges(last.changes)
    setSegmentsOn(last.segments)
    tell('undo')
  }
  const choosePass = (id: string) => { setActive(id); setHovered(null); setMenu(null); tell('pass') }
  const nextPass = () => {
    const at = passes.findIndex((c) => c.id === active)
    if (at < passes.length - 1) { choosePass(passes[at + 1].id); tell('next-pass') }
  }
  useEffect(() => {  // number keys: the passes; Del: out of every class; Z: undo; Space: a clean look
    const onKey = (e: KeyboardEvent) => {
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement).tagName) || e.metaKey || e.ctrlKey || e.altKey) return
      const pass = passes.find((c) => c.key && c.key === e.key)
      if (pass) { e.preventDefault(); choosePass(pass.id); return }
      if (e.key === ' ') { e.preventDefault(); setHidden((h) => !h); tell('overlays'); return }
      if (e.key === 'Escape') { setMenu(null); return }
      if (readOnly) return
      if ((e.key === 'Delete' || e.key === 'Backspace') && target) { e.preventDefault(); giveClass(target, NONE); tell('unpick'); return }
      if (e.key.toLowerCase() === 'z' && !e.repeat) { e.preventDefault(); undo() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  /** Sends these labels (and, with `finish`, finishes the field) in one request: all of them are saved, or none. */
  const send = async (records: Rec[], finish: boolean) => {
    setSaving(true)
    try {
      const result = await api.post<RoundSaved>(`/api/annotate/${name}/tiles/${enc(tile)}/save`, { records: records.map((r) => ({ tile_id: tile, ...r })), submit: finish, own: !!field.own })
      setSaved((current) => {
        const next = { ...current }
        for (const r of result.saved) { if (r.deleted) delete next[r.id]; else next[r.id] = r }
        return next
      })
      setProgress(result.progress)
      setError(null)
      return true
    } catch (e) {
      setError(`Not saved: ${(e as Error).message.replace(/^\d{3}: /, '')} — your changes are still here.`)
      return false
    } finally {
      setSaving(false)
    }
  }
  const saveAll = async (finish: boolean) => {
    if (saving || readOnly || (!finish && !unsaved)) return
    const marks = Object.values(changes).flatMap((c) => (c.mark ? [c.mark] : []))
      .map((m) => (m.candidate ? { task: m.task, candidate: m.candidate, value: m.value, deleted: m.deleted } : m))
    if (segmentsDirty && wandClass) marks.push({ task: 'simple_area', value: wandClass.id, meta: { segments: segmentsOn, reach_um: reach }, deleted: !segmentsOn.length })
    if (!(await send(marks, finish))) return
    setChanges({}); setUndos([]); setRestored(0); setSegmentsSaved(segmentsOn)
    writeDraft(draftAt, null)
    tell(finish ? 'finish' : 'save')
    if (!finish) return
    if (onFinished) { onFinished(); return }
    reload()
    const next = await api.get<{ tile_id: string | null }>(`/api/annotate/${name}/next`).catch(() => ({ tile_id: null }))
    navigate(next.tile_id && next.tile_id !== tile ? `/annotate/${name}/${enc(next.tile_id)}` : `/annotate/${name}`)
  }
  const mayLeave = useLeaveGuard(unsaved)
  useSaveKeys(() => { void saveAll(false) }, () => { void saveAll(true) })
  const skip = async () => {
    if (!mayLeave()) return
    const next = await api.get<{ tile_id: string | null }>(`/api/annotate/${name}/next?skip=${enc(tile)}`).catch(() => ({ tile_id: null }))
    if (next.tile_id && next.tile_id !== tile) navigate(`/annotate/${name}/${enc(next.tile_id)}`)
    else setError('No other field is waiting for you right now.')
  }
  const discardAndClose = () => {
    if (unsaved && !window.confirm(`Discard ${changesText(unsaved)} on this field and close it? They are not saved and cannot be brought back.`)) return
    writeDraft(draftAt, null)
    discarding.current = true
    navigate(`/annotate/${name}`)
  }
  const finishedBy = field.finished ?? []
  const iFinished = finishedBy.some((f) => f.person === detail.annotator)
  const workOnIt = () => {
    const ask = iFinished
      ? 'Redo this field? You change your own saved labels on it; the field stays yours while you work on it.'
      : `Add an extra annotation of this field? You start from detection’s output and your own labels only; ${finishedBy.map((f) => f.person).join(', ')}’s labels stay as they are.`
    if (window.confirm(ask)) onOpen(iFinished ? 'edit' : 'own')
  }
  const openMenu = (_p: ImagePoint, e: React.MouseEvent) => {
    if (readOnly || wandOn || !target) { setMenu(null); return }
    setMenu({ key: target.key, x: e.clientX, y: e.clientY })
  }

  const counts = useMemo(() => {  // how many objects of the field are in each class now
    const out: Record<string, number> = {}
    for (const s of shown) { if (s.candidate?.excluded) continue; const v = callOf(s).value; out[v] = (out[v] ?? 0) + 1 }
    return out
  }, [shown, myCalls, checked])  // eslint-disable-line react-hooks/exhaustive-deps
  const changedCandidates = useMemo(() => new Set(Object.values(changes).map((c) => (c.mark?.candidate as string | undefined) ?? c.label?.candidate ?? '').filter(Boolean)), [changes])
  const isUnsaved = (s: Shown) => (s.candidate ? changedCandidates.has(s.candidate.candidate_id) : !!changes[`simple:${tile}:${rkey(s.x, s.y)}`])
  const drawn = useMemo(() => shown.map((s) => {
    const call = callOf(s)
    const poly = outlineOf(s)
    return { key: s.key, x: s.x, y: s.y, r: s.r, value: call.value, colour: byId.get(call.value)?.colour ?? INACTIVE_GREY, others: call.by === 'others',
      points: poly ? polygonPoints(poly) : null, faded: !!s.candidate?.excluded, unsaved: isUnsaved(s) }
  }), [own, myCalls, checked, changes, byId])  // eslint-disable-line react-hooks/exhaustive-deps
  const passIndex = passes.findIndex((c) => c.id === active)
  const activeClass = byId.get(active)
  const fields = detail.fields
  const position = fields.findIndex((f) => f.tile_id === tile)
  const done = !!saved[`done:${tile}`]

  return (
    <div className="annotate-page round-page fit-page simple-page">
      <div className="annotate-top">
        <div className="annotate-heading">
          <span className="row small"><BackToSets />{!practice && <Link to={`/annotate/${name}`}>All fields of this session</Link>}</span>
          <h1>{meta.title}</h1>
          <div className="muted small">{meta.stain} · Fix the field (simple) · field {position + 1} of {fields.length}</div>
        </div>
        <div className="annotate-progress"><RoundProgressLine progress={progress} me={detail.annotator} /></div>
        <div className="annotate-actions">
          <button className="btn" onClick={() => void skip()} disabled={saving} title="Come back to this field later: the next field waiting for you opens">Skip for now</button>
        </div>
      </div>
      {meta.instructions && <div className="instructions">{meta.instructions}</div>}
      <ErrorNote error={error} />
      {restored > 0 && <div className="found found-warn small round-restored" role="status">{changesText(restored)} you had not saved came back from this browser.
        They are not saved yet. <button className="link-btn small" onClick={() => { setChanges({}); setUndos([]); setRestored(0); setSegmentsOn(segmentsSaved) }}>Discard them</button></div>}
      {readOnly && <div className="found found-good field-finished" role="status">
        <span><b>Done by {finishedText(finishedBy)}.</b> You are looking at it: nothing here can be changed.</span>
        <span className="spacer" />
        {openBy.length > 0 ? <span className="small">{openBy.join(', ')} {openBy.length === 1 ? 'is' : 'are'} working on it now.</span>
          : <button className="btn small primary" onClick={workOnIt}>{iFinished ? 'Redo this field' : 'Add an extra annotation'}</button>}
      </div>}
      {field.own && <div className="found small field-others" role="status">Extra annotation: you start from detection’s output and your own labels only.
        {' '}{finishedBy.length ? `${finishedBy.map((f) => f.person).join(', ')}’s labels stay as they are; yours are added alongside.` : ''}</div>}
      {!readOnly && !field.own && (openBy.length > 0 || doneBy.length > 0) && <div className="found found-warn small field-others" role="status">
        {doneBy.length > 0 ? `${doneBy.join(', ')} already finished this field.` : `${openBy.join(', ')} ${openBy.length === 1 ? 'is' : 'are'} working on this field right now.`}</div>}
      <div className="annotate-workspace">
        <div className="annotate-image" onContextMenu={(e) => e.preventDefault()} onMouseLeave={() => { if (!menu) setHovered(null) }}>
          {menu && byKey.get(menu.key) && <SimpleMenu at={menu} classes={objectClasses} current={callOf(byKey.get(menu.key)!).value}
            onPick={(value) => { const s = byKey.get(menu.key)!; setMenu(null); giveClass(s, value); tell('menu') }} onClose={() => setMenu(null)} />}
          <ZoomViewer fit src={apiUrl(`/api/annotate/${name}/fields/${enc(tile)}/image.jpg`)}
            layers={hidden ? [] : field.layers.filter((l) => layers[l]).map((l) => apiUrl(`/api/annotate/${name}/fields/${enc(tile)}/layer/${l}.png`))}
            title={<span className="row"><span className="muted small">{tile}</span>{done && <span className="badge status-done">finished</span>}
              {activeClass && !readOnly && <span className="badge pass-badge" style={{ borderColor: activeClass.colour }}>
                <span className="look-swatch" style={{ borderColor: activeClass.colour, background: activeClass.colour }} />{activeClass.name}
                {wandOn ? ' — click threads' : ' — click to add or take out'}</span>}
              {hidden && <span className="badge">Overlays hidden · <span className="kbd">Space</span></span>}
              {pending && <span className="small muted">Outlining…</span>}</span>}
            cursor={readOnly ? 'default' : wandOn ? 'crosshair' : target ? 'pointer' : 'copy'}
            onClick={readOnly ? undefined : click} onMove={move} onContext={openMenu}
            under={(zoom, _s, rawUnit) => hidden ? null : <SimpleLayer items={drawn} unit={Number.isFinite(rawUnit) && rawUnit > 0 ? rawUnit : 1} zoom={zoom}
              active={wandOn ? null : active} dim={wandOn ? 0.55 : 1} />}>
            {(zoom, size, rawUnit) => {
              const unit = Number.isFinite(rawUnit) && rawUnit > 0 ? rawUnit : 1
              if (hidden) return null
              const box = field.box
              const lit = target
              const litPoly = lit ? outlineOf(lit) : null
              const litCall = lit ? callOf(lit) : null
              const colour = litCall ? byId.get(litCall.value)?.colour ?? INACTIVE_GREY : '#fff'
              return <>
                {wandOn && lines && <WandLines lines={lines} on={segmentsOn} unit={unit} colour={wandClass!.colour} />}
                {layers.box && <>
                  <path d={`M0 0H${size.w}V${size.h}H0Z M${box.x} ${box.y}v${box.height}h${box.width}v${-box.height}Z`} fill="#000" fillOpacity={0.22} fillRule="evenodd" />
                  <rect x={box.x} y={box.y} width={box.width} height={box.height} fill="none" stroke={BOX_COLOR} strokeWidth={1.5 * unit} strokeDasharray={`${12 * unit} ${8 * unit}`} />
                </>}
                {lit && litCall && (() => {
                  const width = LIT_PX * lineScale(zoom) * unit
                  const shape = (extra: Record<string, unknown>) => litPoly ? <polygon points={polygonPoints(litPoly)} {...extra} />
                    : <circle cx={lit.x} cy={lit.y} r={Math.max(6, lit.r)} {...extra} />
                  const xs = litPoly ? litPoly.map((p) => p[0]) : [lit.x + lit.r]
                  const ys = litPoly ? litPoly.map((p) => p[1]) : [lit.y - lit.r]
                  const tx = Math.min(size.w - 4 * unit, Math.max(...xs) + 6 * unit), ty = Math.max(14 * unit, Math.min(...ys) - 4 * unit)
                  const text = litCall.value === NONE ? 'not an object' : byId.get(litCall.value)?.name ?? litCall.value
                  return <g className="round-object lit" pointerEvents="none">
                    {shape({ fill: 'none', stroke: '#ffffff', strokeOpacity: 0.55, strokeWidth: width + 4.5 * unit })}
                    {shape({ fill: 'none', stroke: '#000', strokeOpacity: 0.6, strokeWidth: width + HALO_PX * unit, strokeDasharray: litCall.value === NONE ? `${5 * unit} ${4 * unit}` : undefined })}
                    {shape({ fill: colour, fillOpacity: 0.2, stroke: colour, strokeWidth: width, strokeDasharray: litCall.value === NONE ? `${5 * unit} ${4 * unit}` : undefined })}
                    <text className="object-tag" x={tx} y={ty} fontSize={13 * unit} strokeWidth={3.5 * unit} textAnchor={tx > size.w * 0.8 ? 'end' : 'start'}>{text}
                      {litCall.by === 'others' ? ` · ${litCall.who?.join(', ')}` : ''}</text>
                  </g>
                })()}
                {pending && <circle className="sam-pending" cx={pending[0]} cy={pending[1]} r={10 * unit} fill="none" stroke="#fff" strokeWidth={2 * unit} />}
              </>
            }}
          </ZoomViewer>
        </div>
        <aside className="annotate-side round-side" aria-label="The classes, the keys and this field">
          {coach}
          <div className="card side-block pass-card" role="group" aria-label="Classes: one at a time">
            <h3>One class at a time</h3>
            <div className="pass-list">
              {passes.map((c, i) => (
                <button key={c.id} className={`pass-button${c.id === active ? ' chosen' : ''}`} onClick={() => choosePass(c.id)} aria-pressed={c.id === active}
                  title={c.description} style={{ borderColor: c.id === active ? c.colour : undefined }}>
                  <span className="kbd key-slot">{c.key || i + 1}</span>
                  <span className="look-swatch" style={{ borderColor: c.colour, background: c.colour }} />
                  <span className="pass-name">{c.name}</span>
                  <span className="small muted">{c.kind === 'wand' ? `${segmentsOn.length} threads` : (counts[c.id] ?? 0)}</span>
                </button>
              ))}
            </div>
            <p className="small muted" style={{ margin: '6px 0 0' }}>{activeClass?.description}</p>
            {!readOnly && <div className="row" style={{ marginTop: 6 }}>
              {passIndex < passes.length - 1 ? <button className="btn" onClick={nextPass}>Next class: {passes[passIndex + 1]?.name} →</button>
                : <span className="small secondary">Last class: when the field looks right, Save and continue.</span>}
            </div>}
          </div>
          {wandOn && <div className="card side-block wand-card" role="group" aria-label="The magic wand">
            <h3>Magic wand</h3>
            <p className="small" style={{ margin: '0 0 6px' }}>Click a thread: it and the threads near it are chosen. Click another thread to add more,
              {' '}<span className="kbd">Shift</span>+click to take some out.</p>
            <label className="field small">How far it reaches: <b>{reach.toFixed(0)} µm</b>
              <input type="range" min={vocab.wand.from} max={vocab.wand.to} step={1} value={reach} onChange={(e) => setReach(Number(e.target.value))} /></label>
            {ladder.pending && <Loading inline label={pendingText(ladder.pending)} />}
            {ladder.data && ladder.data.status !== 'done' && <p className="small muted">No threads were found on this field.</p>}
            <p className="small muted" style={{ margin: '4px 0 0' }}>{segmentsOn.length} of {segments.length} thread pieces chosen.
              {segmentsOn.length > 0 && !readOnly && <> <button className="link-btn small" onClick={() => { remember(); setSegmentsOn([]) }}>Clear</button></>}</p>
          </div>}
          <div className="card side-block">
            <h3>How it works</h3>
            <ul className="key-list small">
              <li><span className="kbd">click</span> an object: into this class, or out of it</li>
              <li><span className="kbd">click</span> empty tissue: a missed object of this class</li>
              <li><span className="kbd">right-click</span> an object: choose its class from a list</li>
              <li><span className="kbd">1</span>–<span className="kbd">{passes.length}</span> the classes · <span className="kbd">Del</span> out of every class</li>
              <li><span className="kbd">Z</span> undo · <span className="kbd">Space</span> hide the drawings · <span className="kbd">+</span> <span className="kbd">−</span> zoom · arrows move</li>
            </ul>
            <ul className="look-legend small secondary">
              <li><span className="look-swatch" style={{ borderColor: activeClass?.colour, background: activeClass?.colour }} />in the class you are on: bright</li>
              <li><span className="look-swatch faint-swatch" style={{ borderColor: objectClasses.find((c) => c.id !== active)?.colour }} />in another class: faint, its own colour</li>
              <li><span className="look-swatch look-removed" style={{ borderColor: INACTIVE_GREY }} />in no class (not an object): dotted grey</li>
            </ul>
          </div>
          <div className="card side-block">
            <h3>This field</h3>
            <p className="small field-counts">{objectClasses.map((c) => <span key={c.id}><b>{counts[c.id] ?? 0}</b> {c.name} · </span>)}
              <b>{counts[NONE] ?? 0}</b> in no class{added.length ? <> · <b>{added.length}</b> added by you</> : null}</p>
            <p className="small muted" style={{ margin: 0 }}>Save and continue keeps every object as you leave it: those in a class in that class, the
              others as “not an object”.</p>
            <details className="side-details compact">
              <summary className="small">Display options</summary>
              {field.layers.map((l) => <label key={l} className="layer-toggle"><input type="checkbox" checked={!!layers[l]} onChange={() => setLayers({ ...layers, [l]: !layers[l] })} />
                <span className="swatch" style={{ background: RASTER_LAYERS[l]?.color }} />{RASTER_LAYERS[l]?.label ?? l}</label>)}
              <label className="layer-toggle"><input type="checkbox" checked={!!layers.box} onChange={() => setLayers({ ...layers, box: !layers.box })} />field boundary (yellow)</label>
            </details>
          </div>
          {readOnly ? <div className="card side-block side-save">
            <p className="small" style={{ margin: '0 0 6px' }}>Done by {finishedText(finishedBy)}.</p>
            <div className="row">{openBy.length > 0 ? <span className="small muted">{openBy.join(', ')} {openBy.length === 1 ? 'is' : 'are'} working on it now.</span>
              : <button className="btn primary" onClick={workOnIt}>{iFinished ? 'Redo this field' : 'Add an extra annotation'}</button>}
              <span className="spacer" /><Link className="link-btn small" to={`/annotate/${name}`}>Back to the fields</Link></div>
          </div> : <div className="card side-block side-save">
            <SaveState unsaved={unsaved} saving={saving} />
            <div className="row side-save-buttons">
              <SaveButtons unsaved={unsaved} saving={saving || !!pending} onSave={() => void saveAll(false)} onContinue={() => void saveAll(true)} canContinue={!saving && !pending}
                continueTitle="Save and finish this field: every object counts as you leave it (Ctrl+Enter)" />
            </div>
            {!onFinished && <div className="row side-discard"><span className="spacer" />
              <button className="link-btn small" onClick={discardAndClose} title="Drop your unsaved changes on this field and go back to the session’s page">Discard and close</button></div>}
          </div>}
          <div className="side-foot small">{!onFinished && <Link to="/tutorial?kind=simple">Tutorial</Link>}<span className="spacer" /><span className="muted">{tile}</span></div>
        </aside>
      </div>
    </div>
  )
}

interface Drawn { key: string; x: number; y: number; r: number; value: string; colour: string; others: boolean; points: string | null; faded: boolean; unsaved: boolean }

/** Every object, drawn once (not on every mouse move): the class now worked on bright, the other classes faint in their
 *  own colour, what is in no class dotted grey. */
const SimpleLayer = memo(function SimpleLayer({ items, unit, zoom, active, dim }: { items: Drawn[]; unit: number; zoom: number; active: string | null; dim: number }) {
  const line = lineScale(zoom)
  return (
    <g opacity={dim} pointerEvents="none">
      {items.map((d) => {
        const none = d.value === NONE
        const on = !none && d.value === active
        const width = (on ? OUTLINE_PX * 1.25 : none ? OUTLINE_PX * 0.9 : OUTLINE_PX * 0.75) * line * unit
        const dash = none ? `${5 * unit} ${4 * unit}` : undefined
        const shape = (extra: Record<string, unknown>) => d.points ? <polygon points={d.points} {...extra} /> : <circle cx={d.x} cy={d.y} r={Math.max(6, d.r)} {...extra} />
        return <g key={d.key} className={`round-object simple-object${on ? ' in-pass' : none ? ' in-none' : ' in-other'}`} data-key={d.key} data-class={d.value}
          opacity={d.faded ? 0.3 : on || none ? 1 : 0.5}>
          {(on || none) && shape({ fill: 'none', stroke: '#000', strokeOpacity: 0.6, strokeWidth: width + HALO_PX * unit, strokeDasharray: dash, strokeLinecap: 'round' })}
          {shape({ fill: d.colour, fillOpacity: on ? 0.16 : 0, stroke: d.colour, strokeWidth: width, strokeDasharray: dash, strokeLinecap: 'round' })}
          {d.others && <circle cx={d.x} cy={d.y} r={2.4 * unit} fill={PALETTE.markers.others} stroke="#000" strokeWidth={0.6 * unit} />}
          {d.unsaved && <circle className="unsaved-mark" cx={d.x + Math.max(6, d.r) * 0.75} cy={d.y - Math.max(6, d.r) * 0.75} r={2.6 * unit} fill={PALETTE.markers.unsaved} strokeWidth={0.8 * unit} />}
        </g>
      })}
    </g>
  )
})

/** The thread segments of the wand's pass: the chosen ones in the class's colour, the others faint white. */
const WandLines = memo(function WandLines({ lines, on, unit, colour }: { lines: number[][]; on: number[]; unit: number; colour: string }) {
  const chosen = useMemo(() => new Set(on), [on])
  const path = (keep: (i: number) => boolean) => lines.map((l, i) => {
    if (!keep(i)) return ''
    let out = `M${l[0]} ${l[1]}`
    for (let k = 2; k < l.length; k += 2) out += `L${l[k]} ${l[k + 1]}`
    return l.length === 2 ? `${out}h0.01` : out
  }).join('')
  const rest = path((i) => !chosen.has(i)), picked = path((i) => chosen.has(i))
  return (
    <g className="wand-lines" pointerEvents="none">
      <path d={rest} fill="none" stroke="#000" strokeOpacity={0.35} strokeWidth={3 * unit} strokeLinecap="round" strokeLinejoin="round" />
      <path d={rest} fill="none" stroke="#ffffff" strokeOpacity={0.85} strokeWidth={1.3 * unit} strokeLinecap="round" strokeLinejoin="round" />
      <path d={picked} fill="none" stroke="#000" strokeOpacity={0.6} strokeWidth={4.4 * unit} strokeLinecap="round" strokeLinejoin="round" />
      <path d={picked} fill="none" stroke={colour} strokeWidth={2.6 * unit} strokeLinecap="round" strokeLinejoin="round" data-chosen={on.length} />
    </g>
  )
})

/** Right-click on an object: every simple class with its key, and "not an object"; the object's class now marked. */
function SimpleMenu({ at, classes, current, onPick, onClose }: { at: { x: number; y: number }; classes: SimpleClassDef[]; current: string
  onPick: (value: string) => void; onClose: () => void }) {
  const box = useRef<HTMLDivElement>(null)
  const [place, setPlace] = useState({ left: at.x, top: at.y })
  useEffect(() => {
    const b = box.current?.getBoundingClientRect()
    if (b) setPlace({ left: Math.max(8, Math.min(at.x, window.innerWidth - b.width - 8)), top: Math.max(8, Math.min(at.y, window.innerHeight - b.height - 8)) })
  }, [at.x, at.y])
  const close = useCallback(onClose, [onClose])
  useEffect(() => {
    const away = (e: MouseEvent) => { if (!box.current?.contains(e.target as Node)) close() }
    window.addEventListener('mousedown', away, true)
    window.addEventListener('wheel', close, true)
    return () => { window.removeEventListener('mousedown', away, true); window.removeEventListener('wheel', close, true) }
  }, [close])
  return (
    <div ref={box} className="class-menu" role="menu" aria-label="Class of this object" style={place} onContextMenu={(e) => e.preventDefault()}>
      <div className="class-menu-group">
        {classes.map((c, i) => (
          <button key={c.id} role="menuitem" className={`class-menu-item${current === c.id ? ' chosen' : ''}`} title={c.description} onClick={() => onPick(c.id)}>
            <span className="kbd key-slot">{c.key || i + 1}</span><span className="look-swatch" style={{ borderColor: c.colour, background: c.colour }} />
            <span className="ellipsis">{c.name}</span>
          </button>
        ))}
      </div>
      <div className="class-menu-group">
        <button role="menuitem" className={`class-menu-item${current === NONE ? ' chosen' : ''}`} onClick={() => onPick(NONE)}>
          <span className="kbd key-slot">Del</span><span className="look-swatch look-removed" style={{ borderColor: INACTIVE_GREY }} />
          <span className="ellipsis">not an object</span>
        </button>
      </div>
    </div>
  )
}
