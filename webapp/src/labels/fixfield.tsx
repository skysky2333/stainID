/* Round 2, "Fix the field": the whole field with every object the AI considered and its outline, always shown: solid
   in the class colour when it counts as a class, dotted grey when it was rejected or removed, dotted in the unsure
   colour when unsure. Everything acts on the object under the cursor (it lights up and its facts show at the side):
   1–9 (and the type keys) choose its class, 0 unsure, Delete removes it (dotted grey), Tab / Shift+Tab switch between its
   three Segment Anything masks at once, Space edits it precisely (everything else locked: right-click a spot to include,
   left-click one to leave out, drag the corners; Space again saves), Z undoes. Clicking empty tissue outlines a missed
   object with one click (dotted, "new — choose a class") until a class key gives it a class. A change of call is
   recorded exactly like an answer in round 1. Finished: every call nobody has decided and every outline left as it is
   counts as checked by this person. What others decide meanwhile shows within seconds ("checked by Ben"). */
import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import type { AddedState, Label, RoundCheck, RoundDetail, RoundField, RoundObject, RoundProgress, RoundSamOptions, SessionDetail } from '../api'
import { api, apiUrl, enc } from '../api'
import { ErrorNote, Loaded, fmt } from '../components'
import { useFetch, useStored } from '../hooks'
import type { ImagePoint } from '../viewer'
import { RASTER_LAYERS, ZoomViewer } from '../viewer'
import type { ClassPick } from './classKeys'
import { resolveClass } from './classKeys'
import type { RoundEvent } from './rounds'
import { ClassLegend, KeyLegend, MaskSwitch, ROUND_TITLE, RoundProgressLine, callText, lookOf, maskName, polygonArea, polygonPoints, stepMask, unsureClass,
  useCandidateKeys, useLive } from './rounds'

const BOX_COLOR = '#ffd21f'
const MAX_HANDLES = 200
const rkey = (x: number, y: number) => `${Math.round(x)}:${Math.round(y)}`
type Meta = SessionDetail['meta'] & RoundDetail
type Mine = Label & { candidate?: string | null }
type Rec = Record<string, unknown>
/** An object someone just added with one click and has not given a class yet (kept on the page only). */
interface Fresh { key: string; x: number; y: number; options: number[][][]; chosen: number }
/** One object on the page: a candidate of the store, one this person added, or one just added without a class. */
interface Shown { key: string; x: number; y: number; r: number; candidate?: RoundObject; added?: Mine; fresh?: Fresh }
/** Precise editing of one object: its polygon and the spots clicked (x, y, 1 include / 0 leave out). */
interface Precise { key: string; points: number[][]; clicks: number[][]; busy: boolean }
/** One change, and how to take it back: records to save again, and what to do on the page. */
interface Undo { records: Rec[]; local?: () => void }

function inside(polygon: number[][], x: number, y: number) {
  let hit = false
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i]
    const [xj, yj] = polygon[j]
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) hit = !hit
  }
  return hit
}

function circle(x: number, y: number, r: number) {
  return Array.from({ length: 20 }, (_, i) => [Math.round(x + r * Math.cos((i * Math.PI) / 10)), Math.round(y + r * Math.sin((i * Math.PI) / 10))])
}

/** `onEvent` hears what the person does (the tutorial follows it); `onFinished` replaces going to the next field. */
interface Hooks { onEvent?: (event: RoundEvent) => void; onFinished?: () => void }

export function FixField({ detail, reload, tile, onEvent, onFinished }: { detail: SessionDetail; reload: () => void; tile: string } & Hooks) {
  const work = useFetch<RoundField>(`/api/annotate/${detail.meta.name}/fields/${enc(tile)}`)
  return <Loaded state={work} label="Opening the field…">{(data) => <Fixer key={tile} detail={detail} reload={reload} tile={tile} field={data}
    onEvent={onEvent} onFinished={onFinished} />}</Loaded>
}

function Fixer({ detail, reload, tile, field, onEvent, onFinished }: { detail: SessionDetail; reload: () => void; tile: string; field: RoundField } & Hooks) {
  const tell = (event: RoundEvent) => onEvent?.(event)
  const navigate = useNavigate()
  const meta = detail.meta as Meta
  const name = meta.name
  const { vocabulary, keys, subtypes: typesAsked } = meta
  const waitMs = Math.round(1000 * vocabulary.subtype_wait_s)
  const unsure = unsureClass(vocabulary)
  const [mine, setMine] = useState<Record<string, Mine>>(() => Object.fromEntries(field.mine.map((m) => [m.id, m])))
  const [checked, setChecked] = useState<Record<string, RoundCheck>>(field.checked)
  const [openBy, setOpenBy] = useState<string[]>(field.open_by)
  const [progress, setProgress] = useState<RoundProgress | null>(meta.progress)
  const [masks, setMasks] = useState<Record<string, number>>(() => Object.fromEntries(field.mine  // the mask chosen per candidate
    .filter((m) => m.task === 'outline' && m.candidate && /^mask \d+$/.test(m.value ?? '')).map((m) => [m.candidate!, Number(m.value.slice(5))])))
  const [prompts, setPrompts] = useState<Record<string, { options: number[][][]; chosen: number }>>({})  // masks of objects added in this visit
  const [fresh, setFresh] = useState<Record<string, Fresh>>({})
  const [hovered, setHovered] = useState<string | null>(null)
  const [precise, setPrecise] = useState<Precise | null>(null)
  const [dragging, setDragging] = useState<number | null>(null)
  const [focus, setFocus] = useState<{ x: number; y: number; key: number } | null>(null)  // where precise editing zooms in
  const [draft, setDraft] = useState<number[][]>([])  // drawing a missed object by hand (when one click cannot outline it)
  const [byHand, setByHand] = useState(false)  // the next click on empty tissue starts an outline drawn by hand
  const [cursor, setCursor] = useState<number[] | null>(null)
  const [pending, setPending] = useState<number[] | null>(null)
  const [undos, setUndos] = useState<Undo[]>([])
  const [saving, setSaving] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [samReady, setSamReady] = useState(field.segment_ready)
  const [addedStates, setAddedStates] = useState<AddedState[]>(field.added)
  const [layers, setLayers] = useStored<Record<string, boolean>>('stainid.round2.layers', { box: true, margin: true })
  const [note, setNote] = useState(() => field.mine.find((m) => m.task === 'note')?.value ?? '')
  const sam = meta.segment_anything && field.segment_anything

  useLive(name, { tile_id: tile }, (live) => {
    if (live.checked) setChecked(live.checked)
    if (live.open_by) setOpenBy(live.open_by)
    if (live.progress) setProgress(live.progress)
  })
  useEffect(() => {  // Segment Anything still preparing this field: look again until clicks answer at once
    if (samReady || !sam) return
    const timer = setInterval(() => {
      api.get<{ ready: boolean }>(`/api/annotate/${name}/fields/${enc(tile)}/segment`).then((s) => { if (s.ready) setSamReady(true) }).catch(() => undefined)
    }, 2500)
    return () => clearInterval(timer)
  }, [samReady, sam, name, tile])

  const classId = (x: number, y: number) => `class:${tile}:${rkey(x, y)}`
  const outlineId = (x: number, y: number) => `outline:${tile}:${rkey(x, y)}`
  const own = useMemo(() => field.objects.filter((o) => o.own), [field.objects])
  const ownKeys = useMemo(() => new Set(own.map((o) => rkey(o.x, o.y))), [own])
  const added = Object.values(mine).filter((m) => m.task === 'class' && !m.candidate && m.x != null && !ownKeys.has(rkey(m.x, m.y!)))
  const shown: Shown[] = [
    ...own.map((o) => ({ key: o.candidate_id, x: o.x, y: o.y, r: o.radius_px, candidate: o })),
    ...added.map((m) => ({ key: `add:${rkey(m.x!, m.y!)}`, x: m.x!, y: m.y!, r: 8, added: m })),
    ...Object.values(fresh).map((f) => ({ key: f.key, x: f.x, y: f.y, r: 8, fresh: f })),
  ]
  const byKey = new Map(shown.map((s) => [s.key, s]))
  const target = precise ? byKey.get(precise.key) : hovered ? byKey.get(hovered) : undefined

  /** Where an object's call stands: this person's, else what others decided, else the AI's (none for a new object). */
  const callOf = (s: Shown): { pick: ClassPick; by: 'me' | 'others' | 'ai' | 'new'; who?: string[] } => {
    if (s.fresh) return { pick: { identity: '', subtype: null }, by: 'new' }
    const my = mine[classId(s.x, s.y)]
    if (my) return { pick: resolveClass(vocabulary, my.value, my.subtype), by: 'me' }
    if (s.candidate && checked[s.candidate.candidate_id]) {
      const c = checked[s.candidate.candidate_id]
      return { pick: { identity: c.identity, subtype: c.subtype || null }, by: 'others', who: c.by }
    }
    return { pick: { identity: s.candidate?.ai_value ?? '', subtype: s.candidate?.ai_subtype || null }, by: 'ai' }
  }
  const choicesOf = (s: Shown | undefined): { list: (number[][] | null)[]; index: number } => {
    if (!s) return { list: [], index: 0 }
    if (s.candidate) return { list: s.candidate.masks, index: masks[s.candidate.candidate_id] ?? s.candidate.mask }
    if (s.fresh) return { list: s.fresh.options, index: s.fresh.chosen }
    const prompt = prompts[s.key]
    return prompt ? { list: prompt.options, index: prompt.chosen } : { list: [], index: 0 }
  }
  const outlineOf = (s: Shown): number[][] | null => {
    if (precise?.key === s.key) return precise.points
    if (s.fresh) return s.fresh.options[s.fresh.chosen] ?? null
    const drawn = mine[outlineId(s.x, s.y)]
    if (drawn?.points) return drawn.points
    const { list, index } = choicesOf(s)
    return list[index] ?? null
  }

  /** Saves records; with `undo`, the change can be taken back with Z (what these labels were before is saved again). */
  const save = useCallback(async (records: Rec[], undo?: { local?: () => void }) => {
    if (undo) {
      const touched = records.flatMap((r) => (r.task === 'class' && r.mask != null ? [r, { ...r, task: 'outline' }] : r.task === 'class' || r.task === 'outline' ? [r] : []))
      const before: Rec[] = touched.map((r) => {
        const at = r.candidate ? own.find((o) => o.candidate_id === r.candidate) : undefined
        const x = Number(at?.x ?? r.x), y = Number(at?.y ?? r.y)
        const old = mine[`${r.task}:${tile}:${rkey(x, y)}`]
        const where = r.candidate ? { candidate: r.candidate } : { x, y }
        return old ? { task: r.task, ...where, value: old.value, subtype: old.subtype ?? '', points: old.points } : { task: r.task, ...where, deleted: true }
      })
      setUndos((u) => [{ records: before, local: undo.local }, ...u].slice(0, 100))
    }
    if (!records.length) return
    setSaving((n) => n + 1)
    try {
      const saved = await api.post<Mine[]>(`/api/annotate/${name}/labels`, { records: records.map((r) => ({ tile_id: tile, ...r })) })
      setMine((current) => {
        const next = { ...current }
        for (const r of saved) { if (r.deleted) delete next[r.id]; else next[r.id] = r }
        return next
      })
      setError(null)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setSaving((n) => n - 1)
    }
  }, [name, tile, mine, own])

  /** The object under the cursor gets this class (a new one is saved with its outline then). */
  const giveClass = useCallback((s: Shown, pick: ClassPick) => {
    const subtype = typesAsked ? pick.subtype ?? '' : ''
    if (s.fresh) {
      const f = s.fresh
      const x = Math.round(f.x), y = Math.round(f.y)
      setFresh((all) => { const next = { ...all }; delete next[f.key]; return next })
      setPrompts((p) => ({ ...p, [`add:${rkey(x, y)}`]: { options: f.options, chosen: f.chosen } }))
      setHovered(`add:${rkey(x, y)}`)
      save([{ task: 'class', x, y, value: pick.identity, subtype }, { task: 'outline', x, y, points: f.options[f.chosen] }],
        { local: () => setFresh((all) => ({ ...all, [f.key]: f })) })
    } else if (s.candidate) {
      const drawn = mine[outlineId(s.x, s.y)]
      const edited = drawn && !drawn.value?.startsWith('mask')
      const mask = !edited && s.candidate.masks.length > 1 ? (masks[s.candidate.candidate_id] ?? s.candidate.mask) : null
      save([{ task: 'class', candidate: s.candidate.candidate_id, value: pick.identity, subtype, mask }], {})
    } else {
      save([{ task: 'class', x: s.x, y: s.y, value: pick.identity, subtype }], {})
    }
  }, [save, typesAsked, mine, masks])  // eslint-disable-line react-hooks/exhaustive-deps

  /** Delete: a candidate becomes "not an object" (dotted grey); an object this person added goes (undoable). */
  const removeTarget = useCallback(() => {
    const s = target
    if (!s) return
    if (s.fresh) { const f = s.fresh; setFresh((all) => { const next = { ...all }; delete next[f.key]; return next }); setPrecise(null); return }
    if (s.candidate) { if (meta.remove_class) giveClass(s, { identity: meta.remove_class, subtype: null }); return }
    save([{ task: 'outline', x: s.x, y: s.y, deleted: true }, { task: 'class', x: s.x, y: s.y, deleted: true }], {})
  }, [target, giveClass, save, meta.remove_class])

  const switchMask = useCallback((bigger: boolean) => {
    const s = target
    if (!s || precise) return
    const { list, index } = choicesOf(s)
    const next = stepMask(list, index, bigger)
    if (next === index || !list[next]) return
    if (s.fresh) { const f = s.fresh; setFresh((all) => ({ ...all, [f.key]: { ...f, chosen: next } })); return }
    if (s.candidate) {
      const id = s.candidate.candidate_id
      setMasks((m) => ({ ...m, [id]: next }))
      save([{ task: 'outline', candidate: id, points: list[next], value: `mask ${next}` }], { local: () => setMasks((m) => ({ ...m, [id]: index })) })
    } else {
      setPrompts((p) => ({ ...p, [s.key]: { ...p[s.key], chosen: next } }))
      save([{ task: 'outline', x: s.x, y: s.y, points: list[next] }], { local: () => setPrompts((p) => ({ ...p, [s.key]: { ...p[s.key], chosen: index } })) })
    }
  }, [target, precise, save, masks, prompts])  // eslint-disable-line react-hooks/exhaustive-deps

  /** Space: precise editing of the object under the cursor, in and (saving) out. */
  const togglePrecise = useCallback(() => {
    if (draft.length) return
    if (!precise) {
      const s = target
      if (!s) return
      setPrecise({ key: s.key, points: (outlineOf(s) ?? circle(s.x, s.y, Math.max(6, s.r))).map((p) => [...p]), clicks: [], busy: false })
      setFocus({ x: s.x, y: s.y, key: Date.now() })  // zoom in on it
      return
    }
    const s = byKey.get(precise.key)
    const points = precise.points.map((p) => p.map(Math.round))
    setPrecise(null)
    if (!s) return
    if (s.fresh) { const f = s.fresh; setFresh((all) => ({ ...all, [f.key]: { ...f, options: [...f.options.slice(0, f.chosen), points, ...f.options.slice(f.chosen + 1)] } })); return }
    save([s.candidate ? { task: 'outline', candidate: s.candidate.candidate_id, points, value: 'edited' } : { task: 'outline', x: s.x, y: s.y, points, value: 'edited' }], {})
  }, [precise, target, draft, save])  // eslint-disable-line react-hooks/exhaustive-deps

  const undo = useCallback(() => {
    if (precise) { setPrecise(null); return }
    const last = undos[0]
    if (!last) return
    setUndos((u) => u.slice(1))
    last.local?.()
    if (last.records.length) save(last.records)
  }, [precise, undos, save])

  const classKeys = useCandidateKeys(keys, waitMs, {
    answering: !!target && !draft.length,
    onClass: (pick) => { if (!target) return; tell(target.fresh ? 'added' : pick.subtype ? 'subtype' : 'class'); giveClass(target, pick) },
    onUnsure: () => { if (target && unsure) { tell('unsure'); giveClass(target, { identity: unsure, subtype: null }) } },
    onRemove: () => { if (target) tell('remove'); removeTarget() },
    onMask: (bigger) => { if (target && !precise) tell('mask'); switchMask(bigger) },
    onPrecise: () => { if (precise) tell('precise-out'); else if (target) tell('precise-in'); togglePrecise() },
    onUndo: () => { tell('undo'); undo() },
    onEnter: () => { if (draft.length >= 3) closeDraft() },
    onEscape: () => {
      if (precise) { setPrecise(null); return }
      if (draft.length || byHand) { setDraft([]); setByHand(false); return }
      if (target?.fresh) removeTarget()
    },
  })

  /** One click on empty tissue: Segment Anything outlines it (three sizes) as a new object without a class. */
  const addAt = useCallback(async (x: number, y: number) => {
    const at = [Math.round(x), Math.round(y)]
    setPending(at)
    try {
      const found = await api.post<RoundSamOptions>(`/api/annotate/${name}/fields/${enc(tile)}/segment`, { points: [[at[0], at[1], 1]] })
      if (!found.options.length) { setError('Segment Anything found no object there: click closer to its middle.'); return }
      const key = `new:${rkey(at[0], at[1])}`
      setFresh((all) => ({ ...all, [key]: { key, x: at[0], y: at[1], options: found.options.map((o) => o.points), chosen: found.chosen } }))
      onEvent?.('add')
      setHovered(key)
      setError(null)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setPending(null)
    }
  }, [name, tile])

  /** Precise editing: Segment Anything again with the spots clicked (the object's own centre included). */
  const refine = useCallback(async (clicks: number[][]) => {
    if (!precise) return
    const s = byKey.get(precise.key)
    if (!s) return
    setPrecise({ ...precise, clicks, busy: true })
    try {
      const points = [[Math.round(s.x), Math.round(s.y), 1], ...clicks.map(([x, y, on]) => [Math.round(x), Math.round(y), on])]
      const found = await api.post<RoundSamOptions>(`/api/annotate/${name}/fields/${enc(tile)}/segment`, { points })
      const best = found.options[found.chosen]
      setPrecise((p) => (p && p.key === s.key ? { ...p, points: best ? best.points : p.points, busy: false } : p))
      setError(best ? null : 'Segment Anything found no outline with these spots: drag the corners instead.')
    } catch (e) {
      setPrecise((p) => (p ? { ...p, busy: false } : p))
      setError((e as Error).message)
    }
  }, [precise, name, tile])  // eslint-disable-line react-hooks/exhaustive-deps

  const closeDraft = () => {
    if (draft.length < 3) return
    const points = draft.map((p) => p.map(Math.round))
    const x = Math.round(points.reduce((a, p) => a + p[0], 0) / points.length), y = Math.round(points.reduce((a, p) => a + p[1], 0) / points.length)
    const key = `new:${rkey(x, y)}`
    setFresh((all) => ({ ...all, [key]: { key, x, y, options: [points], chosen: 0 } }))
    setHovered(key)
    setDraft([])
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

  const click = (p: ImagePoint, event: React.MouseEvent) => {
    if (precise) {  // right-click: include this spot; left-click: leave it out
      if (!precise.busy && sam) { tell('precise-edit'); void refine([...precise.clicks, [p.x, p.y, event.button === 2 ? 1 : 0]]) }
      return
    }
    if (event.button === 2) return
    if (draft.length) {
      if (draft.length >= 3 && Math.hypot(draft[0][0] - p.x, draft[0][1] - p.y) < 10 * p.unit) closeDraft()
      else setDraft([...draft, [p.x, p.y]])
      return
    }
    if (hit(p)) return  // the keys act on what is under the cursor; a click on it does nothing
    if (sam && !byHand) void addAt(p.x, p.y)
    else { setDraft([[p.x, p.y]]); setByHand(false) }
  }
  const move = (p: ImagePoint) => {
    setCursor([p.x, p.y])
    if (dragging !== null && precise) { setPrecise({ ...precise, points: precise.points.map((q, i) => (i === dragging ? [p.x, p.y] : q)) }); return }
    if (!precise && !draft.length) {
      const now = hit(p)?.key ?? null
      if (now && now !== hovered) tell('hover')
      setHovered(now)
    }
  }
  useEffect(() => {
    if (dragging === null) return
    const up = () => setDragging(null)
    window.addEventListener('mouseup', up)
    return () => window.removeEventListener('mouseup', up)
  }, [dragging])

  const finish = async () => {
    setSaving((n) => n + 1)
    try {
      await api.post(`/api/annotate/${name}/fields/${enc(tile)}/finish`, {})
      tell('finish')
      if (onFinished) { onFinished(); return }
      reload()
      const next = await api.get<{ tile_id: string | null }>(`/api/annotate/${name}/next`).catch(() => ({ tile_id: null }))
      navigate(next.tile_id && next.tile_id !== tile ? `/annotate/${name}/${enc(next.tile_id)}` : `/annotate/${name}`)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setSaving((n) => n - 1)
    }
  }
  const reopen = async () => { await save([{ task: 'done', deleted: true }]); reload() }
  const done = !!mine[`done:${tile}`]

  const addedState = (s: Shown) => addedStates.find((a) => rkey(a.x, a.y) === rkey(s.x, s.y))?.state ?? 'measuring'
  const measuring = !(meta as { practice?: boolean }).practice && added.some((m) => addedStates.find((a) => rkey(a.x, a.y) === rkey(m.x!, m.y!))?.state !== 'measured')
  useEffect(() => {  // follow the added objects while the server measures them (so they can train the model)
    if (!measuring) return
    const timer = setTimeout(() => { api.get<AddedState[]>(`/api/annotate/${name}/fields/${enc(tile)}/added`).then(setAddedStates).catch(() => undefined) }, 4000)
    return () => clearTimeout(timer)
  }, [measuring, addedStates, name, tile])

  const fields = detail.fields
  const position = fields.findIndex((f) => f.tile_id === tile)
  const later = [...fields.slice(position + 1), ...fields.slice(0, position)].find((f) => !f.done_by_me && f.prepared && !f.open_by_others && f.tile_id !== tile)
  const changed = Object.values(mine).filter((m) => (m.task === 'class' || m.task === 'outline') && !m.confirmed).length
  const unclassed = Object.keys(fresh).length
  const toggle = (k: string) => setLayers({ ...layers, [k]: !layers[k] })
  const busy = saving > 0 || !!pending
  const px = field.pixel_size_um

  return (
    <div className="annotate-page round-page">
      <div className="annotate-top">
        <div className="annotate-heading">
          <span className="row small"><Link to="/annotate">← All sessions</Link><Link to={`/annotate/${name}`}>All fields of this session</Link></span>
          <h1>{meta.title}</h1>
          <div className="muted small">{meta.stain} · Round 2 · {ROUND_TITLE[2]}</div>
        </div>
        <div className="annotate-progress">
          <RoundProgressLine progress={progress} unit="fields" me={detail.annotator} />
          {openBy.length > 0 && <span className="small warning-text">{openBy.join(', ')} {openBy.length === 1 ? 'is' : 'are'} also on this field</span>}
        </div>
        <div className="annotate-actions">
          <span className="muted small save-state" aria-live="polite">{pending ? 'Outlining…' : saving ? 'Saving…' : 'All changes saved'}</span>
          <button className="btn" onClick={() => later && navigate(`/annotate/${name}/${enc(later.tile_id)}`)} disabled={!later}
            title={later ? 'Come back to this field later' : 'No other field is waiting for you'}>Skip for now</button>
          {done ? <button className="btn" onClick={reopen} disabled={busy}>Reopen this field</button>
            : <button className="btn primary big" disabled={busy || !!precise || draft.length > 0 || unclassed > 0} onClick={finish}
              title={unclassed ? 'Give the new objects a class first (or Delete them)' : 'Every call and outline you left as it is counts as checked by you'}>Finished — next field →</button>}
        </div>
      </div>
      {meta.instructions && <div className="instructions">{meta.instructions}</div>}
      <ErrorNote error={error} />
      <div className="annotate-workspace">
        <div className={`annotate-image${precise ? ' is-precise' : ''}`} onContextMenu={(e) => e.preventDefault()} onMouseLeave={() => { if (!precise) setHovered(null) }}>
          <ZoomViewer src={apiUrl(`/api/annotate/${name}/fields/${enc(tile)}/image.jpg`)}
            layers={field.layers.filter((l) => layers[l]).map((l) => apiUrl(`/api/annotate/${name}/fields/${enc(tile)}/layer/${l}.png`))}
            title={<span className="row"><span className="muted small">{tile}</span>{done && <span className="badge status-done">finished</span>}
              {precise && <span className="badge status-running">precise editing</span>}</span>}
            focus={focus} cursor={precise ? 'crosshair' : target ? 'pointer' : 'copy'} onClick={click} onMove={move}>
            {(_zoom, size, rawUnit) => {
              const unit = Number.isFinite(rawUnit) && rawUnit > 0 ? rawUnit : 1
              const box = field.box
              return <>
                {layers.box && <>
                  <path d={`M0 0H${size.w}V${size.h}H0Z M${box.x} ${box.y}v${box.height}h${box.width}v${-box.height}Z`} fill="#000" fillOpacity={0.22} fillRule="evenodd" />
                  <rect x={box.x} y={box.y} width={box.width} height={box.height} fill="none" stroke={BOX_COLOR} strokeWidth={2 * unit} strokeDasharray={`${12 * unit} ${8 * unit}`} />
                </>}
                {layers.margin && field.objects.filter((o) => !o.own).map((o) => {
                  const poly = o.masks[o.mask]
                  return poly ? <polygon key={o.candidate_id} points={polygonPoints(poly)} fill="none" stroke="#ffffff" strokeOpacity={0.3} strokeWidth={unit}
                    strokeDasharray={`${2 * unit} ${3 * unit}`} pointerEvents="none" /> : null
                })}
                <g opacity={precise ? 0.25 : 1} pointerEvents="none">
                  {shown.map((s) => {
                    if (precise?.key === s.key) return null
                    const call = callOf(s)
                    const look = lookOf(vocabulary, meta.remove_class, call.pick.identity)
                    const poly = outlineOf(s)
                    const lit = s.key === hovered
                    const width = (look.dash ? 1.3 : 1.8) * unit
                    const style = { stroke: look.colour, fill: look.colour }
                    return <g key={s.key} className={`round-object look-${look.look}${lit ? ' lit' : ''}`} data-key={s.key}>
                      {lit && poly && <polygon points={polygonPoints(poly)} fill="none" stroke="#ffffff" strokeOpacity={0.9} strokeWidth={5 * unit} />}
                      {look.dash && !lit && poly && <polygon points={polygonPoints(poly)} fill="none" stroke="#ffffff" strokeOpacity={0.55} strokeWidth={width + 1.4 * unit} />}
                      {poly ? <polygon points={polygonPoints(poly)} style={style} fillOpacity={lit ? 0.22 : look.dash ? 0 : 0.08} strokeWidth={lit ? 2.4 * unit : width}
                        strokeDasharray={look.dash ? `${1.5 * unit} ${3 * unit}` : undefined} strokeLinecap="round" />
                        : <circle cx={s.x} cy={s.y} r={Math.max(6, s.r)} style={style} fillOpacity={0} strokeWidth={width} strokeDasharray={look.dash ? `${1.5 * unit} ${3 * unit}` : undefined} />}
                      {(s.added || s.fresh) && <polygon points={`${s.x},${s.y - 5 * unit} ${s.x + 5 * unit},${s.y} ${s.x},${s.y + 5 * unit} ${s.x - 5 * unit},${s.y}`} style={style} fillOpacity={1} stroke="#000" strokeWidth={0.8 * unit} />}
                      {call.by === 'others' && <circle cx={s.x} cy={s.y} r={2.4 * unit} fill="#4ade80" stroke="#000" strokeWidth={0.6 * unit} />}
                    </g>
                  })}
                </g>
                {precise && <>
                  <polygon points={polygonPoints(precise.points)} fill="#ffffff" fillOpacity={0.12} stroke="#ffffff" strokeWidth={2 * unit} />
                  {precise.points.length <= MAX_HANDLES && precise.points.map((p, i) => (
                    <circle key={i} cx={p[0]} cy={p[1]} r={3.2 * unit} fill="#fff" stroke="#000" strokeWidth={0.7 * unit} style={{ cursor: 'move' }}
                      onMouseDown={(e) => { e.stopPropagation(); tell('precise-edit'); setDragging(i) }} onMouseUp={(e) => e.stopPropagation()} />
                  ))}
                  {precise.clicks.map((c, i) => <circle key={`c${i}`} cx={c[0]} cy={c[1]} r={3.5 * unit} fill={c[2] ? '#4ade80' : '#f87171'} stroke="#000" strokeWidth={unit} />)}
                </>}
                {pending && <circle className="sam-pending" cx={pending[0]} cy={pending[1]} r={10 * unit} fill="none" stroke="#fff" strokeWidth={2 * unit} />}
                {draft.length > 0 && <>
                  <polyline points={polygonPoints([...draft, ...(cursor ? [cursor] : [])])} fill="none" stroke="#fff" strokeWidth={2 * unit} />
                  {draft.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r={(i === 0 ? 5 : 3) * unit} fill="#fff" stroke="#000" strokeWidth={unit} />)}
                </>}
              </>
            }}
          </ZoomViewer>
          <KeyLegend round={2} vocabulary={vocabulary} keys={keys} typesAsked={typesAsked} waitMs={waitMs} sam={sam} />
        </div>
        <aside className="annotate-side">
          <TargetCard s={target} call={target ? callOf(target) : null} meta={meta} precise={precise} px={px} outline={target ? outlineOf(target) : null}
            choices={choicesOf(target)} added={target?.added ? addedState(target) : null}
            onMask={(i) => { const s = target; if (!s) return; const { index } = choicesOf(s); if (i !== index) switchMask(i > index) }} />
          {!target && (
            <div className="card">
              <h3>What to do</h3>
              <ol className="small secondary round-steps">
                <li><b>Point at an object</b> — it lights up. Press its class number, <span className="kbd">0</span> unsure, <span className="kbd">Del</span> not an object.</li>
                <li><b>Outline too big or small?</b> <span className="kbd">Tab</span> / <span className="kbd">⇧ Tab</span>. Still not right: <span className="kbd">Space</span> to edit it precisely.</li>
                <li><b>Missed object?</b> {sam ? 'Click it: its outline appears; then press its class number.' : 'Click its corners one by one, then the first corner (or Enter).'}</li>
                <li><b>Finished</b> when the field looks right: what you left as it is counts as checked by you.</li>
              </ol>
              {sam && !samReady && <p className="small muted">Preparing one-click outlines for this field… the first clicks may take a few seconds.</p>}
              {sam && (byHand ? <p className="small precise-bar">Drawing by hand: click the corners of the object, then the first corner or press Enter.
                {' '}<button className="link-btn small" onClick={() => setByHand(false)}>Cancel</button></p>
                : <button className="btn small" onClick={() => { setPrecise(null); setDraft([]); setByHand(true) }}
                  title="The last resort, when one click and precise editing cannot outline it">Draw an object by hand…</button>)}
            </div>
          )}
          <div className="card">
            <h3>This field</h3>
            <p className="small secondary">{own.length.toLocaleString()} objects the AI considered · {added.length} added by you · {changed} change{changed === 1 ? '' : 's'} by you
              {Object.keys(checked).length > 0 && <> · {Object.keys(checked).length} checked by others (green dot)</>}.
              {unclassed > 0 && <b className="warning-text"> {unclassed} new object{unclassed === 1 ? '' : 's'} still need a class.</b>}</p>
            <div className="tag-list">{meta.tags.map((t) => (
              <label key={t} className="layer-toggle"><input type="checkbox" checked={!!mine[`tag:${tile}:${t}`]}
                onChange={() => save([{ task: 'tag', value: t, deleted: !!mine[`tag:${tile}:${t}`] }])} />{t}</label>
            ))}</div>
            <label className="field" style={{ marginTop: 8 }}>Note (optional)
              <textarea rows={2} value={note} onChange={(e) => setNote(e.target.value)}
                onBlur={() => { if (note !== (Object.values(mine).find((m) => m.task === 'note')?.value ?? '')) save([{ task: 'note', value: note, deleted: !note }]) }} /></label>
          </div>
          <details className="card side-details">
            <summary>Display options</summary>
            <label className="layer-toggle"><input type="checkbox" checked={!!layers.margin} onChange={() => toggle('margin')} />objects of the neighbouring fields</label>
            {field.layers.map((l) => (
              <label key={l} className="layer-toggle"><input type="checkbox" checked={!!layers[l]} onChange={() => toggle(l)} /><span className="swatch" style={{ background: RASTER_LAYERS[l]?.color }} />{RASTER_LAYERS[l]?.label ?? l}</label>
            ))}
            <label className="layer-toggle"><input type="checkbox" checked={!!layers.box} onChange={() => toggle('box')} />field boundary (yellow)</label>
          </details>
          <details className="card side-details">
            <summary>Go to another field</summary>
            <p className="muted small">Green = finished by you; dot = someone else is on it.</p>
            <div className="field-jump">
              {fields.map((f, i) => f.prepared
                ? <Link key={f.tile_id} to={`/annotate/${name}/${enc(f.tile_id)}`} title={f.tile_id}
                  className={`field-chip${f.tile_id === tile ? ' current' : f.done_by_me ? ' done' : ''}`}>{i + 1}{f.open_by_others > 0 && <span className="busy-dot" />}</Link>
                : <span key={f.tile_id} className="field-chip waiting" title="still being prepared">{i + 1}</span>)}
            </div>
          </details>
          <div className="card">
            <h3>Colours and keys</h3>
            <ClassLegend vocabulary={vocabulary} keys={keys} removeClass={meta.remove_class} />
            {classKeys.waiting && <p className="small precise-bar">Now the type: {Object.entries(keys.then[classKeys.waiting.identity] ?? {}).map(([k, t]) =>
              `${k} ${vocabulary.classes.find((c) => c.id === classKeys.waiting!.identity)?.subtypes.find((s) => s.id === t)?.name ?? t}`).join(' · ')} · 0 no type</p>}
          </div>
        </aside>
      </div>
    </div>
  )
}

const ADDED_TEXT: Record<AddedState['state'], string> = {
  measuring: 'Being measured so it can teach the AI…',
  measured: 'Measured — it will help train the next model.',
  could_not_be_measured: 'Could not be measured, so it cannot train the model; it still counts as an object the AI missed.',
  proposed: 'The AI had proposed this object too, so it counts as that object.',
}

/** The facts of the object under the cursor (or being edited precisely). */
function TargetCard({ s, call, meta, precise, px, outline, choices, added, onMask }: {
  s: Shown | undefined; call: { pick: ClassPick; by: string; who?: string[] } | null; meta: Meta; precise: Precise | null; px: number
  outline: number[][] | null; choices: { list: (number[][] | null)[]; index: number }; added: AddedState['state'] | null
  onMask: (i: number) => void
}) {
  if (!s || !call) return null
  const { vocabulary, subtypes: typesAsked } = meta
  const look = lookOf(vocabulary, meta.remove_class, call.pick.identity)
  return (
    <div className={`card round-target look-${look.look}`}>
      <div className="row"><span className={`look-swatch look-${look.look}`} style={{ borderColor: look.colour }} />
        <b>{s.fresh ? 'New — choose a class' : callText(vocabulary, call.pick.identity, call.pick.subtype, typesAsked)}</b></div>
      {s.candidate && <p className="small" style={{ margin: '6px 0 0' }}>The AI: {callText(vocabulary, s.candidate.ai_value, s.candidate.ai_subtype, typesAsked)}
        {s.candidate.ai_probability != null && <span className="muted"> · score {fmt(s.candidate.ai_probability, 2)}</span>}
        {s.candidate.decision !== 'accepted' && <span className="muted"> · left out{s.candidate.reason?.startsWith('rule:') ? ` (${s.candidate.reason.slice(5).replace(/_/g, ' ')})` : ''}</span>}</p>}
      {outline && <p className="small muted" style={{ margin: '4px 0 0' }}>{fmt(polygonArea(outline) * px * px, 3)} µm²
        {choices.list.length > 1 && ` · outline ${maskName(choices.index, choices.list.length)}`}</p>}
      {call.by === 'others' && <p className="small checked-by">✓ Checked by <b>{call.who?.join(', ')}</b> — you can still change it.</p>}
      {call.by === 'me' && <p className="small muted" style={{ margin: '4px 0 0' }}>Changed by you{s.added ? ' (you added it)' : ''}.</p>}
      {added && added !== 'proposed' && <p className={`small added-note${added === 'could_not_be_measured' ? ' warn' : ''}`}><b>The AI never proposed this object.</b> {ADDED_TEXT[added]}</p>}
      {precise ? <p className="small precise-bar"><b>Precise editing.</b> Right-click a spot to include it, left-click one to leave it out, drag the corners.
        {precise.busy ? ' Outlining…' : ''} <span className="kbd">Space</span> saves · <span className="kbd">Esc</span> cancels.</p>
        : <MaskSwitch count={choices.list.length} index={choices.index} onChange={onMask} />}
    </div>
  )
}
