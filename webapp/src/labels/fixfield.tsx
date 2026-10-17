/* Round 2, "Fix the field": the whole field with every object detection considered and its outline, always shown, one
   shape per object: solid in its class colour when it counts as a class (each class its own colour, unsure too), dotted
   light grey when it is not active: rejected by detection, removed by a person, or just added and waiting for its class.
   Everything acts on the object under the cursor (it lights up, its name shows beside it and its facts at the side):
   1–9 (and the type keys) choose its class, 0 unsure, Delete removes it (it turns dotted grey and stays; a class key brings
   it back), Tab / Shift+Tab a bigger or smaller Segment Anything mask at once, Space edits it precisely (everything else
   locked: right-click a spot to add it, left-click one to leave it out, drag the dots; Z undoes a step; Space again keeps
   it, Esc cancels), Z undoes. Clicking empty tissue outlines a missed object with one click: a new candidate, dotted, until
   a class key makes it solid. A change of call is recorded exactly like an answer in round 1. Every change stays on this
   page ("3 unsaved changes", with a draft in this browser in case of a reload; Z undoes) until Save (stays on the field)
   or Save and continue: the field is then finished, every call nobody has decided and every outline left as it is counts
   as checked by this person, and the next field comes. Opening a field holds it for this person, so the others are
   handed other fields; what others decide meanwhile shows within seconds ("checked by Ben"). The page takes the window
   exactly: the picture as large as fits, the side panel scrolling by itself. */
import { memo, useCallback, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import type { AddedState, Label, RoundCheck, RoundDetail, RoundField, RoundObject, RoundProgress, RoundSamOptions, RoundSaved, SessionDetail } from '../api'
import { api, apiUrl, enc, pendingText } from '../api'
import { ErrorNote, Loading, fmt } from '../components'
import { useFetch, useStored } from '../hooks'
import type { ImagePoint } from '../viewer'
import { RASTER_LAYERS, ZoomViewer } from '../viewer'
import type { ClassPick } from './classKeys'
import { resolveClass } from './classKeys'
import type { RoundEvent } from './rounds'
import { BRUSH_SCALE, BRUSH_UM, useBrush } from './brush'
import { ClassLegend, LookLegend, MaskSwitch, ROUND_TITLE, RoundProgressLine, SaveButtons, SaveState, callText, changesText, draftKey, lookOf, maskName,
  polygonArea, polygonPoints, readDraft, simplifyPolygon, stepMask, unsureClass, useCandidateKeys, useFitScreen, useLeaveGuard, useLive, useSaveKeys,
  writeDraft } from './rounds'

const BOX_COLOR = '#ffd21f'
const HANDLES = 48  // at most this many dots to drag in precise editing (the outline is simplified to them)
const rkey = (x: number, y: number) => `${Math.round(x)}:${Math.round(y)}`
type Meta = SessionDetail['meta'] & RoundDetail
type Mine = Label & { candidate?: string | null }
type Rec = Record<string, unknown>
/** An object someone just added with one click and has not given a class yet (kept on the page only); `removed`: they
 *  pressed Delete on it, so it stays dotted grey until a class key brings it back. */
interface Fresh { key: string; x: number; y: number; options: number[][][]; chosen: number; removed?: boolean }
/** One object on the page: a candidate of the store, one this person added, or one just added without a class. */
interface Shown { key: string; x: number; y: number; r: number; candidate?: RoundObject; added?: Mine; fresh?: Fresh }
/** Precise editing of one object: its polygon, the spots clicked (x, y, 1 add / 0 leave out) and the steps before (Z). */
interface Precise { key: string; points: number[][]; clicks: number[][]; busy: boolean; history: { points: number[][]; clicks: number[][] }[] }
/** An unsaved change of one label: what to send on Save (null when another label's record carries it, e.g. the mask
 *  chosen with a class) and how the label looks on the page meanwhile (null: taken away). */
interface Change { mark: Rec | null; label: Mine | null }
/** One step back with Z: the changes of these labels as they were before, and what to do on the page. */
interface Undo { before: Record<string, Change | undefined>; local?: () => void }
const SEND_ORDER: Record<string, number> = { class: 0, outline: 1 }  // a class record that carries its mask goes before an outline chosen after it

/** How many things the unsaved changes touch: one per object (its class and outline together), tag or note. */
function changedThings(changes: Record<string, Change>) {
  return new Set(Object.keys(changes).map((id) => (/^(class|outline):/.test(id) ? id.split(':').slice(-2).join(':') : id))).size
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

function circle(x: number, y: number, r: number) {
  return Array.from({ length: 20 }, (_, i) => [Math.round(x + r * Math.cos((i * Math.PI) / 10)), Math.round(y + r * Math.sin((i * Math.PI) / 10))])
}

/** `onEvent` hears what the person does (the tutorial follows it); `onFinished` replaces going to the next field; `coach`
 *  is shown at the top of the side panel (the tutorial's steps). */
interface Hooks { onEvent?: (event: RoundEvent) => void; onFinished?: () => void; coach?: ReactNode }

export function FixField({ detail, reload, tile, onEvent, onFinished, coach }: { detail: SessionDetail; reload: () => void; tile: string } & Hooks) {
  const work = useFetch<RoundField<number[]>>(`/api/annotate/${detail.meta.name}/fields/${enc(tile)}`)
  // every outline of every object, after the field is drawn (the field brings only each object's default one)
  const shapes = useFetch<Record<string, (number[] | null)[]>>(work.data ? `/api/annotate/${detail.meta.name}/fields/${enc(tile)}/masks` : null)
  const field = useMemo(() => (work.data ? unflatten(work.data, shapes.data) : null), [work.data, shapes.data])
  useFitScreen()
  if (!field) {  // the picture at once, the objects when they have arrived
    return (
      <div className="annotate-page round-page fit-page">
        <div className="annotate-top"><div className="annotate-heading"><h1>{detail.meta.title}</h1>
          <div className="muted small">{detail.meta.stain} · {ROUND_TITLE[2]}</div></div></div>
        <ErrorNote error={work.error} />
        <div className="annotate-workspace">
          <div className="annotate-image field-loading">
            <ZoomViewer fit src={apiUrl(`/api/annotate/${detail.meta.name}/fields/${enc(tile)}/image.jpg`)}
              title={<span className="row"><span className="muted small">{tile}</span><span className="small muted">Loading the objects…</span></span>}>
              {() => null}
            </ZoomViewer>
          </div>
          <aside className="annotate-side round-side">{coach}<div className="card side-block">
            <Loading label={work.pending ? pendingText(work.pending) : 'Opening the field…'} /></div></aside>
        </div>
      </div>
    )
  }
  return <Fixer key={tile} detail={detail} reload={reload} tile={tile} field={field} onEvent={onEvent} onFinished={onFinished} coach={coach} />
}

/** The field with its outlines as [x, y] corners: the server sends them flat ([x0, y0, x1, y1, …], whole pixels), the
 *  default one of each object with the field and all of them (`all`) once they are loaded. */
function unflatten(field: RoundField<number[]>, all: Record<string, (number[] | null)[]> | null): RoundField {
  const corners = (flat: number[] | null | undefined) => {
    if (!flat || flat.length < 6) return null
    const out: number[][] = []
    for (let i = 0; i + 1 < flat.length; i += 2) out.push([flat[i], flat[i + 1]])
    return out
  }
  return { ...field, objects: field.objects.map((o) => ({ ...o, masks: (all?.[o.candidate_id] ?? o.masks).map(corners) })) }
}

/** The field's objects with outlines of at most `SHAPE_POINTS` corners (Segment Anything's come with hundreds): drawn,
 *  hit-tested and dragged much faster, and the same to the eye. */
const SHAPE_POINTS = 64
/** An outline with about `most` corners, evenly spaced along it (one pass: fast enough for thousands of outlines). */
function evenly(points: number[][], most: number): number[][] {
  let length = 0
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) length += Math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1])
  const step = Math.max(1, length / most)
  const out = [points[0]]
  let last = points[0]
  for (let i = 1; i < points.length; i++) {
    const p = points[i]
    if (Math.hypot(p[0] - last[0], p[1] - last[1]) >= step) { out.push(p); last = p }
  }
  return out.length >= 3 ? out : points
}
function lighter(field: RoundField): RoundField {
  return { ...field, objects: field.objects.map((o) => ({ ...o, masks: o.masks.map((m) => (m && m.length > SHAPE_POINTS ? evenly(m, SHAPE_POINTS) : m)) })) }
}

/** One object as the field draws it when it is not under the mouse (the layer of all of them is drawn once, not on
 *  every mouse move). */
interface Drawn { key: string; x: number; y: number; r: number; points: string | null; colour: string; look: string; dash: boolean; diamond: boolean
  others: boolean; unsaved: boolean; fresh: boolean }

const ObjectLayer = memo(function ObjectLayer({ items, unit, dim }: { items: Drawn[]; unit: number; dim: number }) {
  return (
    <g opacity={dim} pointerEvents="none">
      {items.map((d) => {
        const width = (d.dash ? 2.2 : 2.4) * unit
        const dash = d.dash ? `${5 * unit} ${4 * unit}` : undefined
        const shape = (extra: Record<string, unknown>) => d.points
          ? <polygon points={d.points} {...extra} />
          : <circle cx={d.x} cy={d.y} r={Math.max(6, d.r)} {...extra} />
        return <g key={d.key} className={`round-object look-${d.look}${d.unsaved ? ' unsaved' : ''}`} data-key={d.key}>
          {shape({ fill: 'none', stroke: '#000000', strokeOpacity: 0.6, strokeWidth: width + 1.8 * unit, strokeDasharray: dash, strokeLinecap: 'round' })}
          {shape({ style: { stroke: d.colour, fill: d.colour }, fillOpacity: d.dash ? 0 : 0.08, strokeWidth: width, strokeDasharray: dash, strokeLinecap: 'round' })}
          {d.diamond && <polygon points={`${d.x},${d.y - 5 * unit} ${d.x + 5 * unit},${d.y} ${d.x},${d.y + 5 * unit} ${d.x - 5 * unit},${d.y}`}
            fill={d.look === 'class' || d.look === 'unsure' ? d.colour : '#ffffff'} stroke="#000" strokeWidth={0.8 * unit} />}
          {d.others && <circle cx={d.x} cy={d.y} r={2.4 * unit} fill="#4ade80" stroke="#000" strokeWidth={0.6 * unit} />}
          {d.unsaved && !d.fresh && <circle className="unsaved-mark" cx={d.x + Math.max(6, d.r) * 0.75} cy={d.y - Math.max(6, d.r) * 0.75} r={2.6 * unit}
            strokeWidth={0.8 * unit} />}
        </g>
      })}
    </g>
  )
})

function Fixer({ detail, reload, tile, field: arrived, onEvent, onFinished, coach }: { detail: SessionDetail; reload: () => void; tile: string; field: RoundField } & Hooks) {
  const field = useMemo(() => lighter(arrived), [arrived])
  const tell = (event: RoundEvent) => onEvent?.(event)
  const navigate = useNavigate()
  const meta = detail.meta as Meta
  const name = meta.name
  const { vocabulary, keys, subtypes: typesAsked } = meta
  const waitMs = Math.round(1000 * vocabulary.subtype_wait_s)
  const unsure = unsureClass(vocabulary)
  const draftAt = draftKey(detail.annotator, name, tile)
  const [saved, setSaved] = useState<Record<string, Mine>>(() => Object.fromEntries(field.mine.map((m) => [m.id, m])))  // as the server has them
  const [changes, setChanges] = useState<Record<string, Change>>(() => readDraft<{ changes: Record<string, Change> }>(draftAt)?.changes ?? {})  // not saved yet
  const [restored, setRestored] = useState(() => changedThings(changes))
  const mine = useMemo(() => {  // this person's labels as the page shows them: the saved ones with the unsaved changes on top
    const out = { ...saved }
    for (const [id, c] of Object.entries(changes)) { if (c.label) out[id] = c.label; else delete out[id] }
    return out
  }, [saved, changes])
  const [checked, setChecked] = useState<Record<string, RoundCheck>>(field.checked)
  const [openBy, setOpenBy] = useState<string[]>(field.open_by)
  const [doneBy, setDoneBy] = useState<string[]>(field.done_by)
  const [progress, setProgress] = useState<RoundProgress | null>(meta.progress)
  const [masks, setMasks] = useState<Record<string, number>>(() => Object.fromEntries(Object.values(mine)  // the mask chosen per candidate
    .filter((m) => m.task === 'outline' && m.candidate && /^mask \d+$/.test(m.value ?? '')).map((m) => [m.candidate!, Number(m.value.slice(5))])))
  const [prompts, setPrompts] = useState<Record<string, { options: number[][][]; chosen: number }>>({})  // masks of objects added in this visit
  const [fresh, setFresh] = useState<Record<string, Fresh>>({})
  const [hovered, setHovered] = useState<string | null>(null)
  const [precise, setPrecise] = useState<Precise | null>(null)
  const [dragging, setDragging] = useState<number | null>(null)
  const [focus, setFocus] = useState<{ x: number; y: number; key: number; back?: boolean } | null>(null)  // where precise editing zooms in (and back)
  const [draft, setDraft] = useState<number[][]>([])  // drawing a missed object by hand (when one click cannot outline it)
  const [byHand, setByHand] = useState(false)  // the next click on empty tissue starts an outline drawn by hand
  const [cursor, setCursor] = useState<number[] | null>(null)
  const [pending, setPending] = useState<number[] | null>(null)
  const [undos, setUndos] = useState<Undo[]>([])
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [samReady, setSamReady] = useState(field.segment_ready)
  const [addedStates, setAddedStates] = useState<AddedState[]>(field.added)
  const [layers, setLayers] = useStored<Record<string, boolean>>('stainid.round2.layers', { box: true, margin: true })
  const [help, setHelp] = useStored<boolean>('stainid.round2.help', false)
  const [note, setNote] = useState(() => field.mine.find((m) => m.task === 'note')?.value ?? '')
  const sam = meta.segment_anything && field.segment_anything
  // the exclusion brush (E): bad areas painted on a mask, saved with the field's other changes
  const [excluding, setExcluding] = useState(false)
  const [brushUm, setBrushUm] = useStored<number>('stainid.round2.brush', BRUSH_UM.start)
  const [fieldSize, setFieldSize] = useState<{ w: number; h: number } | null>(null)
  const [brushDirty, setBrushDirty] = useState(false)
  const savedArea = useMemo(() => {
    const m = field.mine.find((l) => l.task === 'exclude')
    return m ? { points: m.points as unknown as number[][][] | null, value: m.value ?? '' } : null
  }, [field.mine])
  const brush = useBrush(fieldSize, savedArea)

  useLive(name, { tile_id: tile }, (live) => {
    if (live.checked) setChecked(live.checked)
    if (live.open_by) setOpenBy(live.open_by)
    if (live.done_by) setDoneBy(live.done_by)
    if (live.progress) setProgress(live.progress)
  })
  const unsaved = changedThings(changes) + (brushDirty ? 1 : 0)
  useEffect(() => { writeDraft(draftAt, unsaved ? { changes } : null) }, [draftAt, changes, unsaved])
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

  /** Where an object's call stands: this person's, else what others decided, else detection's (none for a new object). */
  const callOf = (s: Shown): { pick: ClassPick; by: 'me' | 'others' | 'ai' | 'new'; who?: string[] } => {
    if (s.fresh) return { pick: { identity: s.fresh.removed ? meta.remove_class : '', subtype: null }, by: 'new' }
    const my = mine[classId(s.x, s.y)]
    if (my) return { pick: resolveClass(vocabulary, my.value, my.subtype), by: 'me' }
    if (s.candidate && checked[s.candidate.candidate_id]) {
      const c = checked[s.candidate.candidate_id]
      return { pick: { identity: c.identity, subtype: c.subtype || null }, by: 'others', who: c.by }
    }
    return { pick: { identity: s.candidate?.ai_value ?? '', subtype: s.candidate?.ai_subtype || null }, by: 'ai' }
  }
  const samNames = (n: number) => Array.from({ length: n }, (_, i) => `SAM ${i + 1}/${n}`)  // one click: Segment Anything's own three
  const choicesOf = (s: Shown | undefined): { list: (number[][] | null)[]; index: number; names?: string[] } => {
    if (!s) return { list: [], index: 0 }
    if (s.candidate) return { list: s.candidate.masks, index: masks[s.candidate.candidate_id] ?? s.candidate.mask }
    if (s.fresh) return { list: s.fresh.options, index: s.fresh.chosen, names: samNames(s.fresh.options.length) }
    const prompt = prompts[s.key]
    return prompt ? { list: prompt.options, index: prompt.chosen, names: samNames(prompt.options.length) } : { list: [], index: 0 }
  }
  const isUnsaved = (s: Shown) => !!changes[classId(s.x, s.y)] || !!changes[outlineId(s.x, s.y)]
  const outlineOf = (s: Shown): number[][] | null => {
    if (precise?.key === s.key) return precise.points
    if (s.fresh) return s.fresh.options[s.fresh.chosen] ?? null
    const drawn = mine[outlineId(s.x, s.y)]
    if (drawn?.points) return drawn.points
    const { list, index } = choicesOf(s)
    return list[index] ?? null
  }

  /** Puts changes on the page (nothing is sent before Save); each call is one step Z takes back. */
  const save = useCallback((records: Rec[], undo?: { local?: () => void }) => {
    const next: Record<string, Change> = {}
    const label = (id: string, task: Mine['task'], x: number, y: number, more: Partial<Mine>): Mine => ({
      id, source: detail.annotator ?? '', stain: meta.stain, task, tile_id: tile, x, y, value: '', subtype: null, points: null, session: name, confirmed: false,
      time: Date.now() / 1000, ...more })
    const noObject = (identity: unknown) => identity === meta.remove_class || identity === unsure
    for (const r of records) {
      const at = r.candidate ? own.find((o) => o.candidate_id === r.candidate) : undefined
      const x = Number(at?.x ?? r.x), y = Number(at?.y ?? r.y)
      const id = r.task === 'tag' ? `tag:${tile}:${r.value}` : r.task === 'note' || r.task === 'done' ? `${r.task}:${tile}` : `${r.task}:${tile}:${rkey(x, y)}`
      const candidate = (r.candidate as string | undefined) ?? null
      if (r.deleted) { next[id] = { mark: saved[id] ? r : null, label: null }; continue }
      next[id] = { mark: r, label: label(id, r.task as Mine['task'], x, y, { value: String(r.value ?? ''), subtype: (r.subtype as string) || null, points: (r.points as number[][]) ?? null, candidate }) }
      if (r.task === 'class' && at && r.mask != null && !noObject(r.value)) {  // the mask chosen with the call is saved with it
        const mask = Number(r.mask)
        next[outlineId(x, y)] = { mark: null, label: label(outlineId(x, y), 'outline', x, y, { value: `mask ${mask}`, points: at.masks[mask] ?? null, candidate }) }
      } else if (r.task === 'class' && at && changes[outlineId(x, y)]?.label?.value?.startsWith('mask')) {
        next[outlineId(x, y)] = { mark: null, label: null }  // not an object (or unsure): no mask goes with it
      }
    }
    setUndos((u) => [{ before: Object.fromEntries(Object.keys(next).map((id) => [id, changes[id]])), local: undo?.local }, ...u].slice(0, 200))
    const out = { ...changes, ...next }
    for (const [id, c] of Object.entries(next)) if (!c.mark && !c.label) delete out[id]  // taken away before it was ever saved
    setChanges(out)
  }, [name, tile, own, saved, changes, meta.stain, meta.remove_class, unsure, detail.annotator])  // eslint-disable-line react-hooks/exhaustive-deps

  /** The object under the cursor gets this class (a new one is saved with its outline then). */
  const giveClass = useCallback((s: Shown, pick: ClassPick) => {
    const subtype = typesAsked ? pick.subtype ?? '' : ''
    if (s.fresh) {
      const f = s.fresh
      const x = Math.round(f.x), y = Math.round(f.y)
      setFresh((all) => { const next = { ...all }; delete next[f.key]; return next })
      setPrompts((p) => ({ ...p, [`add:${rkey(x, y)}`]: { options: f.options, chosen: f.chosen } }))
      if (!precise) setHovered(`add:${rkey(x, y)}`)
      else setPrecise((p) => (p ? { ...p, key: `add:${rkey(x, y)}` } : p))
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
  }, [save, typesAsked, mine, masks, precise])  // eslint-disable-line react-hooks/exhaustive-deps

  /** Delete: the object turns dotted grey and stays ("not an object"); a class key brings it back, Z undoes. */
  const removeTarget = useCallback(() => {
    const s = target
    if (!s) return
    if (s.fresh) {
      const f = s.fresh
      if (f.removed) return
      setFresh((all) => ({ ...all, [f.key]: { ...f, removed: true } }))
      save([], { local: () => setFresh((all) => (all[f.key] ? { ...all, [f.key]: { ...all[f.key], removed: false } } : all)) })
      return
    }
    if (meta.remove_class && callOf(s).pick.identity !== meta.remove_class) giveClass(s, { identity: meta.remove_class, subtype: null })
  }, [target, giveClass, save, meta.remove_class])  // eslint-disable-line react-hooks/exhaustive-deps

  /** Show this one of the object's outlines (the bigger / smaller keys, or a dot of the side panel). */
  const chooseMask = useCallback((s: Shown, next: number) => {
    if (precise) return
    const { list, index } = choicesOf(s)
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
  }, [precise, save, masks, prompts])  // eslint-disable-line react-hooks/exhaustive-deps
  const switchMask = useCallback((bigger: boolean) => {
    const s = target
    if (!s || precise) return
    const { list, index } = choicesOf(s)
    chooseMask(s, stepMask(list, index, bigger))
  }, [target, precise, chooseMask])  // eslint-disable-line react-hooks/exhaustive-deps

  /** Out of precise editing: the view goes back to exactly where it was before (zoom and position), the object stays
   *  the one the keys act on. */
  const leavePrecise = () => {
    const key = precise?.key
    setPrecise(null)
    setFocus({ x: 0, y: 0, key: Date.now(), back: true })
    if (key) setHovered(key)
  }

  /** Space: precise editing of the object under the cursor, in and (saving) out. */
  const togglePrecise = useCallback(() => {
    if (draft.length) return
    if (!precise) {
      const s = target
      if (!s) return
      const start = outlineOf(s) ?? circle(s.x, s.y, Math.max(6, s.r))
      setPrecise({ key: s.key, points: simplifyPolygon(start.map((p) => [...p]), HANDLES), clicks: [], busy: false, history: [] })
      setFocus({ x: s.x, y: s.y, key: Date.now() })  // zoom in on it
      return
    }
    const s = byKey.get(precise.key)
    const points = precise.points.map((p) => p.map(Math.round))
    leavePrecise()
    if (!s || !precise.history.length) return  // nothing changed
    if (s.fresh) { const f = s.fresh; setFresh((all) => ({ ...all, [f.key]: { ...f, options: [...f.options.slice(0, f.chosen), points, ...f.options.slice(f.chosen + 1)] } })); return }
    save([s.candidate ? { task: 'outline', candidate: s.candidate.candidate_id, points, value: 'edited' } : { task: 'outline', x: s.x, y: s.y, points, value: 'edited' }], {})
  }, [precise, target, draft, save])  // eslint-disable-line react-hooks/exhaustive-deps

  const undo = useCallback(() => {
    if (precise) {  // a step of the precise edit back (Esc leaves it without keeping anything)
      const last = precise.history[precise.history.length - 1]
      if (last) setPrecise({ ...precise, points: last.points, clicks: last.clicks, history: precise.history.slice(0, -1) })
      return
    }
    const last = undos[0]
    if (!last) return
    setUndos((u) => u.slice(1))
    last.local?.()
    setChanges((current) => {
      const out = { ...current }
      for (const [id, before] of Object.entries(last.before)) { if (before) out[id] = before; else delete out[id] }
      return out
    })
  }, [precise, undos])

  const classKeys = useCandidateKeys(keys, waitMs, {
    answering: !!target && !draft.length && !excluding,
    onClass: (pick) => { if (!target) return; tell(target.fresh ? 'added' : pick.subtype ? 'subtype' : 'class'); giveClass(target, pick) },
    onUnsure: () => { if (target && unsure) { tell(target.fresh ? 'added' : 'unsure'); giveClass(target, { identity: unsure, subtype: null }) } },
    onRemove: () => { if (excluding) return; if (target) tell('remove'); removeTarget() },
    onMask: (bigger) => { if (excluding) return; if (target && !precise) tell('mask'); switchMask(bigger) },
    onPrecise: () => { if (excluding) return; if (precise) tell('precise-out'); else if (target) tell('precise-in'); togglePrecise() },
    onUndo: () => { tell('undo'); undo() },
    onEnter: () => { if (draft.length >= 3) closeDraft() },
    onEscape: () => {
      if (excluding) { setExcluding(false); return }
      if (precise) { leavePrecise(); return }
      if (draft.length || byHand) { setDraft([]); setByHand(false); return }
      if (target?.fresh) { const f = target.fresh; setFresh((all) => { const next = { ...all }; delete next[f.key]; return next }) }
    },
  })

  useEffect(() => {  // E: the exclusion brush on and off
    const onKey = (e: KeyboardEvent) => {
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement).tagName) || e.metaKey || e.ctrlKey || e.altKey || e.repeat) return
      if (e.key.toLowerCase() !== 'e' || precise || draft.length) return
      e.preventDefault()
      setExcluding((on) => !on)
      setHovered(null)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [precise, draft.length])
  const brushPx = brushUm / (field.pixel_size_um || 0.27) / 2  // the brush's radius in field pixels
  const press = (p: ImagePoint, event: React.MouseEvent) => {
    if (!excluding) return
    brush.begin(p.x, p.y, brushPx, event.button === 2 || event.shiftKey)
  }
  const release = () => {
    if (!excluding) return
    const before = brush.end()
    if (!before) return
    setBrushDirty(true)
    save([], { local: () => { brush.restore(before); setBrushDirty(true) } })  // one stroke is one step of Z
    tell('exclude')
  }
  const wheelBrush = (deltaY: number) => {
    if (!excluding) return false
    setBrushUm(Math.round(Math.min(BRUSH_UM.max, Math.max(BRUSH_UM.min, brushUm * (deltaY < 0 ? 1.15 : 1 / 1.15)))))
    return true
  }

  /** One click on empty tissue: Segment Anything outlines it (three sizes) as a new candidate without a class. */
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
  }, [name, tile])  // eslint-disable-line react-hooks/exhaustive-deps

  /** Precise editing: Segment Anything again with the spots clicked (the object's own centre included). */
  const refine = useCallback(async (clicks: number[][]) => {
    if (!precise) return
    const s = byKey.get(precise.key)
    if (!s) return
    const before = { points: precise.points, clicks: precise.clicks }
    setPrecise({ ...precise, clicks, busy: true })
    try {
      const points = [[Math.round(s.x), Math.round(s.y), 1], ...clicks.map(([x, y, on]) => [Math.round(x), Math.round(y), on])]
      const found = await api.post<RoundSamOptions>(`/api/annotate/${name}/fields/${enc(tile)}/segment`, { points })
      const best = found.options[found.chosen]
      setPrecise((p) => (p && p.key === s.key ? { ...p, points: best ? simplifyPolygon(best.points, HANDLES) : p.points, busy: false,
        history: [...p.history, before] } : p))
      setError(best ? null : 'Segment Anything found no outline with these spots: drag the dots instead.')
    } catch (e) {
      setPrecise((p) => (p ? { ...p, clicks: before.clicks, busy: false } : p))
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
    if (excluding) return  // painting: a click is a dab of the brush, nothing else
    if (precise) {  // right-click: add this spot; left-click: leave it out
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
    if (excluding) { brush.extend(p.x, p.y); return }
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
  const startDrag = (i: number) => {
    if (!precise) return
    tell('precise-edit')
    setPrecise({ ...precise, history: [...precise.history, { points: precise.points, clicks: precise.clicks }] })
    setDragging(i)
  }

  /** Sends these labels (and, with `finish`, finishes the field) in one request: all of them are saved, or none. */
  const send = async (records: Rec[], finish: boolean) => {
    setSaving(true)
    try {
      const result = await api.post<RoundSaved>(`/api/annotate/${name}/tiles/${enc(tile)}/save`, { records: records.map((r) => ({ tile_id: tile, ...r })), submit: finish })
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
  /** Save (stays on the field) or, with `finish`, Save and continue: the field is finished and the next one opens. */
  const saveAll = async (finish: boolean) => {
    if (saving || (!finish && !unsaved)) return
    const marks = Object.values(changes).flatMap((c) => (c.mark ? [c.mark] : []))
      .sort((a, b) => (SEND_ORDER[String(a.task)] ?? 2) - (SEND_ORDER[String(b.task)] ?? 2))
    if (brushDirty) marks.push({ task: 'exclude', png: brush.png(), scale: BRUSH_SCALE })  // the painted bad area ("" : none)
    if (!(await send(marks, finish))) return
    setChanges({}); setUndos([]); setRestored(0); setBrushDirty(false)
    writeDraft(draftAt, null)
    tell(finish ? 'finish' : 'save')
    if (!finish) return
    if (onFinished) { onFinished(); return }
    reload()
    const next = await api.get<{ tile_id: string | null }>(`/api/annotate/${name}/next`).catch(() => ({ tile_id: null }))
    navigate(next.tile_id && next.tile_id !== tile ? `/annotate/${name}/${enc(next.tile_id)}` : `/annotate/${name}`)
  }
  const reopen = async () => { if (await send([{ task: 'done', deleted: true }], false)) reload() }
  const done = !!saved[`done:${tile}`]

  const addedState = (s: Shown) => addedStates.find((a) => rkey(a.x, a.y) === rkey(s.x, s.y))?.state ?? 'measuring'
  const measuring = !(meta as { practice?: boolean }).practice && added.some((m) => saved[m.id] && addedStates.find((a) => rkey(a.x, a.y) === rkey(m.x!, m.y!))?.state !== 'measured')
  useEffect(() => {  // follow the added objects while the server measures them (so they can train the model)
    if (!measuring) return
    const timer = setTimeout(() => { api.get<AddedState[]>(`/api/annotate/${name}/fields/${enc(tile)}/added`).then(setAddedStates).catch(() => undefined) }, 4000)
    return () => clearTimeout(timer)
  }, [measuring, addedStates, name, tile])

  const drawnItems = useMemo<Drawn[]>(() => shown.filter((s) => precise?.key !== s.key).map((s) => {
    const call = callOf(s)
    const look = lookOf(vocabulary, meta.remove_class, call.pick.identity)
    const poly = outlineOf(s)
    return { key: s.key, x: s.x, y: s.y, r: s.r, points: poly ? polygonPoints(poly) : null, colour: look.colour, look: look.look, dash: look.dash,
      diamond: !!(s.added || s.fresh), others: call.by === 'others', unsaved: isUnsaved(s), fresh: !!s.fresh }
  }), [own, mine, fresh, checked, masks, prompts, precise?.key, precise?.points, changes, vocabulary, meta.remove_class])  // eslint-disable-line react-hooks/exhaustive-deps

  const fields = detail.fields
  const position = fields.findIndex((f) => f.tile_id === tile)
  const later = [...fields.slice(position + 1), ...fields.slice(0, position)].find((f) => !f.done_by_me && f.prepared && !f.open_by_others && f.tile_id !== tile)
  const changed = Object.values(mine).filter((m) => (m.task === 'class' || m.task === 'outline') && !m.confirmed).length
  const checkedNow = own.filter((o) => mine[classId(o.x, o.y)] || checked[o.candidate_id]).length
  const unclassed = Object.values(fresh).filter((f) => !f.removed).length
  const toggle = (k: string) => setLayers({ ...layers, [k]: !layers[k] })
  const busy = saving || !!pending
  const px = field.pixel_size_um
  const mayLeave = useLeaveGuard(unsaved + unclassed)
  const canFinish = !busy && !precise && !draft.length  // detection's calls stand unless changed: no object needs a class first
  useSaveKeys(() => { if (!precise) void saveAll(false) }, () => { if (canFinish) void saveAll(true) })
  const finishTitle = precise ? 'Leave precise editing first (Space keeps the edit, Esc cancels it)'
    : unclassed ? `Save your changes and finish this field. ${unclassed} new candidate${unclassed === 1 ? '' : 's'} without a class ${unclassed === 1 ? 'is' : 'are'} left out (Ctrl+Enter)`
      : 'Save your changes and finish this field: every call and outline you left as it is counts as checked by you (Ctrl+Enter)'
  const locked = precise ? 'Precise editing: only this object can change now. Space keeps the edit, Esc cancels it.' : undefined

  return (
    <div className="annotate-page round-page fit-page">
      <div className="annotate-top">
        <div className="annotate-heading">
          <span className="row small"><Link to="/annotate">← All sessions</Link><Link to={`/annotate/${name}`}>All fields of this session</Link></span>
          <h1>{meta.title}</h1>
          <div className="muted small">{meta.stain} · {ROUND_TITLE[2]} · field {position + 1} of {fields.length}</div>
        </div>
        <div className="annotate-progress">
          <RoundProgressLine progress={progress} me={detail.annotator} />
        </div>
        <div className="annotate-actions">
          <button className="btn" onClick={() => later && mayLeave() && navigate(`/annotate/${name}/${enc(later.tile_id)}`)} disabled={!later || !!precise}
            title={locked ?? (later ? 'Come back to this field later' : 'No other field is waiting for you')}>Skip for now</button>
          {done && <button className="btn" onClick={reopen} disabled={busy || !!precise} title={locked ?? 'Count this field as not finished by you yet'}>Reopen</button>}
        </div>
      </div>
      {meta.instructions && <div className="instructions">{meta.instructions}</div>}
      <ErrorNote error={error} />
      {restored > 0 && <div className="found found-warn small round-restored" role="status">{changesText(restored)} you had not saved came back from this browser.
        They are not saved yet. <button className="link-btn small" onClick={() => { setChanges({}); setUndos([]); setRestored(0) }}>Discard them</button></div>}
      {(openBy.length > 0 || doneBy.length > 0) && <div className="found found-warn small field-others" role="status">
        {doneBy.length > 0 ? `${doneBy.join(', ')} already finished this field.` : `${openBy.join(', ')} ${openBy.length === 1 ? 'is' : 'are'} working on this field right now.`}
        {' '}You can still look and change things here, or <button className="link-btn small" onClick={() => void (async () => {
          if (!mayLeave()) return
          const next = await api.get<{ tile_id: string | null }>(`/api/annotate/${name}/next?skip=${enc(tile)}`).catch(() => ({ tile_id: null }))
          navigate(next.tile_id ? `/annotate/${name}/${enc(next.tile_id)}` : `/annotate/${name}`)
        })()}>go to a field nobody is on</button>.</div>}
      <div className="annotate-workspace">
        <div className={`annotate-image${precise ? ' is-precise' : ''}`} onContextMenu={(e) => e.preventDefault()} onMouseLeave={() => { if (!precise) setHovered(null) }}>
          <ZoomViewer fit src={apiUrl(`/api/annotate/${name}/fields/${enc(tile)}/image.jpg`)}
            layers={field.layers.filter((l) => layers[l]).map((l) => apiUrl(`/api/annotate/${name}/fields/${enc(tile)}/layer/${l}.png`))}
            title={<span className="row"><span className="muted small">{tile}</span>{done && <span className="badge status-done">finished</span>}
              {precise && <span className="badge status-running">precise editing — Space keeps, Esc cancels</span>}
              {pending && <span className="small muted">Outlining…</span>}</span>}
            focus={focus} cursor={excluding ? 'none' : precise ? 'crosshair' : target ? 'pointer' : 'copy'} onClick={click} onMove={move}
            onPress={press} onRelease={release} panLock={excluding} onWheelOver={wheelBrush} onSize={setFieldSize}
            under={(_z, _s, rawUnit) => <ObjectLayer items={drawnItems} unit={Number.isFinite(rawUnit) && rawUnit > 0 ? rawUnit : 1}
              dim={precise ? 0.25 : excluding ? 0.55 : 1} />}>
            {(_zoom, size, rawUnit) => {
              const unit = Number.isFinite(rawUnit) && rawUnit > 0 ? rawUnit : 1
              const box = field.box
              const lit = target && !precise ? target : undefined
              const litPoly = lit ? outlineOf(lit) : null
              const litCall = lit ? callOf(lit) : null
              return <>
                {layers.box && <>
                  <path d={`M0 0H${size.w}V${size.h}H0Z M${box.x} ${box.y}v${box.height}h${box.width}v${-box.height}Z`} fill="#000" fillOpacity={0.22} fillRule="evenodd" />
                  <rect x={box.x} y={box.y} width={box.width} height={box.height} fill="none" stroke={BOX_COLOR} strokeWidth={2 * unit} strokeDasharray={`${12 * unit} ${8 * unit}`} />
                </>}
                {layers.margin && field.objects.filter((o) => !o.own).map((o) => {
                  const poly = o.masks[o.mask]
                  return poly ? <polygon key={o.candidate_id} points={polygonPoints(poly)} fill="none" stroke="#ffffff" strokeOpacity={0.35} strokeWidth={1.2 * unit}
                    pointerEvents="none" /> : null
                })}
                {lit && !precise && (() => {  // the object under the mouse, on top: thicker, with a glow
                  const look = lookOf(vocabulary, meta.remove_class, litCall!.pick.identity)
                  const width = 3.4 * unit
                  const dash = look.dash ? `${5 * unit} ${4 * unit}` : undefined
                  const shape = (extra: Record<string, unknown>) => litPoly
                    ? <polygon points={polygonPoints(litPoly)} {...extra} />
                    : <circle cx={lit.x} cy={lit.y} r={Math.max(6, lit.r)} {...extra} />
                  return <g className={`round-object look-${look.look} lit`} data-key={lit.key} pointerEvents="none">
                    {shape({ fill: 'none', stroke: '#ffffff', strokeOpacity: 0.55, strokeWidth: width + 6 * unit })}
                    {shape({ fill: 'none', stroke: '#000000', strokeOpacity: 0.6, strokeWidth: width + 1.8 * unit, strokeDasharray: dash, strokeLinecap: 'round' })}
                    {shape({ style: { stroke: look.colour, fill: look.colour }, fillOpacity: 0.22, strokeWidth: width, strokeDasharray: dash, strokeLinecap: 'round' })}
                  </g>
                })()}
                {lit && litCall && (() => {  // the name of the object under the cursor, beside it
                  const xs = litPoly ? litPoly.map((p) => p[0]) : [lit.x + lit.r]
                  const ys = litPoly ? litPoly.map((p) => p[1]) : [lit.y - lit.r]
                  const text = lit.fresh && !lit.fresh.removed ? 'new — press a class key' : callText(vocabulary, litCall.pick.identity, litCall.pick.subtype, typesAsked)
                  const score = lit.candidate?.ai_probability
                  const tx = Math.min(size.w - 4 * unit, Math.max(...xs) + 6 * unit), ty = Math.max(14 * unit, Math.min(...ys) - 4 * unit)
                  return <text className="object-tag" x={tx} y={ty} fontSize={13 * unit} strokeWidth={3.5 * unit} pointerEvents="none"
                    textAnchor={tx > size.w * 0.8 ? 'end' : 'start'}>{text}{score != null ? ` · ${fmt(score, 2)}` : ''}</text>
                })()}
                {precise && <>
                  <polygon points={polygonPoints(precise.points)} fill="#ffffff" fillOpacity={0.12} stroke="#000" strokeOpacity={0.6} strokeWidth={4 * unit} />
                  <polygon points={polygonPoints(precise.points)} fill="none" stroke="#ffffff" strokeWidth={2.2 * unit} />
                  {precise.points.map((p, i) => (
                    <circle key={i} cx={p[0]} cy={p[1]} r={3.6 * unit} fill="#fff" stroke="#000" strokeWidth={0.8 * unit} style={{ cursor: 'move' }}
                      onMouseDown={(e) => { if (e.button !== 0) return; e.stopPropagation(); startDrag(i) }} onMouseUp={(e) => e.stopPropagation()} />
                  ))}
                  {precise.clicks.map((c, i) => <circle key={`c${i}`} cx={c[0]} cy={c[1]} r={3.8 * unit} fill={c[2] ? '#4ade80' : '#f87171'} stroke="#000" strokeWidth={unit} />)}
                </>}
                {brush.url && <>  {/* the areas marked as bad: a translucent red hatch */}
                  <defs>
                    <pattern id="exclude-hatch" width={9 * unit} height={9 * unit} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                      <rect width={9 * unit} height={9 * unit} fill="#ff3b3b" fillOpacity={0.22} />
                      <line x1={0} y1={0} x2={0} y2={9 * unit} stroke="#ff3b3b" strokeOpacity={0.85} strokeWidth={2.4 * unit} />
                    </pattern>
                    <mask id="exclude-mask" maskUnits="userSpaceOnUse" x={0} y={0} width={size.w} height={size.h}>
                      <image href={brush.url} x={0} y={0} width={Math.ceil(size.w / BRUSH_SCALE) * BRUSH_SCALE} height={Math.ceil(size.h / BRUSH_SCALE) * BRUSH_SCALE} preserveAspectRatio="none" />
                    </mask>
                  </defs>
                  <rect x={0} y={0} width={size.w} height={size.h} fill="url(#exclude-hatch)" mask="url(#exclude-mask)" pointerEvents="none" />
                </>}
                {excluding && cursor && <circle cx={cursor[0]} cy={cursor[1]} r={brushPx} fill="#ff3b3b" fillOpacity={0.12} stroke="#ffffff" strokeWidth={1.6 * unit}
                  strokeDasharray={`${4 * unit} ${3 * unit}`} pointerEvents="none" />}
                {pending && <circle className="sam-pending" cx={pending[0]} cy={pending[1]} r={10 * unit} fill="none" stroke="#fff" strokeWidth={2 * unit} />}
                {draft.length > 0 && <>
                  <polyline points={polygonPoints([...draft, ...(cursor ? [cursor] : [])])} fill="none" stroke="#fff" strokeWidth={2 * unit} />
                  {draft.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r={(i === 0 ? 5 : 3) * unit} fill="#fff" stroke="#000" strokeWidth={unit} />)}
                </>}
              </>
            }}
          </ZoomViewer>
        </div>
        <aside className="annotate-side round-side" aria-label="The object, the keys and this field">
          {coach}
          {excluding && <div className="card side-block exclude-bar" role="status">
            <b>Excluding</b> · brush {brushUm} µm · <span className="kbd">E</span> to exit
            <p className="small secondary" style={{ margin: '4px 0 0' }}>Left-drag paints a bad area (fold, smudge, empty glass), right-drag or Shift-drag
              erases, scroll changes the brush, the middle button moves the picture, <span className="kbd">Z</span> takes back a stroke. Objects in it are left
              out of scoring and training; saved with Save.</p>
          </div>}
          <TargetCard s={target} call={target ? callOf(target) : null} meta={meta} precise={precise} px={px} outline={target ? outlineOf(target) : null}
            choices={choicesOf(target)} added={target?.added && saved[target.added.id] ? addedState(target) : null} unsaved={!!target && isUnsaved(target)}
            onMask={(i) => { if (target) chooseMask(target, i) }} />
          <div className="card side-block">
            <h3>Classes</h3>
            <ClassLegend vocabulary={vocabulary} keys={keys} removeClass={meta.remove_class} waiting={classKeys.waiting?.identity} />
          </div>
          <div className="card side-block">
            <h3>Keys</h3>
            <Controls sam={sam} precise={!!precise} excluding={excluding} />
            <LookLegend />
          </div>
          <div className="card side-block">
            <h3>This field</h3>
            <p className="small field-counts"><b>{checkedNow.toLocaleString()}</b> of {own.length.toLocaleString()} checked · <b>{changed}</b> changed by you ·{' '}
              <b>{added.length}</b> added</p>
            {unclassed > 0 && <p className="small warning-text" style={{ margin: '0 0 4px' }}>{unclassed} new candidate{unclassed === 1 ? '' : 's'} without a class:
              {' '}left out when you save, unless you give {unclassed === 1 ? 'it' : 'them'} a class.</p>}
            <p className="small muted" style={{ margin: 0 }}>Detection’s calls stand unless you change them: nothing needs a class first. Save and continue
              counts the whole field as checked by you, every object you left as it is with detection’s call.</p>
            <details className="side-details compact">
              <summary className="small">Tags and a note</summary>
              <div className="tag-list">{meta.tags.map((t) => (
                <label key={t} className="layer-toggle"><input type="checkbox" checked={!!mine[`tag:${tile}:${t}`]} disabled={!!precise}
                  onChange={() => save([{ task: 'tag', value: t, deleted: !!mine[`tag:${tile}:${t}`] }])} />{t}</label>
              ))}</div>
              <label className="field" style={{ marginTop: 6 }}>Note (optional)
                <textarea rows={2} value={note} disabled={!!precise} onChange={(e) => setNote(e.target.value)}
                  onBlur={() => { if (note !== (Object.values(mine).find((m) => m.task === 'note')?.value ?? '')) save([{ task: 'note', value: note, deleted: !note }]) }} /></label>
            </details>
          </div>
          <div className="card side-block side-save">
            <SaveState unsaved={unsaved} saving={saving} />
            <div className="row side-save-buttons">
              <SaveButtons unsaved={unsaved} saving={busy || !!precise} onSave={() => void saveAll(false)} onContinue={() => void saveAll(true)} canContinue={canFinish}
                continueTitle={finishTitle} />
            </div>
          </div>
          <div className="side-foot small">
            <button className={`btn small help-toggle${help ? ' chosen' : ''}`} aria-expanded={help} onClick={() => setHelp(!help)} title="How this page works">?</button>
            {!onFinished && <Link to="/annotate/tutorial">Tutorial</Link>}
            <span className="spacer" />
            <span className="muted">{tile}</span>
          </div>
          {help && (
            <div className="card side-block round-help">
              <ol className="small secondary round-steps">
                <li><b>Point at an object</b>: it lights up, its name shows beside it and its facts above. Press its class number, <span className="kbd">0</span> unsure,
                  {' '}<span className="kbd">Del</span> not an object (it turns dotted grey and stays; a class number brings it back).</li>
                <li><b>Outline too big or small?</b> <span className="kbd">Tab</span> / <span className="kbd">⇧ Tab</span>. Still not right: <span className="kbd">Space</span>,
                  then right-click spots to add, left-click spots to leave out, or drag the dots; <span className="kbd">Space</span> again keeps it.</li>
                <li><b>Missed object?</b> {sam ? 'Click it: its outline appears as a new candidate (dotted); then press its class number.' : 'Click its corners one by one, then the first corner (or Enter).'}</li>
                <li><b>Nothing is saved until you press Save</b> (a small orange dot marks each object you changed). <b>Save and continue</b> when the field
                  looks right: what you left as it is counts as checked by you, and the next field opens.</li>
                <li>Scroll to zoom; drag (or hold the middle button) to move the picture.</li>
              </ol>
              {sam && !samReady && <p className="small muted">Preparing one-click outlines for this field… the first clicks may take a few seconds.</p>}
              {sam && (byHand ? <p className="small precise-bar">Drawing by hand: click the corners of the object, then the first corner or press Enter.
                {' '}<button className="link-btn small" onClick={() => setByHand(false)}>Cancel</button></p>
                : <button className="btn small" disabled={!!precise} onClick={() => { setPrecise(null); setDraft([]); setByHand(true) }}
                  title="The last resort, when one click and precise editing cannot outline it">Draw an object by hand…</button>)}
              <details className="side-details compact">
                <summary className="small">Display options</summary>
                <label className="layer-toggle"><input type="checkbox" checked={!!layers.margin} onChange={() => toggle('margin')} />objects of the neighbouring fields</label>
                {field.layers.map((l) => (
                  <label key={l} className="layer-toggle"><input type="checkbox" checked={!!layers[l]} onChange={() => toggle(l)} /><span className="swatch" style={{ background: RASTER_LAYERS[l]?.color }} />{RASTER_LAYERS[l]?.label ?? l}</label>
                ))}
                <label className="layer-toggle"><input type="checkbox" checked={!!layers.box} onChange={() => toggle('box')} />field boundary (yellow)</label>
              </details>
              <details className="side-details compact">
                <summary className="small">Go to another field</summary>
                <p className="muted small">Green = finished by you; dot = someone else is on it.</p>
                <div className="field-jump">
                  {fields.map((f, i) => f.prepared
                    ? <Link key={f.tile_id} to={`/annotate/${name}/${enc(f.tile_id)}`} title={f.tile_id}
                      className={`field-chip${f.tile_id === tile ? ' current' : f.done_by_me ? ' done' : ''}`}>{i + 1}{f.open_by_others > 0 && <span className="busy-dot" />}</Link>
                    : <span key={f.tile_id} className="field-chip waiting" title="still being prepared">{i + 1}</span>)}
                </div>
              </details>
            </div>
          )}
        </aside>
      </div>
    </div>
  )
}

/** The keys of the field, one per line, with what they do; while precise editing, only its own. */
function Controls({ sam, precise, excluding }: { sam: boolean; precise: boolean; excluding: boolean }) {
  if (excluding) {
    return (
      <ul className="key-list small">
        <li className="key-now"><b>Excluding</b> — outlines are locked</li>
        <li><span className="kbd">left-drag</span> paint a bad area</li>
        <li><span className="kbd">right-drag</span> or <span className="kbd">⇧ drag</span> erase</li>
        <li><span className="kbd">scroll</span> brush size · <span className="kbd">middle</span> move the picture</li>
        <li><span className="kbd">Z</span> one stroke back · <span className="kbd">E</span> or <span className="kbd">Esc</span> done</li>
      </ul>
    )
  }
  if (precise) {
    return (
      <ul className="key-list small">
        <li className="key-now"><b>Precise editing</b> — only this object can change</li>
        {sam && <li><span className="kbd">right-click</span> add a spot to it</li>}
        {sam && <li><span className="kbd">left-click</span> leave a spot out</li>}
        <li><span className="kbd">drag</span> a dot to move the outline</li>
        <li><span className="kbd">Z</span> one step back</li>
        <li><span className="kbd">Space</span> keep the edit · <span className="kbd">Esc</span> cancel</li>
      </ul>
    )
  }
  return (
    <ul className="key-list small">
      <li><span className="kbd">Tab</span> <span className="kbd">⇧ Tab</span> bigger / smaller outline</li>
      <li><span className="kbd">Del</span> remove — it turns dotted grey</li>
      <li><span className="kbd">click</span> empty tissue: {sam ? 'a new candidate' : 'draw a new one by hand'}</li>
      {sam && <li><span className="kbd">Space</span> precise edit: right-click add, left-click remove, drag dots, <span className="kbd">Space</span> done</li>}
      <li><span className="kbd">E</span> mark a bad area with a brush (folds, smudges): its objects are left out</li>
      <li><span className="kbd">Z</span> undo · <span className="kbd">Esc</span> cancel</li>
      <li><span className="kbd">Ctrl S</span> save · <span className="kbd">Ctrl ⏎</span> save and continue</li>
    </ul>
  )
}

const ADDED_TEXT: Record<AddedState['state'], string> = {
  measuring: 'Being measured so it can teach the model…',
  measured: 'Measured — it will help train the next model.',
  could_not_be_measured: 'Could not be measured, so it cannot train the model; it still counts as an object detection missed.',
  proposed: 'Detection had proposed this object too, so it counts as that object.',
}

/** The facts of the object under the cursor (or being edited precisely): its class, detection's call and score, its
 *  area and which of its masks shows. */
function TargetCard({ s, call, meta, precise, px, outline, choices, added, unsaved, onMask }: {
  s: Shown | undefined; call: { pick: ClassPick; by: string; who?: string[] } | null; meta: Meta; precise: Precise | null; px: number
  outline: number[][] | null; choices: { list: (number[][] | null)[]; index: number; names?: string[] }; added: AddedState['state'] | null; unsaved: boolean
  onMask: (i: number) => void
}) {
  if (!s || !call) {
    return (
      <div className="card side-block round-target empty">
        <p className="small muted" style={{ margin: 0 }}>Point at an object: it lights up and its facts show here. Every key acts on it.</p>
      </div>
    )
  }
  const { vocabulary, subtypes: typesAsked } = meta
  const look = lookOf(vocabulary, meta.remove_class, call.pick.identity)
  const c = s.candidate
  const title = s.fresh && !s.fresh.removed ? 'New candidate — press a class key' : callText(vocabulary, call.pick.identity, call.pick.subtype, typesAsked)
  const who = call.by === 'me' ? (unsaved ? 'your change, not saved yet' : 'your call') : call.by === 'others' ? `checked by ${call.who?.join(', ')}`
    : call.by === 'ai' ? 'detection’s call' : 'added now'
  return (
    <div className={`card side-block round-target look-${look.look}`} style={{ borderLeftColor: look.look === 'class' || look.look === 'unsure' ? look.colour : undefined }}>
      <div className="row target-head"><span className={`look-swatch look-${look.look}`} style={{ borderColor: look.colour, background: look.dash ? 'transparent' : look.colour }} />
        <b className="ellipsis">{title}</b></div>
      <dl className="target-facts small">
        <dt>Now</dt><dd>{who}</dd>
        {c && <><dt>Detection</dt><dd>{callText(vocabulary, c.ai_value, c.ai_subtype, typesAsked)}{c.decision !== 'accepted' && <span className="muted">
          {' '}(rejected{c.reason?.startsWith('rule:') ? `: ${c.reason.slice(5).replace(/_/g, ' ')}` : ''})</span>}</dd>
          <dt>Score</dt><dd>{c.ai_probability != null ? fmt(c.ai_probability, 2) : '—'}</dd></>}
        {outline && <><dt>Area</dt><dd>{fmt(polygonArea(outline) * px * px, 3)} µm²</dd></>}
        {choices.list.length > 0 && !precise && <><dt>Outline</dt><dd>{maskName(choices.index, choices.list.length, choices.names)}</dd></>}
      </dl>
      {added && added !== 'proposed' && <p className={`small added-note${added === 'could_not_be_measured' ? ' warn' : ''}`}><b>Detection never proposed this object.</b> {ADDED_TEXT[added]}</p>}
      {precise ? <p className="small precise-bar"><b>Precise editing.</b> {precise.busy ? 'Outlining…' : 'Right-click adds a spot, left-click leaves one out, drag the dots.'}</p>
        : <MaskSwitch masks={choices.list} index={choices.index} onChange={onMask} names={choices.names} />}
    </div>
  )
}
