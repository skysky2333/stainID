/* Round 2, "Fix the field". Step 1 of every field is the acceptance threshold (fieldStart.ts): a slider starting at
   detection's own threshold re-decides, live, which objects start solid and which dotted grey; it locks on "Start
   editing" (Enter) or by itself on the first edit, and only "Reset field" unlocks it (discarding the field's unsaved
   edits). Then the whole field with every object detection considered and its outline, always shown, one
   shape per object: solid in its type's colour (each counted type and each named look-alike its own colour), dotted
   light grey when it is not active: rejected by detection or by a person (Del), or just added and waiting for its type;
   dotted amber with a "?" when someone is unsure. Objects in an excluded area are drawn faint.
   Everything acts on the object under the cursor (it lights up, its name shows beside it and its facts at the side):
   1–4 pick its counted type, Q W E R a named look-alike (the server's `vocabulary.assign_keys`), ` (or 0) unsure,
   Delete rejects it (it turns dotted grey and stays; a number key makes it solid again), Tab switches its outline between Segment Anything's and the stain outline (the choice is saved per candidate), a click on it edits it precisely
   (everything else locked: a click adds a spot to it, Shift+click leaves one out, drag the dots; Z undoes a step; a
   click outside its neighbourhood, drawn faintly, or Enter keeps the edit, Esc leaves without keeping), Space hides and shows everything drawn over the
   picture (a clean look at the tissue), B changes the bad areas (the brush, brush.ts: a drag adds to the chosen type's
   area, Shift-drag takes out of it, Tab / Shift+Tab the brush size; people's areas teach and test bad-area detection,
   the analysis keeps detection's own, drawn hatched as "Excluded from analysis"), S splits an object in two (a click:
   a preview, Enter keeps it; or a line drawn across objects), M merges objects (click or stroke them, Enter), Z undoes. Clicking empty tissue
   outlines a missed object with one click: a new candidate, dotted, until
   a class key makes it solid. A change of call is recorded exactly like an answer in round 1. Every change stays on this
   page ("3 unsaved changes", with a draft in this browser in case of a reload; Z undoes) until Save (stays on the field)
   or Save and continue: the field is then finished, every call nobody has decided and every outline left as it is counts
   as checked by this person, and the next field comes. Opening a field holds it for this person, so the others are
   handed other fields; what others decide meanwhile shows within seconds ("checked by Ben"). The page takes the window
   exactly: the picture as large as fits, the side panel scrolling by itself. */
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import type { AddedState, FieldAreas, Label, PeopleAreas, RoundCheck, RoundDetail, RoundField, RoundObject, RoundProgress, RoundSamOptions, RoundSaved, SessionDetail } from '../api'
import { api, apiUrl, enc, pendingText } from '../api'
import { ErrorNote, Loading, fmt } from '../components'
import { PALETTE } from '../palette'
import { useFetch, useStored } from '../hooks'
import type { ImagePoint } from '../viewer'
import { RASTER_LAYERS, ZoomViewer } from '../viewer'
import type { ClassPick } from './classKeys'
import { resolveClass, roleOf } from './classKeys'
import type { RoundEvent } from './rounds'
import { BRUSH_KEY, BRUSH_PX, BRUSH_SCALE, CLEARED, EXCLUDE_TYPES, INCLUDE, brushStep, useBrush } from './brush'
import type { Slider } from './fieldStart'
import { SLIDER_STEP, acceptedAt, lock, moveTo, slider, sliderText, startPick, unlock } from './fieldStart'
import { LockIcon, ThresholdSlider } from './threshold'
import { reasonText, whyStart } from './reasons'
import { peopleChecks } from './aiLabels'
import type { NotCounted, ThreadLadder } from './threads'
import { THREAD_KEY, ThreadLines, ThreadsCard, stepOf } from './threads'
import { BackToSets, CHOICE_IDS, CHOICE_NAMES, PIECE_IDS, PIECE_NAMES, tabNext, ClassLegend, LookLegend, MaskSwitch, ROUND_TITLE, RoundProgressLine, SaveButtons, SaveState, callText, changesText, draftKey, lookOf, maskName,
  polygonArea, polygonPoints, readDraft, simplifyPolygon, stepMask, unsureClass, useCandidateKeys, useFitScreen, useLeaveGuard, useLive, useSaveKeys,
  writeDraft } from './rounds'

const BOX_COLOR = PALETTE.markers.box
const HANDLES = 48  // at most this many dots to drag in precise editing (the outline is simplified to them)
const rkey = (x: number, y: number) => `${Math.round(x)}:${Math.round(y)}`
type Meta = SessionDetail['meta'] & RoundDetail
type Mine = Label & { candidate?: string | null }
type Rec = Record<string, unknown>
/** The areas excluded from analysis over a field (GET …/fields/{tile}/excluded, store_view.field_excluded): the field's
 *  part of the core's automatic layer (the same pixels as the Cores view and the analysis), where its picture lies in field pixels
 *  and the types on it. */
interface FieldExcluded {
  status: string; scale?: number; x?: number; y?: number; width?: number; height?: number; version?: string | number
  kinds: { id: string; label: string; color: string; hatch: string; type: string; fraction: number | null }[]
}
/** An object someone just added with one click and has not given a type yet (kept on the page only); `removed`: they
 *  pressed Delete on it, so it stays dotted grey until a number key makes it solid. */
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

/** The neighbourhood of an object in precise editing: its bounding box grown by half its size each way, at least
 *  `least` (image px): a click inside adds or leaves out a spot, a click beyond it ends the editing. */
function neighbourhood(polygon: number[][], least: number) {
  const xs = polygon.map((p) => p[0]), ys = polygon.map((p) => p[1])
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys)
  const gx = Math.max(least, (x1 - x0) / 2), gy = Math.max(least, (y1 - y0) / 2)
  return { x: x0 - gx, y: y0 - gy, w: x1 - x0 + 2 * gx, h: y1 - y0 + 2 * gy }
}

/** The length of a drawn line (field px). */
function lineLength(line: number[][]) {
  let n = 0
  for (let i = 1; i < line.length; i++) n += Math.hypot(line[i][0] - line[i - 1][0], line[i][1] - line[i - 1][1])
  return n
}

/** A polygon's centre (its area-weighted centroid; the corners' mean for a degenerate one), rounded to 0.1 px. */
function centroid(points: number[][]): number[] {
  let a = 0, cx = 0, cy = 0
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const f = points[j][0] * points[i][1] - points[i][0] * points[j][1]
    a += f; cx += (points[j][0] + points[i][0]) * f; cy += (points[j][1] + points[i][1]) * f
  }
  if (Math.abs(a) < 1e-9) return [points.reduce((t, q) => t + q[0], 0) / points.length, points.reduce((t, q) => t + q[1], 0) / points.length].map((v) => Math.round(v * 10) / 10)
  return [cx / (3 * a), cy / (3 * a)].map((v) => Math.round(v * 10) / 10)
}

/** Whether the boxes around an outline and a line meet (the objects a split line may cut; the server checks it crosses). */
function boxesMeet(polygon: number[][], line: number[][]) {
  const box = (q: number[][]) => [Math.min(...q.map((v) => v[0])), Math.min(...q.map((v) => v[1])), Math.max(...q.map((v) => v[0])), Math.max(...q.map((v) => v[1]))]
  const [a0, b0, a1, b1] = box(polygon), [c0, d0, c1, d1] = box(line)
  return a0 <= c1 && c0 <= a1 && b0 <= d1 && d0 <= b1
}

function circle(x: number, y: number, r: number) {
  return Array.from({ length: 20 }, (_, i) => [Math.round(x + r * Math.cos((i * Math.PI) / 10)), Math.round(y + r * Math.sin((i * Math.PI) / 10))])
}

/** Whether an outline label names one of a candidate's outlines (CHOICE_IDS, or "sam" of earlier releases). */
const isChoice = (value: unknown) => typeof value === 'string' && (CHOICE_IDS.includes(value as typeof CHOICE_IDS[number]) || value === 'sam')
/** Its index into the candidate's outlines; "sam" of earlier releases: the Segment Anything output shown first, else medium. */
const choiceIndex = (value: unknown, o?: { mask: number }) => (value === 'sam' ? (o && o.mask > 0 ? o.mask : 2) : CHOICE_IDS.indexOf(value as typeof CHOICE_IDS[number]))

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
  others: boolean; unsaved: boolean; fresh: boolean; faded: boolean; piece?: string }

/** Who checked an object, as the page says it: people by name (never the AI's labels: aiLabels.ts). */
export function checkedByText(who: string[] = []): string {
  return who.join(', ')
}

/** People's areas as SVG paths (rings in field px: an outer ring, then its holes). */
const ringsPath = (rings: number[][][]) => rings.map((r) => `M${r.map((p) => p.join(',')).join('L')}Z`).join('')

/** People's areas over the field: painted ones outlined in their type's colour, the parts marked fine dashed green.
 *  They teach and test bad-area detection (and leave objects in them out of training and scoring); the analysis never
 *  reads them. */
function PeopleAreaLines({ areas, unit }: { areas: PeopleAreas; unit: number }) {
  return (
    <g pointerEvents="none" className="people-areas">
      {areas.regions.map((r, k) => r.kind === INCLUDE
        ? <path key={k} d={ringsPath(r.rings)} fill="none" fillRule="evenodd" stroke={CLEARED.colour} strokeWidth={2 * unit} strokeDasharray={`${7 * unit} ${5 * unit}`} />
        : <path key={k} d={ringsPath(r.rings)} fill="none" fillRule="evenodd" stroke={EXCLUDE_TYPES.find((t) => t.key === r.kind)?.colour ?? '#fff'} strokeWidth={2.2 * unit} />)}
    </g>
  )
}

const ObjectLayer = memo(function ObjectLayer({ items, unit, dim }: { items: Drawn[]; unit: number; dim: number }) {
  return (
    <g opacity={dim} pointerEvents="none">
      {items.map((d) => {
        const width = (d.dash ? 2.2 : 2.4) * unit
        const dash = d.dash ? `${5 * unit} ${4 * unit}` : undefined
        const shape = (extra: Record<string, unknown>) => d.points
          ? <polygon points={d.points} {...extra} />
          : <circle cx={d.x} cy={d.y} r={Math.max(6, d.r)} {...extra} />
        return <g key={d.key} className={`round-object look-${d.look}${d.unsaved ? ' unsaved' : ''}`} data-key={d.key} opacity={d.faded ? 0.35 : undefined}>
          {shape({ fill: 'none', stroke: '#000000', strokeOpacity: 0.6, strokeWidth: width + 1.8 * unit, strokeDasharray: dash, strokeLinecap: 'round' })}
          {shape({ style: { stroke: d.colour, fill: d.colour }, fillOpacity: d.dash ? 0 : 0.08, strokeWidth: width, strokeDasharray: dash, strokeLinecap: 'round' })}
          {d.diamond && <polygon points={`${d.x},${d.y - 5 * unit} ${d.x + 5 * unit},${d.y} ${d.x},${d.y + 5 * unit} ${d.x - 5 * unit},${d.y}`}
            fill={d.look === 'class' || d.look === 'unsure' ? d.colour : '#ffffff'} stroke="#000" strokeWidth={0.8 * unit} />}
          {d.look === 'unsure' && <text className="unsure-mark" x={d.x} y={d.y + 4.5 * unit} fontSize={13 * unit} textAnchor="middle" fill={d.colour}
            stroke="#000" strokeWidth={2.5 * unit} paintOrder="stroke">?</text>}
          {d.others && <circle cx={d.x} cy={d.y} r={2.4 * unit} fill={PALETTE.markers.others} stroke="#000" strokeWidth={0.6 * unit} />}
          {d.piece && <text className="piece-mark" x={d.x} y={d.y + 3.5 * unit} fontSize={9 * unit} textAnchor="middle" fill="#fff" stroke="#000"
            strokeWidth={2.2 * unit} paintOrder="stroke">{d.piece === 'split' ? '½' : '∪'}</text>}
          {d.unsaved && !d.fresh && <circle className="unsaved-mark" cx={d.x + Math.max(6, d.r) * 0.75} cy={d.y - Math.max(6, d.r) * 0.75} r={2.6 * unit}
            fill={PALETTE.markers.unsaved} strokeWidth={0.8 * unit} />}
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
  const unsure = unsureClass(vocabulary)
  const draftAt = (meta as { practice?: boolean }).practice ? '' : draftKey(detail.annotator, name, tile)  // the practice keeps no draft: a reload starts it afresh with the coach
  const [saved, setSaved] = useState<Record<string, Mine>>(() => Object.fromEntries(field.mine.map((m) => [m.id, m])))  // as the server has them
  const [changes, setChanges] = useState<Record<string, Change>>(() => readDraft<{ changes: Record<string, Change> }>(draftAt)?.changes ?? {})  // not saved yet
  // step 1, the acceptance threshold: where the objects start (fieldStart.ts); locked before the first edit
  const [start, setStart] = useState<Slider | null>(() => {
    const detection = field.detection_threshold
    if (detection == null) return null
    const kept = readDraft<{ slider?: Slider }>(draftAt)?.slider
    return field.threshold == null && kept ? { ...kept, detection } : slider(detection, field.threshold)
  })
  const lockStart = useCallback(() => setStart((s) => (s && !s.locked ? lock(s) : s)), [])
  const [mode, setMode] = useState<'split' | 'merge' | null>(null)  // S and M
  // AT8 threads mode (H, threads.tsx): the field's thread network at one thread threshold, drawn only in this mode
  const threadsHere = field.threads != null
  const [threading, setThreading] = useState(false)
  const [threadsAsked, setThreadsAsked] = useState(false)  // fetched once threads mode first opens (made ahead on the server)
  const ladder = useFetch<ThreadLadder>(threadsHere && threadsAsked ? `/api/annotate/${name}/fields/${enc(tile)}/threads` : null)
  const [threadSaved, setThreadSaved] = useState<number | null>(field.threads?.factor ?? null)
  const [threadStep, setThreadStep] = useState<number | null>(null)
  const notCounted = useFetch<NotCounted>(threadsHere && threadsAsked ? `/api/annotate/${name}/fields/${enc(tile)}/threads/areas` : null)
  const ladderDone = ladder.data?.status === 'done' && ladder.data.steps.length ? ladder.data : null
  const threadIndex = ladderDone ? (threadStep ?? stepOf(ladderDone.factors, threadSaved, ladderDone.detection)) : 0
  const [split, setSplit] = useState<{ key: string; pieces: number[][][]; busy: boolean } | null>(null)  // the preview of a split
  const [chosenForMerge, setChosenForMerge] = useState<string[]>([])
  const [stroke, setStroke] = useState<number[][] | null>(null)  // a line drawn across objects (split) or over them (merge)
  const [restored, setRestored] = useState(() => changedThings(changes))
  const mine = useMemo(() => {  // this person's labels as the page shows them: the saved ones with the unsaved changes on top
    const out = { ...saved }
    for (const [id, c] of Object.entries(changes)) { if (c.label) out[id] = c.label; else delete out[id] }
    return out
  }, [saved, changes])
  const [checked, setChecked] = useState<Record<string, RoundCheck>>(() => peopleChecks(field.checked))  // people's only: never the AI's labels
  const [openBy, setOpenBy] = useState<string[]>(field.open_by)
  const [doneBy, setDoneBy] = useState<string[]>(field.done_by)
  const [progress, setProgress] = useState<RoundProgress | null>(meta.progress)
  const chosenMasks = (labels: Record<string, Mine>) => Object.fromEntries(Object.values(labels)  // the outline chosen per candidate (Tab)
    .filter((m) => m.task === 'outline' && m.candidate && isChoice(m.value))
    .map((m) => [m.candidate!, choiceIndex(m.value, field.objects.find((o) => o.candidate_id === m.candidate))]))
  const [masks, setMasks] = useState<Record<string, number>>(() => chosenMasks(mine))
  const [liveSam, setLiveSam] = useState<Record<string, (number[][] | null)[]>>({})  // Segment Anything's sizes asked live, for a candidate the store has none for
  const [pieceSam, setPieceSam] = useState<Record<string, (number[][] | null)[]>>({})  // a split piece's SAM small and medium, asked live
  const [prompts, setPrompts] = useState<Record<string, { options: number[][][]; chosen: number }>>({})  // masks of objects added in this visit
  const [fresh, setFresh] = useState<Record<string, Fresh>>({})
  const [hovered, setHovered] = useState<string | null>(null)
  const [precise, setPrecise] = useState<Precise | null>(null)
  const [hidden, setHidden] = useState(false)  // Space: nothing drawn over the picture, for a clean look at the tissue
  const [dragging, setDragging] = useState<number | null>(null)
  // precise editing, a dot being dragged: press on it, move, let go anywhere (the dot holds the pointer: `setPointerCapture`)
  const dragRef = useRef<{ index: number; pointer: number; before: number[][]; target: Element } | null>(null)
  const dragEndedAt = useRef(0)  // the click a drag's release makes is not a click on the picture
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
  const [layers, setLayers] = useStored<Record<string, boolean>>('stainid.round2.layers', { box: true, margin: true, excluded: true })
  const [help, setHelp] = useStored<boolean>('stainid.round2.help', false)
  const [note, setNote] = useState(() => field.mine.find((m) => m.task === 'note')?.value ?? '')
  const sam = meta.segment_anything && field.segment_anything
  // the brush (B): bad areas painted on a mask, saved with the field's other changes as people's areas
  const [excluding, setExcluding] = useState(false)
  const [brushSize, setBrushSize] = useStored<number>(BRUSH_KEY, BRUSH_PX.start)  // the brush's diameter on screen (px)
  const [brushKind, setBrushKind] = useStored<string>('stainid.round2.brush.type', '1')  // the type painted (keys 1–4 with the brush on)
  // what the analysis leaves out over the field: its part of detection's automatic layer, as on the Cores view
  const excluded = useFetch<FieldExcluded>(`/api/annotate/${name}/fields/${enc(tile)}/excluded`)
  // people's areas over the field (painted and marked fine), outlined; they never change the layer above
  const people = useFetch<PeopleAreas>(`/api/annotate/${name}/fields/${enc(tile)}/annotated`)
  // each brush type's current area (detection's with people's saved edits merged in): what the brush adds to and takes from
  const areas = useFetch<FieldAreas>(`/api/annotate/${name}/fields/${enc(tile)}/areas`)
  const brushType = EXCLUDE_TYPES.find((t) => t.key === brushKind) ?? EXCLUDE_TYPES[0]
  const [shiftDown, setShiftDown] = useState(false)  // Shift held while the brush is on: it erases (the cursor shows it)
  useEffect(() => {
    const on = (e: KeyboardEvent) => setShiftDown(e.shiftKey)
    window.addEventListener('keydown', on)
    window.addEventListener('keyup', on)
    return () => { window.removeEventListener('keydown', on); window.removeEventListener('keyup', on) }
  }, [])
  const [fieldSize, setFieldSize] = useState<{ w: number; h: number } | null>(null)
  // the strokes since the last Save over each type's current area, drawn on a canvas over the field once a frame (the
  // chosen type strong while the brush is on, every type at the same strength once it is off)
  const brush = useBrush(fieldSize, areas.data, excluding ? brushKind : '', excluding ? 0.3 : 0.55)
  const ring = useRef<SVGCircleElement | null>(null)  // the brush under the mouse: moved directly, no React update per move
  const resetAfter = useRef<FieldAreas | null | undefined>(undefined)  // after Save: the current areas before it
  useEffect(() => {  // the server has the saved edit: the strokes are in the current areas now
    if (resetAfter.current === undefined || areas.data === resetAfter.current) return
    resetAfter.current = undefined
    brush.reset()
  }, [areas.data, brush.reset])

  useLive(name, { tile_id: tile }, (live) => {
    if (live.checked) setChecked(peopleChecks(live.checked))
    if (live.open_by) setOpenBy(live.open_by)
    if (live.done_by) setDoneBy(live.done_by)
    if (live.progress) setProgress(live.progress)
  })
  const unsaved = changedThings(changes) + (brush.dirty ? 1 : 0)
  const discarding = useRef(false)  // Discard and close: no draft is kept again on the way out
  useEffect(() => { if (!discarding.current) writeDraft(draftAt, unsaved ? { changes, slider: start } : null) }, [draftAt, changes, unsaved, start])
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
  const replaced = new Set(Object.values(mine).filter((m) => m.task === 'replaced' && m.candidate).map((m) => m.candidate!))  // split or merged
  const added = Object.values(mine).filter((m) => m.task === 'class' && !m.candidate && m.x != null && !ownKeys.has(rkey(m.x, m.y!)))
  const shown: Shown[] = [
    ...own.filter((o) => !replaced.has(o.candidate_id)).map((o) => ({ key: o.candidate_id, x: o.x, y: o.y, r: o.radius_px, candidate: o })),
    ...added.map((m) => ({ key: `add:${rkey(m.x!, m.y!)}`, x: m.x!, y: m.y!, r: 8, added: m })),
    ...Object.values(fresh).map((f) => ({ key: f.key, x: f.x, y: f.y, r: 8, fresh: f })),
  ]
  const byKey = new Map(shown.map((s) => [s.key, s]))
  const target = precise ? byKey.get(precise.key) : hovered ? byKey.get(hovered) : undefined

  /** Where a candidate starts: detection's call, at the acceptance threshold of step 1. */
  const startOf = (o: RoundObject): ClassPick => (start ? startPick(o, start.value, start.detection, meta.remove_class)
    : { identity: o.ai_value, subtype: o.ai_subtype || null })
  /** Where an object's call stands: this person's, else what others decided, else where it starts (none for a new object). */
  const callOf = (s: Shown): { pick: ClassPick; by: 'me' | 'others' | 'ai' | 'new'; who?: string[] } => {
    if (s.fresh) return { pick: { identity: s.fresh.removed ? meta.remove_class : '', subtype: null }, by: 'new' }
    const my = mine[classId(s.x, s.y)]
    if (my) return { pick: resolveClass(vocabulary, my.value, my.subtype), by: 'me' }
    if (s.candidate && checked[s.candidate.candidate_id]) {
      const c = checked[s.candidate.candidate_id]
      return { pick: { identity: c.identity, subtype: c.subtype || null }, by: 'others', who: c.by }
    }
    return { pick: s.candidate ? startOf(s.candidate) : { identity: '', subtype: null }, by: 'ai' }
  }
  const samNames = (n: number) => Array.from({ length: n }, (_, i) => `SAM ${i + 1}/${n}`)  // one click: Segment Anything's own three
  /** A split piece (an object of a split, `meta.edit`): its split outline, the outline it shows now (PIECE_IDS) and its
   *  siblings (the other pieces of the split). */
  const pieceOf = (s: Shown) => {
    const m = s.added
    if (!m || m.meta?.edit !== 'split') return null
    const drawn = mine[outlineId(s.x, s.y)]
    const at = PIECE_IDS.indexOf((drawn?.value ?? '') as typeof PIECE_IDS[number])
    const split = drawn?.meta?.split_outline ?? (at > 0 ? null : drawn?.points) ?? prompts[s.key]?.options[0] ?? null
    const siblings = added.filter((o) => o.meta?.group && o.meta.group === m.meta?.group && o.id !== m.id)
      .map((o) => ({ x: o.x!, y: o.y!, points: mine[outlineId(o.x!, o.y!)]?.points ?? null }))
    return { split, index: at > 0 ? at : 0, siblings, group: m.meta.group }
  }
  const choicesOf = (s: Shown | undefined): { list: (number[][] | null)[]; index: number; names?: string[]; first?: number } => {
    if (!s) return { list: [], index: 0 }
    if (s.candidate) {  // its outlines (CHOICE_IDS: stain, SAM small, medium, large), the stain's default first
      const id = s.candidate.candidate_id
      const m = s.candidate.masks, live = liveSam[id]
      return { list: [m[0] ?? null, m[1] ?? live?.[0] ?? null, m[2] ?? live?.[1] ?? null, m[3] ?? live?.[2] ?? null], index: masks[id] ?? s.candidate.mask,
        names: CHOICE_NAMES, first: s.candidate.mask }
    }
    const piece = pieceOf(s)
    if (piece) return { list: [piece.split, ...(pieceSam[s.key] ?? [null, null])], index: piece.index, names: PIECE_NAMES, first: 0 }
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
    lockStart()  // the first edit of any kind locks the acceptance threshold
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
      next[id] = { mark: r, label: label(id, r.task as Mine['task'], x, y, { value: String(r.value ?? ''), subtype: (r.subtype as string) || null, points: (r.points as number[][]) ?? null, candidate,
        meta: (r.meta as Mine['meta']) ?? null }) }
      if (r.task === 'class' && at && r.mask != null && !noObject(r.value)) {  // the mask chosen with the call is saved with it
        const mask = Number(r.mask)
        next[outlineId(x, y)] = { mark: null, label: label(outlineId(x, y), 'outline', x, y, { value: CHOICE_IDS[mask] ?? '', points: at.masks[mask] ?? null, candidate }) }
      } else if (r.task === 'class' && at && isChoice(changes[outlineId(x, y)]?.label?.value)) {
        next[outlineId(x, y)] = { mark: null, label: null }  // not an object (or unsure): no mask goes with it
      }
    }
    setUndos((u) => [{ before: Object.fromEntries(Object.keys(next).map((id) => [id, changes[id]])), local: undo?.local }, ...u].slice(0, 200))
    const out = { ...changes, ...next }
    for (const [id, c] of Object.entries(next)) if (!c.mark && !c.label) delete out[id]  // taken away before it was ever saved
    setChanges(out)
  }, [name, tile, own, saved, changes, meta.stain, meta.remove_class, unsure, detail.annotator, lockStart])  // eslint-disable-line react-hooks/exhaustive-deps

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
      const edited = drawn && !isChoice(drawn.value) && !drawn.value?.startsWith('mask')
      const shownAt = masks[s.candidate.candidate_id] ?? s.candidate.mask
      // the outline it is on goes with the call when the store has it (one Segment Anything drew live is its own outline record)
      const mask = !edited && s.candidate.masks.length > 1 && s.candidate.masks[shownAt] ? shownAt : null
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

  /** Show this one of the object's outlines (Tab, or a button of the side panel). A candidate's choice is kept with
   *  the field (its option by name, `CHOICE_IDS`); Segment Anything where the store has none is asked live. */
  const chooseMask = useCallback((s: Shown, next: number) => {
    if (precise) return
    const { list, index } = choicesOf(s)
    if (next === index) return
    if (s.candidate && !list[next] && s.candidate.more_masks && s.candidate.masks.filter(Boolean).length <= 1) { setError('The other outlines are still loading: press Tab again in a moment.'); return }
    if (s.candidate && !list[next] && next > 0 && sam) {  // no Segment Anything outline stored: one click of it, at the object's centre, its sizes
      const c = s.candidate
      void (async () => {
        try {
          const found = await api.post<RoundSamOptions>(`/api/annotate/${name}/fields/${enc(tile)}/segment`, { points: [[Math.round(c.x), Math.round(c.y), 1]] })
          const sizes = [0, 1, 2].map((i) => found.options[i]?.points ?? null)
          const points = sizes[next - 1]
          if (!points) { setError(`Segment Anything found no ${CHOICE_NAMES[next]} there: this object keeps its outline.`); return }
          setLiveSam((all) => ({ ...all, [c.candidate_id]: sizes }))
          setMasks((m) => ({ ...m, [c.candidate_id]: next }))
          save([{ task: 'outline', candidate: c.candidate_id, points, value: CHOICE_IDS[next] }], { local: () => setMasks((m) => ({ ...m, [c.candidate_id]: index })) })
          tell('mask')
        } catch (e) { setError(`${(e as Error).message} This object keeps its outline.`) }
      })()
      return
    }
    const piece = pieceOf(s)
    if (piece) {  // a split piece: its split outline, or Segment Anything's small or medium with the split as context
      const keep = (points: number[][]) => {
        save([{ task: 'outline', x: s.x, y: s.y, points: points.map((q) => q.map(Math.round)), value: PIECE_IDS[next],
          meta: { edit: 'split', group: piece.group, split_outline: piece.split ?? undefined } }], {})
        tell('mask')
      }
      if (next === 0 || list[next]) { if (list[next]) keep(list[next]!); return }
      if (!sam || !piece.split) { setError('One-click outlines are not available here: this piece keeps its split outline.'); return }
      const xs = piece.split.map((q) => q[0]), ys = piece.split.map((q) => q[1])
      const pad = Math.max(8, 0.15 * Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)))
      const box = [Math.min(...xs) - pad, Math.min(...ys) - pad, Math.max(...xs) + pad, Math.max(...ys) + pad].map(Math.round)
      const [cx, cy] = centroid(piece.split).map(Math.round)
      void (async () => {
        try {
          const found = await api.post<RoundSamOptions>(`/api/annotate/${name}/fields/${enc(tile)}/segment`, {
            points: [[cx, cy, 1], ...piece.siblings.map((o) => [Math.round(o.x), Math.round(o.y), 0])], box, sizes: true,
            avoid: piece.siblings.map((o) => o.points).filter((p): p is number[][] => !!p && p.length >= 3) })
          const sizes = [0, 1].map((i) => found.options[i]?.points ?? null)
          setPieceSam((all) => ({ ...all, [s.key]: sizes }))
          const points = sizes[next - 1]
          if (!points) { setError(`Segment Anything found no ${PIECE_NAMES[next]} for this piece: it keeps its split outline.`); return }
          keep(points)
        } catch (e) { setError(`${(e as Error).message} This piece keeps its split outline.`) }
      })()
      return
    }
    if (!list[next]) {
      if (s.candidate) setError(`This object has no ${CHOICE_NAMES[next]}${next > 0 ? ', and one-click outlines are not available here' : ''}.`)
      return
    }
    if (s.fresh) { const f = s.fresh; setFresh((all) => ({ ...all, [f.key]: { ...f, chosen: next } })); return }
    if (s.candidate) {
      const id = s.candidate.candidate_id
      setMasks((m) => ({ ...m, [id]: next }))
      save([{ task: 'outline', candidate: id, points: list[next], value: CHOICE_IDS[next] }], { local: () => setMasks((m) => ({ ...m, [id]: index })) })
    } else {
      setPrompts((p) => ({ ...p, [s.key]: { ...p[s.key], chosen: next } }))
      save([{ task: 'outline', x: s.x, y: s.y, points: list[next] }], { local: () => setPrompts((p) => ({ ...p, [s.key]: { ...p[s.key], chosen: index } })) })
    }
  }, [precise, save, masks, prompts])  // eslint-disable-line react-hooks/exhaustive-deps

  /** Tab: the object under the mouse moves to its next outline (a candidate: stain outline -> SAM small -> SAM medium;
   *  a split piece: split outline -> SAM small -> SAM medium, asked with the other pieces as context; an object added
   *  with one click: Segment Anything's next size). Nothing under the mouse: nothing happens. */
  const switchMask = useCallback((bigger: boolean) => {
    const s = target
    if (!s || precise) return
    const { list, index, names, first } = choicesOf(s)
    chooseMask(s, names === CHOICE_NAMES || names === PIECE_NAMES ? tabNext(index, first ?? 0) : stepMask(list, index, bigger))
  }, [target, precise, chooseMask])  // eslint-disable-line react-hooks/exhaustive-deps

  /** Out of precise editing: the view goes back to exactly where it was before (zoom and position), the object stays
   *  the one the keys act on. */
  const leavePrecise = () => {
    const key = precise?.key
    setPrecise(null)
    setFocus({ x: 0, y: 0, key: Date.now(), back: true })
    if (key) setHovered(key)
  }

  /** A click on an object: precise editing of it (zoomed in; the view before comes back on leaving). */
  const enterPrecise = (s: Shown) => {
    if (draft.length || precise) return
    lockStart()
    const start = outlineOf(s) ?? circle(s.x, s.y, Math.max(6, s.r))
    setPrecise({ key: s.key, points: simplifyPolygon(start.map((p) => [...p]), HANDLES), clicks: [], busy: false, history: [] })
    setFocus({ x: s.x, y: s.y, key: Date.now() })  // zoom in on it
    setHidden(false)
  }

  /** A click outside the object: precise editing ends, keeping the edit. */
  const finishPrecise = useCallback(() => {
    if (!precise) return
    const s = byKey.get(precise.key)
    const points = precise.points.map((p) => p.map(Math.round))
    leavePrecise()
    if (!s || !precise.history.length) return  // nothing changed
    if (s.fresh) { const f = s.fresh; setFresh((all) => ({ ...all, [f.key]: { ...f, options: [...f.options.slice(0, f.chosen), points, ...f.options.slice(f.chosen + 1)] } })); return }
    save([s.candidate ? { task: 'outline', candidate: s.candidate.candidate_id, points, value: 'edited' } : { task: 'outline', x: s.x, y: s.y, points, value: 'edited' }], {})
  }, [precise, save])  // eslint-disable-line react-hooks/exhaustive-deps

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

  const confirmThreads = async () => {  // Enter in threads mode: this field's thread threshold (one label, replaced when set again)
    if (!ladderDone || saving) return
    const factor = ladderDone.factors[threadIndex]
    if (await send([{ task: 'thread_threshold', value: factor.toFixed(4) }], false)) {
      setThreadSaved(factor); setThreadStep(null); setThreading(false)
      tell('threads-saved')
    }
  }
  useCandidateKeys(keys, {
    answering: !!target && !draft.length && !excluding && !threading,
    onSize: (bigger) => {  // the brush: Tab smaller, Shift+Tab bigger (the outline keys rest while it is on)
      if (!excluding) return false
      setBrushSize(brushStep(brushSize, bigger))
      tell('brush-size')
      return true
    },
    onClass: (pick) => {
      if (!target) return
      const before = callOf(target).pick
      tell(target.fresh ? 'added' : before.identity === pick.identity && before.identity !== meta.remove_class ? 'subtype' : 'class')
      giveClass(target, pick)
    },
    onUnsure: () => { if (target && unsure) { tell(target.fresh ? 'added' : 'unsure'); giveClass(target, { identity: unsure, subtype: null }) } },
    onRemove: () => { if (excluding || threading) return; if (target) tell('remove'); removeTarget() },
    onMask: (bigger) => { if (excluding || mode || threading) return; if (target && !precise) tell('mask'); switchMask(bigger) },  // the brush's Tab sizes it first (onSize)
    onPrecise: () => { tell('overlays'); setHidden((h) => !h) },  // Space: hide and show everything drawn over the picture
    onUndo: () => { tell('undo'); undo() },
    onEnter: () => {
      if (threading) { void confirmThreads(); return }  // threads mode: "Threads look right"
      if (start && !start.locked) { lockStart(); tell('start'); return }  // step 1 done: "Start editing"
      if (mode === 'split') { if (split && !split.busy) acceptSplit(); return }
      if (mode === 'merge') { void mergeChosen(); return }
      if (precise) { tell('precise-out'); finishPrecise(); return }
      if (draft.length >= 3) closeDraft()
    },
    onEscape: () => {
      if (dragRef.current) { endDrag(true); return }  // during a drag: only that drag is undone, the editing goes on
      if (threading) { setThreading(false); return }  // the thread layer goes with the mode
      if (split) { setSplit(null); return }  // the preview goes; a second Esc leaves split mode
      if (mode) { leaveMode(); return }
      if (excluding) { setExcluding(false); return }
      if (precise) { leavePrecise(); return }
      if (draft.length || byHand) { setDraft([]); setByHand(false); return }
      if (target?.fresh) { const f = target.fresh; setFresh((all) => { const next = { ...all }; delete next[f.key]; return next }) }
    },
  })

  const leaveMode = () => { setMode(null); setSplit(null); setChosenForMerge([]); setStroke(null) }
  useEffect(() => {  // B the brush, S split, M merge on and off; [ ] the acceptance threshold in fine steps
    const onKey = (e: KeyboardEvent) => {
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement).tagName) || e.metaKey || e.ctrlKey || e.altKey) return
      if (e.key === '[' || e.key === ']') {
        if (threading && ladderDone) {  // threads mode: one step of the thread threshold
          e.preventDefault()
          setThreadStep(Math.max(0, Math.min(ladderDone.factors.length - 1, threadIndex + (e.key === ']' ? 1 : -1)))); tell('threads-step')
        } else if (start && !start.locked) { e.preventDefault(); setStart((s) => (s ? moveTo(s, s.value + (e.key === ']' ? SLIDER_STEP : -SLIDER_STEP)) : s)); tell('threshold') }
        return
      }
      if (e.repeat) return
      if (excluding && ['1', '2', '3', '4'].includes(e.key)) { e.preventDefault(); setBrushKind(e.key); return }  // the type of area painted
      const key = e.key.toLowerCase()
      if (key === THREAD_KEY && threadsHere && !precise && !draft.length && !e.shiftKey) {  // H: threads mode on and off
        e.preventDefault(); setHovered(null); leaveMode(); setExcluding(false); setThreadsAsked(true)
        setThreading((on) => !on); tell('threads-mode')
        return
      }
      if (!['b', 's', 'm'].includes(key) || precise || draft.length || e.shiftKey) return
      e.preventDefault()
      setHovered(null)
      setThreading(false)
      if (key === 'b') { leaveMode(); setExcluding((on) => !on); return }
      setExcluding(false)
      const next = key === 's' ? 'split' : 'merge'
      setSplit(null); setChosenForMerge([]); setStroke(null)
      setMode((m) => (m === next ? null : next))
      tell(next === 'split' ? 'split-mode' : 'merge-mode')
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [precise, draft.length, excluding, start, threading, ladderDone, threadIndex])  // eslint-disable-line react-hooks/exhaustive-deps
  const brushRadius = (unit: number) => (brushSize / 2) * unit  // in field px at this zoom (`unit`: field px per screen px): constant on screen
  const press = (p: ImagePoint, event: React.MouseEvent) => {
    if (event.button !== 0) return  // the left button paints (Shift: takes out) or draws a line; the middle one pans
    if (mode) { setStroke([[p.x, p.y]]); return }
    if (!excluding) return
    lockStart()
    brush.begin(p.x, p.y, brushRadius(p.unit), brushKind, event.shiftKey)  // Shift: out of the chosen type's area
  }
  const release = () => {
    if (mode) {
      const line = stroke
      setStroke(null)
      if (line && line.length > 1 && lineLength(line) > 8) { if (mode === 'split') void splitAlong(line); else pickAlong(line) }
      return
    }
    if (!excluding) return
    const before = brush.end()
    if (!before) return
    save([], { local: () => brush.restore(before) })  // one stroke is one step of Z
    tell('exclude')
  }

  /* ---------- split (S) and merge (M): object edits, saved like the others (labels/objectedit.py) ---------- */
  const editable = (s: Shown | undefined | null): s is Shown => !!s && !s.fresh && !!outlineOf(s)
  /** These objects are replaced by `pieces` (outlines, each with its type): the pieces are objects of their own (class
   *  and outline labels, `meta.edit`), each original gets a `replaced` label. One step of Z. */
  const replaceBy = (originals: Shown[], pieces: { points: number[][]; pick: ClassPick }[], kind: 'split' | 'merge') => {
    const group = Math.random().toString(16).slice(2, 10)
    const into = pieces.map((piece) => centroid(piece.points))
    const records: Rec[] = []
    for (const o of originals) {  // first the originals (a piece may land where an added original was)
      if (o.candidate) {
        records.push({ task: 'replaced', candidate: o.candidate.candidate_id, value: kind, meta: { group, into } })
        if (mine[classId(o.x, o.y)]) records.push({ task: 'class', candidate: o.candidate.candidate_id, deleted: true })
        if (mine[outlineId(o.x, o.y)]) records.push({ task: 'outline', candidate: o.candidate.candidate_id, deleted: true })
      } else {
        records.push({ task: 'replaced', x: o.x, y: o.y, value: kind, meta: { group, into } })
        records.push({ task: 'class', x: o.x, y: o.y, deleted: true }, { task: 'outline', x: o.x, y: o.y, deleted: true })
      }
    }
    pieces.forEach((piece, i) => {
      const [x, y] = into[i]
      const points = piece.points.map((q) => q.map(Math.round))
      records.push({ task: 'class', x, y, value: piece.pick.identity, subtype: typesAsked ? piece.pick.subtype ?? '' : '', meta: { edit: kind, group } },
        { task: 'outline', x, y, points, meta: { edit: kind, group } })
      setPrompts((all) => ({ ...all, [`add:${rkey(x, y)}`]: { options: [points], chosen: 0 } }))
    })
    save(records, {})
    tell(kind)
  }
  const [merging, setMerging] = useState(false)  // a merge or a cut along a line is with the server
  const splitAt = async (s: Shown, seed?: number[]) => {
    const points = outlineOf(s)
    if (!points) return
    lockStart()
    setSplit({ key: s.key, pieces: split?.key === s.key ? split.pieces : [], busy: true })
    try {
      const found = await api.post<{ pieces: number[][][] }>(`/api/annotate/${name}/fields/${enc(tile)}/split`, { points, seed: seed ?? null })
      setSplit({ key: s.key, pieces: found.pieces, busy: false })
      setError(null)
      tell('split-preview')
    } catch (e) {
      setSplit(null)
      setError((e as Error).message.replace(/^\d{3}: /, ''))
    }
  }
  const acceptSplit = () => {
    const s = split ? byKey.get(split.key) : undefined
    if (!split || !s || split.pieces.length !== 2) return
    const pick = callOf(s).pick
    replaceBy([s], split.pieces.map((points) => ({ points, pick })), 'split')
    setSplit(null)
  }
  /** A line drawn in split mode: every object it crosses right through is cut along it (the server checks). */
  const splitAlong = async (line: number[][]) => {
    const near = shown.filter((s) => editable(s) && boxesMeet(outlineOf(s)!, line))
    if (!near.length) return
    setMerging(true)
    try {
      const found = await api.post<{ pieces: (number[][][] | null)[] }>(`/api/annotate/${name}/fields/${enc(tile)}/split`,
        { line, outlines: near.map((s) => outlineOf(s)) })
      const cut = near.map((s, i) => ({ s, pieces: found.pieces[i] })).filter((c) => c.pieces && c.pieces.length === 2)
      if (!cut.length) { setError('The line crosses no object right through: draw it from outside an object, across it, to outside again.'); return }
      setError(null)
      for (const c of cut) replaceBy([c.s], c.pieces!.map((points) => ({ points, pick: callOf(c.s).pick })), 'split')
    } catch (e) {
      setError((e as Error).message.replace(/^\d{3}: /, ''))
    } finally {
      setMerging(false)
    }
  }
  const toggleMerge = (s: Shown) => {
    setChosenForMerge((all) => (all.includes(s.key) ? all.filter((k) => k !== s.key) : [...all, s.key]))
    lockStart()
    tell('merge-pick')
  }
  const pickAlong = (line: number[][]) => {
    const touched = shown.filter((s) => editable(s) && line.some(([x, y]) => inside(outlineOf(s)!, x, y))).map((s) => s.key)
    if (!touched.length) return
    lockStart()
    setChosenForMerge((all) => [...all, ...touched.filter((k) => !all.includes(k))])
    tell('merge-pick')
  }
  const mergeChosen = async () => {
    if (merging) return
    const chosen = chosenForMerge.map((k) => byKey.get(k)).filter(editable)
    if (chosen.length < 2) { setError('Choose two or more objects to merge (click them, or drag over them).'); return }
    setMerging(true)
    try {
      const found = await api.post<{ outline: number[][] }>(`/api/annotate/${name}/fields/${enc(tile)}/merge`, { outlines: chosen.map((s) => outlineOf(s)) })
      const largest = [...chosen].sort((a, b) => polygonArea(outlineOf(b)!) - polygonArea(outlineOf(a)!))[0]
      replaceBy(chosen, [{ points: found.outline, pick: callOf(largest).pick }], 'merge')
      setChosenForMerge([])
      setError(null)
    } catch (e) {
      setError((e as Error).message.replace(/^\d{3}: /, ''))
    } finally {
      setMerging(false)
    }
  }

  /** "Reset field" (after a confirmation): this field's unsaved edits discarded and the acceptance threshold free again. */
  const resetField = () => {
    if (!window.confirm('Reset this field? Your unsaved changes on it are discarded and you can set the acceptance threshold again.')) return
    setChanges({}); setUndos([]); setRestored(0); setFresh({}); setPrompts({}); setPrecise(null); setDraft([]); setByHand(false)
    setMasks(chosenMasks(saved)); leaveMode(); brush.reset(); setError(null)
    setStart((s) => (s ? unlock(s) : s))
    writeDraft(draftAt, null)
    tell('reset')
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
    } catch (e) {  // one-click outlines not working here: this click is the first corner of an outline drawn by hand
      setError(`${(e as Error).message} This click is the first corner: click the object's other corners, then the first one (or Enter).`)
      setDraft([at])
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
    if (event.button !== 0) return  // only the left button edits (the middle one pans; a right-click means nothing)
    if (mode) {
      const on = hit(p)
      if (!editable(on)) return
      if (mode === 'merge') { toggleMerge(on); return }
      if (split?.key === on.key && !split.busy) void splitAt(on, [p.x, p.y])  // a second click on it: the split moves
      else if (!split?.busy) void splitAt(on)
      return
    }
    if (precise) {  // a click adds this spot, Shift+click leaves it out; beyond the object's neighbourhood: done
      if (dragRef.current || performance.now() - dragEndedAt.current < 300) return  // the end of a drag, wherever it is let go
      const n = neighbourhood(precise.points, 24 * p.unit)
      if (p.x < n.x || p.x > n.x + n.w || p.y < n.y || p.y > n.y + n.h) { tell('precise-out'); finishPrecise(); return }
      if (!precise.busy && sam) { tell('precise-spot'); void refine([...precise.clicks, [p.x, p.y, event.shiftKey ? 0 : 1]]) }
      return
    }
    if (draft.length) {
      if (draft.length >= 3 && Math.hypot(draft[0][0] - p.x, draft[0][1] - p.y) < 10 * p.unit) closeDraft()
      else setDraft([...draft, [p.x, p.y]])
      return
    }
    const on = hit(p)
    if (on) { tell('precise-in'); enterPrecise(on); return }  // a click on an object: precise editing of it (the keys act on what is under the mouse)
    if (sam && !byHand) void addAt(p.x, p.y)
    else { setDraft([[p.x, p.y]]); setByHand(false) }
  }
  const move = (p: ImagePoint) => {
    if (excluding) {  // painting: the ring and the canvas change directly (at most once a frame), the page is not drawn again
      const r = ring.current
      if (r) { r.setAttribute('cx', String(p.x)); r.setAttribute('cy', String(p.y)); r.setAttribute('r', String(brushRadius(p.unit))); r.style.visibility = 'visible' }
      brush.extend(p.x, p.y)
      return
    }
    setCursor([p.x, p.y])
    if (mode && stroke) { setStroke([...stroke, [p.x, p.y]]); return }
    if (dragRef.current) return  // the dot follows the pointer itself (`dragMove`)
    if (!precise && !draft.length) {
      const now = hit(p)?.key ?? null
      if (now && now !== hovered) tell('hover')
      setHovered(now)
    }
  }
  /** The end of a dot's drag: let go anywhere (inside or beyond the frame, beyond the picture), the pointer lost, the
   *  window left. It never ends precise editing. With `cancel` (Esc) the dot goes back to where it was. */
  const endDrag = useCallback((cancel = false) => {
    const d = dragRef.current
    if (!d) return
    dragRef.current = null
    dragEndedAt.current = performance.now()
    try { if (d.target.hasPointerCapture(d.pointer)) d.target.releasePointerCapture(d.pointer) } catch { /* gone already */ }
    setDragging(null)
    if (cancel) setPrecise((p) => (p ? { ...p, points: d.before, history: p.history.slice(0, -1) } : p))
  }, [])
  useEffect(() => {  // a backstop: a release the dot does not hear (another window, a lost capture) still ends the drag
    if (dragging === null) return
    const up = () => endDrag()
    const away = () => endDrag()
    window.addEventListener('pointerup', up, true)
    window.addEventListener('pointercancel', up, true)
    window.addEventListener('mouseup', up, true)
    window.addEventListener('blur', away)
    return () => {
      window.removeEventListener('pointerup', up, true)
      window.removeEventListener('pointercancel', up, true)
      window.removeEventListener('mouseup', up, true)
      window.removeEventListener('blur', away)
    }
  }, [dragging, endDrag])
  /** A press on a dot: the drag starts (one undo step per drag); the dot holds the pointer until it is let go. */
  const dragDown = (e: React.PointerEvent<SVGCircleElement>, i: number) => {
    if (e.button !== 0 || !precise) return
    e.stopPropagation()
    e.preventDefault()  // no mousedown for the picture: it neither pans nor counts the release as a click
    try { e.currentTarget.setPointerCapture(e.pointerId) } catch { /* the window's listeners end it then */ }
    dragRef.current = { index: i, pointer: e.pointerId, before: precise.points, target: e.currentTarget }
    tell('precise-drag')
    setPrecise({ ...precise, history: [...precise.history, { points: precise.points, clicks: precise.clicks }] })
    setDragging(i)
  }
  const dragMove = (e: React.PointerEvent<SVGCircleElement>) => {
    const d = dragRef.current
    if (!d || e.pointerId !== d.pointer) return
    e.stopPropagation()
    const svg = e.currentTarget.ownerSVGElement
    const matrix = svg?.getScreenCTM()
    if (!svg || !matrix) return
    const at = new DOMPoint(e.clientX, e.clientY).matrixTransform(matrix.inverse())  // the picture's px
    setPrecise((p) => (p ? { ...p, points: p.points.map((q, k) => (k === d.index ? [at.x, at.y] : q)) } : p))
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
    const painted = brush.dirty
    if (painted) marks.push({ task: 'exclude', pngs: brush.pngs(), scale: BRUSH_SCALE })  // added to and taken out of each type's area
    if (start) marks.push({ task: 'threshold', value: start.value.toFixed(3) })  // the acceptance threshold of step 1, locked now
    setStart((s) => (s ? lock(s) : s))
    if (!(await send(marks, finish))) return
    if (painted && !(meta as { practice?: boolean }).practice) {  // the edit is saved: the current areas have it now
      resetAfter.current = areas.data ?? null
      areas.reload()
      people.reload()
    } else if (painted) brush.reset()
    setChanges({}); setUndos([]); setRestored(0)
    writeDraft(draftAt, null)
    tell(finish ? 'finish' : 'save')
    if (!finish) return
    if (onFinished) { onFinished(); return }
    reload()
    const next = await api.get<{ tile_id: string | null }>(`/api/annotate/${name}/next`).catch(() => ({ tile_id: null }))
    navigate(next.tile_id && next.tile_id !== tile ? `/annotate/${name}/${enc(next.tile_id)}` : `/annotate/${name}`)
  }
  /** Discard and close: this field's unsaved changes (and their copy in this browser) dropped, back to the session's
   *  page. One confirmation when there is anything to lose. */
  const discardAndClose = () => {
    const lost = unsavedNow()
    if (lost && !window.confirm(`Discard ${changesText(lost)} on this field and close it? They are not saved and cannot be brought back.`)) return
    writeDraft(draftAt, null)
    discarding.current = true
    navigate(`/annotate/${name}`)
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
    const look = lookOf(vocabulary, meta.remove_class, call.pick.identity, call.pick.subtype)
    const poly = outlineOf(s)
    return { key: s.key, x: s.x, y: s.y, r: s.r, points: poly ? polygonPoints(poly) : null, colour: look.colour, look: look.look, dash: look.dash,
      diamond: !!((s.added && !s.added.meta?.edit) || s.fresh), others: call.by === 'others' && !!call.who?.length, unsaved: isUnsaved(s), fresh: !!s.fresh,
      faded: !!s.candidate?.excluded, piece: s.added?.meta?.edit }
  }), [own, mine, fresh, checked, masks, prompts, precise?.key, precise?.points, changes, vocabulary, meta.remove_class, start])  // eslint-disable-line react-hooks/exhaustive-deps

  const fields = detail.fields
  const position = fields.findIndex((f) => f.tile_id === tile)
  const later = [...fields.slice(position + 1), ...fields.slice(0, position)].find((f) => !f.done_by_me && f.prepared && !f.open_by_others && f.tile_id !== tile)
  const changed = Object.values(mine).filter((m) => (m.task === 'class' || m.task === 'outline') && !m.confirmed).length
  const checkedNow = own.filter((o) => mine[classId(o.x, o.y)] || checked[o.candidate_id]).length
  const startable = own.filter((o) => !o.excluded)  // what step 1 counts: the field's own objects, not those in excluded areas
  const positive = (identity: string) => roleOf(vocabulary, identity) === 'positive'
  const unclassed = Object.values(fresh).filter((f) => !f.removed).length
  const toggle = (k: string) => setLayers({ ...layers, [k]: !layers[k] })
  const busy = saving || !!pending
  const px = field.pixel_size_um
  const mayLeave = useLeaveGuard(unsaved + unclassed)
  const unsavedNow = () => unsaved + unclassed + (precise || draft.length ? 1 : 0)
  const canFinish = !busy && !precise && !draft.length  // detection's calls stand unless changed: no object needs a class first
  useSaveKeys(() => { if (!precise) void saveAll(false) }, () => { if (canFinish) void saveAll(true) })
  const finishTitle = precise ? 'Leave precise editing first (a click beyond the frame or Enter keeps the edit, Esc cancels it)'
    : unclassed ? `Save your changes and finish this field. ${unclassed} new candidate${unclassed === 1 ? '' : 's'} without a class ${unclassed === 1 ? 'is' : 'are'} left out (Ctrl+Enter)`
      : 'Save your changes and finish this field: every call and outline you left as it is counts as checked by you (Ctrl+Enter)'
  const locked = precise ? 'Precise editing: only this object can change now. A click beyond the frame or Enter keeps the edit, Esc cancels it.' : undefined

  return (
    <div className="annotate-page round-page fit-page">
      <div className="annotate-top">
        <div className="annotate-heading">
          <span className="row small"><BackToSets />{!(meta as { practice?: boolean }).practice && <Link to={`/annotate/${name}`}>All fields of this session</Link>}</span>
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
        {doneBy.length > 0 ? `${checkedByText(doneBy)} already finished this field.` : `${openBy.join(', ')} ${openBy.length === 1 ? 'is' : 'are'} working on this field right now.`}
        {' '}You can still look and change things here, or <button className="link-btn small" onClick={() => void (async () => {
          if (!mayLeave()) return
          const next = await api.get<{ tile_id: string | null }>(`/api/annotate/${name}/next?skip=${enc(tile)}`).catch(() => ({ tile_id: null }))
          navigate(next.tile_id ? `/annotate/${name}/${enc(next.tile_id)}` : `/annotate/${name}`)
        })()}>go to a field nobody is on</button>.</div>}
      <div className="annotate-workspace">
        <div className={`annotate-image${precise ? ' is-precise' : ''}`} onContextMenu={(e) => e.preventDefault()} onMouseLeave={() => { if (!precise) setHovered(null) }}>
          <ZoomViewer fit src={apiUrl(`/api/annotate/${name}/fields/${enc(tile)}/image.jpg`)}
            layers={hidden ? [] : field.layers.filter((l) => layers[l]).map((l) => apiUrl(`/api/annotate/${name}/fields/${enc(tile)}/layer/${l}.png`))}
            title={<span className="row"><span className="muted small">{tile}</span>{done && <span className="badge status-done">finished</span>}
              {precise && <span className="badge status-running">precise editing — click beyond the frame or Enter to finish, Esc cancels</span>}
              {hidden && <span className="badge">Overlays hidden · <span className="kbd">Space</span></span>}
              {pending && <span className="small muted">Outlining…</span>}
              {mode === 'split' && <span className="badge status-running">split — click an object, or draw a line across · Esc leaves</span>}
              {mode === 'merge' && <span className="badge status-running">merge — {chosenForMerge.length} selected · Enter to merge · Esc leaves</span>}
              {start && !start.locked && <span className="badge">Step 1: set the acceptance threshold</span>}
              {target && (target.candidate || pieceOf(target)) && !precise && !excluding && <span className="badge outline-badge" title="Tab switches the outline of the object under the mouse">
                Outline: {(choicesOf(target).names ?? CHOICE_NAMES)[choicesOf(target).index]} · Tab</span>}</span>}
            focus={focus} cursor={threading ? 'default' : excluding ? 'none' : mode ? 'crosshair' : precise ? 'crosshair' : target ? 'pointer' : 'copy'} onClick={threading ? undefined : click} onMove={move}
            onPress={press} onRelease={release} panLock={excluding || !!mode} onSize={setFieldSize}
            paint={!hidden && fieldSize && (excluding || brush.dirty) ? (  // each type's area now: one merged region per type (brush.ts), from the first stroke
              <div ref={brush.host} className="exclude-painted" style={{ position: 'absolute', left: 0, top: 0,
                width: `${(100 * Math.ceil(fieldSize.w / BRUSH_SCALE) * BRUSH_SCALE) / fieldSize.w}%`, height: `${(100 * Math.ceil(fieldSize.h / BRUSH_SCALE) * BRUSH_SCALE) / fieldSize.h}%` }} />
            ) : null}
            under={(_z, _s, rawUnit) => hidden ? null : <>
              {layers.excluded !== false && !excluding && excluded.data?.status === 'done' && excluded.data.width && excluded.data.height ? (  // excluded from analysis
                <image href={apiUrl(`/api/annotate/${name}/fields/${enc(tile)}/excluded.png?v=${excluded.data.version ?? 0}`)} x={excluded.data.x} y={excluded.data.y}
                  width={excluded.data.width} height={excluded.data.height} preserveAspectRatio="none" style={{ imageRendering: 'pixelated' }} />
              ) : null}
              {layers.excluded !== false && !excluding && people.data && <PeopleAreaLines areas={people.data} unit={Number.isFinite(rawUnit) && rawUnit > 0 ? rawUnit : 1} />}
              <ObjectLayer items={drawnItems} unit={Number.isFinite(rawUnit) && rawUnit > 0 ? rawUnit : 1} dim={precise ? 0.25 : threading ? 0.3 : excluding ? 0.55 : split ? 0.6 : 1} />
            </>}>
            {(_zoom, size, rawUnit) => {
              const unit = Number.isFinite(rawUnit) && rawUnit > 0 ? rawUnit : 1
              if (hidden) return null  // a clean look at the tissue
              const box = field.box
              const lit = target && !precise ? target : undefined
              const litPoly = lit ? outlineOf(lit) : null
              const litCall = lit ? callOf(lit) : null
              return <>
                {threading && ladderDone && <ThreadLines lines={ladderDone.steps[threadIndex].lines} unit={unit} areas={notCounted.data} />}
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
                  const look = lookOf(vocabulary, meta.remove_class, litCall!.pick.identity, litCall!.pick.subtype)
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
                  {(() => {  // where a click still edits; beyond it a click ends the editing
                    const n = neighbourhood(precise.points, 24 * unit)
                    return <rect x={n.x} y={n.y} width={n.w} height={n.h} rx={6 * unit} fill="none" stroke="#ffffff" strokeOpacity={0.55}
                      strokeWidth={1.2 * unit} strokeDasharray={`${6 * unit} ${5 * unit}`} pointerEvents="none" />
                  })()}
                  <polygon points={polygonPoints(precise.points)} fill="#ffffff" fillOpacity={0.12} stroke="#000" strokeOpacity={0.6} strokeWidth={4 * unit} />
                  <polygon points={polygonPoints(precise.points)} fill="none" stroke="#ffffff" strokeWidth={2.2 * unit} />
                  {precise.points.map((p, i) => (
                    <circle key={i} className="precise-dot" cx={p[0]} cy={p[1]} r={(dragging === i ? 4.6 : 3.6) * unit} fill="#fff" stroke="#000" strokeWidth={0.8 * unit}
                      style={{ cursor: 'move', touchAction: 'none' }} data-dot={i}
                      onPointerDown={(e) => dragDown(e, i)} onPointerMove={dragMove} onPointerUp={(e) => { e.stopPropagation(); endDrag() }}
                      onPointerCancel={() => endDrag()} onLostPointerCapture={() => endDrag()}
                      onMouseDown={(e) => e.stopPropagation()} onMouseUp={(e) => e.stopPropagation()} onClick={(e) => e.stopPropagation()} />
                  ))}
                  {precise.clicks.map((c, i) => <circle key={`c${i}`} cx={c[0]} cy={c[1]} r={3.8 * unit} fill={c[2] ? PALETTE.markers.spot_added : PALETTE.markers.spot_removed} stroke="#000" strokeWidth={unit} />)}
                </>}
                {mode === 'merge' && chosenForMerge.map((k) => {  // chosen to merge
                  const s = byKey.get(k)
                  const poly = s ? outlineOf(s) : null
                  return poly ? <polygon key={`m-${k}`} points={polygonPoints(poly)} fill={PALETTE.markers.merge} fillOpacity={0.18} stroke={PALETTE.markers.merge} strokeWidth={3 * unit} pointerEvents="none" /> : null
                })}
                {split && split.pieces.map((piece, i) => (  // the preview of a split: Enter keeps it
                  <polygon key={`s-${i}`} points={polygonPoints(piece)} fill={i ? PALETTE.markers.split_b : PALETTE.markers.split_a} fillOpacity={0.22} stroke={i ? PALETTE.markers.split_b : PALETTE.markers.split_a}
                    strokeWidth={2.4 * unit} pointerEvents="none" />
                ))}
                {stroke && stroke.length > 1 && <polyline points={polygonPoints(stroke)} fill="none" stroke={mode === 'split' ? PALETTE.markers.split_b : PALETTE.markers.merge}
                  strokeWidth={2.4 * unit} strokeDasharray={`${6 * unit} ${4 * unit}`} pointerEvents="none" />}
                {excluding && <circle ref={ring} r={brushRadius(unit)} fill={shiftDown ? CLEARED.colour : brushType.colour} fillOpacity={0.15} style={{ visibility: 'hidden' }}
                  stroke="#ffffff" strokeWidth={1.6 * unit} strokeDasharray={`${4 * unit} ${3 * unit}`} pointerEvents="none" />}
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
          {start && (!start.locked ? (
            <div className="card side-block start-card" role="group" aria-label="Step 1: the acceptance threshold">
              <h3>Step 1: set the acceptance threshold</h3>
              <ThresholdSlider value={start.value} detection={start.detection} onChange={(v) => { setStart((s) => (s ? moveTo(s, v) : s)); tell('threshold') }} />
              <p className="small" style={{ margin: '4px 0' }}>{sliderText(acceptedAt(startable, start.value, start.detection, meta.remove_class, positive), start.value,
                acceptedAt(startable, start.detection, start.detection, meta.remove_class, positive), start.detection)}</p>
              <p className="small muted" style={{ margin: '0 0 6px' }}>Objects scoring at or above it start solid in detection’s type, those below dotted grey (at 0 every object that no rule of detection decided; point at one to see its rule).
                It only sets where this field starts (it is saved with the field to tune thresholds later); the analysis keeps detection’s threshold.</p>
              <div className="row"><button className="btn primary" onClick={() => { lockStart(); tell('start') }}>Start editing</button>
                <span className="small muted"><span className="kbd">Enter</span> · <span className="kbd">[</span> <span className="kbd">]</span> fine steps</span></div>
            </div>
          ) : (
            <div className="card side-block start-card locked small row">
              <LockIcon /><span>Acceptance threshold <b>{start.value.toFixed(2)}</b>{Math.abs(start.value - start.detection) > 1e-9 ? ` (detection ${start.detection.toFixed(2)})` : ' (detection’s)'}</span>
              <span className="spacer" /><button className="link-btn small" onClick={resetField} disabled={saving}
                title="Discard this field’s unsaved changes and set the threshold again">Reset field</button>
            </div>
          ))}
          {threading && <ThreadsCard ladder={ladder.data} pending={ladder.pending ? pendingText(ladder.pending) : null} error={ladder.error} index={threadIndex}
            onChange={(i) => { setThreadStep(i); tell('threads-step') }} onConfirm={() => void confirmThreads()} saving={saving} saved={threadSaved} notCounted={!!notCounted.data?.png} />}
          {mode === 'split' && <div className="card side-block exclude-bar" role="status">
            <b>Split</b>{split?.busy || merging ? <> · <Loading inline label="Splitting…" /></> : null} — <span className="kbd">S</span> or <span className="kbd">Esc</span> to leave
            <ul className="key-list small" style={{ marginTop: 6 }}>
              <li><span className="kbd">click</span> an object: it is split in two (a preview)</li>
              <li><span className="kbd">click</span> elsewhere on it: the split moves · <span className="kbd">Enter</span> keeps it · <span className="kbd">Esc</span> drops it</li>
              <li><span className="kbd">drag</span> a line across objects: each one it crosses right through is cut along it</li>
            </ul>
            <p className="small secondary" style={{ margin: '4px 0 0' }}>The pieces keep the object’s type (the keys retype them); <span className="kbd">Z</span> undoes.
              They are your labels: detection’s objects stay what the analysis counts.</p>
          </div>}
          {mode === 'merge' && <div className="card side-block exclude-bar" role="status">
            <b>Merge</b> — <b>{chosenForMerge.length}</b> selected{merging ? <> · <Loading inline label="Merging…" /></> : chosenForMerge.length >= 2 ? ' · Enter to merge' : ''} · <span className="kbd">M</span> or <span className="kbd">Esc</span> to leave
            <ul className="key-list small" style={{ marginTop: 6 }}>
              <li><span className="kbd">click</span> objects to add or take them out · <span className="kbd">drag</span> over objects adds every one it touches</li>
              <li><span className="kbd">Enter</span> merges them into one: the type of the largest (the keys retype it)</li>
            </ul>
          </div>}
          {excluding && <div className="card side-block exclude-bar" role="status">
            <b>Excluding</b> · <span className="look-swatch" style={{ borderColor: brushType.colour, background: brushType.colour }} /> {brushType.name} · brush <b>{brushSize} px</b> on screen (zoom in for a finer one)
            {' '}· <span className="kbd">B</span> to exit
            <ul className="class-legend small" style={{ marginTop: 6 }}>{EXCLUDE_TYPES.map((t) => (
              <li key={t.key} className={t.key === brushKind ? 'waiting' : ''}><span className="kbd key-slot">{t.key}</span>
                <span className="look-swatch" style={{ borderColor: t.colour, background: t.colour }} />{t.name}</li>
            ))}
              <li><span className="kbd key-slot">⇧</span><span className="look-swatch" style={{ borderColor: CLEARED.colour, background: CLEARED.colour }} />Shift-drag:
                take it out of the {brushType.name} area, detection’s part too</li></ul>
            <p className="small secondary" style={{ margin: '4px 0 0' }}>The hatched region is the {brushType.name} area as it stands: detection’s with everyone’s
              edits. A drag adds to it (one region, however many strokes; another type’s pixels become this type), Shift-drag takes out of it.
              While the brush is on, 1–4 choose the type and the class keys rest; <span className="kbd">Tab</span> smaller, <span className="kbd">⇧ Tab</span> bigger
              brush; the wheel zooms, the middle button moves the picture, <span className="kbd">Z</span> takes back a stroke. Saved with Save: people’s
              areas teach and test bad-area detection, and objects in them are left out of training and scoring. The analysis keeps detection’s own.</p>
          </div>}
          <TargetCard s={target} call={target ? callOf(target) : null} meta={meta} precise={precise} px={px} outline={target ? outlineOf(target) : null}
            why={target?.candidate ? whyStart(target.candidate, start ? start.value : null, start ? start.detection : null) : ''}
            choices={choicesOf(target)} added={target?.added && !target.added.meta?.edit && saved[target.added.id] ? addedState(target) : null} unsaved={!!target && isUnsaved(target)}
            onMask={(i) => { if (target) chooseMask(target, i) }} />
          <div className="card side-block">
            <h3>Classes</h3>
            <ClassLegend vocabulary={vocabulary} keys={keys} removeClass={meta.remove_class} active={target ? callOf(target).pick : null} />
            <p className="small muted" style={{ margin: '4px 0 0' }}><span className="kbd">B</span> brush for bad areas (then 1–4 are area types) ·
              {' '}<span className="kbd">S</span> split · <span className="kbd">M</span> merge{threadsHere && <> · <span className="kbd">H</span> threads
                {threadSaved != null ? ` (set: ${threadSaved === 1 ? 'detection’s' : `${threadSaved.toFixed(2)}×`})` : ''}</>}</p>
          </div>
          <div className="card side-block">
            <h3>Keys</h3>
            <Controls sam={sam} precise={!!precise} excluding={excluding} mode={mode} />
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
          {(() => {  // what the hatches and outlines over the field are
            const shown = layers.excluded !== false
            const found = shown && excluded.data?.status === 'done' ? excluded.data.kinds.filter((k) => k.fraction == null || k.fraction > 0) : []
            const theirs = shown ? [...new Set((people.data?.regions ?? []).map((r) => r.kind))] : []
            const mine = brush.dirty ? EXCLUDE_TYPES.filter((t) => brush.types.includes(t.key)) : []
            if (!found.length && !theirs.length && !mine.length) return null
            return (
              <div className="card side-block">
                {found.length > 0 && <><h3>Excluded from analysis</h3>
                  <ul className="look-legend small secondary" style={{ marginTop: 0, paddingTop: 0, borderTop: 'none' }}>
                    {found.map((k) => <li key={k.id}><span className="look-swatch hatched" style={{ borderColor: k.color, color: k.color }} />{k.label}</li>)}
                  </ul>
                  <p className="small muted" style={{ margin: '2px 0 6px' }}>Detection’s own areas (hatched): what the analysis leaves out.</p></>}
                {(theirs.length > 0 || mine.length > 0) && <><h3>People’s areas</h3>
                  <ul className="look-legend small secondary" style={{ marginTop: 0, paddingTop: 0, borderTop: 'none' }}>
                    {theirs.map((kind) => {
                      const t = kind === INCLUDE ? CLEARED : EXCLUDE_TYPES.find((e) => e.key === kind)
                      return <li key={`theirs-${kind}`}><span className={`look-swatch${kind === INCLUDE ? ' look-removed' : ''}`} style={{ borderColor: t?.colour, background: 'transparent' }} />
                        {kind === INCLUDE ? 'taken out' : t?.name ?? kind} (outlined)</li>
                    })}
                    {mine.map((t) => <li key={`mine-${t.key}`}><span className="look-swatch hatched" style={{ borderColor: t.colour, color: t.colour }} />
                      {t.name} with your strokes (not saved yet)</li>)}
                  </ul>
                  <p className="small muted" style={{ margin: '4px 0 0' }}>They teach and test bad-area detection, and objects in them are left out of training
                    and scoring; the analysis keeps detection’s areas. <span className="kbd">B</span> changes them (keys 1–4 the type), Shift-drag takes out.</p></>}
              </div>
            )
          })()}
          <div className="card side-block side-save">
            <SaveState unsaved={unsaved} saving={saving} />
            <div className="row side-save-buttons">
              <SaveButtons unsaved={unsaved} saving={busy || !!precise} onSave={() => void saveAll(false)} onContinue={() => void saveAll(true)} canContinue={canFinish}
                continueTitle={finishTitle} />
            </div>
            {!onFinished && <div className="row side-discard"><span className="spacer" />
              <button className="link-btn small" onClick={discardAndClose}
                title="Drop your unsaved changes on this field and go back to the session’s page">Discard and close</button></div>}
          </div>
          <div className="side-foot small">
            <button className={`btn small help-toggle${help ? ' chosen' : ''}`} aria-expanded={help} onClick={() => setHelp(!help)} title="How this page works">?</button>
            {!onFinished && <Link to="/tutorial">Tutorial</Link>}
            <span className="spacer" />
            <span className="muted">{tile}</span>
          </div>
          {help && (
            <div className="card side-block round-help">
              <ol className="small secondary round-steps">
                <li><b>Point at an object</b>: it lights up, its name shows beside it and its facts above. Press its type’s number (or a look-alike’s letter),
                  {' '}<span className="kbd">`</span> unsure (dotted amber), <span className="kbd">Del</span> reject (it turns dotted grey and stays, like
                  detection’s own rejects; a number key makes it solid again).</li>
                <li><b>Outline too big or small?</b> <span className="kbd">Tab</span> / <span className="kbd">⇧ Tab</span>. Still not right: click the object,
                  then click spots to add, Shift+click spots to leave out, or drag the dots; a click beyond the faint frame (or Enter) keeps it.</li>
                <li><b>A clean look at the tissue?</b> <span className="kbd">Space</span> hides everything drawn over the picture; <span className="kbd">Space</span> again brings it back.</li>
                <li><b>Two objects outlined as one?</b> <span className="kbd">S</span>, then click it: a split shows (click elsewhere on it to move it), <span className="kbd">Enter</span> keeps it.
                  Or draw a line across it. <b>One object in pieces?</b> <span className="kbd">M</span>, click or drag over the pieces, <span className="kbd">Enter</span>.</li>
                <li><b>Missed object?</b> {sam ? 'Click it: its outline appears as a new candidate (dotted); then press its type’s key.' : 'Click its corners one by one, then the first corner (or Enter).'}</li>
                <li><b>Nothing is saved until you press Save</b> (a small gold dot marks each object you changed). <b>Save and continue</b> when the field
                  looks right: what you left as it is counts as checked by you, and the next field opens.</li>
                <li><span className="look-swatch" style={{ borderColor: PALETTE.markers.others, background: PALETTE.markers.others, borderRadius: '50%' }} /> a small
                  blue dot: someone else checked it.</li>
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
                <label className="layer-toggle"><input type="checkbox" checked={layers.excluded !== false} onChange={() => setLayers({ ...layers, excluded: layers.excluded === false })} />
                  excluded areas (detection’s, hatched) and people’s areas (outlined)</label>
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
function Controls({ sam, precise, excluding, mode }: { sam: boolean; precise: boolean; excluding: boolean; mode: 'split' | 'merge' | null }) {
  if (excluding) {
    return (
      <ul className="key-list small">
        <li className="key-now"><b>Excluding</b> — outlines are locked</li>
        <li><span className="kbd">1</span>–<span className="kbd">4</span> the type: fold, artifact, no tissue, out of focus</li>
        <li><span className="kbd">drag</span> add to that type’s area</li>
        <li><span className="kbd">⇧ drag</span> take out of it (detection’s part too)</li>
        <li><span className="kbd">Tab</span> smaller · <span className="kbd">⇧ Tab</span> bigger brush · <span className="kbd">scroll</span> zoom · <span className="kbd">middle</span> move</li>
        <li><span className="kbd">Z</span> one stroke back · <span className="kbd">B</span> or <span className="kbd">Esc</span> done</li>
      </ul>
    )
  }
  if (mode) {
    return (
      <ul className="key-list small">
        <li className="key-now"><b>{mode === 'split' ? 'Split' : 'Merge'}</b> — the class keys still retype the object under the mouse</li>
        {mode === 'split' ? <>
          <li><span className="kbd">click</span> split an object · <span className="kbd">click</span> again on it to move the split</li>
          <li><span className="kbd">drag</span> a line: cut every object it crosses</li>
        </> : <li><span className="kbd">click</span> / <span className="kbd">drag</span> choose objects</li>}
        <li><span className="kbd">Enter</span> keep · <span className="kbd">Esc</span> drop / leave · <span className="kbd">Z</span> undo</li>
      </ul>
    )
  }
  if (precise) {
    return (
      <ul className="key-list small">
        <li className="key-now"><b>Precise editing</b> — only this object can change</li>
        {sam && <li><span className="kbd">click</span> add a spot to it</li>}
        {sam && <li><span className="kbd">⇧ click</span> leave a spot out</li>}
        <li><span className="kbd">drag</span> a dot to move the outline</li>
        <li><span className="kbd">Z</span> one step back</li>
        <li><span className="kbd">click beyond the frame</span> or <span className="kbd">Enter</span> keep the edit · <span className="kbd">Esc</span> cancel</li>
      </ul>
    )
  }
  return (
    <ul className="key-list small">
      <li><span className="kbd">Tab</span> switch its outline: Segment Anything / stain</li>
      <li><span className="kbd">Del</span> reject — dotted grey · <span className="kbd">`</span> unsure — dotted amber ?</li>
      <li><span className="kbd">click</span> empty tissue: {sam ? 'a new candidate' : 'draw a new one by hand'}</li>
      <li><span className="kbd">click</span> an object: precise edit{sam ? ' (click add, ⇧ click remove, drag dots, click beyond the frame done)' : ' (drag dots, click beyond the frame done)'}</li>
      <li><span className="kbd">Space</span> hide / show the drawings</li>
      <li><span className="kbd">S</span> split an object · <span className="kbd">M</span> merge objects</li>
      <li><span className="kbd">B</span> brush for bad areas (1–4 the type; ⇧ drag takes out): their objects are left out of training</li>
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
function TargetCard({ s, call, meta, precise, px, outline, choices, added, unsaved, onMask, why }: {
  s: Shown | undefined; call: { pick: ClassPick; by: string; who?: string[] } | null; meta: Meta; precise: Precise | null; px: number
  outline: number[][] | null; choices: { list: (number[][] | null)[]; index: number; names?: string[]; first?: number }; added: AddedState['state'] | null; unsaved: boolean
  onMask: (i: number) => void
  /** why it starts as it does when that is not the acceptance threshold alone (reasons.ts) */
  why?: string
}) {
  if (!s || !call) {
    return (
      <div className="card side-block round-target empty">
        <p className="small muted" style={{ margin: 0 }}>Point at an object: it lights up and its facts show here. Every key acts on it.</p>
      </div>
    )
  }
  const { vocabulary, subtypes: typesAsked } = meta
  const look = lookOf(vocabulary, meta.remove_class, call.pick.identity, call.pick.subtype)
  const c = s.candidate
  const title = s.fresh && !s.fresh.removed ? 'New candidate — press a class key' : callText(vocabulary, call.pick.identity, call.pick.subtype, typesAsked)
  const who = call.by === 'me' ? (unsaved ? 'your change, not saved yet' : 'your call') : call.by === 'others' ? `checked by ${checkedByText(call.who)}`
    : call.by === 'ai' ? 'detection’s call' : 'added now'
  return (
    <div className={`card side-block round-target look-${look.look}`} style={{ borderLeftColor: look.look === 'class' || look.look === 'unsure' ? look.colour : undefined }}>
      <div className="row target-head"><span className={`look-swatch look-${look.look}`} style={{ borderColor: look.colour, background: look.dash ? 'transparent' : look.colour }} />
        <b className="ellipsis">{title}</b></div>
      <dl className="target-facts small">
        <dt>Now</dt><dd>{who}</dd>
        {c && <><dt>Detection</dt><dd>{callText(vocabulary, c.ai_value, c.ai_subtype, typesAsked)}{c.decision !== 'accepted' && c.ai_value === meta.remove_class && <span className="muted">
          {' '}(rejected{c.reason?.startsWith('rule:') ? `: ${reasonText(c.reason)}` : ''})</span>}</dd>
          <dt>Score</dt><dd>{c.ai_probability != null ? fmt(c.ai_probability, 2) : '—'}</dd></>}
        {why && <><dt>Why</dt><dd className="target-why">{why}</dd></>}
        {outline && <><dt>Area</dt><dd>{fmt(polygonArea(outline) * px * px, 3)} µm²</dd></>}
        {choices.list.length > 0 && !precise && choices.names !== CHOICE_NAMES && choices.names !== PIECE_NAMES && <><dt>Outline</dt><dd>{maskName(choices.index, choices.list.length, choices.names)}</dd></>}
      </dl>
      {added && added !== 'proposed' && <p className={`small added-note${added === 'could_not_be_measured' ? ' warn' : ''}`}><b>Detection never proposed this object.</b> {ADDED_TEXT[added]}</p>}
      {precise ? <p className="small precise-bar"><b>Precise editing.</b> {precise.busy ? 'Outlining…' : 'A click adds a spot, Shift+click leaves one out, drag the dots; a click beyond the faint frame (or Enter) finishes.'}</p>
        : <MaskSwitch masks={choices.list} index={choices.index} onChange={onMask} names={choices.names} first={choices.first} />}
    </div>
  )
}
