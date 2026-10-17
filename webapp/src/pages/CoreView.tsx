import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import type { CoreDetail } from '../api'
import { apiUrl, coreThumb, enc } from '../api'
import { ErrorNote, GroupBadge, Loaded, Loading, PageHead, Progress, fmt } from '../components'
import type { Line, Mark, Pyramid, ViewTarget } from '../deepzoom'
import { DeepZoom } from '../deepzoom'
import { useFetch, useStored } from '../hooks'
import { useProject } from '../project'
import { Advanced } from '../setup/parts'
import { CLASS_STYLE, RASTER_LAYERS } from '../viewer'
import { INACTIVE_GREY, MODEL_CLASSES, UNSURE_AMBER, classColour } from '../classPalette'
import type { Vocabulary } from '../labels/classKeys'

interface CoreInfo extends CoreDetail {
  images: (CoreDetail['images'][number] & { pixel_width_um?: number })[]
  neighbours: { previous: string | null; next: string | null }
}
type PictureState = ({ ready: true } & Pyramid) | { ready: false; state: 'waiting' | 'preparing' | 'failed'; ahead?: number; error?: string; thumb_size: number | null }
/** What detection measured on an AT8 core's neuropil threads (all tiles). */
interface ThreadTotals { length_mm: number; skeleton_px: number }
interface CoreObjects {
  status: 'done' | 'partial' | 'not_run'; tiles_done: number; tiles_total: number; columns: string[]; rows: (string | number | null)[][]
  tiles: number[][]; outlines: boolean; threads: ThreadTotals | null
  /** "v2": the candidate store (every candidate with its call and outline); "current": the detection steps' outputs */
  source?: 'v2' | 'current'; unproposed?: number | null
}
/** Stained regions no v2 candidate covers: rows of [x, y, area µm², DAB / threshold, rings] in core-image pixels. */
/** The areas the analysis leaves out (detection's automatic layer, `scale` core px per layer px, and the tissue in no
 *  analysed tile): each kind with its colour, hatching and share of the core's area; the picture is excluded.png. */
interface ExcludedKind { id: string; label: string; color: string; hatch: string; fraction: number | null }
interface CoreExcluded { status: 'done' | 'not_made'; scale?: number; width?: number; height?: number; version?: string | number; kinds: ExcludedKind[] }
/** What people said against detection on this core (GET …/annotations, labels/compare.py: the comparison scoring and the
 *  Labels overview use). Labels train and evaluate models only: nothing here changes what the analysis counts. */
interface AnnotatorAnswer { person: string; identity: string; subtype: string; role: string; time: number }
interface FixedOutline { person: string; time: number; points: number[][] | null; mask: number | null }
interface AnnotatedObject {
  candidate_id: string; x_core?: number; y_core?: number; kind: 'agree' | 'changed' | 'rejected' | 'unsure' | 'outline'
  detection_identity?: string; detection_subtype?: string; identity?: string; subtype?: string; people?: AnnotatorAnswer[]; outline_fixed: FixedOutline[]
}
interface AddedObject { person: string; time: number; identity: string; subtype: string; x_core: number; y_core: number; points: number[][] | null }
/** People's painted (types "1"–"4") and cleared ("include") areas: rings in core px, the first outer, the rest holes. */
interface PeopleArea { kind: string; rings: number[][][] }
interface AreaAgreement { cells_people_excluded: number; cells_auto_excluded: number; cells_both: number; cleared_auto_cells: number; painted_new_cells: number; cells_analysed?: number; f1: number | null }
interface CoreAnnotations {
  objects: AnnotatedObject[]; added: AddedObject[]
  counts: { agree: number; changed: number; rejected: number; unsure: number; added: number; outline_fixed: number }
  areas?: { regions: PeopleArea[]; agreement: AreaAgreement | null; edits: { person: string; time: number; kinds: string[] }[] }
}
/** The review status of a core image's tiles (GET …/status, labels/status.py: the reader the Labels page uses too). */
type TileState = 'reviewed' | 'in_progress' | 'not_annotated'
interface StatusTile { tile_id: string; x: number; y: number; width: number; height: number; state: TileState; by: string[]; working: string[] }
interface CoreStatus {
  width: number | null; height: number | null; tiles: StatusTile[]; not_analysed: { x: number; y: number; width: number; height: number }[]
  counts: Record<TileState | 'not_analysed', number>; areas_checked_by: string[]; sessions: { name: string; title: string; round: number; tiles: string[] }[]
}
interface ViewRect { x0: number; y0: number; x1: number; y1: number }
interface CoreUnproposed { status: 'done' | 'not_run'; columns: string[]; rows: [number, number, number, number, number[][]][] }
/** Detection's outlines: per object (its exact x, y) the outlined area and polygons, in core-image pixels. The view
 *  always shows each object's own outline as the analysis counts it (the stain's default, `outline_default`); people's
 *  fixed outlines are only drawn while pointing at their object (the Annotations layer), never instead of it. */
interface CoreOutlines { status: 'done' | 'not_made'; columns: string[]; rows: [number, number, number | null, number[][]][] }
const typing = (target: EventTarget | null) => {
  const element = target as HTMLElement | null
  return !!element && (['INPUT', 'SELECT', 'TEXTAREA'].includes(element.tagName) || element.isContentEditable)
}
/** AT8 neuropil threads traced on the server (the first time, a few minutes; tiles done so far meanwhile): per piece its
 *  mean width (µm) and polyline, in core-image pixels. */
interface CoreThreads {
  status: 'done' | 'waiting' | 'tracing' | 'failed' | 'not_run' | 'not_started'; tiles_done: number; tiles_total: number
  pieces: number; length_mm: number; measured: ThreadTotals | null; rows: [number, number[]][]; error?: string; ahead?: number
}
interface Outline { area: number | null; rings: number[][] }
interface Found { index: number; x: number; y: number; r: number; cls: string; color: string; p: number | null; area: number | null; outline?: Outline; reason?: string; candidate?: string }

// exact outlines are drawn instead of circles wherever the Outline objects step has made them; every candidate is shown,
// rejected ones dotted grey (stored under a new key, so the choice of hiding them made before applies no more)
const LAYER_DEFAULTS = { objects: true, rejected: true, outlines: true, threads: false, unproposed: false, excluded: true, annotations: true, dab: false, tiles: false }
type Layers = typeof LAYER_DEFAULTS
/* The colours of the objects on the core picture: saturated, and clear of DAB brown and haematoxylin blue, each drawn
   over a thin dark halo (deepzoom.tsx). Rejected candidates are dotted light grey. The names stay those of CLASS_STYLE. */
const UNPROPOSED_COLOR = '#b98cff'  // stained areas no candidate covers: no label class
/** An object's colour: its label type's, the same on every page (classPalette.ts; the store's class, else detection's
 *  class name mapped onto the label classes and types: a compact plaque is the compact colour); rejected candidates
 *  dotted light grey. */
function objectColor(vocabulary: Vocabulary | undefined, cls: string, identity?: string): string {
  if (cls === 'rejected') return INACTIVE_GREY
  if (cls === 'unproposed') return UNPROPOSED_COLOR
  const mapped = MODEL_CLASSES[cls]
  const id = identity && vocabulary?.classes.some((c) => c.id === identity) ? identity : mapped?.[0]
  const type = mapped && mapped[0] === id ? mapped[1] || null : null
  return vocabulary && id ? classColour(vocabulary, id, type) : CLASS_STYLE[cls]?.color ?? '#ffffff'
}
/** A legend swatch drawn as the object is (dotted grey for rejected candidates). */
const swatchOf = (color: string, dotted = false) => (dotted ? { background: 'transparent', border: `2px dotted ${color}`, boxShadow: '0 0 0 1px rgb(0 0 0 / 35%)' }
  : { background: color, boxShadow: '0 0 0 1px rgb(0 0 0 / 35%)' })
const TILE_COLOR = 'rgb(30 30 30 / 55%)'
const THREAD_COLOR = '#00e5ff'  // cyan over a dark halo: the shared mid blue of RASTER_LAYERS.threads vanished on haematoxylin
const TRACING = ['waiting', 'tracing']
/** A thread's line on screen (px), by its measured width: thin, medium (1–2 µm) or thick (2 µm or more). */
const threadLine = (widthUm: number) => (widthUm < 1 ? 1.2 : widthUm < 2 ? 1.9 : 2.8)
/** Why detection kept or rejected a candidate (the v2 reason codes), in plain words. */
const REASON_TEXT: Record<string, string> = {
  accepted: 'kept by the model', rejected_by_model: 'the model judged it not an object', 'rule:speck': 'smaller than a plaque (a speck)',
  'rule:vascular_or_edge': 'on a vessel wall or tissue edge', 'rule:duplicate': 'a second detection of a nearby object',
  'rule:excluded': 'in an area left out of the analysis', 'rule:near_profile': 'a nucleus beside a stained neuron',
  'rule:pale_nucleus': 'a pale nucleus without stain', 'rule:shape': 'too small, too large or too long for a neuron',
  'rule:low_contrast': 'too faint against its surroundings', 'rule:small': 'too small for a tau+ neuron',
  'rule:artifact_shape': 'shaped like an artifact', 'rule:thread_like': 'lies on a thread',
  'rule:edge': 'a brown rim along a tissue edge or hole', 'rule:irregular': 'too irregular for a cell body (thread tangle)',
  'rule:fold': 'on a tissue fold',
}
const reasonText = (code: string) => REASON_TEXT[code] ?? code.replace(/^rule:/, '').replace(/_/g, ' ')
/** The colours of people's areas (the brush's, labels/brush.ts EXCLUDE_TYPES), and of a part they marked as fine. */
const AREA_COLOURS: Record<string, string> = { '1': '#ffb000', '2': '#a855f7', '3': '#9ca3af', '4': '#00e5ff', include: '#22c55e' }
const AREA_NAMES: Record<string, string> = { '1': 'fold or tissue damage', '2': 'artifact', '3': 'no tissue', '4': 'out of focus or bad staining', include: 'marked fine' }
/** The tile states of the minimap, with their fills. */
const TILE_STATES: { id: TileState | 'not_analysed'; label: string; fill: string }[] = [
  { id: 'reviewed', label: 'reviewed', fill: 'rgb(34 197 94 / 45%)' },
  { id: 'in_progress', label: 'in progress', fill: 'rgb(250 204 21 / 45%)' },
  { id: 'not_annotated', label: 'not annotated', fill: 'rgb(255 255 255 / 6%)' },
  { id: 'not_analysed', label: 'not analysed', fill: 'url(#minimap-hatch)' },
]
/** A class (and type) by its names in the stain's label classes. */
function className(vocabulary: Vocabulary | undefined, identity?: string, subtype?: string): string {
  const found = vocabulary?.classes.find((c) => c.id === identity)
  const sub = found?.subtypes.find((t) => t.id === subtype)
  return [sub?.name, found?.name ?? identity].filter(Boolean).join(' ') || '—'
}
const day = (t: number) => (t ? new Date(t * 1000).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' }) : '')
/** A flat ring (x0, y0, x1, y1, …) from [[x, y], …]. */
const flat = (ring: number[][]) => ring.flatMap(([x, y]) => [Math.round(x), Math.round(y)])
/** A "+" (or, turned, an "×") as a flat ring around (x, y), `arm` long each way. */
function plus(x: number, y: number, arm: number, turned = false): number[] {
  const t = arm / 3.5
  const shape = [[-t, -arm], [t, -arm], [t, -t], [arm, -t], [arm, t], [t, t], [t, arm], [-t, arm], [-t, t], [-arm, t], [-arm, -t], [-t, -t]]
  const c = turned ? Math.SQRT1_2 : 1, n = turned ? Math.SQRT1_2 : 0
  return shape.flatMap(([dx, dy]) => [x + dx * c - dy * n, y + dx * n + dy * c])
}

export default function CoreView() {
  const { coreId = '' } = useParams()
  const [params, setParams] = useSearchParams()
  const navigate = useNavigate()
  const { tmaName } = useProject()
  const core = useFetch<CoreInfo>(`/api/cores/${enc(coreId)}`)
  const info = core.data?.core_id === coreId ? core.data : null
  const stains = (info?.images ?? []).map((i) => i.stain)
  const asked = params.get('stain')
  const stain = asked && stains.includes(asked) ? asked : stains[0]
  const back = info ? `/cohort?tma=${enc(info.tma)}${stain ? `&stain=${enc(stain)}` : ''}` : '/cohort'
  const open = (id: string | null) => id && navigate(`/cores/${enc(id)}${stain ? `?stain=${enc(stain)}` : ''}`)
  const openRef = useRef(open)
  openRef.current = open
  const neighbours = info?.neighbours
  useEffect(() => {  // ← and →: the previous and the next core (the core list's order), the same stain and layers
    const onKey = (e: KeyboardEvent) => {
      if ((e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') || e.ctrlKey || e.metaKey || e.altKey || e.shiftKey || typing(e.target)) return
      const id = e.key === 'ArrowLeft' ? neighbours?.previous : neighbours?.next
      e.preventDefault()
      e.stopPropagation()  // before the picture's own arrow keys (they move it): ↑ and ↓ still do
      if (id) openRef.current(id)
    }
    window.addEventListener('keydown', onKey, { capture: true })
    return () => window.removeEventListener('keydown', onKey, { capture: true })
  }, [neighbours?.previous, neighbours?.next])

  return (
    <>
      <Link to={back} className="back-link">← All cores{info ? ` of ${tmaName(info.tma)}` : ''}</Link>
      <PageHead title={info ? `${tmaName(info.tma)} · ${info.core_label}` : coreId}
        subtitle={info ? <span className="row core-facts"><GroupBadge group={info.disease_group} />
          {info.donor_id ? <span>donor {info.donor_id}</span> : <span>control tissue</span>}
          {info.region && <span>{info.region}</span>}{info.technical_replicate && <span>replicate {info.technical_replicate}</span>}</span> : 'Loading the core…'}>
        {info && <span className="row">
          <button className="btn" disabled={!info.neighbours.previous} onClick={() => open(info.neighbours.previous)} title="The core before this one on the TMA (key ←)">← Previous core</button>
          <button className="btn" disabled={!info.neighbours.next} onClick={() => open(info.neighbours.next)} title="The core after this one on the TMA (key →)">Next core →</button>
        </span>}
      </PageHead>
      <Loaded state={core} label="Loading the core…">
        {() => !info ? <Loading label="Loading the core…" /> : !stain ? (
          <div className="empty-state">There is no picture of this core yet. Pictures of the cores are made by the <b>Cut out cores</b> step; see <Link to="/workflow">Run</Link>.</div>
        ) : (
          <>
            <div className="tabs" role="tablist" aria-label="Stain">
              {stains.map((s) => <button key={s} role="tab" aria-selected={s === stain} className={`tab${s === stain ? ' active' : ''}`}
                onClick={() => setParams({ stain: s }, { replace: true })}>{s}</button>)}
            </div>
            <CoreImage key={`${coreId}-${stain}`} core={info} stain={stain} start={params.get('x') && params.get('y') ? { x: Number(params.get('x')), y: Number(params.get('y')) } : null} />
          </>
        )}
      </Loaded>
    </>
  )
}

/** One stain of one core: the zoomable picture with the detections on top, and what is shown beside it. */
function CoreImage({ core, stain, start }: { core: CoreInfo; stain: string; start: { x: number; y: number } | null }) {
  const base = `/api/cores/${enc(core.core_id)}/${enc(stain)}`
  const navigate = useNavigate()
  const [poll, setPoll] = useState(true)
  const picture = useFetch<PictureState>(`${base}/picture`, poll ? 1500 : undefined)
  const found = useFetch<CoreObjects>(`${base}/objects`)
  const defaults = useFetch<{ vocabularies: Record<string, Vocabulary> }>('/api/labels/defaults')  // the label classes, for their colours
  const vocabulary = defaults.data?.vocabularies[stain]
  const [stored, setLayers] = useStored<Layers>('stainid.core.layers.v5', LAYER_DEFAULTS)
  const layers = { ...LAYER_DEFAULTS, ...stored }
  const outlinesMade = !!found.data?.outlines
  const [wantOutlines, setWantOutlines] = useState(false)  // fetched once asked for, kept when the layer is turned off
  useEffect(() => { if (layers.outlines) setWantOutlines(true) }, [layers.outlines])
  const shapes = useFetch<CoreOutlines>(wantOutlines && outlinesMade ? `${base}/outlines?outline=default` : null)  // detection's own outline of each
  const threadsMeasured = found.data?.threads ?? null
  const [threadPoll, setThreadPoll] = useState(true)
  const traced = useFetch<CoreThreads>(layers.threads && threadsMeasured ? `${base}/threads` : null, threadPoll ? 3000 : undefined)
  const tracing = TRACING.includes(traced.data?.status ?? 'waiting')
  const unproposedCount = found.data?.unproposed ?? null
  const stainedOnly = useFetch<CoreUnproposed>(layers.unproposed && unproposedCount !== null ? `${base}/unproposed` : null)
  const excluded = useFetch<CoreExcluded>(layers.excluded ? `${base}/excluded` : null)
  const annotations = useFetch<CoreAnnotations>(layers.annotations ? `${base}/annotations` : null)
  const status = useFetch<CoreStatus>(`${base}/status`, 15000)  // the minimap: follows people's submits
  useEffect(() => {  // and at once when the window comes back (someone may have submitted meanwhile)
    const again = () => status.reload()
    window.addEventListener('focus', again)
    return () => window.removeEventListener('focus', again)
  }, [status.reload])  // eslint-disable-line react-hooks/exhaustive-deps
  const [hidden, setHidden] = useState(false)  // Space: a clean look at the tissue
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== ' ' || e.ctrlKey || e.metaKey || e.altKey || typing(e.target)) return
      if ((e.target as HTMLElement | null)?.tagName === 'BUTTON') return  // Space presses a focused button
      e.preventDefault()
      setHidden((h) => !h)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
  const excludedData = excluded.data
  const overlays = useMemo(() => (!hidden && layers.excluded && excludedData?.status === 'done' && excludedData.width && excludedData.height && excludedData.scale
    ? [{ src: apiUrl(`${base}/excluded.png?v=${excludedData.version ?? 0}`), width: excludedData.width * excludedData.scale,
      height: excludedData.height * excludedData.scale }] : []), [hidden, layers.excluded, excludedData, base])
  useEffect(() => { setThreadPoll(tracing) }, [tracing])
  const [selected, setSelected] = useState<number | null>(null)
  const [target, setTarget] = useState<ViewTarget | null>(start ? { ...start, scale: 0.5, key: 0 } : null)
  const [viewRect, setViewRect] = useState<ViewRect | null>(null)
  const image = core.images.find((i) => i.stain === stain)
  const ready = picture.data?.ready ? picture.data : null
  const settled = !!ready || (picture.data && !picture.data.ready && picture.data.state === 'failed')
  useEffect(() => { if (settled) setPoll(false) }, [settled, picture.data])  // also after a tile went missing (onTileError)
  const layout = useRef<HTMLDivElement>(null)
  const fit = useFitWindow(layout, !!ready)

  const objects = useMemo<Found[]>(() => {
    const data = found.data
    if (!data) return []
    const col = Object.fromEntries(data.columns.map((c, i) => [c, i]))
    const outlines = new Map((shapes.data?.rows ?? []).map(([x, y, area, rings]) => [`${x},${y}`, { area, rings }]))
    return data.rows.map((r, index) => ({ index, x: Number(r[col.x]), y: Number(r[col.y]), r: Number(r[col.radius_px] ?? 6), cls: String(r[col.model_class]),
      color: objectColor(vocabulary, String(r[col.model_class]), col.class === undefined ? undefined : String(r[col.class] ?? '')),
      p: r[col.probability] === null ? null : Number(r[col.probability]), area: r[col.area_um2] === null ? null : Number(r[col.area_um2]),
      outline: outlines.get(`${r[col.x]},${r[col.y]}`), reason: col.reason === undefined ? undefined : String(r[col.reason] ?? ''),
      candidate: col.candidate_id === undefined ? undefined : String(r[col.candidate_id] ?? '') }))
  }, [found.data, shapes.data, vocabulary])
  const colorOfClass = useMemo(() => Object.fromEntries(objects.map((o) => [o.cls, o.color])), [objects])
  const counts = useMemo(() => objects.reduce<Record<string, number>>((acc, o) => ({ ...acc, [o.cls]: (acc[o.cls] ?? 0) + 1 }), {}), [objects])
  const classes = Object.keys(counts).filter((c) => c !== 'rejected').sort()
  const showOutlines = layers.outlines && outlinesMade && !!shapes.data
  const drawn = useMemo(() => objects.filter((o) => (layers.rejected || o.cls !== 'rejected') && (layers.objects || (showOutlines && o.outline))),
    [objects, layers.objects, layers.rejected, showOutlines])
  // an object with an exact outline shown is drawn by its outline instead of the circle
  const marks = useMemo<Mark[]>(() => drawn.map((o) => {
    const outlined = showOutlines && !!o.outline
    return { x: o.x, y: o.y, r: o.r, color: o.color, dashed: o.cls === 'rejected', rings: outlined ? o.outline!.rings : undefined, circle: !outlined }
  }), [drawn, showOutlines])
  const regions = useMemo<Mark[]>(() => (layers.unproposed && stainedOnly.data ? stainedOnly.data.rows.map(([x, y, , , rings]) =>
    ({ x, y, color: UNPROPOSED_COLOR, rings, circle: false })) : []), [layers.unproposed, stainedOnly.data])

  // the Annotations layer: a small badge where people differ from detection, the objects they added (dashed) and their areas
  const notes = annotations.data
  const byCandidate = useMemo(() => new Map((notes?.objects ?? []).map((o) => [o.candidate_id, o])), [notes])
  const objectOf = useMemo(() => new Map(objects.filter((o) => o.candidate).map((o) => [o.candidate!, o])), [objects])
  const annotated = useMemo(() => {
    if (!layers.annotations || !notes) return { marks: [] as Mark[], about: [] as AnnotationMark[] }
    const out: Mark[] = []
    const about: AnnotationMark[] = []
    for (const a of notes.objects) {
      if (a.kind === 'agree') continue
      const o = objectOf.get(a.candidate_id)
      const x = o?.x ?? a.x_core, y = o?.y ?? a.y_core
      if (x === undefined || y === undefined) continue
      const r = o?.r ?? 8
      const rejected = a.kind === 'rejected', unsure = a.kind === 'unsure'
      const color = rejected ? INACTIVE_GREY : unsure ? UNSURE_AMBER : a.kind === 'outline'
        ? classColour(vocabulary, a.detection_identity ?? o?.cls, a.detection_subtype || null) : classColour(vocabulary, a.identity, a.subtype || null)
      out.push({ x: x + r * 0.8, y: y - r * 0.8, r: Math.max(2.5, r * 0.3), color, dashed: rejected || unsure, circle: true })
      about.push({ object: a, at: o })
      if (rejected) {  // crossed out: not an object
        out.push({ x, y, color, rings: [plus(x, y, r * 0.75, true)], circle: false })
        about.push({ object: a, at: o })
      }
    }
    const typical = objects.length ? [...objects].sort((a, b) => a.r - b.r)[Math.floor(objects.length / 2)].r : 10
    for (const a of notes.added) {
      const color = classColour(vocabulary, a.identity, a.subtype || null)
      const ring = a.points && a.points.length >= 3 ? [flat(a.points)] : undefined
      const r = ring ? Math.max(...a.points!.map(([x, y]) => Math.hypot(x - a.x_core, y - a.y_core))) : typical
      out.push({ x: a.x_core, y: a.y_core, r, color, dashed: true, rings: ring, circle: !ring })
      const at = r * 0.75, arm = Math.max(3, Math.min(r * 0.25, typical * 0.3))  // its small "+", at the upper right
      out.push({ x: a.x_core + at, y: a.y_core - at, color, rings: [plus(a.x_core + at, a.y_core - at, arm)], circle: false })
      about.push({ added: a }, { added: a })
    }
    for (const area of notes.areas?.regions ?? []) {
      if (!area.rings.length || !area.rings[0].length) continue
      out.push({ x: area.rings[0][0][0], y: area.rings[0][0][1], color: AREA_COLOURS[area.kind] ?? '#ffffff', dashed: area.kind === 'include',
        rings: area.rings.map(flat), circle: false })
      about.push({ area })
    }
    return { marks: out, about }
  }, [layers.annotations, notes, objectOf, objects, vocabulary])
  const shown = useMemo(() => (hidden ? [] : [...marks, ...regions, ...annotated.marks]), [hidden, marks, regions, annotated.marks])
  const lines = useMemo<Line[]>(() => (!hidden && layers.threads && traced.data ? traced.data.rows.map(([width, points]) => ({ points, width: threadLine(width) })) : []),
    [hidden, layers.threads, traced.data])
  const boxes = useMemo<Mark[]>(() => (!hidden && layers.tiles ? (found.data?.tiles ?? []).map(([x, y, w, h]) => ({ x, y, w, h, color: TILE_COLOR, dashed: true })) : []),
    [hidden, layers.tiles, found.data])
  const toggle = (key: keyof Layers) => setLayers({ ...layers, [key]: !layers[key] })
  const pick = drawn.find((o) => o.index === selected) ?? null

  // pointing at an object with an outline someone fixed: their outline, thin and dashed, beside detection's (never instead)
  const [fixed, setFixed] = useState<{ points: number[][]; color: string } | null>(null)
  const peopleLines = (a: AnnotatedObject) => [
    `Detection: ${className(vocabulary, a.detection_identity, a.detection_subtype)}`,
    ...(a.people ?? []).map((p) => `Annotator (${p.person}, ${day(p.time)}): ${p.role === 'skip' ? 'unsure' : className(vocabulary, p.identity, p.subtype)}`),
    ...a.outline_fixed.map((f) => `Outline fixed by ${f.person} (${day(f.time)})${f.mask !== null ? `: chose mask ${f.mask}` : ''}`),
  ]
  const describe = (i: number) => {
    const o = drawn[i]
    const pointing = (a: AnnotatedObject | undefined) => {
      const outline = a?.outline_fixed.find((f) => f.points && f.points.length >= 3)
      setFixed(outline ? { points: outline.points!, color: o?.color ?? '#ffffff' } : null)
    }
    if (o) {
      const a = layers.annotations && o.candidate ? byCandidate.get(o.candidate) : undefined
      pointing(a)
      const own = [CLASS_STYLE[o.cls]?.label ?? o.cls, o.p === null ? null : `${Math.round(o.p * 100)}% sure`, o.area === null ? null : `${area(o.area)} µm²`,
        showOutlines && o.outline?.area != null ? `outlined ${area(o.outline.area)} µm²` : null].filter(Boolean).join(' · ')
      return a && a.kind !== 'agree' ? [own, ...peopleLines(a)].join('\n') : own
    }
    const k = i - drawn.length
    if (k < regions.length) { setFixed(null); return `${CLASS_STYLE.unproposed.label} · ${area(stainedOnly.data?.rows[k]?.[2] ?? 0)} µm²` }
    const about = annotated.about[k - regions.length]
    if (about?.object) { pointing(about.object); return peopleLines(about.object).join('\n') }
    setFixed(null)
    if (about?.added) return `+ Added by ${about.added.person} (${day(about.added.time)}): ${className(vocabulary, about.added.identity, about.added.subtype)} — detection missed it`
    if (about?.area) return `People’s area: ${AREA_NAMES[about.area.kind] ?? about.area.kind}`
    return ''
  }
  const go = (x: number, y: number, scale?: number) => setTarget((t) => ({ x, y, scale, key: (t?.key ?? 0) + 1 }))
  const tileUrl = ready ? (level: number, col: number, row: number, layer?: string) =>
    apiUrl(`/api/images/cores/${enc(core.core_id)}/${enc(stain)}/tiles/${level}/${col}_${row}${layer ? `_${layer}.png` : '.jpg'}?v=${ready.version}`) : null
  const pictureBox = useRef<HTMLDivElement>(null)
  const screenScale = () => {  // screen px per image px now
    const element = pictureBox.current?.querySelector('.dz')
    return element && viewRect ? element.clientWidth / Math.max(1e-6, viewRect.x1 - viewRect.x0) : undefined
  }
  const openTile = (tile: StatusTile) => {
    const session = status.data?.sessions.find((s) => s.tiles.includes(tile.tile_id))
    if (session) navigate(session.round === 2 ? `/annotate/${enc(session.name)}/${enc(tile.tile_id)}` : `/annotate/${enc(session.name)}`)
  }

  return (
    <div ref={layout} className="core-layout" style={fit}>
      <div ref={pictureBox} className="card core-picture" onPointerDown={() => setFixed(null)} onPointerLeave={() => setFixed(null)}>
        {ready && tileUrl ? (
          <DeepZoom pyramid={ready} tileUrl={tileUrl} layers={!hidden && layers.dab ? ['dab'] : []} marks={shown} lines={lines} lineColor={THREAD_COLOR}
            boxes={boxes} overlays={overlays} pixelUm={image?.pixel_width_um} height="auto"
            selected={!hidden && pick ? drawn.indexOf(pick) : null} describe={describe} target={target} onView={setViewRect}
            onPick={(i) => {
              if (i !== null && i >= drawn.length) {  // a badge: its object
                const at = annotated.about[i - drawn.length - regions.length]?.at
                setSelected(at ? at.index : null)
                return
              }
              setSelected(i === null ? null : drawn[i].index)
            }} onTileError={() => setPoll(true)}
            toolbar={<span className="row dz-toolbar-info"><span className="small muted">Scroll to zoom · drag to move · click an object for details ·
              {' '}<span className="kbd">Space</span> hides what is drawn · <span className="kbd">←</span> <span className="kbd">→</span> other cores</span>
              {hidden && <span className="badge">Overlays hidden · <span className="kbd">Space</span></span>}</span>} />
        ) : (
          <Preparing core={core} stain={stain} state={picture.data} error={picture.error} />
        )}
        {!hidden && fixed && viewRect && <FixedOutlineHint box={pictureBox.current} view={viewRect} points={fixed.points} color={fixed.color} />}
      </div>
      <div className="core-side">
        <Minimap core={core.core_id} stain={stain} status={status} view={viewRect}
          onJump={(x, y) => go(x, y, screenScale())} onOpen={openTile} />
        <div className="card">
          <h3>On the picture</h3>
          <Detections status={found.data} error={found.error} stain={stain} tma={core.tma} tissue={image?.tissue_status} />
          {found.data && found.data.status !== 'not_run' && (
            <>
              <label className="layer-toggle"><input type="checkbox" checked={layers.objects} onChange={() => toggle('objects')} />
                <b>What detection found</b><span className="spacer" /><span className="muted">{(objects.length - (counts.rejected ?? 0)).toLocaleString()}</span></label>
              {classes.map((c) => (
                <div key={c} className="layer-toggle indent">
                  <span className="swatch" style={swatchOf(colorOfClass[c] ?? objectColor(vocabulary, c))} />{CLASS_STYLE[c]?.label ?? c}<span className="spacer" /><span className="muted">{counts[c].toLocaleString()}</span>
                </div>
              ))}
              {!!counts.rejected && (
                <label className="layer-toggle indent" title="Candidates detection looked at and decided were not real objects (drawn dotted grey)">
                  <input type="checkbox" checked={layers.rejected} disabled={!layers.objects} onChange={() => toggle('rejected')} />
                  <span className="swatch" style={swatchOf(INACTIVE_GREY, true)} />also show rejected candidates<span className="spacer" /><span className="muted">{counts.rejected.toLocaleString()}</span>
                </label>
              )}
              <OutlinesToggle made={outlinesMade} on={layers.outlines} onToggle={() => toggle('outlines')} state={shapes}
                count={drawn.filter((o) => o.outline).length} v2={found.data.source === 'v2'} />
              {stain === 'AT8' && <ThreadsToggle measured={threadsMeasured} on={layers.threads} onToggle={() => toggle('threads')} state={traced} />}
              {unproposedCount !== null && <UnproposedToggle count={unproposedCount} on={layers.unproposed} onToggle={() => toggle('unproposed')} state={stainedOnly} />}
            </>
          )}
          <ExcludedToggle on={layers.excluded} onToggle={() => toggle('excluded')} state={excluded} />
          {found.data?.source === 'v2' && <AnnotationsToggle on={layers.annotations} onToggle={() => toggle('annotations')} state={annotations} />}
          <label className="layer-toggle"><input type="checkbox" checked={layers.dab} onChange={() => toggle('dab')} />
            <span className="swatch" style={{ background: RASTER_LAYERS.dab.color }} />Stained area (brown above this slide’s threshold)</label>
          {(found.data?.tiles.length ?? 0) > 0 && <label className="layer-toggle"><input type="checkbox" checked={layers.tiles} onChange={() => toggle('tiles')} />
            <span className="swatch outline dashed" />Tile grid ({found.data!.tiles.length} tiles analysed)</label>}
        </div>
        {pick && (
          <div className="card">
            <h3>Selected object</h3>
            <p className="row" style={{ margin: '0 0 6px' }}><span className="swatch" style={swatchOf(pick.color, pick.cls === 'rejected')} /><b>{CLASS_STYLE[pick.cls]?.label ?? pick.cls}</b></p>
            <dl className="facts">
              {pick.p !== null && <><dt>Detection score</dt><dd>{Math.round(pick.p * 100)}%</dd></>}
              {pick.reason && <><dt>Why</dt><dd>{reasonText(pick.reason)}</dd></>}
              {pick.area !== null && <><dt>Area</dt><dd>{area(pick.area)} µm²</dd></>}
              {showOutlines && pick.outline?.area != null && <><dt title="The area inside detection's outline">Outlined area</dt><dd>{area(pick.outline.area)} µm²</dd></>}
              <dt>Position</dt><dd>{Math.round(pick.x).toLocaleString()}, {Math.round(pick.y).toLocaleString()} px</dd>
            </dl>
            {(() => {
              const a = pick.candidate ? byCandidate.get(pick.candidate) : undefined
              return a && layers.annotations ? <p className="small secondary" style={{ margin: '0 0 6px', whiteSpace: 'pre-line' }}>{peopleLines(a).slice(1).join('\n')}</p> : null
            })()}
            <div className="row"><button className="btn small" onClick={() => go(pick.x, pick.y, 1)}>Zoom to it</button>
              <button className="btn small" onClick={() => setSelected(null)}>Clear</button></div>
          </div>
        )}
        <Advanced title="Details">
          <dl className="facts">
            <dt>Core</dt><dd><code>{core.core_id}</code></dd>
            {image && <><dt>Image size</dt><dd>{image.native_width_px.toLocaleString()} × {image.native_height_px.toLocaleString()} px</dd></>}
            {image?.pixel_width_um && <><dt>Pixel size</dt><dd>{fmt(image.pixel_width_um, 3)} µm</dd></>}
            {image?.tissue_fraction !== undefined && <><dt>Tissue</dt><dd>{Math.round(image.tissue_fraction * 100)}% of the core{image.tissue_status ? ` (${image.tissue_status})` : ''}</dd></>}
            {core.sample_region_id && <><dt>Sample</dt><dd><code>{core.sample_region_id}</code></dd></>}
            {found.data && <><dt>Detection</dt><dd>{found.data.tiles_done} of {found.data.tiles_total} tiles</dd></>}
          </dl>
          {drawn.length > 0 && <ObjectList objects={drawn} selected={selected} onPick={(o) => { setSelected(o.index); go(o.x, o.y, 1) }} />}
        </Advanced>
      </div>
    </div>
  )
}

/** What a mark of the Annotations layer is about (for its tooltip and for clicking it). */
interface AnnotationMark { object?: AnnotatedObject; at?: Found; added?: AddedObject; area?: PeopleArea }

/** The page fits the window (no scrolling): the square picture takes the height left below the page's head, the side
 *  panel scrolls within the same height. Measured, so the head's height (wrapping, a note) does not matter; nothing on a
 *  narrow window, where the two stack. */
function useFitWindow(layout: React.RefObject<HTMLDivElement | null>, ready: boolean): React.CSSProperties | undefined {
  const [fit, setFit] = useState<{ square: number; side: number } | null>(null)
  const measure = useCallback(() => {
    const element = layout.current
    if (!element || window.innerWidth <= 1180) { setFit(null); return }
    const top = element.getBoundingClientRect().top + window.scrollY
    const below = 16  // room under the layout
    const height = Math.max(320, window.innerHeight - top - below)
    const card = element.querySelector('.core-picture') as HTMLElement | null
    const square = element.querySelector('.core-picture .dz, .core-picture .dz-preparing') as HTMLElement | null
    const chrome = card && square ? card.offsetHeight - square.offsetHeight : 70
    const next = { square: Math.max(280, Math.floor(height - chrome)), side: Math.floor(height) }
    setFit((old) => (old && old.square === next.square && old.side === next.side ? old : next))
  }, [layout])
  useLayoutEffect(() => {
    measure()
    window.addEventListener('resize', measure)
    return () => window.removeEventListener('resize', measure)
  }, [measure, ready])
  useLayoutEffect(() => {  // what the page still sticks out by (margins below the layout): taken off once more
    if (!fit) return
    const over = (document.scrollingElement?.scrollHeight ?? 0) - window.innerHeight
    if (over > 0) setFit({ square: Math.max(280, fit.square - over), side: Math.max(280, fit.side - over) })
  }, [fit])
  return fit ? ({ '--core-square': `${fit.square}px`, '--core-side': `${fit.side}px` } as React.CSSProperties) : undefined
}

/** An outline someone fixed, thin and dashed over the picture while pointing at its object (screen space, from the view). */
function FixedOutlineHint({ box, view, points, color }: { box: HTMLDivElement | null; view: ViewRect; points: number[][]; color: string }) {
  const dz = box?.querySelector('.dz') as HTMLElement | null
  if (!box || !dz) return null
  const outer = box.getBoundingClientRect(), inner = dz.getBoundingClientRect()
  const k = inner.width / Math.max(1e-6, view.x1 - view.x0)
  const path = points.map(([x, y]) => `${((x + 0.5 - view.x0) * k).toFixed(1)},${((y + 0.5 - view.y0) * k).toFixed(1)}`).join(' ')
  return (
    <svg className="core-fixed-outline" style={{ left: inner.left - outer.left, top: inner.top - outer.top, width: inner.width, height: inner.height }} aria-hidden="true">
      <polygon points={path} fill="none" stroke="rgb(0 0 0 / 60%)" strokeWidth={2.4} strokeDasharray="5 4" />
      <polygon points={path} fill="none" stroke={color} strokeWidth={1.2} strokeDasharray="5 4" />
    </svg>
  )
}

/** The minimap: the whole core with its tiles shaded by how far their review is (labels/status.py), the squares no
 *  analysed tile covers hatched, and the main picture's view as a rectangle; a click (or a drag) moves the view there, a
 *  double-click on a tile opens it in the annotation session that has it. Not hidden by Space. */
function Minimap({ core, stain, status, view, onJump, onOpen }: {
  core: string; stain: string; status: { data: CoreStatus | null; error: string | null }; view: ViewRect | null
  onJump: (x: number, y: number) => void; onOpen: (tile: StatusTile) => void
}) {
  const data = status.data
  const svg = useRef<SVGSVGElement>(null)
  const dragging = useRef(false)
  if (!data || !data.width || !data.height) {
    return <div className="card minimap-card">{status.error ? <ErrorNote error={status.error} /> : <Loading inline label="Loading the review status…" />}</div>
  }
  const { width, height } = data
  const at = (e: React.PointerEvent | React.MouseEvent) => {
    const rect = svg.current!.getBoundingClientRect()
    return { x: ((e.clientX - rect.left) / rect.width) * width, y: ((e.clientY - rect.top) / rect.height) * height }
  }
  const tileAt = (x: number, y: number) => data.tiles.find((t) => x >= t.x && x < t.x + t.width && y >= t.y && y < t.y + t.height)
  const total = data.tiles.length
  const line = (width / 256) * 1.5
  return (
    <div className="card minimap-card">
      <svg ref={svg} className="minimap" viewBox={`0 0 ${width} ${height}`} style={{ aspectRatio: `${width} / ${height}` }} role="img" aria-label="The whole core: tiles by review status and the part in view"
        onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); dragging.current = true; const p = at(e); onJump(p.x, p.y) }}
        onPointerMove={(e) => { if (dragging.current) { const p = at(e); onJump(p.x, p.y) } }}
        onPointerUp={() => { dragging.current = false }}
        onDoubleClick={(e) => { const p = at(e); const tile = tileAt(p.x, p.y); if (tile) onOpen(tile) }}>
        <defs>
          <pattern id="minimap-hatch" width={width / 40} height={width / 40} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width={width / 40} height={width / 40} fill="rgb(100 116 139 / 25%)" />
            <line x1={0} y1={0} x2={0} y2={width / 40} stroke="rgb(100 116 139 / 85%)" strokeWidth={width / 160} />
          </pattern>
        </defs>
        <image href={apiUrl(`/api/images/cores/${enc(core)}/${enc(stain)}.jpg?size=256`)} x={0} y={0} width={width} height={height} preserveAspectRatio="none" />
        {data.not_analysed.map((t, i) => <rect key={`n${i}`} x={t.x} y={t.y} width={t.width} height={t.height} fill="url(#minimap-hatch)" />)}
        {data.tiles.map((t) => (
          <rect key={t.tile_id} x={t.x} y={t.y} width={t.width} height={t.height} fill={TILE_STATES.find((s) => s.id === t.state)!.fill}
            stroke="rgb(0 0 0 / 35%)" strokeWidth={line / 2}>
            <title>{`${t.tile_id}: ${t.state.replace('_', ' ')}${t.by.length ? ` by ${t.by.join(', ')}` : ''}${t.working.length ? ` (on it: ${t.working.join(', ')})` : ''}`}</title>
          </rect>
        ))}
        {view && <rect x={Math.max(0, view.x0)} y={Math.max(0, view.y0)} width={Math.max(0, Math.min(width, view.x1) - Math.max(0, view.x0))}
          height={Math.max(0, Math.min(height, view.y1) - Math.max(0, view.y0))} fill="none" stroke="#ffffff" strokeWidth={line * 1.4}
          style={{ filter: 'drop-shadow(0 0 1.5px rgb(0 0 0 / 90%))' }} />}
      </svg>
      <ul className="minimap-legend small">
        <li><b>{data.counts.reviewed} of {total} tiles reviewed</b></li>
        {TILE_STATES.map((s) => (
          <li key={s.id}><span className="swatch minimap-swatch" style={s.id === 'not_analysed' ? { background: 'repeating-linear-gradient(45deg, rgb(100 116 139 / 85%) 0 1.5px, rgb(100 116 139 / 25%) 1.5px 4px)' } : { background: s.fill }} />
            {s.label}<span className="spacer" /><span className="muted">{data.counts[s.id] ?? 0}</span></li>
        ))}
      </ul>
      <p className="small secondary minimap-areas">{data.areas_checked_by.length ? <>Bad areas checked by: {data.areas_checked_by.join(', ')}</> : 'Bad areas: not checked yet'}</p>
      <p className="small muted" style={{ margin: 0 }}>Click to move there · double-click a tile to open it in annotation</p>
    </div>
  )
}

/** The "Annotations" layer: where people's labels differ from detection (a small badge in the class they gave; grey
 *  dashed: not an object; white dashed: unsure), the objects they added that detection missed (dashed), their bad
 *  areas (outlines by type; dashed green: marked fine), and how they compare. Labels train and evaluate models only. */
function AnnotationsToggle({ on, onToggle, state }: { on: boolean; onToggle: () => void; state: { data: CoreAnnotations | null; error: string | null; reload: () => void } }) {
  const data = state.data
  const c = data?.counts
  const agreement = data?.areas?.agreement
  const cells = agreement ? Math.max(1, agreement.cells_analysed ?? agreement.cells_auto_excluded) : 1  // shares of the analysed area
  const rows: [string, number | undefined, React.CSSProperties][] = c ? [
    ['agree with detection', c.agree, { background: 'transparent', border: '1px solid var(--border)' }],
    ['changed class', c.changed, swatchOf('#ffb000')],
    ['not an object (rejected)', c.rejected, swatchOf(INACTIVE_GREY, true)],
    ['unsure', c.unsure, swatchOf('#ffffff', true)],
    ['added (detection missed it)', c.added, swatchOf('#ffffff', true)],
    ['outline fixed (shown when pointing at it)', c.outline_fixed, { background: 'transparent', border: '1.5px dashed var(--text-muted)' }],
  ] : []
  return (
    <>
      <label className="layer-toggle" title="What annotators said about these objects, against detection; labels train and evaluate the models and never change what the analysis counts">
        <input type="checkbox" checked={on} onChange={onToggle} />
        <span className="swatch" style={{ background: 'transparent', border: '2px solid var(--text-muted)', borderRadius: '50%' }} />Annotations</label>
      {on && !data && !state.error && <div className="layer-toggle indent"><Loading inline label="Loading the annotations…" /></div>}
      {on && data && rows.filter(([, n]) => n).map(([label, n, style]) => (
        <div key={label} className="layer-toggle indent"><span className="swatch" style={style} />{label}<span className="spacer" /><span className="muted">{n!.toLocaleString()}</span></div>
      ))}
      {on && data && !rows.some(([, n]) => n) && <p className="layer-note small">Nobody has labelled objects of this core yet.</p>}
      {on && agreement && (data?.areas?.regions.length ?? 0) > 0 && (
        <div className="layer-toggle indent wrap" title="People's painted and cleared areas against detection's automatic excluded areas (a check on the fold and artifact finders; the analysis uses detection's)">
          <span className="swatch" style={{ background: 'transparent', border: `2px solid ${AREA_COLOURS['1']}` }} />
          people’s areas: {agreement.f1 == null ? '' : `F1 ${agreement.f1.toFixed(2)} with detection’s; `}
          {(100 * agreement.cleared_auto_cells / cells).toFixed(1)}% cleared, {(100 * agreement.painted_new_cells / cells).toFixed(1)}% added
        </div>
      )}
      {on && state.error && <div className="layer-toggle indent wrap"><ErrorNote error={state.error} />
        <button className="btn small" onClick={state.reload}>Try again</button></div>}
    </>
  )
}

/** While the server prepares the zoomable picture (first view of a core only): the small picture, dimmed, and why. */
function Preparing({ core, stain, state, error }: { core: CoreInfo; stain: string; state: PictureState | null; error: string | null }) {
  const failed = error ?? (state && !state.ready && state.state === 'failed' ? state.error ?? 'unknown error' : null)
  const ahead = state && !state.ready ? state.ahead ?? 0 : 0
  return (
    <div className="dz-preparing">
      {state && !state.ready && state.thumb_size && <img src={coreThumb(core.core_id, stain, state.thumb_size)} alt="" />}
      <div className="dz-preparing-note">
        {failed ? <><b>This picture could not be prepared.</b><ErrorNote error={failed} /></>
          : !state ? <Loading label="Loading the picture…" />
          : <Loading label={<span><b>Preparing this core for viewing…</b><br />
              <span className="small">{ahead ? `${ahead} other picture${ahead === 1 ? '' : 's'} first, then this one. ` : ''}The full scan is read once (about 10 s); after that it opens instantly.</span></span>} />}
      </div>
    </div>
  )
}

/** The "Exact outlines" layer: on wherever the outlines are made (drawn instead of circles); where they are not, why not. */
function OutlinesToggle({ made, on, onToggle, state, count, v2 = false }: {
  made: boolean; on: boolean; onToggle: () => void; state: { data: CoreOutlines | null; error: string | null; reload: () => void }; count: number
  v2?: boolean
}) {
  const ready = made && state.data?.status !== 'not_made'
  return (
    <>
      <label className="layer-toggle" title={v2 ? 'The exact shape of each object: its Segment Anything outline where the outline step has made one, else the outline detection found it with'
        : 'The exact shape of each object, drawn by Segment Anything (the optional Outline objects step)'}>
        <input type="checkbox" checked={on && ready} disabled={!ready} onChange={onToggle} />
        <span className="swatch outline" />{v2 ? 'Exact outlines' : 'Exact outlines (Segment Anything)'}<span className="spacer" />
        {on && ready && state.data && <span className="muted">{count.toLocaleString()}</span>}
      </label>
      {!ready && <p className="layer-note small">Not made for this core yet, so each object is marked with a circle. Exact outlines come from
        the optional <Link to="/workflow#masks">Outline objects</Link> step, which has not reached this core.</p>}
      {on && ready && !state.data && !state.error && <div className="layer-toggle indent"><Loading inline label="Loading the outlines…" /></div>}
      {on && ready && state.error && <div className="layer-toggle indent wrap"><ErrorNote error={state.error} />
        <button className="btn small" onClick={state.reload}>Try again</button></div>}
    </>
  )
}

/** The AT8 "Neuropil threads" layer: what detection measured, and (when turned on) the threads traced on the picture. */
function ThreadsToggle({ measured, on, onToggle, state }: {
  measured: ThreadTotals | null; on: boolean; onToggle: () => void; state: { data: CoreThreads | null; error: string | null; reload: () => void }
}) {
  const data = state.data
  const tracing = !!data && TRACING.includes(data.status)
  return (
    <>
      <label className="layer-toggle" title="Thin tau-positive threads in the neuropil, as detection traced them; line thickness shows each thread’s width">
        <input type="checkbox" checked={on && !!measured} disabled={!measured} onChange={onToggle} />
        <span className="swatch" style={{ background: THREAD_COLOR }} />Neuropil threads<span className="spacer" />
        {measured && <span className="muted" title="Total length of the threads detection measured in this core">{lengthMm(measured.length_mm)}</span>}
      </label>
      {!measured && <p className="layer-note small">Threads are measured when detection finds tau on this core; it has not finished this core yet.</p>}
      {on && measured && !data && !state.error && <div className="layer-toggle indent"><Loading inline label="Loading the threads…" /></div>}
      {on && tracing && (
        <div className="layer-note small">
          {data!.status === 'waiting' ? 'Waiting for another core’s threads to finish, then tracing this one… '
            : `Tracing the threads on this core (first time only, a few minutes)… ${data!.tiles_done} of ${data!.tiles_total} tiles.`}
          <Progress done={data!.tiles_done} total={data!.tiles_total} />
        </div>
      )}
      {on && data?.status === 'done' && <p className="layer-note small">{data.pieces.toLocaleString()} thread pieces drawn; thicker lines are thicker threads (1 µm, 2 µm and up).</p>}
      {on && (state.error || data?.status === 'failed') && <div className="layer-toggle indent wrap"><ErrorNote error={state.error ?? data?.error ?? 'unknown error'} />
        <button className="btn small" onClick={state.reload}>Try again</button></div>}
    </>
  )
}

/** The "Stained but never proposed" layer: brown regions at least object-sized that no candidate covers (a check on the
 *  candidate finder, from the v2 proposals). */
function UnproposedToggle({ count, on, onToggle, state }: {
  count: number; on: boolean; onToggle: () => void; state: { data: CoreUnproposed | null; error: string | null; reload: () => void }
}) {
  return (
    <>
      <label className="layer-toggle" title="Brown areas above this slide's threshold, at least the size of an object, that detection never looked at">
        <input type="checkbox" checked={on} onChange={onToggle} />
        <span className="swatch" style={swatchOf(UNPROPOSED_COLOR)} />Stained but never proposed<span className="spacer" />
        <span className="muted">{count.toLocaleString()}</span>
      </label>
      {on && !state.data && !state.error && <div className="layer-toggle indent"><Loading inline label="Loading the stained areas…" /></div>}
      {on && state.error && <div className="layer-toggle indent wrap"><ErrorNote error={state.error} />
        <button className="btn small" onClick={state.reload}>Try again</button></div>}
    </>
  )
}

/** A legend swatch hatched like the excluded-areas picture (\ / | - or a plain translucent fill). */
function hatchSwatch(kind: ExcludedKind) {
  const angle: Record<string, string> = { '\\': '135deg', '/': '45deg', '|': '90deg', '-': '0deg' }
  const fill = `${kind.color}55`
  return { background: kind.hatch ? `repeating-linear-gradient(${angle[kind.hatch] ?? '45deg'}, ${kind.color} 0 1.5px, ${fill} 1.5px 4px)` : fill,
    boxShadow: '0 0 0 1px rgb(0 0 0 / 35%)' }
}

/** The "Excluded from analysis" layer: what the analysis leaves out of this core, as detection found it (folds, linear
 *  artifacts, empty glass, the AT8 edge guard) and the tissue no analysed tile covers, hatched over the picture in the
 *  kinds' colours, with each kind's share of the core's area. People's painted and cleared areas are not in it (they
 *  train and evaluate the finders; the Annotations layer shows them). */
function ExcludedToggle({ on, onToggle, state }: { on: boolean; onToggle: () => void; state: { data: CoreExcluded | null; error: string | null; reload: () => void } }) {
  const data = state.data
  return (
    <>
      <label className="layer-toggle" title="What the analysis leaves out of this core, as detection found it: folds, linear artifacts, empty glass, the AT8 edge guard, and tissue in no analysed tile">
        <input type="checkbox" checked={on} onChange={onToggle} />
        <span className="swatch" style={hatchSwatch({ id: '', label: '', color: '#ff3b3b', hatch: '\\', fraction: null })} />Excluded from analysis</label>
      {on && !data && !state.error && <div className="layer-toggle indent"><Loading inline label="Loading the excluded areas…" /></div>}
      {on && data?.status === 'not_made' && <p className="layer-note small">Not made for this core yet: the new detection’s “Find candidates”
        step maps them (<Link to="/workflow">Run</Link>).</p>}
      {on && data?.status === 'done' && data.kinds.filter((k) => k.fraction !== 0).map((k) => (
        <div key={k.id} className="layer-toggle indent" title={`${k.label}: its share of the core’s area`}>
          <span className="swatch" style={hatchSwatch(k)} />{k.label}<span className="spacer" />
          <span className="muted">{k.fraction == null ? '' : k.fraction > 0 && k.fraction < 0.001 ? '<0.1%' : `${(k.fraction * 100).toFixed(1)}%`}</span>
        </div>
      ))}
      {on && state.error && <div className="layer-toggle indent wrap"><ErrorNote error={state.error} />
        <button className="btn small" onClick={state.reload}>Try again</button></div>}
    </>
  )
}

/** A length given in mm: in µm below 1 mm. */
function lengthMm(mm: number) {
  return mm >= 1 ? `${fmt(mm, 3)} mm` : `${Math.round(mm * 1000).toLocaleString()} µm`
}

/** An area in µm², with thousands separators. */
function area(um2: number) {
  return um2 >= 100 ? Math.round(um2).toLocaleString() : fmt(um2, 3)
}

function Detections({ status, error, stain, tma, tissue }: { status: CoreObjects | null; error: string | null; stain: string; tma: string; tissue?: string | null }) {
  const { tmaName } = useProject()
  if (error) return <ErrorNote error={error} />
  if (!status) return <Loading inline label="Loading the detections…" />
  if (status.status === 'done') return null
  if (status.status === 'partial') {
    return (
      <div className="note-box">
        <b>Detection is part-way through this core.</b> Showing what it has found so far ({status.tiles_done} of {status.tiles_total} tiles).
        <Progress done={status.tiles_done} total={status.tiles_total} />
      </div>
    )
  }
  if (!status.tiles_total && (tissue === 'empty' || tissue === 'sparse')) {
    return (
      <div className="note-box">
        <b>Skipped: the tissue check found {tissue === 'empty' ? 'no' : 'too little'} tissue on this {stain} picture.</b> Detection only looks
        at pictures with tissue, so nothing was detected here. If you can see tissue, the check was wrong: ask whoever runs the analysis
        to look at this core again.
      </div>
    )
  }
  return (
    <div className="note-box">
      <b>Not detected yet.</b> Detection has not looked at the {stain} slide of {tmaName(tma)} yet{status.tiles_total ? '' : ' (no tiles were chosen for it)'}, so only the picture is shown.
      {' '}<Link to="/workflow">See progress on the Run page.</Link>
    </div>
  )
}

function ObjectList({ objects, selected, onPick }: { objects: Found[]; selected: number | null; onPick: (o: Found) => void }) {
  const [page, setPage] = useState(0)
  const size = 100
  const sorted = useMemo(() => [...objects].sort((a, b) => (b.p ?? 0) - (a.p ?? 0)), [objects])
  const pages = Math.ceil(sorted.length / size)
  return (
    <>
      <div className="row small" style={{ margin: '10px 0 6px' }}><b>All objects ({objects.length.toLocaleString()})</b><span className="spacer" />
        {pages > 1 && <><button className="btn small" disabled={page === 0} onClick={() => setPage(page - 1)}>‹</button>
          <span className="muted">{page + 1} / {pages}</span><button className="btn small" disabled={page >= pages - 1} onClick={() => setPage(page + 1)}>›</button></>}</div>
      <div className="table-wrap" style={{ maxHeight: 320 }}>
        <table className="compact">
          <thead><tr><th>Class</th><th className="num">Sure</th><th className="num">µm²</th></tr></thead>
          <tbody>
            {sorted.slice(page * size, (page + 1) * size).map((o) => (
              <tr key={o.index} className={`clickable${o.index === selected ? ' selected' : ''}`} onClick={() => onPick(o)}>
                <td><span className="swatch" style={swatchOf(o.color, o.cls === 'rejected')} /> {CLASS_STYLE[o.cls]?.short ?? o.cls}</td>
                <td className="num">{o.p === null ? '—' : `${Math.round(o.p * 100)}%`}</td><td className="num">{o.area === null ? '—' : area(o.area)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}
