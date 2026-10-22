/* The one brush of bad areas: the field tool (key B) and "Mark bad areas" (always on) share it. People change the
   existing bad areas, they do not draw separate new ones.

   For each type (`EXCLUDE_TYPES`: 1 fold or tissue damage, 2 artifact, 3 no tissue, 4 out of focus or bad staining;
   the keys 1–4 choose it while the brush is on) the current area is detection's area of that type with everyone's
   saved edits merged in (GET …/fields/{tile}/areas or …/cores/{core}/{stain}/areas, `exclusions.final_say`). A drag
   adds the brushed pixels to the chosen type's area (a union: one region, however many strokes; pixels of another type
   are given to the chosen one); Shift-drag takes them out of the chosen type's area, detection's included ("marked
   fine" is no more than that: taken out of a type). The chosen type's current area is drawn as one merged region (the
   others faintly). Tab makes the brush smaller, Shift+Tab bigger (+ / − zoom the picture; its size in µm shows in the panel and as
   the ring under the mouse); the wheel always zooms. Each stroke is one step Z takes back.

   `BrushMasks` holds, at 1/`scale` of the picture, per type what was added and what was taken out since the last Save,
   over the type's current area (`setBase`). Save sends them as PNGs ("1"–"4" added, "-1"–"-4" taken out;
   labels/exclusions.py turns them into one edit in the label log, resolved per type into masks by
   `stainid.exclusions`); then they are emptied (`reset`) once the server's current areas have the edit. Nothing is
   encoded while painting (a core image is large: a 12,000 px core makes 1,500 px masks). People's areas teach and
   test bad-area detection and leave the objects in them out of training and scoring; the analysis never reads them
   (it keeps detection's own). */
import { useCallback, useEffect, useRef, useState } from 'react'
import type { FieldAreas } from '../api'
import { PALETTE, areaColour } from '../palette'

export const BRUSH_SCALE = 4  // the field tool: masks at a quarter of the field picture in each direction
export const AREA_SCALE = 8  // "Mark bad areas": core px per mask px (stainid.exclusions.SCALE)
/** The brush diameter in screen px: the ring keeps its size on screen at every zoom, so zooming in gives a finer brush
 *  on the tissue. A stroke is kept in tissue units as painted (its dabs at the zoom of the moment it started). */
export const BRUSH_PX = { min: 4, max: 400, start: 36 }
export const BRUSH_KEY = 'stainid.brush.screen'  // the size, kept in this browser: one for Fix the field and Mark bad areas
export const BRUSH_STEP = 1.2  // Tab / Shift+Tab: the diameter this much smaller / bigger
export const INCLUDE = 'include'
/** The types of excluded area, with their colours (the one palette's, which the layer's, store_view.EXCLUDED_KINDS, are too). */
export const EXCLUDE_TYPES: { key: string; name: string; colour: string }[] = [
  { key: '1', name: 'fold or tissue damage', colour: PALETTE.areas['1'] },
  { key: '2', name: 'artifact (scratch, bubble, debris, linear)', colour: PALETTE.areas['2'] },
  { key: '3', name: 'no tissue / empty', colour: PALETTE.areas['3'] },
  { key: '4', name: 'out of focus or bad staining', colour: PALETTE.areas['4'] },
]
/** Taking out (Shift-drag): the brush cursor's colour; the older "include" edits (cleared) in the history too. */
export const CLEARED = { key: INCLUDE, name: 'taken out', colour: PALETTE.areas.taken_out }
const TYPES = EXCLUDE_TYPES.map((t) => t.key)
const MASKS = [...TYPES, ...TYPES.map((k) => `-${k}`)]
const UNDO_STEPS = 40

/** The brush diameter (screen px) after Tab (smaller) or Shift+Tab (bigger), within `BRUSH_PX`. */
export function brushStep(px: number, bigger: boolean): number {
  const next = Math.round(bigger ? Math.max(px + 1, px * BRUSH_STEP) : Math.min(px - 1, px / BRUSH_STEP))
  return Math.min(BRUSH_PX.max, Math.max(BRUSH_PX.min, next))
}

export const colourOfType = areaColour

/** The masks as they were in a box before a stroke (for Z). */
export interface BrushStep { x: number; y: number; w: number; h: number; data: Record<string, ImageData>; painted: Record<string, boolean> }

const hatches = new Map<string, CanvasPattern | string>()
/** A type's fill: diagonal lines in its colour over a faint wash of it. */
function hatchOf(g: CanvasRenderingContext2D, kind: string): CanvasPattern | string {
  const known = hatches.get(kind)
  if (known) return known
  const colour = colourOfType(kind)
  const tile = document.createElement('canvas')
  tile.width = tile.height = 4
  const t = ctx(tile)
  t.fillStyle = colour
  t.globalAlpha = 0.3
  t.fillRect(0, 0, 4, 4)
  t.globalAlpha = 1
  for (let k = 0; k < 4; k++) t.fillRect(k, 3 - k, 1, 1)
  const made = g.createPattern(tile, 'repeat') ?? colour
  hatches.set(kind, made)
  return made
}

type Box = [number, number, number, number]  // x0, y0, x1, y1 in mask px
/** The 2D context of a mask that is read back (for Z and Save): kept in memory. */
const ctx = (c: HTMLCanvasElement) => c.getContext('2d', { willReadFrequently: true })!
const grow = (a: Box | null, b: Box): Box => (a ? [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]), Math.max(a[3], b[3])] : b)

/** The strokes since Save over each type's current area, on one picture (`width` × `height` picture px).

    Painting stays smooth on a large picture: each type's area now is kept as its own hatched canvas (`shown`), changed
    dab by dab (the brush's line stroked in with the type's hatch, or cut out of the others), and the page stacks these
    canvases (`attach`) with the chosen type strong and the others faint (`look`, CSS opacity). Nothing is copied from
    one canvas to another, encoded or read back while painting (copying a canvas that just changed copies all of it:
    measured ~4 ms a draw on a 600 px mask, against well under 1 ms for a dab); the areas are put together again only
    after a new base from the server, Z or Reset (`remake`). */
export class BrushMasks {
  readonly w: number
  readonly h: number
  readonly scale: number
  private masks: Record<string, HTMLCanvasElement> = {}
  private base: Record<string, HTMLCanvasElement> = {}
  private shown: Record<string, HTMLCanvasElement> = {}
  private present: Record<string, boolean> = {}
  private painted: Record<string, boolean> = {}
  private stroke: { kind: string; erase: boolean; last: number[]; radius: number; box: Box | null; before: Record<string, HTMLCanvasElement>; painted: Record<string, boolean> } | null = null
  private undos: BrushStep[] = []

  constructor(width: number, height: number, scale: number) {
    this.scale = scale
    this.w = Math.max(1, Math.ceil(width / scale))
    this.h = Math.max(1, Math.ceil(height / scale))
    for (const k of MASKS) this.masks[k] = this.canvas(true)
    for (const k of TYPES) {
      this.base[k] = this.canvas(true)
      const c = this.canvas(false)
      c.dataset.kind = k
      c.style.cssText = 'position:absolute;left:0;top:0;width:100%;height:100%;image-rendering:pixelated;pointer-events:none'
      this.shown[k] = c
    }
  }

  /** `kept`: a mask that is read back (for Z and Save), kept in memory. */
  private canvas(kept: boolean) {
    const c = document.createElement('canvas')
    c.width = this.w
    c.height = this.h
    if (kept) ctx(c)
    return c
  }

  /** Something added to or taken out of a type since Save (what Save would send). */
  get dirty() { return Object.values(this.painted).some(Boolean) }
  get painting() { return !!this.stroke }
  get steps() { return this.undos.length }

  /** Each type's area now, put together again from its current area and the masks (after a new base, Z or Reset):
   *  the current area, plus what was added to it, less what was taken out of it and what was given to another type. */
  private remake() {
    for (const k of TYPES) {
      const g = this.shown[k].getContext('2d')!
      g.globalCompositeOperation = 'source-over'
      g.clearRect(0, 0, this.w, this.h)
      g.drawImage(this.base[k], 0, 0)
      g.drawImage(this.masks[k], 0, 0)
      g.globalCompositeOperation = 'destination-out'
      g.drawImage(this.masks[`-${k}`], 0, 0)
      for (const j of TYPES) if (j !== k) g.drawImage(this.masks[j], 0, 0)
      g.globalCompositeOperation = 'source-in'
      g.fillStyle = hatchOf(g, k)
      g.fillRect(0, 0, this.w, this.h)
      g.globalCompositeOperation = 'source-over'
    }
  }

  /** Each type's current area from the server (`areas`), drawn onto the masks' grid; resolves once drawn. */
  setBase(areas: FieldAreas | null): Promise<void> {
    const jobs = TYPES.map((k) => new Promise<void>((done) => {
      const c = this.base[k]
      const g = ctx(c)
      g.clearRect(0, 0, c.width, c.height)
      this.present[k] = !!this.painted[k]
      const url = areas?.kinds?.[k]
      if (!areas || !url || areas.status !== 'done') { done(); return }
      const image = new Image()
      image.onload = () => {
        g.imageSmoothingEnabled = false
        g.drawImage(image, (areas.x ?? 0) / this.scale, (areas.y ?? 0) / this.scale, (areas.width ?? 0) / this.scale, (areas.height ?? 0) / this.scale)
        this.present[k] = true
        done()
      }
      image.onerror = () => done()
      image.src = url
    }))
    return Promise.all(jobs).then(() => this.remake())
  }

  /** One dab: painting `kind` adds to it (and takes the pixels off the other types and off what was taken out of
   *  it); erasing takes the pixels out of `kind` (also out of what was added to it). The areas shown change with it. */
  private dab(from: number[], to: number[]) {
    const s = this.stroke!
    const line = (c: HTMLCanvasElement, on: boolean, style: string | CanvasPattern = '#ffffff') => {
      const g = c.getContext('2d')!
      g.globalCompositeOperation = on ? 'source-over' : 'destination-out'
      g.strokeStyle = style
      g.lineCap = 'round'
      g.lineWidth = Math.max(1, (2 * s.radius) / this.scale)
      g.beginPath()
      g.moveTo(from[0] / this.scale, from[1] / this.scale)
      g.lineTo(to[0] / this.scale + 0.01, to[1] / this.scale)
      g.stroke()
      g.globalCompositeOperation = 'source-over'
    }
    if (s.erase) {
      line(this.masks[`-${s.kind}`], true)
      line(this.masks[s.kind], false)
      line(this.shown[s.kind], false)
      this.painted[`-${s.kind}`] = true
    } else {
      line(this.masks[s.kind], true)
      line(this.masks[`-${s.kind}`], false)
      const shown = this.shown[s.kind]
      line(shown, false)  // a clean cut first, so the hatch is not laid twice where strokes cross
      line(shown, true, hatchOf(shown.getContext('2d')!, s.kind))
      for (const j of TYPES) if (j !== s.kind) { line(this.masks[j], false); line(this.shown[j], false) }
      this.painted[s.kind] = true
      this.present[s.kind] = true
    }
    const r = s.radius / this.scale + 2
    s.box = grow(s.box, [Math.min(from[0], to[0]) / this.scale - r, Math.min(from[1], to[1]) / this.scale - r,
      Math.max(from[0], to[0]) / this.scale + r, Math.max(from[1], to[1]) / this.scale + r])
  }

  /** A stroke starts at this picture point (`radius` in picture px): adding to `kind`, or with `erase` taking out of it. */
  begin(x: number, y: number, radius: number, kind: string, erase: boolean) {
    const before: Record<string, HTMLCanvasElement> = {}
    for (const [k, c] of Object.entries(this.masks)) {  // a copy to take the touched part from when the stroke ends (for Z)
      const copy = this.canvas(true)
      ctx(copy).drawImage(c, 0, 0)
      before[k] = copy
    }
    this.stroke = { kind, erase, last: [x, y], radius, box: null, before, painted: { ...this.painted } }
    this.dab([x, y], [x, y])
  }

  extend(x: number, y: number) {
    const s = this.stroke
    if (!s) return
    this.dab(s.last, [x, y])
    s.last = [x, y]
  }

  /** The stroke ends: the masks as they were before it in the box it touched (one step for Z), or null when no stroke
   *  was on. */
  end(): BrushStep | null {
    const s = this.stroke
    this.stroke = null
    if (!s || !s.box) return null
    const x = Math.max(0, Math.floor(s.box[0])), y = Math.max(0, Math.floor(s.box[1]))
    const w = Math.max(1, Math.min(this.w, Math.ceil(s.box[2])) - x), h = Math.max(1, Math.min(this.h, Math.ceil(s.box[3])) - y)
    const data = Object.fromEntries(Object.entries(s.before).map(([k, c]) => [k, ctx(c).getImageData(x, y, w, h)]))
    const step = { x, y, w, h, data, painted: s.painted }
    this.undos = [...this.undos.slice(-(UNDO_STEPS - 1)), step]
    return step
  }

  /** Puts a step back (the field tool's own Z). */
  restore(step: BrushStep) {
    for (const [k, data] of Object.entries(step.data)) { const c = this.masks[k]; if (c) ctx(c).putImageData(data, step.x, step.y) }
    this.painted = { ...step.painted }
    this.undos = this.undos.filter((u) => u !== step)
    this.remake()
  }

  /** Z: the last stroke taken back; false when there is none. */
  undo(): boolean {
    const last = this.undos[this.undos.length - 1]
    if (!last) return false
    this.restore(last)
    return true
  }

  /** Empty masks (after Save, once the server's current areas have the edit; or "Reset field"). */
  reset() {
    for (const c of Object.values(this.masks)) ctx(c).clearRect(0, 0, c.width, c.height)
    this.painted = {}
    this.undos = []
    this.remake()
  }

  /** Whether a type has any area now (detection's, saved edits or strokes since Save). */
  has(kind: string) { return !!this.present[kind] }
  /** The types with an area now. */
  types(): string[] { return TYPES.filter((k) => this.present[k]) }

  /** Puts the types' canvases into `host` (a positioned element the size of the picture), or takes them out (null). */
  attach(host: HTMLElement | null) {
    for (const k of TYPES) {
      const c = this.shown[k]
      if (host) { if (c.parentElement !== host) host.appendChild(c) } else c.remove()
    }
  }

  /** The chosen type at full strength (on top), the others at `others`. */
  look(chosen: string, others = 0.35) {
    for (const k of TYPES) {
      const c = this.shown[k]
      c.style.opacity = String(k === chosen ? 1 : others)
      c.style.zIndex = k === chosen ? '1' : '0'
    }
  }

  /** What Save sends: a PNG per type with something added ("1"–"4") or taken out ("-1"–"-4"); null when nothing. */
  pngs(): Record<string, string> | null {
    const out: Record<string, string> = {}
    for (const k of MASKS) if (this.painted[k]) out[k] = this.masks[k].toDataURL('image/png')
    return Object.keys(out).length ? out : null
  }
}

/** The field tool's brush: `BrushMasks` over one field (`size` in field px, masks at 1/`BRUSH_SCALE`), shown in an
 *  element the page lays over the field (`host`, a ref callback): no React update per pointer move. `types` (the
 *  types with an area) and `dirty` change only when a stroke ends or the areas load. */
export function useBrush(size: { w: number; h: number } | null, areas: FieldAreas | null, chosen: string, others: number) {
  const masks = useRef<BrushMasks | null>(null)
  const target = useRef<HTMLElement | null>(null)
  const [types, setTypes] = useState<string[]>([])
  const [dirty, setDirty] = useState(false)

  const settle = useCallback(() => {  // after a stroke, a new base, Z or Reset: what the page lists
    const m = masks.current
    if (!m) return
    setTypes((was) => { const now = m.types(); return now.join() === was.join() ? was : now })
    setDirty(m.dirty)
  }, [])
  useEffect(() => { masks.current?.look(chosen, others) }, [chosen, others, types])

  useEffect(() => {  // empty masks the size of the field
    if (!size || masks.current) return
    masks.current = new BrushMasks(size.w, size.h, BRUSH_SCALE)
    masks.current.attach(target.current)
    settle()
  }, [size, settle])

  useEffect(() => {  // the current areas from the server
    const m = masks.current
    if (!m || !areas) return
    let live = true
    void m.setBase(areas).then(() => { if (live) settle() })
    return () => { live = false }
  }, [areas, size, settle])

  const look = useRef({ chosen, others })
  look.current = { chosen, others }
  const host = useCallback((h: HTMLElement | null) => {
    target.current = h
    masks.current?.attach(h)
    masks.current?.look(look.current.chosen, look.current.others)
  }, [])
  const begin = useCallback((x: number, y: number, radius: number, kind: string, erase: boolean) => {
    masks.current?.begin(x, y, radius, kind, erase)
  }, [])
  const extend = useCallback((x: number, y: number) => {
    const m = masks.current
    if (m?.painting) m.extend(x, y)
  }, [])
  /** The stroke ends: the step to put back for Z, or null when no stroke was on. */
  const end = useCallback((): BrushStep | null => {
    const step = masks.current?.end() ?? null
    if (step) settle()
    return step
  }, [settle])
  const restore = useCallback((step: BrushStep | null) => { if (step) { masks.current?.restore(step); settle() } }, [settle])
  const reset = useCallback(() => { masks.current?.reset(); settle() }, [settle])
  const pngs = useCallback(() => masks.current?.pngs() ?? {}, [])
  return { types, dirty, host, painting: () => !!masks.current?.painting, begin, extend, end, restore, reset, pngs }
}
