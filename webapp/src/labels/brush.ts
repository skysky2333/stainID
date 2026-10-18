/* The one brush of bad areas: the field tool (key B) and "Mark bad areas" (always on) share it. People change the
   existing bad areas, they do not draw separate new ones.

   For each type (`EXCLUDE_TYPES`: 1 fold or tissue damage, 2 artifact, 3 no tissue, 4 out of focus or bad staining;
   the keys 1–4 choose it while the brush is on) the current area is detection's area of that type with everyone's
   saved edits merged in (GET …/fields/{tile}/areas or …/cores/{core}/{stain}/areas, `exclusions.final_say`). A drag
   adds the brushed pixels to the chosen type's area (a union: one region, however many strokes; pixels of another type
   are given to the chosen one); Shift-drag takes them out of the chosen type's area, detection's included ("marked
   fine" is no more than that: taken out of a type). The chosen type's current area is drawn as one merged region (the
   others faintly). Tab makes the brush smaller, Shift+Tab bigger (+ / − too; its size in µm shows in the panel and as
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

export const BRUSH_SCALE = 4  // the field tool: masks at a quarter of the field picture in each direction
export const AREA_SCALE = 8  // "Mark bad areas": core px per mask px (stainid.exclusions.SCALE)
export const BRUSH_UM = { min: 5, max: 2000, start: 40 }  // brush diameter, µm
export const BRUSH_STEP = 1.2  // Tab / Shift+Tab: the diameter this much smaller / bigger
export const INCLUDE = 'include'
/** The types of excluded area, with their colours (the same as the layer's, store_view.EXCLUDED_KINDS). */
export const EXCLUDE_TYPES: { key: string; name: string; colour: string }[] = [
  { key: '1', name: 'fold or tissue damage', colour: '#ffb000' },
  { key: '2', name: 'artifact (scratch, bubble, debris, linear)', colour: '#a855f7' },
  { key: '3', name: 'no tissue / empty', colour: '#9ca3af' },
  { key: '4', name: 'out of focus or bad staining', colour: '#00e5ff' },
]
/** Taking out (Shift-drag): the brush cursor's colour; the older "include" edits (cleared) in the history too. */
export const CLEARED = { key: INCLUDE, name: 'taken out', colour: '#22c55e' }
const TYPES = EXCLUDE_TYPES.map((t) => t.key)
const MASKS = [...TYPES, ...TYPES.map((k) => `-${k}`)]
const UNDO_STEPS = 40

/** The brush diameter after Tab (smaller) or Shift+Tab (bigger), within `BRUSH_UM`. */
export function brushStep(um: number, bigger: boolean): number {
  return Math.round(Math.min(BRUSH_UM.max, Math.max(BRUSH_UM.min, bigger ? um * BRUSH_STEP : um / BRUSH_STEP)))
}

export const colourOfType = (kind: string) => (kind === INCLUDE ? CLEARED.colour : EXCLUDE_TYPES.find((t) => t.key === kind)?.colour ?? '#fff')

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
  const t = tile.getContext('2d')!
  t.fillStyle = colour
  t.globalAlpha = 0.3
  t.fillRect(0, 0, 4, 4)
  t.globalAlpha = 1
  for (let k = 0; k < 4; k++) t.fillRect(k, 3 - k, 1, 1)
  const made = g.createPattern(tile, 'repeat') ?? colour
  hatches.set(kind, made)
  return made
}

/** The strokes since Save over each type's current area, on one picture (`width` × `height` picture px). */
export class BrushMasks {
  readonly w: number
  readonly h: number
  readonly scale: number
  private masks: Record<string, HTMLCanvasElement> = {}
  private base: Record<string, HTMLCanvasElement> = {}
  private hasBase: Record<string, boolean> = {}
  private painted: Record<string, boolean> = {}
  private stroke: { kind: string; erase: boolean; last: number[]; radius: number; box: number[]; before: Record<string, HTMLCanvasElement>; painted: Record<string, boolean> } | null = null
  private undos: BrushStep[] = []
  private scratch: HTMLCanvasElement
  private layer: HTMLCanvasElement

  constructor(width: number, height: number, scale: number) {
    this.scale = scale
    this.w = Math.max(1, Math.ceil(width / scale))
    this.h = Math.max(1, Math.ceil(height / scale))
    for (const k of MASKS) this.masks[k] = this.canvas()
    for (const k of TYPES) this.base[k] = this.canvas()
    this.scratch = this.canvas()
    this.layer = this.canvas()
  }

  private canvas() {
    const c = document.createElement('canvas')
    c.width = this.w
    c.height = this.h
    return c
  }

  /** Something added to or taken out of a type since Save (what Save would send). */
  get dirty() { return Object.values(this.painted).some(Boolean) }
  get painting() { return !!this.stroke }
  get steps() { return this.undos.length }

  /** Each type's current area from the server (`areas`), drawn onto the masks' grid; resolves once drawn. */
  setBase(areas: FieldAreas | null): Promise<void> {
    const jobs = TYPES.map((k) => new Promise<void>((done) => {
      const c = this.base[k]
      const g = c.getContext('2d')!
      g.clearRect(0, 0, c.width, c.height)
      this.hasBase[k] = false
      const url = areas?.kinds?.[k]
      if (!areas || !url || areas.status !== 'done') { done(); return }
      const image = new Image()
      image.onload = () => {
        g.imageSmoothingEnabled = false
        g.drawImage(image, (areas.x ?? 0) / this.scale, (areas.y ?? 0) / this.scale, (areas.width ?? 0) / this.scale, (areas.height ?? 0) / this.scale)
        this.hasBase[k] = true
        done()
      }
      image.onerror = () => done()
      image.src = url
    }))
    return Promise.all(jobs).then(() => undefined)
  }

  /** One dab: painting `kind` adds to it (and takes the pixels off the other types and off what was taken out of
   *  it); erasing takes the pixels out of `kind` (also out of what was added to it). */
  private dab(from: number[], to: number[]) {
    const s = this.stroke!
    const line = (k: string, on: boolean) => {
      const g = this.masks[k].getContext('2d')!
      g.globalCompositeOperation = on ? 'source-over' : 'destination-out'
      g.strokeStyle = '#ffffff'
      g.lineCap = 'round'
      g.lineWidth = Math.max(1, (2 * s.radius) / this.scale)
      g.beginPath()
      g.moveTo(from[0] / this.scale, from[1] / this.scale)
      g.lineTo(to[0] / this.scale + 0.01, to[1] / this.scale)
      g.stroke()
      g.globalCompositeOperation = 'source-over'
    }
    if (s.erase) {
      line(`-${s.kind}`, true)
      line(s.kind, false)
      this.painted[`-${s.kind}`] = true
    } else {
      line(s.kind, true)
      line(`-${s.kind}`, false)
      for (const j of TYPES) if (j !== s.kind) line(j, false)
      this.painted[s.kind] = true
    }
    const r = s.radius / this.scale + 2
    const xs = [from[0] / this.scale, to[0] / this.scale], ys = [from[1] / this.scale, to[1] / this.scale]
    s.box = [Math.min(s.box[0], Math.min(...xs) - r), Math.min(s.box[1], Math.min(...ys) - r),
      Math.max(s.box[2], Math.max(...xs) + r), Math.max(s.box[3], Math.max(...ys) + r)]
  }

  /** A stroke starts at this picture point (`radius` in picture px): adding to `kind`, or with `erase` taking out of it. */
  begin(x: number, y: number, radius: number, kind: string, erase: boolean) {
    const before: Record<string, HTMLCanvasElement> = {}
    for (const [k, c] of Object.entries(this.masks)) {  // a copy to take the touched part from when the stroke ends (for Z)
      const copy = this.canvas()
      copy.getContext('2d')!.drawImage(c, 0, 0)
      before[k] = copy
    }
    this.stroke = { kind, erase, last: [x, y], radius, box: [Infinity, Infinity, -Infinity, -Infinity], before, painted: { ...this.painted } }
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
    if (!s) return null
    const x = Math.max(0, Math.floor(s.box[0])), y = Math.max(0, Math.floor(s.box[1]))
    const w = Math.max(1, Math.min(this.w, Math.ceil(s.box[2])) - x), h = Math.max(1, Math.min(this.h, Math.ceil(s.box[3])) - y)
    const data = Object.fromEntries(Object.entries(s.before).map(([k, c]) => [k, c.getContext('2d')!.getImageData(x, y, w, h)]))
    const step = { x, y, w, h, data, painted: s.painted }
    this.undos = [...this.undos.slice(-(UNDO_STEPS - 1)), step]
    return step
  }

  /** Puts a step back (the field tool's own Z). */
  restore(step: BrushStep) {
    for (const [k, data] of Object.entries(step.data)) this.masks[k]?.getContext('2d')!.putImageData(data, step.x, step.y)
    this.painted = { ...step.painted }
    this.undos = this.undos.filter((u) => u !== step)
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
    for (const c of Object.values(this.masks)) c.getContext('2d')!.clearRect(0, 0, c.width, c.height)
    this.painted = {}
    this.undos = []
  }

  /** Whether a type has any area now (detection's, saved edits or strokes since Save). */
  has(kind: string) { return !!this.hasBase[kind] || !!this.painted[kind] }

  /** A type's area now, white on transparent on the masks' grid: its current area, plus what was added to it, less what
   *  was taken out of it and what was given to another type. The canvas is reused by the next call. */
  area(kind: string): HTMLCanvasElement {
    const g = this.scratch.getContext('2d')!
    g.globalCompositeOperation = 'source-over'
    g.clearRect(0, 0, this.w, this.h)
    g.drawImage(this.base[kind], 0, 0)
    g.drawImage(this.masks[kind], 0, 0)
    g.globalCompositeOperation = 'destination-out'
    g.drawImage(this.masks[`-${kind}`], 0, 0)
    for (const j of TYPES) if (j !== kind) g.drawImage(this.masks[j], 0, 0)
    g.globalCompositeOperation = 'source-over'
    return this.scratch
  }

  /** Draws every type's area now on `c` (the masks' size): each as a hatch in its colour, `chosen` at full strength and
   *  the others at `others`. */
  draw(c: HTMLCanvasElement, chosen: string, others = 0.35) {
    if (c.width !== this.w || c.height !== this.h) { c.width = this.w; c.height = this.h }
    const out = c.getContext('2d')!
    out.clearRect(0, 0, this.w, this.h)
    const t = this.layer.getContext('2d')!
    for (const kind of [...TYPES.filter((k) => k !== chosen), chosen]) {
      if (!this.has(kind)) continue
      t.globalCompositeOperation = 'source-over'
      t.clearRect(0, 0, this.w, this.h)
      t.drawImage(this.area(kind), 0, 0)
      t.globalCompositeOperation = 'source-in'
      t.fillStyle = hatchOf(t, kind)
      t.fillRect(0, 0, this.w, this.h)
      t.globalCompositeOperation = 'source-over'
      out.globalAlpha = kind === chosen ? 1 : others
      out.drawImage(this.layer, 0, 0)
      out.globalAlpha = 1
    }
  }

  /** What Save sends: a PNG per type with something added ("1"–"4") or taken out ("-1"–"-4"); null when nothing. */
  pngs(): Record<string, string> | null {
    const out: Record<string, string> = {}
    for (const k of MASKS) if (this.painted[k]) out[k] = this.masks[k].toDataURL('image/png')
    return Object.keys(out).length ? out : null
  }
}

function painted(c: HTMLCanvasElement): boolean {
  const data = c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data
  for (let i = 3; i < data.length; i += 4) if (data[i] > 127) return true
  return false
}

/** The field tool's brush: `BrushMasks` over one field (`size` in field px, masks at 1/`BRUSH_SCALE`), with each type's
 *  area now as a data URL (`current`) for the field's drawing. */
export function useBrush(size: { w: number; h: number } | null, areas: FieldAreas | null) {
  const masks = useRef<BrushMasks | null>(null)
  const [current, setCurrent] = useState<Record<string, string>>({})  // per type: its area now (data URL), with the strokes since Save
  const [dirty, setDirty] = useState(false)

  const show = useCallback(() => {
    const m = masks.current
    if (!m) return
    const out: Record<string, string> = {}
    for (const k of TYPES) {
      if (!m.has(k)) continue
      const c = m.area(k)
      if (painted(c)) out[k] = c.toDataURL('image/png')
    }
    setCurrent(out)
    setDirty(m.dirty)
  }, [])

  useEffect(() => {  // empty masks the size of the field
    if (!size || masks.current) return
    masks.current = new BrushMasks(size.w, size.h, BRUSH_SCALE)
    show()
  }, [size, show])

  useEffect(() => {  // the current areas from the server
    const m = masks.current
    if (!m || !areas) return
    let live = true
    void m.setBase(areas).then(() => { if (live) show() })
    return () => { live = false }
  }, [areas, size, show])

  const begin = useCallback((x: number, y: number, radius: number, kind: string, erase: boolean) => {
    masks.current?.begin(x, y, radius, kind, erase)
    show()
  }, [show])
  const frame = useRef(0)
  const extend = useCallback((x: number, y: number) => {
    const m = masks.current
    if (!m?.painting) return
    m.extend(x, y)
    if (!frame.current) frame.current = requestAnimationFrame(() => { frame.current = 0; show() })  // at most once a frame
  }, [show])
  /** The stroke ends: the step to put back for Z, or null when no stroke was on. */
  const end = useCallback((): BrushStep | null => {
    const step = masks.current?.end() ?? null
    if (step) show()
    return step
  }, [show])
  const restore = useCallback((step: BrushStep | null) => { if (step) { masks.current?.restore(step); show() } }, [show])
  const reset = useCallback(() => { masks.current?.reset(); show() }, [show])
  const pngs = useCallback(() => masks.current?.pngs() ?? {}, [])
  return { current, dirty, painting: () => !!masks.current?.painting, begin, extend, end, restore, reset, pngs }
}
