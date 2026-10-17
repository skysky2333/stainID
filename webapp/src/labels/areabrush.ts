/* The brush of "Mark bad areas": a whole core image's bad areas painted on masks at 1/`AREA_SCALE` of the image (the
   grid of the excluded-areas layer, 8 core px per mask px), one mask per type (`EXCLUDE_TYPES`, keys 1–4) and one for
   the parts cleared ("include", Shift-drag: marked fine). The masks hold only what was painted since the core was
   opened, one set per stain image; Save sends them as PNGs (labels/exclusions.py turns them into one edit of people's
   areas per stain image). People's areas teach and test the bad-area detection; the analysis keeps detection's own.

   A core image is large (a 12,000 px core: 1,500 px masks), so nothing is encoded while painting: the strokes are drawn
   on the visible canvas (`draw`, the size of the masks, laid over the picture at image size) as a hatch in each type's
   colour and a faint green fill for the parts marked fine, and Z takes back a stroke from a copy of just the part it
   touched. */
import { CLEARED, EXCLUDE_TYPES, INCLUDE } from './brush'

export const AREA_SCALE = 8  // core px per mask px (stainid.exclusions.SCALE)
export const AREA_BRUSH_PX = { min: 6, max: 400, start: 48 }  // brush diameter, screen px
const KINDS = [...EXCLUDE_TYPES.map((t) => t.key), INCLUDE]
const UNDO_STEPS = 40

interface Snapshot { x: number; y: number; w: number; h: number; data: Record<string, ImageData>; painted: Record<string, boolean> }
interface Stroke { kind: string; clear: boolean; last: number[]; radius: number; box: number[]; before: Record<string, HTMLCanvasElement>; painted: Record<string, boolean> }

const fills = new Map<string, CanvasPattern | string>()
/** A type's fill: diagonal lines in its colour over a faint wash of it; the parts marked fine a faint green. */
function fillOf(g: CanvasRenderingContext2D, kind: string): CanvasPattern | string {
  if (kind === INCLUDE) return 'rgb(34 197 94 / 30%)'
  const known = fills.get(kind)
  if (known) return known
  const colour = EXCLUDE_TYPES.find((t) => t.key === kind)?.colour ?? CLEARED.colour
  const tile = document.createElement('canvas')
  tile.width = tile.height = 4
  const t = tile.getContext('2d')!
  t.fillStyle = colour
  t.globalAlpha = 0.3
  t.fillRect(0, 0, 4, 4)
  t.globalAlpha = 1
  for (let k = 0; k < 4; k++) t.fillRect(k, 3 - k, 1, 1)
  const made = g.createPattern(tile, 'repeat') ?? colour
  fills.set(kind, made)
  return made
}

/** The pending strokes on one stain image of a core (`width` × `height` image px). */
export class AreaMasks {
  readonly w: number
  readonly h: number
  private masks: Record<string, HTMLCanvasElement> = {}
  private painted: Record<string, boolean> = {}
  private stroke: Stroke | null = null
  private undos: Snapshot[] = []
  private scratch: HTMLCanvasElement

  constructor(width: number, height: number) {
    this.w = Math.max(1, Math.ceil(width / AREA_SCALE))
    this.h = Math.max(1, Math.ceil(height / AREA_SCALE))
    for (const kind of KINDS) this.masks[kind] = this.canvas()
    this.scratch = this.canvas()
  }

  private canvas() {
    const c = document.createElement('canvas')
    c.width = this.w
    c.height = this.h
    return c
  }

  /** Anything painted or marked fine (that Save would send). */
  get dirty() { return Object.values(this.painted).some(Boolean) }
  get steps() { return this.undos.length }
  get painting() { return !!this.stroke }

  private dab(from: number[], to: number[]) {
    const s = this.stroke!
    for (const [k, c] of Object.entries(this.masks)) {  // a cell has one say: painting one type takes the others off it
      const g = c.getContext('2d')!
      g.globalCompositeOperation = (s.clear ? k === INCLUDE : k === s.kind) ? 'source-over' : 'destination-out'
      g.strokeStyle = '#ffffff'
      g.lineCap = 'round'
      g.lineWidth = Math.max(1, (2 * s.radius) / AREA_SCALE)
      g.beginPath()
      g.moveTo(from[0] / AREA_SCALE, from[1] / AREA_SCALE)
      g.lineTo(to[0] / AREA_SCALE + 0.01, to[1] / AREA_SCALE)
      g.stroke()
      g.globalCompositeOperation = 'source-over'
    }
    this.painted[s.clear ? INCLUDE : s.kind] = true
    const r = s.radius / AREA_SCALE + 2
    const xs = [from[0] / AREA_SCALE, to[0] / AREA_SCALE], ys = [from[1] / AREA_SCALE, to[1] / AREA_SCALE]
    s.box = [Math.min(s.box[0], Math.min(...xs) - r), Math.min(s.box[1], Math.min(...ys) - r),
      Math.max(s.box[2], Math.max(...xs) + r), Math.max(s.box[3], Math.max(...ys) + r)]
  }

  /** A stroke starts at this image point (`radius` in image px): painting `kind`, or with `clear` marking the part fine. */
  begin(x: number, y: number, radius: number, kind: string, clear: boolean) {
    const before: Record<string, HTMLCanvasElement> = {}
    for (const [k, c] of Object.entries(this.masks)) {  // a copy to take the touched part from when the stroke ends (for Z)
      const copy = this.canvas()
      copy.getContext('2d')!.drawImage(c, 0, 0)
      before[k] = copy
    }
    this.stroke = { kind, clear, last: [x, y], radius, box: [Infinity, Infinity, -Infinity, -Infinity], before, painted: { ...this.painted } }
    this.dab([x, y], [x, y])
  }

  extend(x: number, y: number) {
    const s = this.stroke
    if (!s) return
    this.dab(s.last, [x, y])
    s.last = [x, y]
  }

  /** The stroke ends (it becomes one step for Z); false when none was on. */
  end(): boolean {
    const s = this.stroke
    this.stroke = null
    if (!s) return false
    const x = Math.max(0, Math.floor(s.box[0])), y = Math.max(0, Math.floor(s.box[1]))
    const w = Math.min(this.w, Math.ceil(s.box[2])) - x, h = Math.min(this.h, Math.ceil(s.box[3])) - y
    if (w > 0 && h > 0) {
      const data = Object.fromEntries(Object.entries(s.before).map(([k, c]) => [k, c.getContext('2d')!.getImageData(x, y, w, h)]))
      this.undos = [...this.undos.slice(-(UNDO_STEPS - 1)), { x, y, w, h, data, painted: s.painted }]
    }
    return true
  }

  /** Z: the last stroke taken back; false when there is none. */
  undo(): boolean {
    const last = this.undos.pop()
    if (!last) return false
    for (const [k, data] of Object.entries(last.data)) this.masks[k].getContext('2d')!.putImageData(data, last.x, last.y)
    this.painted = last.painted
    return true
  }

  /** Draws the strokes on the visible canvas: each type as a hatch in its colour, the parts marked fine faint green. */
  draw(c: HTMLCanvasElement) {
    if (c.width !== this.w || c.height !== this.h) { c.width = this.w; c.height = this.h }
    const g = c.getContext('2d')!
    g.clearRect(0, 0, this.w, this.h)
    const t = this.scratch.getContext('2d')!
    for (const kind of KINDS) {
      if (!this.painted[kind]) continue
      t.globalCompositeOperation = 'source-over'
      t.clearRect(0, 0, this.w, this.h)
      t.drawImage(this.masks[kind], 0, 0)
      t.globalCompositeOperation = 'source-in'
      t.fillStyle = fillOf(t, kind)
      t.fillRect(0, 0, this.w, this.h)
      t.globalCompositeOperation = 'source-over'
      g.drawImage(this.scratch, 0, 0)
    }
  }

  /** What Save sends for this stain image: one PNG per type painted (null when nothing was). */
  pngs(): Record<string, string> | null {
    const out: Record<string, string> = {}
    for (const k of KINDS) if (this.painted[k]) out[k] = this.masks[k].toDataURL('image/png')
    return Object.keys(out).length ? out : null
  }
}
