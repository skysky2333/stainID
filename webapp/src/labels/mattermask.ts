/* The white-matter brush of "Grey / white matter" (labels/matter.tsx): one area per stain image, kept whole. A drag
   paints white matter, the eraser (or Shift-drag) takes it away; Z takes back a stroke. Unlike the bad-area brush
   (brush.ts: what was added and taken out since Save), Save sends the whole area as it is now (one PNG at 1/`scale`
   of the core image), which the server keeps as that stain image's white matter (labels/matter.py). The area is held
   on a canvas the page lays over the picture (`attach`): changed dab by dab, nothing encoded while painting. */
import { PALETTE } from '../palette'

export const MATTER_SCALE = 8  // core px per mask px (stainid.labels.matter.SCALE)
export const MATTER_COLOUR = PALETTE.areas.white_matter
const UNDO_STEPS = 40

const ctx = (c: HTMLCanvasElement) => c.getContext('2d', { willReadFrequently: true })!
type Box = [number, number, number, number]
const grow = (a: Box | null, b: Box): Box => (a ? [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]), Math.max(a[3], b[3])] : b)

let hatch: CanvasPattern | string | null = null
/** The area's look: diagonal lines in its colour over a wash of it. */
function hatchOf(g: CanvasRenderingContext2D): CanvasPattern | string {
  if (hatch) return hatch
  const tile = document.createElement('canvas')
  tile.width = tile.height = 4
  const t = ctx(tile)
  t.fillStyle = MATTER_COLOUR
  t.globalAlpha = 0.45
  t.fillRect(0, 0, 4, 4)
  t.globalAlpha = 1
  for (let k = 0; k < 4; k++) t.fillRect(k, 3 - k, 1, 1)
  hatch = g.createPattern(tile, 'repeat') ?? MATTER_COLOUR
  return hatch
}

export class MatterMask {
  readonly w: number
  readonly h: number
  readonly scale: number
  private mask: HTMLCanvasElement
  private shown: HTMLCanvasElement
  private stroke: { erase: boolean; last: number[]; radius: number; box: Box | null; before: HTMLCanvasElement } | null = null
  private undos: { x: number; y: number; data: ImageData }[] = []
  private changed = false

  constructor(width: number, height: number, scale = MATTER_SCALE) {
    this.scale = scale
    this.w = Math.max(1, Math.ceil(width / scale))
    this.h = Math.max(1, Math.ceil(height / scale))
    this.mask = this.canvas()
    this.shown = this.canvas()
    this.shown.style.cssText = 'position:absolute;left:0;top:0;width:100%;height:100%;image-rendering:pixelated;pointer-events:none'
  }

  private canvas() {
    const c = document.createElement('canvas')
    c.width = this.w
    c.height = this.h
    ctx(c)
    return c
  }

  /** Painted or erased since the area was loaded (what Save would change). */
  get dirty() { return this.changed }
  get painting() { return !!this.stroke }

  /** The area shown again from the mask (after a new start, Z or a reset). */
  private remake() {
    const g = this.shown.getContext('2d')!
    g.globalCompositeOperation = 'source-over'
    g.clearRect(0, 0, this.w, this.h)
    g.drawImage(this.mask, 0, 0)
    g.globalCompositeOperation = 'source-in'
    g.fillStyle = hatchOf(g)
    g.fillRect(0, 0, this.w, this.h)
    g.globalCompositeOperation = 'source-over'
  }

  /** The area to start from (a white-on-transparent PNG, `url`, over `width` x `height` core px; none: no white
   *  matter); forgets the strokes. Resolves once drawn. */
  start(url: string | null | undefined, width?: number, height?: number): Promise<void> {
    const g = ctx(this.mask)
    g.clearRect(0, 0, this.w, this.h)
    this.undos = []
    this.changed = false
    if (!url) { this.remake(); return Promise.resolve() }
    return new Promise<void>((done) => {
      const image = new Image()
      image.onload = () => {
        g.imageSmoothingEnabled = false
        g.drawImage(image, 0, 0, (width ?? this.w * this.scale) / this.scale, (height ?? this.h * this.scale) / this.scale)
        this.remake()
        done()
      }
      image.onerror = () => { this.remake(); done() }
      image.src = url
    })
  }

  private dab(from: number[], to: number[]) {
    const s = this.stroke!
    for (const [c, style] of [[this.mask, '#ffffff'], [this.shown, hatchOf(this.shown.getContext('2d')!)]] as const) {
      const g = c.getContext('2d')!
      g.globalCompositeOperation = s.erase ? 'destination-out' : 'source-over'
      if (!s.erase && c === this.shown) {  // a clean cut first, so the hatch is not laid twice where strokes cross
        g.globalCompositeOperation = 'destination-out'
        this.line(g, from, to, s.radius, '#ffffff')
        g.globalCompositeOperation = 'source-over'
      }
      this.line(g, from, to, s.radius, style)
      g.globalCompositeOperation = 'source-over'
    }
    const r = s.radius / this.scale + 2
    s.box = grow(s.box, [Math.min(from[0], to[0]) / this.scale - r, Math.min(from[1], to[1]) / this.scale - r,
      Math.max(from[0], to[0]) / this.scale + r, Math.max(from[1], to[1]) / this.scale + r])
    this.changed = true
  }

  private line(g: CanvasRenderingContext2D, from: number[], to: number[], radius: number, style: string | CanvasPattern) {
    g.strokeStyle = style
    g.lineCap = 'round'
    g.lineWidth = Math.max(1, (2 * radius) / this.scale)
    g.beginPath()
    g.moveTo(from[0] / this.scale, from[1] / this.scale)
    g.lineTo(to[0] / this.scale + 0.01, to[1] / this.scale)
    g.stroke()
  }

  /** A stroke starts at this picture point (`radius` in picture px): painting, or with `erase` taking away. */
  begin(x: number, y: number, radius: number, erase: boolean) {
    const before = this.canvas()
    ctx(before).drawImage(this.mask, 0, 0)
    this.stroke = { erase, last: [x, y], radius, box: null, before }
    this.dab([x, y], [x, y])
  }

  extend(x: number, y: number) {
    const s = this.stroke
    if (!s) return
    this.dab(s.last, [x, y])
    s.last = [x, y]
  }

  /** The stroke ends (one step for Z); false when none was on. */
  end(): boolean {
    const s = this.stroke
    this.stroke = null
    if (!s || !s.box) return false
    const x = Math.max(0, Math.floor(s.box[0])), y = Math.max(0, Math.floor(s.box[1]))
    const w = Math.max(1, Math.min(this.w, Math.ceil(s.box[2])) - x), h = Math.max(1, Math.min(this.h, Math.ceil(s.box[3])) - y)
    this.undos = [...this.undos.slice(-(UNDO_STEPS - 1)), { x, y, data: ctx(s.before).getImageData(x, y, w, h) }]
    return true
  }

  /** Z: the last stroke taken back; false when there is none. */
  undo(): boolean {
    const last = this.undos.pop()
    if (!last) return false
    ctx(this.mask).putImageData(last.data, last.x, last.y)
    this.changed = this.undos.length > 0
    this.remake()
    return true
  }

  /** Puts the area's canvas into `host` (positioned, the size of the picture), or takes it out (null). */
  attach(host: HTMLElement | null) {
    if (host) { if (this.shown.parentElement !== host) host.appendChild(this.shown) } else this.shown.remove()
  }

  /** Whether any white matter is painted now. */
  any(): boolean {
    const data = ctx(this.mask).getImageData(0, 0, this.w, this.h).data
    for (let i = 3; i < data.length; i += 4) if (data[i] > 127) return true
    return false
  }

  /** What Save sends: the whole area as a PNG ("" when there is none). */
  png(): string { return this.any() ? this.mask.toDataURL('image/png') : '' }
}
