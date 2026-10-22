import { useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { HALO_PX, LIT_PX, MAX_ZOOM, OUTLINE_PX, lineScale, useViewKeys } from './viewkeys'
import { FullScreenButton } from './viewer'

export interface PyramidLevel { width: number; height: number; cols: number; rows: number }
/** A core image the server has cut into tiles: level 0 is full resolution, each next level half the size. */
export interface Pyramid { width: number; height: number; tile: number; version: string; levels: PyramidLevel[] }
/** Something drawn over the image, in image pixels: a circle (radius `r`) or, with `w`/`h`, a box. `rings` adds the
 *  object's exact outline (polygons as flat [x0, y0, x1, y1, …] pixel indices); `circle: false` leaves the circle out, so
 *  an outlined object is drawn by one shape. `dashed`: drawn with long dashes (rejected candidates; tile boxes). */
export interface Mark { x: number; y: number; r?: number; w?: number; h?: number; color: string; dashed?: boolean; rings?: number[][]; circle?: boolean }
/** An open line over the image (a traced thread): flat [x0, y0, x1, y1, …] pixel indices, drawn `width` screen px wide. */
export interface Line { points: number[]; width: number }
/** Where to look: the image point to centre, and optionally the zoom (screen px per image px). Change `key` to go again. */
export interface ViewTarget { x: number; y: number; scale?: number; key: number }
interface View { s: number; x: number; y: number }  // s = screen px per image px; (x, y) = screen position of the image corner
interface TileRef { level: number; col: number; row: number }

const MAX_SCALE = 4
const FAR_SCALE = 0.25  // below this zoom lines are drawn as faint hairlines
/* Object strokes, in screen px whatever the zoom: every line has a thin dark halo under it so it reads on brown (DAB)
   and on blue (haematoxylin) tissue; dashed marks (rejected candidates) use long dashes so they stay visible. */
const STROKE = OUTLINE_PX  // an outline's width on screen at 1× (the whole core); `lineScale` thickens it a little zoomed in
const STROKE_HOVER = LIT_PX
const STROKE_SELECTED = LIT_PX + 0.5
const HALO = HALO_PX  // extra width of the dark line under each stroke
const HALO_COLOR = 'rgb(0 0 0 / 55%)'
const DASH = [5, 4]

/** Strokes a shape twice: a dark halo, then the colour (`glow`: a soft light around it, for hover and selection).
 *  `unit` = user units per screen px (1 / zoom under the image transform, 1 in screen space). */
function strokeWithHalo(g: CanvasRenderingContext2D, draw: () => void, color: string, width: number, unit: number, dashed = false, glow = false) {
  const dash = dashed ? DASH.map((d) => d * unit) : []
  g.save()
  g.setLineDash(dash)
  g.strokeStyle = HALO_COLOR
  g.lineWidth = (width + HALO) * unit
  draw()
  g.strokeStyle = color
  g.lineWidth = width * unit
  if (glow) { g.shadowColor = color; g.shadowBlur = 10 }
  draw()
  g.restore()
}
const NICE_UM = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]

/** The coarsest level that still gives at least one image pixel per device pixel. */
function chooseLevel(pyramid: Pyramid, scale: number) {
  const ratio = 1 / (scale * (window.devicePixelRatio || 1))
  return Math.max(0, Math.min(pyramid.levels.length - 1, Math.floor(Math.log2(Math.max(1, ratio)))))
}

/** The outlines of some marks as one path, in image pixels (a polygon runs through its edge pixels' centres). */
function ringPath(marks: Mark[]) {
  const path = new Path2D()
  for (const m of marks) for (const ring of m.rings ?? []) {
    path.moveTo(ring[0] + 0.5, ring[1] + 0.5)
    for (let k = 2; k < ring.length; k += 2) path.lineTo(ring[k] + 0.5, ring[k + 1] + 0.5)
    path.closePath()
  }
  return path
}

const LINE_CELL = 1024  // lines are grouped in squares of this many image px, so only the squares in view are drawn

/** Lines grouped by width and by the square their first point lies in: one path (image pixels) and its bounds each. */
function linePaths(lines: Line[]) {
  const groups = new Map<string, { width: number; path: Path2D; x0: number; y0: number; x1: number; y1: number }>()
  for (const line of lines) {
    const p = line.points
    const key = `${line.width}|${Math.floor(p[0] / LINE_CELL)}|${Math.floor(p[1] / LINE_CELL)}`
    if (!groups.has(key)) groups.set(key, { width: line.width, path: new Path2D(), x0: Infinity, y0: Infinity, x1: -Infinity, y1: -Infinity })
    const group = groups.get(key)!
    group.path.moveTo(p[0] + 0.5, p[1] + 0.5)
    for (let k = 0; k < p.length; k += 2) {
      if (k) group.path.lineTo(p[k] + 0.5, p[k + 1] + 0.5)
      group.x0 = Math.min(group.x0, p[k]); group.x1 = Math.max(group.x1, p[k] + 1)
      group.y0 = Math.min(group.y0, p[k + 1]); group.y1 = Math.max(group.y1, p[k + 1] + 1)
    }
  }
  return [...groups.values()]
}

/** Whether image point (x, y) lies inside one of the mark's outline polygons (even-odd rule). */
function insideRings(m: Mark, x: number, y: number) {
  let inside = false
  for (const ring of m.rings ?? []) {
    for (let k = 0, j = ring.length - 2; k < ring.length; j = k, k += 2) {
      const [xi, yi, xj, yj] = [ring[k] + 0.5, ring[k + 1] + 0.5, ring[j] + 0.5, ring[j + 1] + 0.5]
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
    }
  }
  return inside
}

function visibleTiles(pyramid: Pyramid, level: number, view: View, box: { w: number; h: number }): TileRef[] {
  const info = pyramid.levels[level]
  const span = pyramid.tile * 2 ** level  // image px per tile
  const range = (offset: number, length: number, count: number) =>
    [Math.max(0, Math.floor(-offset / view.s / span)), Math.min(count - 1, Math.floor((length - offset) / view.s / span))]
  const [c0, c1] = range(view.x, box.w, info.cols)
  const [r0, r1] = range(view.y, box.h, info.rows)
  const out: TileRef[] = []
  for (let row = r0; row <= r1; row++) for (let col = c0; col <= c1; col++) out.push({ level, col, row })
  return out
}

/* Drawing many objects (10–30 thousand candidates of a busy core) smoothly:
   - the objects, lines and boxes are drawn once into a cached canvas that covers the view and half of it again on every
     side; panning only moves that picture (and the tiles, by a CSS transform), without React rendering anything;
   - while zooming the cached picture is stretched and drawn sharp again once the zoom rests (SETTLE_MS);
   - objects are found through a grid of CELL-px squares: only the squares in view are drawn, and hovering or clicking
     tests only the objects near the pointer (at most once per frame);
   - zoomed far out (an object smaller than DOT_PX on screen) every object is a dot; the outlines come when zoomed in. */
const CELL = 512
const MARGIN = 0.5
const SETTLE_MS = 140
const COMMIT_MS = 120  // while moving, the tiles, the zoom label and the scale bar follow at most this often
const DOT_PX = 5

interface Bounds { x0: number; y0: number; x1: number; y1: number }
/** The marks prepared for drawing and hit-testing: each one's bounds, the grid squares (drawn by the square of their
 *  centre, hit-tested in every square their bounds touch), their outline paths per square and style, their typical size. */
interface Prepared {
  bounds: Bounds[]
  home: Map<number, number[]>
  hits: Map<number, number[]>
  paths: Map<number, Map<string, Path2D>>
  styles: { key: string; color: string; dashed: boolean }[]
  typical: number
  reach: number  // the largest half size of a mark (image px): how far beyond a square its marks can reach
}
const cellKey = (cx: number, cy: number) => cx * 65536 + cy
const styleKey = (m: Mark) => `${m.color}|${m.dashed ? 1 : 0}`

function prepare(marks: Mark[]): Prepared {
  const bounds: Bounds[] = []
  const home = new Map<number, number[]>()
  const hits = new Map<number, number[]>()
  const styles = new Map<string, { key: string; color: string; dashed: boolean }>()
  const sizes: number[] = []
  let reach = 0
  marks.forEach((m, i) => {
    let b: Bounds
    if (m.rings?.length) {
      b = { x0: Infinity, y0: Infinity, x1: -Infinity, y1: -Infinity }
      for (const ring of m.rings) for (let k = 0; k < ring.length; k += 2) {
        if (ring[k] < b.x0) b.x0 = ring[k]
        if (ring[k] + 1 > b.x1) b.x1 = ring[k] + 1
        if (ring[k + 1] < b.y0) b.y0 = ring[k + 1]
        if (ring[k + 1] + 1 > b.y1) b.y1 = ring[k + 1] + 1
      }
      sizes.push(Math.max(b.x1 - b.x0, b.y1 - b.y0))
      const key = styleKey(m)
      if (!styles.has(key)) styles.set(key, { key, color: m.color, dashed: !!m.dashed })
    } else {
      const r = m.r ?? 4
      b = { x0: m.x - r, y0: m.y - r, x1: m.x + r, y1: m.y + r }
      sizes.push(2 * r)
    }
    bounds.push(b)
    reach = Math.max(reach, Math.abs(m.x - b.x0), Math.abs(b.x1 - m.x), Math.abs(m.y - b.y0), Math.abs(b.y1 - m.y))
    const own = cellKey(Math.floor(m.x / CELL), Math.floor(m.y / CELL))
    if (!home.has(own)) home.set(own, [])
    home.get(own)!.push(i)
    for (let cx = Math.floor(b.x0 / CELL); cx <= Math.floor(b.x1 / CELL); cx++) {
      for (let cy = Math.floor(b.y0 / CELL); cy <= Math.floor(b.y1 / CELL); cy++) {
        const key = cellKey(cx, cy)
        if (!hits.has(key)) hits.set(key, [])
        hits.get(key)!.push(i)
      }
    }
  })
  sizes.sort((a, b) => a - b)
  // dashed (rejected) first, so the solid outlines of kept objects lie on top
  const ordered = [...styles.values()].sort((a, b) => Number(b.dashed) - Number(a.dashed))
  return { bounds, home, hits, paths: new Map(), styles: ordered, typical: sizes.length ? sizes[sizes.length >> 1] : 10, reach: Math.min(reach, 4 * CELL) }
}

/** The outline paths of one grid square's marks, one per style (built when the square is first drawn). */
function cellPaths(prepared: Prepared, marks: Mark[], key: number): Map<string, Path2D> {
  let out = prepared.paths.get(key)
  if (out) return out
  out = new Map()
  for (const i of prepared.home.get(key) ?? []) {
    const m = marks[i]
    if (!m.rings?.length) continue
    const style = styleKey(m)
    if (!out.has(style)) out.set(style, new Path2D())
    const path = out.get(style)!
    for (const ring of m.rings) {
      path.moveTo(ring[0] + 0.5, ring[1] + 0.5)
      for (let k = 2; k < ring.length; k += 2) path.lineTo(ring[k] + 0.5, ring[k + 1] + 0.5)
      path.closePath()
    }
  }
  prepared.paths.set(key, out)
  return out
}

/** The grid squares whose marks may show in the image rectangle (marks reach `reach` px beyond their square). */
function cellsIn(prepared: Prepared, x0: number, y0: number, x1: number, y1: number): number[] {
  const out: number[] = []
  const pad = prepared.reach
  for (let cx = Math.floor((x0 - pad) / CELL); cx <= Math.floor((x1 + pad) / CELL); cx++) {
    for (let cy = Math.floor((y0 - pad) / CELL); cy <= Math.floor((y1 + pad) / CELL); cy++) {
      const key = cellKey(cx, cy)
      if (prepared.home.has(key)) out.push(key)
    }
  }
  return out
}

/** A picture drawn over the whole image (e.g. the excluded areas), `width` × `height` image px, any resolution. */
export interface Overlay { src: string; width: number; height: number; opacity?: number }
/** A pointer position for painting (`onPress`, `onMove`): image px, the zoom (screen px per image px) and Shift. */
export interface PaintPoint { x: number; y: number; s: number; shift: boolean }

/**
 * A whole core image to zoom (wheel, double-click, + / − keys and buttons, up to `MAX_ZOOM` times the whole core) and drag
 * around (or hold the arrow keys: `useViewKeys`). Only the tiles in view are
 * fetched, at the screen's resolution; the coarse picture underneath stays visible while sharper tiles arrive.
 * `marks` (detected objects, with their outlines), `lines` (in `lineColor`, thin at any zoom, faint when far out) and
 * `boxes` (tiles, fields) are drawn on a canvas on top; `overlays` are pictures laid over the whole image. A click near
 * a mark or inside its outline reports its index to `onPick` (null for empty space), hovering shows `describe(index)`.
 */
export function DeepZoom({ pyramid, tileUrl, layers = [], marks = [], lines, lineColor = '#fff', boxes = [], overlays = [], selected = null, describe, onPick, target, pixelUm, toolbar, height = 'min(72vh, 760px)', onTileError, onView,
  onPress, onMove, onRelease, panLock = false, onWheelOver, over, cursor }: {
  pyramid: Pyramid
  tileUrl: (level: number, col: number, row: number, layer?: string) => string
  layers?: string[]
  marks?: Mark[]
  lines?: Line[]
  lineColor?: string
  boxes?: Mark[]
  overlays?: Overlay[]
  selected?: number | null
  describe?: (index: number) => string
  onPick?: (index: number | null) => void
  target?: ViewTarget | null
  pixelUm?: number | null
  toolbar?: ReactNode
  /** the picture's height (CSS); 'fill': as tall as its (flex column) parent lets it be */
  height?: string
  /** A tile could not be loaded (the server cleaned up the picture to save space): ask for the picture again. */
  onTileError?: () => void
  /** The image-pixel rectangle in view, whenever the view settles (at most every COMMIT_MS while it moves). */
  onView?: (view: { x0: number; y0: number; x1: number; y1: number }) => void
  /** Painting (with `panLock`): the left button pressed at an image point; the middle button still pans. */
  onPress?: (p: PaintPoint) => void
  /** The pointer moved over the image (also while the left button is held after `onPress`). */
  onMove?: (p: PaintPoint | null) => void
  /** The left button was let go after `onPress`. */
  onRelease?: () => void
  /** A left drag does not move the picture (it paints); the middle button does, and a double click does not zoom. */
  panLock?: boolean
  /** The wheel turned over the picture: return true when it was used (e.g. for the brush size), so it does not zoom. */
  onWheelOver?: (deltaY: number, e: WheelEvent) => boolean
  /** Drawn on the image under the objects, in image px (e.g. an SVG or a canvas the size of the image); `unit` is one
   *  screen px in image px. */
  over?: (unit: number) => ReactNode
  /** The CSS cursor over the picture. */
  cursor?: string
}) {
  const box = useRef<HTMLDivElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const plane = useRef<HTMLDivElement>(null)
  const drag = useRef<{ x: number; y: number; vx: number; vy: number; moved: boolean } | null>(null)
  const [size, setSize] = useState({ w: 0, h: 0 })
  // the view: `viewRef` follows every move at once (drawing, the plane's transform); the state follows it at most every
  // COMMIT_MS, for what React draws (the tiles in view, the zoom label, the scale bar)
  const viewRef = useRef<View | null>(null)
  const [committed, setCommitted] = useState<View | null>(null)
  const [loaded, setLoaded] = useState<Set<string>>(() => new Set())
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null)
  const [dragging, setDragging] = useState(false)
  const [hovered, setHovered] = useState<number | null>(null)
  const prepared = useMemo(() => prepare(marks), [marks])
  const strokes = useMemo(() => linePaths(lines ?? []), [lines])
  useEffect(() => setHovered(null), [marks])
  const cache = useRef<{ canvas: HTMLCanvasElement; view: View; w: number; h: number } | null>(null)
  const frame = useRef(0)
  const settle = useRef(0)
  const commitTimer = useRef(0)
  const latest = useRef({ prepared, marks, strokes, boxes, lineColor, size, hovered, selected })
  latest.current = { prepared, marks, strokes, boxes, lineColor, size, hovered, selected }
  const view = viewRef.current ?? committed

  /** Draws the objects, lines and boxes for this view into the cached picture (the view plus MARGIN on every side). */
  const renderCache = (v: View) => {
    const { prepared: p, marks: ms, strokes: ls, boxes: bs, lineColor: lc, size: sz } = latest.current
    const dpr = window.devicePixelRatio || 1
    const mx = sz.w * MARGIN, my = sz.h * MARGIN
    const w = sz.w + 2 * mx, h = sz.h + 2 * my
    const c = cache.current?.canvas ?? document.createElement('canvas')
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr) }
    const g = c.getContext('2d')!
    g.setTransform(dpr, 0, 0, dpr, 0, 0)
    g.clearRect(0, 0, w, h)
    const ox = v.x + mx, oy = v.y + my  // where the image corner lies in the cached picture
    const [ix0, iy0, ix1, iy1] = [-ox / v.s, -oy / v.s, (w - ox) / v.s, (h - oy) / v.s]
    const z = lineScale(v.s / fitScale())  // outlines a little thicker zoomed in
    g.lineWidth = 1.2
    for (const b of bs) {
      g.strokeStyle = b.color
      g.setLineDash(b.dashed ? [5, 4] : [])
      g.strokeRect(b.x * v.s + ox, b.y * v.s + oy, (b.w ?? 0) * v.s, (b.h ?? 0) * v.s)
    }
    g.setLineDash([])
    if (ls.length) {
      g.save()
      g.setTransform(dpr * v.s, 0, 0, dpr * v.s, dpr * ox, dpr * oy)
      g.lineJoin = 'round'
      g.lineCap = 'round'
      const far = v.s < FAR_SCALE  // zoomed far out: hairlines, faint, so the picture and objects stay visible
      g.globalAlpha = far ? 0.4 : 0.9
      const inView = ls.filter((s) => !(s.x1 < ix0 || s.y1 < iy0 || s.x0 > ix1 || s.y0 > iy1))
      if (!far) {  // a dark halo under the lines, so they read on light and dark tissue
        g.strokeStyle = HALO_COLOR
        for (const s of inView) { g.lineWidth = (s.width + HALO) / v.s; g.stroke(s.path) }
      }
      g.strokeStyle = lc
      for (const s of inView) { g.lineWidth = (far ? Math.min(0.8, s.width) : s.width) / v.s; g.stroke(s.path) }
      g.restore()
    }
    const cells = cellsIn(p, ix0, iy0, ix1, iy1)
    if (p.typical * v.s < DOT_PX) {  // far out: a dot per object, rejected (dashed) ones first and smaller
      const dots = new Map<string, { color: string; dashed: boolean; path: Path2D }>()
      for (const key of cells) for (const i of p.home.get(key)!) {
        const m = ms[i]
        const x = m.x * v.s + ox, y = m.y * v.s + oy
        if (x < -4 || y < -4 || x > w + 4 || y > h + 4) continue
        const style = styleKey(m)
        if (!dots.has(style)) dots.set(style, { color: m.color, dashed: !!m.dashed, path: new Path2D() })
        const radius = m.dashed ? 1.8 : 2.4
        const path = dots.get(style)!.path
        path.moveTo(x + radius, y)
        path.arc(x, y, radius, 0, 2 * Math.PI)
      }
      g.lineWidth = 1
      g.strokeStyle = HALO_COLOR
      for (const d of [...dots.values()].sort((a, b) => Number(b.dashed) - Number(a.dashed))) { g.fillStyle = d.color; g.fill(d.path); g.stroke(d.path) }
      return { canvas: c, view: v, w, h }
    }
    g.save()
    g.setTransform(dpr * v.s, 0, 0, dpr * v.s, dpr * ox, dpr * oy)
    g.lineJoin = 'round'
    const unit = 1 / v.s  // one screen px in image px: the strokes keep their width on screen at any zoom
    const groups = cells.map((key) => cellPaths(p, ms, key))
    for (const style of p.styles) {
      const paths = groups.map((group) => group.get(style.key)).filter((x): x is Path2D => !!x)
      if (!paths.length) continue
      g.setLineDash(style.dashed ? DASH.map((d) => d * unit) : [])
      g.strokeStyle = HALO_COLOR
      g.lineWidth = (STROKE * z + HALO) * unit
      for (const path of paths) g.stroke(path)
      g.strokeStyle = style.color
      g.lineWidth = STROKE * z * unit
      for (const path of paths) g.stroke(path)
    }
    g.restore()
    g.lineJoin = 'round'
    for (const key of cells) for (const i of p.home.get(key)!) {  // objects without an outline: a circle, or a dot when small
      const m = ms[i]
      if (m.circle === false) continue
      const x = m.x * v.s + ox, y = m.y * v.s + oy
      const r = (m.r ?? 4) * v.s + 2
      if (x < -r - 4 || y < -r - 4 || x > w + r + 4 || y > h + r + 4) continue
      if (r < 4) {
        g.beginPath()
        g.arc(x, y, 2.4, 0, 2 * Math.PI)
        g.fillStyle = m.color
        g.strokeStyle = HALO_COLOR
        g.lineWidth = 1
        g.fill()
        g.stroke()
        continue
      }
      strokeWithHalo(g, () => { g.beginPath(); g.arc(x, y, Math.max(r, 5), 0, 2 * Math.PI); g.stroke() }, m.color, STROKE * z, 1, m.dashed)
    }
    return { canvas: c, view: v, w, h }
  }

  /** One frame: the cached picture moved (and stretched while zooming) to the current view, then the hovered and the
   *  selected object on top, thicker and glowing. */
  const draw = () => {
    frame.current = 0
    const c = canvas.current
    const g = c?.getContext('2d')
    const v = viewRef.current
    const { size: sz, marks: ms, hovered: hov, selected: sel } = latest.current
    if (!c || !g || !v || !sz.w) return
    const dpr = window.devicePixelRatio || 1
    g.setTransform(dpr, 0, 0, dpr, 0, 0)
    g.clearRect(0, 0, sz.w, sz.h)
    let k = cache.current ? v.s / cache.current.view.s : 0
    const mx = sz.w * MARGIN, my = sz.h * MARGIN
    const covers = (c: { view: View; w: number; h: number }) => {  // the cached picture, moved and stretched, still fills the view
      const dx = (-mx - c.view.x) * k + v.x, dy = (-my - c.view.y) * k + v.y
      return dx <= 0 && dy <= 0 && dx + c.w * k >= sz.w && dy + c.h * k >= sz.h
    }
    const stale = !cache.current || k > 3 || k < 1 / 3 || !covers(cache.current)
    if (stale) { cache.current = renderCache(v); k = 1 }
    const cached = cache.current!
    if (k !== 1) {  // zooming: stretch the cached picture now, draw it sharp once the zoom rests
      window.clearTimeout(settle.current)
      settle.current = window.setTimeout(() => { cache.current = renderCache(viewRef.current!); requestDraw() }, SETTLE_MS)
    }
    g.imageSmoothingEnabled = true
    g.drawImage(cached.canvas, (-mx - cached.view.x) * k + v.x, (-my - cached.view.y) * k + v.y, cached.w * k, cached.h * k)
    const z = lineScale(v.s / fitScale())
    const lit: [number | null, number][] = [[hov, STROKE_HOVER * z], [sel, STROKE_SELECTED * z]]
    for (const [i, width] of lit) {
      const m = i === null ? null : ms[i]
      if (!m) continue
      if (m.rings?.length && m.circle !== true) {
        const path = ringPath([m])
        g.save()
        g.setTransform(dpr * v.s, 0, 0, dpr * v.s, dpr * v.x, dpr * v.y)
        g.lineJoin = 'round'
        strokeWithHalo(g, () => g.stroke(path), m.color, width, 1 / v.s, m.dashed, true)
        g.restore()
      } else {
        const x = m.x * v.s + v.x, y = m.y * v.s + v.y
        const radius = Math.max((m.r ?? 4) * v.s + 2, 5) + (i === sel ? 3 : 0)
        strokeWithHalo(g, () => { g.beginPath(); g.arc(x, y, radius, 0, 2 * Math.PI); g.stroke() }, m.color, width, 1, m.dashed, true)
      }
    }
  }
  const requestDraw = () => { if (!frame.current) frame.current = requestAnimationFrame(draw) }

  /** Moves the view at once (drawing, the plane's transform); React follows within COMMIT_MS, or now with `now`. */
  const setView = (next: View | null, now = false) => {
    viewRef.current = next
    if (next && plane.current) plane.current.style.transform = `translate(${next.x}px, ${next.y}px) scale(${next.s})`
    requestDraw()
    if (now) { window.clearTimeout(commitTimer.current); commitTimer.current = 0; setCommitted(next); return }
    if (!commitTimer.current) commitTimer.current = window.setTimeout(() => { commitTimer.current = 0; setCommitted(viewRef.current) }, COMMIT_MS)
  }
  useEffect(() => () => { cancelAnimationFrame(frame.current); window.clearTimeout(settle.current); window.clearTimeout(commitTimer.current) }, [])

  const fit = (): View => {
    const s = fitScale()
    return { s, x: (size.w - pyramid.width * s) / 2, y: (size.h - pyramid.height * s) / 2 }
  }
  /** The whole core in view: the zoom lines and `MAX_ZOOM` count from (1×). */
  function fitScale(): number {
    const { size: sz } = latest.current
    return sz.w && sz.h ? Math.min(sz.w / pyramid.width, sz.h / pyramid.height) * 0.96 : 1
  }
  /** At most `MAX_ZOOM` times the whole core, and never less than every pixel of the scan at MAX_SCALE. */
  const maxScale = () => Math.max(MAX_SCALE, fitScale() * MAX_ZOOM)
  const centred = (t: ViewTarget): View => {
    const s = Math.min(maxScale(), Math.max(fit().s, t.scale ?? Math.max(viewRef.current?.s ?? 0, 0.5)))
    return { s, x: size.w / 2 - t.x * s, y: size.h / 2 - t.y * s }
  }
  const zoomAt = (factor: number, px: number, py: number, now = false) => {
    const v = viewRef.current
    if (!v) return
    const s = Math.min(maxScale(), Math.max(fit().s * 0.5, v.s * factor))
    setView({ s, x: px - ((px - v.x) * s) / v.s, y: py - ((py - v.y) * s) / v.s }, now)
  }
  const zoomCentre = (factor: number) => zoomAt(factor, size.w / 2, size.h / 2, true)
  const keys = useViewKeys({ zoomBy: zoomCentre, panBy: (dx, dy) => { const v = viewRef.current; if (v) setView({ ...v, x: v.x + dx, y: v.y + dy }) } })

  useEffect(() => {
    const element = box.current
    if (!element) return
    const resize = new ResizeObserver(() => setSize({ w: element.clientWidth, h: element.clientHeight }))
    resize.observe(element)
    return () => resize.disconnect()
  }, [])
  useEffect(() => {  // the canvas has the box's size (in device pixels); changing it clears it, so only when the size changes
    const c = canvas.current
    const dpr = window.devicePixelRatio || 1
    if (c && size.w) { c.width = Math.round(size.w * dpr); c.height = Math.round(size.h * dpr) }
    cache.current = null
    requestDraw()
  }, [size.w, size.h])  // eslint-disable-line react-hooks/exhaustive-deps
  // what is drawn changed: the cached picture is made again; hovering and selecting only redraw the frame
  useEffect(() => { cache.current = null; requestDraw() }, [prepared, strokes, boxes, lineColor])  // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { requestDraw() }, [hovered, selected])  // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {  // the rectangle in view, for whoever shows it elsewhere (a minimap)
    if (onView && committed && size.w) onView({ x0: -committed.x / committed.s, y0: -committed.y / committed.s, x1: (size.w - committed.x) / committed.s, y1: (size.h - committed.y) / committed.s })
  }, [committed, size.w, size.h])  // eslint-disable-line react-hooks/exhaustive-deps

  // a new image starts from the whole core (or the requested spot) as soon as the box has a size
  useEffect(() => { setView(null, true); setLoaded(new Set()) }, [pyramid.version, pyramid.width])  // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (size.w && size.h && !viewRef.current) setView(target ? centred(target) : fit(), true) })
  useEffect(() => { if (target && size.w) setView(centred(target), true) }, [target?.key])  // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const element = box.current
    if (!element) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const rect = element.getBoundingClientRect()
      const delta = e.deltaMode === 1 ? e.deltaY * 30 : e.deltaY
      if (onWheelOver?.(delta, e)) return
      zoomAt(Math.exp(-Math.max(-300, Math.min(300, delta)) * 0.0022), e.clientX - rect.left, e.clientY - rect.top)
    }
    element.addEventListener('wheel', onWheel, { passive: false })
    return () => element.removeEventListener('wheel', onWheel)
  })

  const local = (e: React.PointerEvent | React.MouseEvent) => {
    const rect = box.current!.getBoundingClientRect()
    return { px: e.clientX - rect.left, py: e.clientY - rect.top }
  }
  /** The mark under a screen point: the nearest whose circle (or centre) is close or whose outline holds the point,
   *  among the marks of the grid squares around it. */
  const nearest = (px: number, py: number): number | null => {
    const v = viewRef.current
    if (!v) return null
    const ix = (px - v.x) / v.s
    const iy = (py - v.y) / v.s
    const near = Math.min(2 * CELL, 8 / v.s)
    const seen = new Set<number>()
    let best: number | null = null
    let bestDistance = Infinity
    for (let cx = Math.floor((ix - near) / CELL); cx <= Math.floor((ix + near) / CELL); cx++) {
      for (let cy = Math.floor((iy - near) / CELL); cy <= Math.floor((iy + near) / CELL); cy++) {
        for (const i of prepared.hits.get(cellKey(cx, cy)) ?? []) {
          if (seen.has(i)) continue
          seen.add(i)
          const m = marks[i]
          const d = Math.hypot(m.x * v.s + v.x - px, m.y * v.s + v.y - py)
          const b = prepared.bounds[i]
          const hit = d < (m.circle === false ? 8 : Math.max(8, (m.r ?? 4) * v.s + 6))
            || (!!m.rings?.length && ix >= b.x0 && ix <= b.x1 && iy >= b.y0 && iy <= b.y1 && insideRings(m, ix, iy))
          if (hit && d < bestDistance) { best = i; bestDistance = d }
        }
      }
    }
    return best
  }
  // hovering: the pointer's last position, hit-tested once per frame
  const pointer = useRef<{ px: number; py: number } | null>(null)
  const hoverFrame = useRef(0)
  const painting = useRef(false)
  const at = (e: React.PointerEvent): PaintPoint | null => {
    const v = viewRef.current
    if (!v) return null
    const { px, py } = local(e)
    return { x: (px - v.x) / v.s, y: (py - v.y) / v.s, s: v.s, shift: e.shiftKey }
  }
  const hoverTest = () => {
    hoverFrame.current = 0
    const at = pointer.current
    if (!at || drag.current?.moved || onPress) return
    const hit = nearest(at.px, at.py)
    setHovered(hit)
    if (describe) setTip(hit === null ? null : { x: at.px, y: at.py, text: describe(hit) })
  }
  useEffect(() => () => cancelAnimationFrame(hoverFrame.current), [])

  const top = pyramid.levels.length - 1
  const level = view ? chooseLevel(pyramid, view.s) : top
  const sharp = view && size.w ? visibleTiles(pyramid, level, view, size) : []
  const shown = view && size.w ? [
    ...(level < top ? visibleTiles(pyramid, top, view, size) : []),
    ...(level + 1 < top ? visibleTiles(pyramid, level + 1, view, size) : []),
    ...sharp,
  ] : []
  const waiting = sharp.some((t) => !loaded.has(`${t.level}-${t.col}-${t.row}`))
  const tileStyle = (t: TileRef) => {
    const f = 2 ** t.level
    const info = pyramid.levels[t.level]
    const pad = view ? 0.6 / view.s : 0  // a hair of overlap hides the seams between tiles
    return { left: t.col * pyramid.tile * f, top: t.row * pyramid.tile * f,
      width: Math.min(pyramid.tile, info.width - t.col * pyramid.tile) * f + pad, height: Math.min(pyramid.tile, info.height - t.row * pyramid.tile) * f + pad }
  }
  const markLoaded = (key: string) => setLoaded((old) => (old.has(key) ? old : new Set(old).add(key)))

  let scaleBar = null
  if (committed && pixelUm) {
    const umPerPx = pixelUm / committed.s
    const um = NICE_UM.find((n) => n / umPerPx >= 60) ?? NICE_UM[NICE_UM.length - 1]
    scaleBar = <div className="dz-scale"><span style={{ width: um / umPerPx }} />{um >= 1000 ? `${um / 1000} mm` : `${um} µm`}</div>
  }

  return (
    <div className={`dz-wrap${height === 'fill' ? ' dz-fill' : ''}`}>
      <div className="dz-toolbar">
        {toolbar}
        <span className="spacer" />
        <span className="row dz-zoom-buttons">
          <button className="btn small" onClick={() => zoomCentre(1 / 1.6)} aria-label="Zoom out" title="Zoom out (−)">−</button>
          <span className="small muted dz-zoom" title="100% shows every pixel of the scan">{committed ? `${committed.s >= 0.1 ? Math.round(committed.s * 100) : (committed.s * 100).toFixed(1)}%` : ''}</span>
          <button className="btn small" onClick={() => zoomCentre(1.6)} aria-label="Zoom in" title={`Zoom in (+), up to ${MAX_ZOOM}× the whole core`}>+</button>
          <button className="btn small" onClick={() => setView(fit(), true)} title="Show the whole core">Whole core</button>
          {height === 'fill' && <FullScreenButton />}
        </span>
      </div>
      <div ref={box} className={`dz${dragging ? ' dragging' : ''}`} style={{ ...(height === 'fill' ? {} : { height }), ...(cursor ? { cursor } : {}) }} tabIndex={0}
        aria-label="Core image: scroll or use + and − to zoom, drag or hold the arrow keys to move"
        onMouseDown={(e) => { if (e.button === 1) e.preventDefault() }}  // no autoscroll on the middle button
        onPointerEnter={keys.activate}
        onPointerDown={(e) => {
          keys.activate()
          const v = viewRef.current
          if (!v || (e.button !== 0 && !(panLock && e.button === 1))) return
          e.currentTarget.setPointerCapture(e.pointerId)
          if (panLock && e.button === 0) {  // painting
            const p = at(e)
            painting.current = true
            if (p) onPress?.(p)
            return
          }
          drag.current = { x: e.clientX, y: e.clientY, vx: v.x, vy: v.y, moved: false }
        }}
        onPointerMove={(e) => {
          if (onMove) onMove(at(e))
          if (painting.current) return
          const d = drag.current
          if (d) {
            if (!d.moved && Math.hypot(e.clientX - d.x, e.clientY - d.y) > 4) { d.moved = true; setDragging(true); setTip(null); setHovered(null) }
            const v = viewRef.current
            if (d.moved && v) setView({ ...v, x: d.vx + e.clientX - d.x, y: d.vy + e.clientY - d.y })
            return
          }
          pointer.current = local(e)
          if (!hoverFrame.current) hoverFrame.current = requestAnimationFrame(hoverTest)
        }}
        onPointerUp={(e) => {
          if (painting.current) { painting.current = false; onRelease?.(); return }
          const d = drag.current
          drag.current = null
          setDragging(false)
          if (d?.moved) setView(viewRef.current, true)
          if (d && !d.moved && onPick && e.button === 0) { const { px, py } = local(e); onPick(nearest(px, py)) }
        }}
        onPointerLeave={() => { pointer.current = null; setTip(null); setHovered(null); onMove?.(null) }}
        onDoubleClick={(e) => { if (panLock) return; const { px, py } = local(e); zoomAt(2, px, py, true) }}>
        {view && (
          <div ref={plane} className="dz-plane" style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.s})` }}>
            {shown.map((t) => {
              const key = `${t.level}-${t.col}-${t.row}`
              return <img key={key} src={tileUrl(t.level, t.col, t.row)} style={tileStyle(t)} alt="" draggable={false} onLoad={() => markLoaded(key)} onError={onTileError} />
            })}
            {layers.flatMap((layer) => sharp.map((t) => (
              <img key={`${layer}-${t.level}-${t.col}-${t.row}`} src={tileUrl(t.level, t.col, t.row, layer)} style={tileStyle(t)} alt="" draggable={false} />
            )))}
            {overlays.map((o) => (
              <img key={o.src} className="dz-overlay" src={o.src} alt="" draggable={false}
                style={{ left: 0, top: 0, width: o.width, height: o.height, opacity: o.opacity ?? 1 }} />
            ))}
            {over?.(1 / view.s)}
          </div>
        )}
        <canvas ref={canvas} className="dz-canvas" style={{ width: size.w, height: size.h }} />
        {scaleBar}
        {waiting && <span className="dz-busy"><span className="spinner" aria-hidden="true" />loading detail…</span>}
        {tip && <div className="dz-tip" style={{ left: Math.max(4, Math.min(tip.x + 14, size.w - 230)), top: tip.y + 14 }}>{tip.text}</div>}
      </div>
    </div>
  )
}
