import { useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'

export interface PyramidLevel { width: number; height: number; cols: number; rows: number }
/** A core image the server has cut into tiles: level 0 is full resolution, each next level half the size. */
export interface Pyramid { width: number; height: number; tile: number; version: string; levels: PyramidLevel[] }
/** Something drawn over the image, in image pixels: a circle (radius `r`) or, with `w`/`h`, a box. */
export interface Mark { x: number; y: number; r?: number; w?: number; h?: number; color: string; dashed?: boolean }
/** Where to look: the image point to centre, and optionally the zoom (screen px per image px). Change `key` to go again. */
export interface ViewTarget { x: number; y: number; scale?: number; key: number }
interface View { s: number; x: number; y: number }  // s = screen px per image px; (x, y) = screen position of the image corner
interface TileRef { level: number; col: number; row: number }

const MAX_SCALE = 4
const NICE_UM = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]

/** The coarsest level that still gives at least one image pixel per device pixel. */
function chooseLevel(pyramid: Pyramid, scale: number) {
  const ratio = 1 / (scale * (window.devicePixelRatio || 1))
  return Math.max(0, Math.min(pyramid.levels.length - 1, Math.floor(Math.log2(Math.max(1, ratio)))))
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

/**
 * A whole core image to zoom (wheel, double-click, + / −, keyboard) and drag around. Only the tiles in view are
 * fetched, at the screen's resolution; the coarse picture underneath stays visible while sharper tiles arrive.
 * `marks` (detected objects) and `boxes` (tiles, fields) are drawn on a canvas on top; a click near a mark reports its
 * index to `onPick` (null for empty space), hovering shows `describe(index)`.
 */
export function DeepZoom({ pyramid, tileUrl, layers = [], marks = [], boxes = [], selected = null, describe, onPick, target, pixelUm, toolbar, height = 'min(72vh, 760px)' }: {
  pyramid: Pyramid
  tileUrl: (level: number, col: number, row: number, layer?: string) => string
  layers?: string[]
  marks?: Mark[]
  boxes?: Mark[]
  selected?: number | null
  describe?: (index: number) => string
  onPick?: (index: number | null) => void
  target?: ViewTarget | null
  pixelUm?: number | null
  toolbar?: ReactNode
  height?: string
}) {
  const box = useRef<HTMLDivElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const drag = useRef<{ x: number; y: number; vx: number; vy: number; moved: boolean } | null>(null)
  const [size, setSize] = useState({ w: 0, h: 0 })
  const [view, setView] = useState<View | null>(null)
  const [loaded, setLoaded] = useState<Set<string>>(() => new Set())
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null)
  const [dragging, setDragging] = useState(false)

  const fit = (): View => {
    const s = Math.min(size.w / pyramid.width, size.h / pyramid.height) * 0.96
    return { s, x: (size.w - pyramid.width * s) / 2, y: (size.h - pyramid.height * s) / 2 }
  }
  const centred = (t: ViewTarget): View => {
    const s = Math.min(MAX_SCALE, Math.max(fit().s, t.scale ?? Math.max(view?.s ?? 0, 0.5)))
    return { s, x: size.w / 2 - t.x * s, y: size.h / 2 - t.y * s }
  }
  const zoomAt = (factor: number, px: number, py: number) => setView((v) => {
    if (!v) return v
    const s = Math.min(MAX_SCALE, Math.max(fit().s * 0.5, v.s * factor))
    return { s, x: px - ((px - v.x) * s) / v.s, y: py - ((py - v.y) * s) / v.s }
  })
  const zoomCentre = (factor: number) => zoomAt(factor, size.w / 2, size.h / 2)

  useEffect(() => {
    const element = box.current
    if (!element) return
    const resize = new ResizeObserver(() => setSize({ w: element.clientWidth, h: element.clientHeight }))
    resize.observe(element)
    return () => resize.disconnect()
  }, [])

  // a new image starts from the whole core (or the requested spot) as soon as the box has a size
  useEffect(() => { setView(null); setLoaded(new Set()) }, [pyramid.version, pyramid.width])
  useEffect(() => { if (size.w && size.h && !view) setView(target ? centred(target) : fit()) })
  useEffect(() => { if (target && size.w) setView(centred(target)) }, [target?.key])

  useEffect(() => {
    const element = box.current
    if (!element) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const rect = element.getBoundingClientRect()
      const delta = e.deltaMode === 1 ? e.deltaY * 30 : e.deltaY
      zoomAt(Math.exp(-Math.max(-300, Math.min(300, delta)) * 0.0022), e.clientX - rect.left, e.clientY - rect.top)
    }
    element.addEventListener('wheel', onWheel, { passive: false })
    return () => element.removeEventListener('wheel', onWheel)
  })

  // the drawing on top: boxes, then the marks inside the view (tiny dots when zoomed far out)
  useEffect(() => {
    const c = canvas.current
    const g = c?.getContext('2d')
    if (!c || !g || !view) return
    const dpr = window.devicePixelRatio || 1
    c.width = size.w * dpr
    c.height = size.h * dpr
    g.setTransform(dpr, 0, 0, dpr, 0, 0)
    g.clearRect(0, 0, size.w, size.h)
    g.lineWidth = 1.5
    for (const b of boxes) {
      g.strokeStyle = b.color
      g.setLineDash(b.dashed ? [5, 4] : [])
      g.strokeRect(b.x * view.s + view.x, b.y * view.s + view.y, (b.w ?? 0) * view.s, (b.h ?? 0) * view.s)
    }
    g.setLineDash([])
    marks.forEach((m, i) => {
      const x = m.x * view.s + view.x
      const y = m.y * view.s + view.y
      const r = (m.r ?? 4) * view.s + 2
      if (x < -r - 4 || y < -r - 4 || x > size.w + r + 4 || y > size.h + r + 4) return
      g.beginPath()
      if (r < 4 && i !== selected) {
        g.fillStyle = m.color
        g.arc(x, y, 1.8, 0, 2 * Math.PI)
        g.fill()
        return
      }
      g.strokeStyle = m.color
      g.lineWidth = i === selected ? 3.5 : 1.8
      g.arc(x, y, Math.max(r, 4) + (i === selected ? 3 : 0), 0, 2 * Math.PI)
      g.stroke()
    })
  }, [view, size, marks, boxes, selected])

  const local = (e: React.PointerEvent | React.MouseEvent) => {
    const rect = box.current!.getBoundingClientRect()
    return { px: e.clientX - rect.left, py: e.clientY - rect.top }
  }
  const nearest = (px: number, py: number): number | null => {
    if (!view) return null
    let best: number | null = null
    let bestDistance = Infinity
    marks.forEach((m, i) => {
      const d = Math.hypot(m.x * view.s + view.x - px, m.y * view.s + view.y - py)
      if (d < Math.max(8, (m.r ?? 4) * view.s + 6) && d < bestDistance) { best = i; bestDistance = d }
    })
    return best
  }

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
  if (view && pixelUm) {
    const umPerPx = pixelUm / view.s
    const um = NICE_UM.find((n) => n / umPerPx >= 60) ?? NICE_UM[NICE_UM.length - 1]
    scaleBar = <div className="dz-scale"><span style={{ width: um / umPerPx }} />{um >= 1000 ? `${um / 1000} mm` : `${um} µm`}</div>
  }

  return (
    <div className="dz-wrap">
      <div className="dz-toolbar">
        {toolbar}
        <span className="spacer" />
        <span className="row dz-zoom-buttons">
          <button className="btn small" onClick={() => zoomCentre(1 / 1.6)} aria-label="Zoom out" title="Zoom out (−)">−</button>
          <span className="small muted dz-zoom" title="100% shows every pixel of the scan">{view ? `${view.s >= 0.1 ? Math.round(view.s * 100) : (view.s * 100).toFixed(1)}%` : ''}</span>
          <button className="btn small" onClick={() => zoomCentre(1.6)} aria-label="Zoom in" title="Zoom in (+)">+</button>
          <button className="btn small" onClick={() => setView(fit())} title="Show the whole core">Whole core</button>
        </span>
      </div>
      <div ref={box} className={`dz${dragging ? ' dragging' : ''}`} style={{ height }} tabIndex={0} aria-label="Core image: scroll or use + and − to zoom, drag or use the arrow keys to move"
        onPointerDown={(e) => {
          if (!view || e.button !== 0) return
          e.currentTarget.setPointerCapture(e.pointerId)
          drag.current = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y, moved: false }
        }}
        onPointerMove={(e) => {
          const d = drag.current
          if (d) {
            if (!d.moved && Math.hypot(e.clientX - d.x, e.clientY - d.y) > 4) { d.moved = true; setDragging(true); setTip(null) }
            if (d.moved) setView((v) => v && { ...v, x: d.vx + e.clientX - d.x, y: d.vy + e.clientY - d.y })
            return
          }
          if (!describe) return
          const { px, py } = local(e)
          const hit = nearest(px, py)
          setTip(hit === null ? null : { x: px, y: py, text: describe(hit) })
        }}
        onPointerUp={(e) => {
          const d = drag.current
          drag.current = null
          setDragging(false)
          if (d && !d.moved && onPick) { const { px, py } = local(e); onPick(nearest(px, py)) }
        }}
        onPointerLeave={() => setTip(null)}
        onDoubleClick={(e) => { const { px, py } = local(e); zoomAt(2, px, py) }}
        onKeyDown={(e) => {
          const moves: Record<string, [number, number]> = { ArrowLeft: [80, 0], ArrowRight: [-80, 0], ArrowUp: [0, 80], ArrowDown: [0, -80] }
          const move = moves[e.key]
          if (move) { e.preventDefault(); setView((v) => v && { ...v, x: v.x + move[0], y: v.y + move[1] }) }
          if (e.key === '+' || e.key === '=') zoomCentre(1.6)
          if (e.key === '-') zoomCentre(1 / 1.6)
        }}>
        {view && (
          <div className="dz-plane" style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.s})` }}>
            {shown.map((t) => {
              const key = `${t.level}-${t.col}-${t.row}`
              return <img key={key} src={tileUrl(t.level, t.col, t.row)} style={tileStyle(t)} alt="" draggable={false} onLoad={() => markLoaded(key)} />
            })}
            {layers.flatMap((layer) => sharp.map((t) => (
              <img key={`${layer}-${t.level}-${t.col}-${t.row}`} src={tileUrl(t.level, t.col, t.row, layer)} style={tileStyle(t)} alt="" draggable={false} />
            )))}
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
