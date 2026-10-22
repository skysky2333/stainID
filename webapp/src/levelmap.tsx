/* Three zoom levels side by side, like a genome browser: the whole core (overview), a middle level around the place
   worked on, and the detailed view (the field page's ZoomViewer, the Cores view's DeepZoom). Each smaller panel draws
   the next level's extent as a rectangle; a click or a drag on a panel moves the next level there (the detailed view
   follows, and with it the middle level), and panning or zooming the detailed view moves the rectangles. One source
   of truth: the detailed view's centre and extent; the middle level is always centred on it. Outlines, states and the
   lock show only in the detailed view: the panels show the picture, the session's field squares and the rectangles. */
import { useRef } from 'react'
import type { ReactNode } from 'react'

export interface Rect { x0: number; y0: number; x1: number; y1: number }
/** A square of the overview: a field of the session at its place in the core, coloured by its state. */
export interface Square { key: string; x: number; y: number; w: number; h: number; colour: string; current?: boolean; title?: string; onOpen?: () => void }

/** The middle level's window: `times` the detailed view's extent (at least `least` across), square, centred on it, kept
 *  inside the picture (`width` × `height`). */
export function middleWindow(view: Rect, width: number, height: number, times = 3.5, least = 0): Rect {
  const cx = (view.x0 + view.x1) / 2, cy = (view.y0 + view.y1) / 2
  const side = Math.min(Math.max(width, height), Math.max(least, (view.x1 - view.x0) * times, (view.y1 - view.y0) * times))
  const half = side / 2
  const x0 = Math.max(0, Math.min(cx - half, width - side)), y0 = Math.max(0, Math.min(cy - half, height - side))
  return { x0, y0, x1: x0 + side, y1: y0 + side }
}

/** One panel: the picture (`src`, `width` × `height` in the coordinates everything else is in) shown over `window`,
 *  the squares, and `next` (the next level's extent). A press or a drag reports where the next level should be centred. */
export function LevelMap({ src, width, height, window: win, next, squares = [], onMove, label, children }: {
  src: string; width: number; height: number; window: Rect; next: Rect | null; squares?: Square[]
  onMove: (x: number, y: number, done: boolean) => void; label: string; children?: ReactNode
}) {
  const svg = useRef<SVGSVGElement>(null)
  const press = useRef<{ x: number; y: number; moved: boolean; square?: Square } | null>(null)
  const at = (e: React.PointerEvent) => {
    const m = svg.current?.getScreenCTM()
    if (!m) return null
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(m.inverse())
    return { x: p.x, y: p.y }
  }
  const w = win.x1 - win.x0, h = win.y1 - win.y0
  const stroke = w / 160
  return (
    <svg ref={svg} className="level-map" viewBox={`${win.x0} ${win.y0} ${w} ${h}`} preserveAspectRatio="xMidYMid meet" role="img" aria-label={label}
      onPointerDown={(e) => {
        if (e.button !== 0) return
        const p = at(e)
        if (!p) return
        e.currentTarget.setPointerCapture(e.pointerId)
        const square = squares.find((s) => !s.current && s.onOpen && p.x >= s.x && p.x <= s.x + s.w && p.y >= s.y && p.y <= s.y + s.h)
        press.current = { x: e.clientX, y: e.clientY, moved: false, square }
        if (!square) onMove(p.x, p.y, false)
      }}
      onPointerMove={(e) => {
        const d = press.current
        if (!d) return
        if (Math.hypot(e.clientX - d.x, e.clientY - d.y) > 4) d.moved = true
        const p = at(e)
        if (p && (d.moved || !d.square)) onMove(p.x, p.y, false)
      }}
      onPointerUp={(e) => {
        const d = press.current
        press.current = null
        if (!d) return
        if (d.square && !d.moved) { d.square.onOpen?.(); return }
        const p = at(e)
        if (p) onMove(p.x, p.y, true)
      }}
      onPointerCancel={() => { press.current = null }}>
      <rect x={0} y={0} width={width} height={height} fill="var(--surface-0)" />
      <image href={src} x={0} y={0} width={width} height={height} preserveAspectRatio="none" />
      {squares.map((s) => (
        <rect key={s.key} x={s.x} y={s.y} width={s.w} height={s.h} fill={s.colour} fillOpacity={s.current ? 0 : 0.22} stroke={s.current ? 'var(--accent)' : s.colour}
          strokeWidth={stroke * (s.current ? 2.2 : 1)} className={`level-square${s.current ? ' current' : ''}${s.onOpen ? ' opens' : ''}`}>
          {s.title && <title>{s.title}</title>}
        </rect>
      ))}
      {children}
      {next && <>
        <rect x={next.x0} y={next.y0} width={next.x1 - next.x0} height={next.y1 - next.y0} fill="none" stroke="#000" strokeOpacity={0.55} strokeWidth={stroke * 3} />
        <rect className="level-next" x={next.x0} y={next.y0} width={next.x1 - next.x0} height={next.y1 - next.y0} fill="#ffffff" fillOpacity={0.08}
          stroke="#ffffff" strokeWidth={stroke * 1.4} />
      </>}
    </svg>
  )
}
