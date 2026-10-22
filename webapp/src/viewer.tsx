import { useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { PALETTE } from './palette'
import { MAX_ZOOM, useViewKeys } from './viewkeys'

/** The names of detection's own classes where no label class names them (the current outputs); their colours are the
 *  label classes' (classPalette.ts `MODEL_CLASSES`), these only for what has no class. */
export const CLASS_STYLE: Record<string, { color: string; label: string; short?: string }> = {
  neuron: { color: PALETTE.states.unknown, label: 'NeuN+ neuron', short: 'neuron' },
  rejected: { color: PALETTE.states.inactive, label: 'rejected candidate', short: 'rejected' },
  compact: { color: PALETTE.states.unknown, label: 'compact plaque', short: 'compact' },
  diffuse: { color: PALETTE.states.unknown, label: 'diffuse plaque', short: 'diffuse' },
  small_plaque: { color: PALETTE.states.unknown, label: 'small plaque (10–15 µm)', short: 'small' },
  tau_neuron_ring: { color: PALETTE.states.unknown, label: 'tau+ neuron', short: 'tau+ neuron' },
  tau_neuron_dense: { color: PALETTE.states.unknown, label: 'tau+ neuron', short: 'tau+ neuron' },
  unproposed: { color: PALETTE.overlays.unproposed, label: 'stained but never proposed', short: 'not proposed' },
}

export const RASTER_LAYERS: Record<string, { label: string; color: string }> = {
  dab: { label: 'DAB above slide threshold', color: PALETTE.overlays.dab },
  exclusions: { label: 'excluded (non-tissue, artifacts, folds)', color: PALETTE.overlays.excluded },
  threads: { label: 'neuropil threads (computed on demand)', color: PALETTE.overlays.threads },
}

export interface ImagePoint { x: number; y: number; unit: number }
/** What the smaller zoom levels may do to a viewer: centre it on a spot (image px), at least at a zoom. */
export interface ViewControl { centre: (x: number, y: number, zoom?: number) => void }

type View = { zoom: number; x: number; y: number }

/**
 * A field image with raster layers and an SVG overlay in image pixel coordinates. Scroll, + / − (keys and buttons)
 * zoom from the picture as it fits up to `MAX_ZOOM`; dragging (the left button where nothing else takes it, the middle
 * one always) and the arrow keys (held: smoothly, `useViewKeys`) pan. A left press without dragging is reported as a
 * click in image pixels (`unit` = image pixels per screen pixel); the right button calls `onContext` (a menu). The page
 * never scrolls from it: the wheel is taken, and the middle button's autoscroll never starts. With `fit`, the viewer
 * takes all the room its card is given (the annotation pages size that to the window), the picture fitted inside it
 * and centred, so a zoomed-in picture uses the whole room; else the picture is as wide as the card.
 */
export function ZoomViewer({ src, layers = [], title, focus, cursor, onClick, onMove, children, fit = false, onPress, onRelease, onWheelOver, onSize, panLock = false, under, paint, onContext,
  onView, control, aside }: {
  src: string
  layers?: string[]
  title: ReactNode
  /** zoom in on this spot (remembering the view before); with `back`, return to the view remembered */
  focus?: { x: number; y: number; key: number; back?: boolean } | null
  cursor?: string
  onClick?: (point: ImagePoint, event: React.MouseEvent) => void
  onMove?: (point: ImagePoint) => void
  children: (zoom: number, size: { w: number; h: number }, unit: number) => ReactNode
  fit?: boolean
  /** a button pressed / let go on the picture (painting with a brush); `panLock`: only the middle button moves the picture */
  onPress?: (point: ImagePoint, event: React.MouseEvent) => void
  onRelease?: (point: ImagePoint, event: React.MouseEvent) => void
  panLock?: boolean
  /** the wheel over the picture: return true when the page used it (e.g. the brush size), so it does not zoom */
  onWheelOver?: (deltaY: number) => boolean
  /** the picture's size in pixels, once known */
  onSize?: (size: { w: number; h: number }) => void
  /** drawn below `children` in a layer of its own (painted once, not again when what is above it changes) */
  under?: (zoom: number, size: { w: number; h: number }, unit: number) => ReactNode
  /** HTML (the brush's canvases) between `under` and `children`, in a layer of its own: a change to it repaints only it */
  paint?: ReactNode
  /** the right button on the picture (a menu at the pointer) */
  onContext?: (point: ImagePoint, event: React.MouseEvent) => void
  /** the part of the picture in view (image px), whenever it changes: the smaller zoom levels beside it (levelmap.tsx) */
  onView?: (rect: { x0: number; y0: number; x1: number; y1: number }) => void
  /** set here: `centre(x, y, zoom)` centres the view on a spot (image px), at least at `zoom` (the smaller zoom levels move it) */
  control?: { current: ViewControl | null }
  /** beside the picture, in the same room (the smaller zoom levels) */
  aside?: ReactNode
}) {
  // the view: `x`, `y` move the picture from where it sits centred at 1× (screen px), `zoom` scales it
  const [view, setView] = useState<View>({ zoom: 1, x: 0, y: 0 })
  const [size, setSize] = useState({ w: 2304, h: 2304 })
  const [loaded, setLoaded] = useState(false)
  const [boxSize, setBoxSize] = useState({ w: 800, h: 800 })
  const drag = useRef<{ x: number; y: number; ox: number; oy: number; moved: boolean; pan: boolean } | null>(null)
  const box = useRef<HTMLDivElement>(null)
  const room = useRef<HTMLDivElement>(null)
  const [space, setSpace] = useState<{ w: number; h: number } | null>(null)  // the room a fitted picture has
  const { zoom } = view
  const viewNow = useRef(view)
  viewNow.current = view
  const before = useRef<View | null>(null)  // the view before a `focus` zoomed in
  const wheelOver = useRef(onWheelOver)
  wheelOver.current = onWheelOver
  const sizeHeard = useRef(onSize)
  sizeHeard.current = onSize

  useEffect(() => {
    const element = room.current
    if (!fit || !element) return
    const measure = () => setSpace({ w: element.clientWidth, h: element.clientHeight })
    measure()
    const resize = new ResizeObserver(measure)
    resize.observe(element)
    return () => resize.disconnect()
  }, [fit])

  useEffect(() => {
    setLoaded(false)
    setView({ zoom: 1, x: 0, y: 0 })
    const img = new Image()
    img.onload = () => { setSize({ w: img.naturalWidth, h: img.naturalHeight }); setLoaded(true); sizeHeard.current?.({ w: img.naturalWidth, h: img.naturalHeight }) }
    img.src = src
  }, [src])

  const ratio = size.w / Math.max(1, size.h)
  const filled = fit && space && space.w > 0 && space.h > 0 ? space : null
  const shown = filled  // the picture at 1×, in screen px
    ? (filled.w / filled.h > ratio ? { width: Math.floor(filled.h * ratio), height: filled.h } : { width: filled.w, height: Math.floor(filled.w / ratio) })
    : { width: boxSize.w, height: boxSize.w / ratio }
  const origin = { x: Math.max(0, (boxSize.w - shown.width) / 2), y: Math.max(0, (boxSize.h - shown.height) / 2) }  // where it sits at 1×
  const layout = useRef({ origin, shown, boxSize })
  layout.current = { origin, shown, boxSize }

  /** The view zoomed by `factor` about a point of the viewer (screen px from its corner); 1× is the picture as it fits. */
  const zoomedAt = (v: View, factor: number, px: number, py: number): View => {
    const next = Math.min(MAX_ZOOM, Math.max(1, v.zoom * factor))
    if (next === 1) return { zoom: 1, x: 0, y: 0 }
    const o = layout.current.origin
    const ax = px - o.x, ay = py - o.y
    return { zoom: next, x: ax - ((ax - v.x) * next) / v.zoom, y: ay - ((ay - v.y) * next) / v.zoom }
  }

  useEffect(() => {
    const element = box.current
    if (!element) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      if (wheelOver.current?.(e.deltaY)) return
      const rect = element.getBoundingClientRect()
      const px = e.clientX - rect.left
      const py = e.clientY - rect.top
      setView((v) => zoomedAt(v, e.deltaY < 0 ? 1.15 : 1 / 1.15, px, py))
    }
    element.addEventListener('wheel', onWheel, { passive: false })
    const resize = new ResizeObserver(() => setBoxSize({ w: element.clientWidth, h: element.clientHeight }))
    resize.observe(element)
    return () => { element.removeEventListener('wheel', onWheel); resize.disconnect() }
  }, [])  // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!focus) return
    if (focus.back) {  // back to exactly where the person was before the zoom-in
      if (before.current) setView(before.current)
      before.current = null
      return
    }
    before.current = before.current ?? viewNow.current
    const { origin: o, shown: s, boxSize: b } = layout.current
    const scale = s.width / size.w
    setView({ zoom: 4, x: b.w / 2 - o.x - focus.x * scale * 4, y: b.h / 2 - o.y - focus.y * scale * 4 })
  }, [focus, size.w])

  const imageWidth = useRef(size.w)
  imageWidth.current = size.w
  const imageHeight = useRef(size.h)
  imageHeight.current = size.h
  useEffect(() => {  // a smaller zoom level moves the view: straight to the viewer, nothing else is drawn again
    if (!control) return
    control.current = {
      centre: (x, y, least = 1) => {
        const { origin: o, shown: s, boxSize: b } = layout.current
        const scale = s.width / imageWidth.current
        const z = Math.min(MAX_ZOOM, Math.max(least, viewNow.current.zoom))
        if (z <= 1) { setView({ zoom: 1, x: 0, y: 0 }); return }
        const w = imageWidth.current, h = imageHeight.current
        const keep = (at: number, half: number, most: number) => (most <= 2 * half ? most / 2 : Math.min(most - half, Math.max(half, at)))  // the view stays on the picture
        const cx = keep(x, b.w / 2 / (scale * z), w), cy = keep(y, b.h / 2 / (scale * z), h)
        setView({ zoom: z, x: b.w / 2 - o.x - cx * scale * z, y: b.h / 2 - o.y - cy * scale * z })
      },
    }
    return () => { control.current = null }
  }, [control])
  const viewHeard = useRef(onView)
  viewHeard.current = onView
  useEffect(() => {  // the part in view, for the smaller zoom levels
    if (!viewHeard.current || !loaded) return
    const per = size.w / Math.max(1, shown.width) / view.zoom
    const at = (sx: number, sy: number) => [(sx - origin.x - view.x) * per, (sy - origin.y - view.y) * per]
    const [x0, y0] = at(0, 0), [x1, y1] = at(boxSize.w, boxSize.h)
    viewHeard.current({ x0: Math.max(0, x0), y0: Math.max(0, y0), x1: Math.min(size.w, x1), y1: Math.min(size.h, y1) })
  }, [view, boxSize.w, boxSize.h, shown.width, origin.x, origin.y, size.w, size.h, loaded])

  const zoomBy = (factor: number) => setView((v) => zoomedAt(v, factor, layout.current.boxSize.w / 2, layout.current.boxSize.h / 2))
  const keys = useViewKeys({ zoomBy, panBy: (dx, dy) => setView((v) => (v.zoom > 1 ? { ...v, x: v.x + dx, y: v.y + dy } : v)) })

  const unit = size.w / Math.max(1, shown.width) / zoom  // image px per screen px
  const toImage = (e: React.MouseEvent): ImagePoint => {
    const rect = box.current!.getBoundingClientRect()
    return { x: (e.clientX - rect.left - origin.x - view.x) * unit, y: (e.clientY - rect.top - origin.y - view.y) * unit, unit }
  }
  const onPicture = (p: ImagePoint) => p.x >= 0 && p.y >= 0 && p.x <= size.w && p.y <= size.h  // not the room around it

  const viewer = (
    <div ref={box} className={`viewer${filled ? ' viewer-filled' : ''}`}
      style={{ ...(filled ? { width: filled.w, height: filled.h } : { aspectRatio: `${size.w} / ${size.h}` }), cursor: cursor ?? (zoom > 1 ? 'grab' : 'default') }}
      onContextMenu={(e) => { e.preventDefault(); if (onContext) { const p = toImage(e); if (onPicture(p)) onContext(p, e) } }}
      onMouseEnter={keys.activate}
      onMouseDown={(e) => {
        keys.activate()
        if (e.button === 2) return  // the right button: the menu (onContext); the middle one pans
        if (e.button === 1) e.preventDefault()  // no autoscroll: the middle button pans the picture
        drag.current = { x: e.clientX, y: e.clientY, ox: view.x, oy: view.y, moved: false, pan: !panLock || e.button === 1 }
        if (e.button !== 1 && onPress) { const p = toImage(e); if (onPicture(p)) onPress(p, e) }
      }}
      onAuxClick={(e) => e.preventDefault()}
      onMouseMove={(e) => {
        const d = drag.current
        if (d && Math.hypot(e.clientX - d.x, e.clientY - d.y) > 4) d.moved = true
        if (d?.moved && d.pan && zoom > 1) setView((v) => ({ ...v, x: d.ox + e.clientX - d.x, y: d.oy + e.clientY - d.y }))
        onMove?.(toImage(e))
      }}
      onMouseUp={(e) => {
        if (e.button === 2) return
        if (e.button !== 1) onRelease?.(toImage(e), e)
        if (drag.current && !drag.current.moved && e.button !== 1) { const p = toImage(e); if (onPicture(p)) onClick?.(p, e) }
        drag.current = null
      }}
      onMouseLeave={(e) => { if (drag.current && onRelease) onRelease(toImage(e), e); drag.current = null }}>
      {!loaded && <div className="viewer-loading">Loading the field image… the first field of a core takes a few seconds; the others then open instantly.</div>}
      <div className="viewer-plane" style={{ width: shown.width, height: shown.height, transform: `translate(${origin.x + view.x}px, ${origin.y + view.y}px) scale(${zoom})` }}>
        <img src={src} alt="field" draggable={false} />
        {layers.map((layer) => <img key={layer} src={layer} alt="" draggable={false} style={{ imageRendering: 'pixelated' }} />)}
        {under && <svg className="viewer-under" viewBox={`0 0 ${size.w} ${size.h}`}>{under(zoom, size, unit)}</svg>}
        {paint && <div className="viewer-paint">{paint}</div>}
        <svg viewBox={`0 0 ${size.w} ${size.h}`}>{children(zoom, size, unit)}</svg>
      </div>
    </div>
  )

  return (
    <div className={`card${fit ? ' viewer-card-fit' : ''}`}>
      <div className="row viewer-head">  {/* one line always: what it says never moves or resizes the picture */}
        <div className="viewer-title">{title}</div>
        <button className="btn small" onClick={() => zoomBy(1 / 1.5)} title="Zoom out (−)" aria-label="Zoom out">−</button>
        <span className="small muted viewer-zoom">{zoom.toFixed(1)}×</span>
        <button className="btn small" onClick={() => zoomBy(1.5)} title={`Zoom in (+), up to ${MAX_ZOOM}×`} aria-label="Zoom in">+</button>
        <button className="btn small" onClick={() => setView({ zoom: 1, x: 0, y: 0 })} title="The whole picture">Reset</button>
      </div>
      {fit ? <div className={`viewer-row${aside ? ' with-aside' : ''}`}>{aside}<div ref={room} className="viewer-room">{viewer}</div></div> : viewer}
    </div>
  )
}
