import { useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'

export const CLASS_STYLE: Record<string, { color: string; label: string; short?: string }> = {
  neuron: { color: '#00d5ff', label: 'NeuN+ neuron', short: 'neuron' },
  rejected: { color: '#ff3b3b', label: 'rejected candidate', short: 'rejected' },
  compact: { color: '#ff00c8', label: 'compact plaque', short: 'compact' },
  diffuse: { color: '#00dcff', label: 'diffuse plaque', short: 'diffuse' },
  small_plaque: { color: '#c8c8c8', label: 'small plaque (10–15 µm)', short: 'small' },
  tau_neuron_ring: { color: '#00e050', label: 'tau+ neuron (nucleus-ring)', short: 'ring' },
  tau_neuron_dense: { color: '#ffd000', label: 'tau+ neuron (dense body)', short: 'dense body' },
  unproposed: { color: '#ff9f1a', label: 'stained but never proposed', short: 'not proposed' },
}

export const RASTER_LAYERS: Record<string, { label: string; color: string }> = {
  dab: { label: 'DAB above slide threshold', color: '#ff2828' },
  exclusions: { label: 'excluded (non-tissue, artifacts, folds)', color: '#5a6eff' },
  threads: { label: 'neuropil threads (computed on demand)', color: '#286eff' },
}

export interface ImagePoint { x: number; y: number; unit: number }

/**
 * A field image with raster layers and an SVG overlay in image pixel coordinates. Scroll zooms towards the pointer,
 * dragging (any button, the middle one too) pans; a left or right press without dragging is reported as a click in image
 * pixels (`unit` = image pixels per screen pixel). The page never scrolls from it: the wheel is taken, and the middle
 * button's autoscroll never starts. With `fit`, the picture is as large as fits the space its card is given (the
 * annotation pages size that to the window), instead of the card's width.
 */
export function ZoomViewer({ src, layers = [], title, focus, cursor, onClick, onMove, children, fit = false, onPress, onRelease, onWheelOver, onSize, panLock = false }: {
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
}) {
  const [view, setView] = useState({ zoom: 1, x: 0, y: 0 })
  const [size, setSize] = useState({ w: 2304, h: 2304 })
  const [loaded, setLoaded] = useState(false)
  const [width, setWidth] = useState(800)
  const drag = useRef<{ x: number; y: number; ox: number; oy: number; moved: boolean; pan: boolean } | null>(null)
  const box = useRef<HTMLDivElement>(null)
  const room = useRef<HTMLDivElement>(null)
  const [space, setSpace] = useState<{ w: number; h: number } | null>(null)  // the room a fitted picture has
  const { zoom } = view
  const viewNow = useRef(view)
  viewNow.current = view
  const before = useRef<{ zoom: number; x: number; y: number } | null>(null)  // the view before a `focus` zoomed in
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

  useEffect(() => {
    const element = box.current
    if (!element) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      if (wheelOver.current?.(e.deltaY)) return
      const rect = element.getBoundingClientRect()
      const px = e.clientX - rect.left
      const py = e.clientY - rect.top
      setView((v) => {
        const next = Math.min(16, Math.max(1, v.zoom * (e.deltaY < 0 ? 1.15 : 1 / 1.15)))
        if (next === 1) return { zoom: 1, x: 0, y: 0 }
        return { zoom: next, x: px - ((px - v.x) * next) / v.zoom, y: py - ((py - v.y) * next) / v.zoom }
      })
    }
    element.addEventListener('wheel', onWheel, { passive: false })
    const resize = new ResizeObserver(() => setWidth(element.clientWidth))
    resize.observe(element)
    return () => { element.removeEventListener('wheel', onWheel); resize.disconnect() }
  }, [])

  useEffect(() => {
    if (!focus) return
    if (focus.back) {  // back to exactly where the person was before the zoom-in
      if (before.current) setView(before.current)
      before.current = null
      return
    }
    before.current = before.current ?? viewNow.current
    const width = box.current?.clientWidth ?? 600
    const scale = width / size.w
    setView({ zoom: 4, x: width / 2 - focus.x * scale * 4, y: width / 2 - focus.y * scale * 4 })
  }, [focus, size.w])

  const zoomBy = (factor: number) => setView((v) => {
    const width = box.current?.clientWidth ?? 600
    const next = Math.min(16, Math.max(1, v.zoom * factor))
    if (next === 1) return { zoom: 1, x: 0, y: 0 }
    return { zoom: next, x: width / 2 - ((width / 2 - v.x) * next) / v.zoom, y: width / 2 - ((width / 2 - v.y) * next) / v.zoom }
  })

  const toImage = (e: React.MouseEvent): ImagePoint => {
    const rect = box.current!.getBoundingClientRect()
    const unit = size.w / rect.width / zoom
    return { x: (e.clientX - rect.left - view.x) * unit, y: (e.clientY - rect.top - view.y) * unit, unit }
  }

  const ratio = size.w / Math.max(1, size.h)
  const fitted = fit && space && space.w > 0 && space.h > 0
    ? (space.w / space.h > ratio ? { width: Math.floor(space.h * ratio), height: space.h } : { width: space.w, height: Math.floor(space.w / ratio) })
    : null
  const viewer = (
    <div ref={box} className="viewer" style={{ ...(fitted ?? { aspectRatio: `${size.w} / ${size.h}` }), cursor: cursor ?? (zoom > 1 ? 'grab' : 'default') }}
      onMouseDown={(e) => {
        if (e.button === 1) e.preventDefault()  // no autoscroll: the middle button pans the picture
        drag.current = { x: e.clientX, y: e.clientY, ox: view.x, oy: view.y, moved: false, pan: !panLock || e.button === 1 }
        if (e.button !== 1) onPress?.(toImage(e), e)
      }}
      onAuxClick={(e) => e.preventDefault()}
      onMouseMove={(e) => {
        const d = drag.current
        if (d && Math.hypot(e.clientX - d.x, e.clientY - d.y) > 4) d.moved = true
        if (d?.moved && d.pan && zoom > 1) setView((v) => ({ ...v, x: d.ox + e.clientX - d.x, y: d.oy + e.clientY - d.y }))
        onMove?.(toImage(e))
      }}
      onMouseUp={(e) => {
        if (e.button !== 1) onRelease?.(toImage(e), e)
        if (drag.current && !drag.current.moved && e.button !== 1) onClick?.(toImage(e), e)
        drag.current = null
      }}
      onMouseLeave={(e) => { if (drag.current && onRelease) onRelease(toImage(e), e); drag.current = null }}>
      {!loaded && <div className="viewer-loading">Loading the field image… the first field of a core takes a few seconds; the others then open instantly.</div>}
      <div style={{ position: 'absolute', inset: 0, transform: `translate(${view.x}px, ${view.y}px) scale(${zoom})`, transformOrigin: '0 0' }}>
        <img src={src} alt="field" draggable={false} />
        {layers.map((layer) => <img key={layer} src={layer} alt="" draggable={false} style={{ imageRendering: 'pixelated' }} />)}
        <svg viewBox={`0 0 ${size.w} ${size.h}`}>{children(zoom, size, size.w / width / zoom)}</svg>
      </div>
    </div>
  )

  return (
    <div className={`card${fit ? ' viewer-card-fit' : ''}`}>
      <div className="row" style={{ marginBottom: 10 }}>
        {title}
        <span className="spacer" />
        <button className="btn small" onClick={() => zoomBy(1 / 1.5)}>−</button>
        <span className="small muted">{zoom.toFixed(1)}×</span>
        <button className="btn small" onClick={() => zoomBy(1.5)}>+</button>
        <button className="btn small" onClick={() => setView({ zoom: 1, x: 0, y: 0 })}>Reset</button>
      </div>
      {fit ? <div ref={room} className="viewer-room">{viewer}</div> : viewer}
    </div>
  )
}
