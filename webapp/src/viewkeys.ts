import { useEffect, useRef } from 'react'

/** Every image viewer (fields: viewer.tsx ZoomViewer; core images: deepzoom.tsx DeepZoom) zooms up to this many times
 *  the picture as it first fits. */
export const MAX_ZOOM = 10
/** One + / − key press (buttons step 1.5×). */
export const KEY_ZOOM = 1.25
/** The arrow keys move the picture this many screen px a second while held. */
const PAN_PX_PER_S = 900
const ARROWS: Record<string, [number, number]> = { ArrowLeft: [1, 0], ArrowRight: [-1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] }

/** The viewers on the page, oldest first; the keys act on the one last pointed at, else the newest. */
const viewers: symbol[] = []
let active: symbol | null = null

function typing(target: EventTarget | null): boolean {
  const t = target as HTMLElement | null
  return !!t && (['INPUT', 'TEXTAREA', 'SELECT'].includes(t.tagName) || t.isContentEditable)
}

export interface ViewKeys {
  /** move the picture by this many screen px (positive: the picture moves right / down, showing what is left / above) */
  panBy: (dx: number, dy: number) => void
  zoomBy: (factor: number) => void
}

/**
 * The keys every image viewer shares: the arrow keys move the picture smoothly while held (one step per animation
 * frame, stopping on release), + / − (and =) zoom in and out about the middle. Not while typing in a field, nor with
 * Ctrl, Cmd or Alt (Shift+arrows are left to the page). Several viewers on a page: the one last pointed at (`activate`).
 */
export function useViewKeys(actions: ViewKeys): { activate: () => void } {
  const me = useRef(Symbol('viewer'))
  const latest = useRef(actions)
  latest.current = actions
  useEffect(() => {
    const id = me.current
    viewers.push(id)
    const held = new Set<string>()
    let frame = 0
    let last = 0
    const mine = () => (active !== null && viewers.includes(active) ? active : viewers[viewers.length - 1]) === id
    const step = (t: number) => {
      const dt = last ? Math.min(0.05, (t - last) / 1000) : 1 / 60
      last = t
      let dx = 0, dy = 0
      for (const key of held) { dx += ARROWS[key][0]; dy += ARROWS[key][1] }
      if (dx || dy) latest.current.panBy(dx * PAN_PX_PER_S * dt, dy * PAN_PX_PER_S * dt)
      if (held.size) frame = requestAnimationFrame(step)
      else { frame = 0; last = 0 }
    }
    const down = (e: KeyboardEvent) => {
      if (!mine() || typing(e.target) || e.ctrlKey || e.metaKey || e.altKey) return
      if (ARROWS[e.key] && !e.shiftKey) {
        e.preventDefault()
        held.add(e.key)
        if (!frame) frame = requestAnimationFrame(step)
        return
      }
      if (e.key === '+' || e.key === '=') { e.preventDefault(); latest.current.zoomBy(KEY_ZOOM) }
      else if (e.key === '-' || e.key === '_' || e.key === '−') { e.preventDefault(); latest.current.zoomBy(1 / KEY_ZOOM) }
    }
    const up = (e: KeyboardEvent) => { held.delete(e.key) }
    const stop = () => held.clear()
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    window.addEventListener('blur', stop)
    return () => {
      viewers.splice(viewers.indexOf(id), 1)
      if (active === id) active = null
      cancelAnimationFrame(frame)
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
      window.removeEventListener('blur', stop)
    }
  }, [])
  return { activate: () => { active = me.current } }
}

/** An outline's width on screen (px) at a zoom (1 = the picture as it first fits): thin at 1×, a little thicker zoomed
 *  in (up to 1.6× the base at 10×), so outlines stay light on the tissue and readable up close. */
export function lineScale(zoom: number): number {
  return Math.min(1.6, Math.max(1, Math.pow(Math.max(1, zoom), 0.2)))
}
/** The base widths (screen px at 1×): an object's outline, the dark halo under it, the one under the mouse. */
export const OUTLINE_PX = 1.35
export const HALO_PX = 1.1
export const LIT_PX = 2.2
