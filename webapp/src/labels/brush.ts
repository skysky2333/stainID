/* The exclusion brush of the field tool (key E): a person paints the bad areas of a field on masks at a quarter of the
   field picture's size, one mask per type (`EXCLUDE_TYPES`: 1 fold or tissue damage, 2 artifact, 3 no tissue, 4 out of
   focus or bad staining; the keys 1–4 choose it while the brush is on). A drag paints the chosen type, Shift-drag erases
   and marks the part as "fine" (an include override, also over what detection excluded by itself); each stroke is one
   step Z takes back. The masks are sent as PNGs with the field's other changes on Save, and the server keeps them as
   typed rings in field pixels (labels/exclusions.py), which come back here when the field opens again. Objects whose
   centre lies in a painted area are left out of scoring and training. */
import { useCallback, useEffect, useRef, useState } from 'react'

export const BRUSH_SCALE = 4  // the masks are a quarter of the field picture in each direction
export const BRUSH_UM = { min: 5, max: 400, start: 40 }  // brush diameter, µm
export const INCLUDE = 'include'
/** The types of excluded area, with their colours (the same as the Cores view's layer, store_view.TYPE_COLOR). */
export const EXCLUDE_TYPES: { key: string; name: string; colour: string }[] = [
  { key: '1', name: 'fold or tissue damage', colour: '#ffb000' },
  { key: '2', name: 'artifact (scratch, bubble, debris, linear)', colour: '#a855f7' },
  { key: '3', name: 'no tissue / empty', colour: '#9ca3af' },
  { key: '4', name: 'out of focus or bad staining', colour: '#00e5ff' },
]
export const CLEARED = { key: INCLUDE, name: 'fine after all (cleared)', colour: '#22c55e' }
const KINDS = [...EXCLUDE_TYPES.map((t) => t.key), INCLUDE]

/** A saved mark: its rings in field pixels, each ring's type and whether it is a hole (`value` "kinds:1,2;holes:0,1"). */
export interface MarkedArea { points: number[][][] | null; value: string }

function parts(saved: MarkedArea | null): { ring: number[][]; hole: boolean; kind: string }[] {
  const fields = Object.fromEntries((saved?.value ?? '').split(';').filter((p) => p.includes(':')).map((p) => [p.slice(0, p.indexOf(':')), p.slice(p.indexOf(':') + 1)]))
  const holes = (fields.holes ?? '').split(',')
  const kinds = (fields.kinds ?? '').split(',')
  return (saved?.points ?? []).map((ring, i) => ({ ring, hole: holes[i] === '1', kind: kinds[i] || '1' }))
}

export function useBrush(size: { w: number; h: number } | null, saved: MarkedArea | null) {
  const canvases = useRef<Record<string, HTMLCanvasElement> | null>(null)
  const stroke = useRef<{ kind: string; clear: boolean; last: number[]; before: Record<string, ImageData>; radius: number } | null>(null)
  const [urls, setUrls] = useState<Record<string, string>>({})

  const show = useCallback(() => {
    const all = canvases.current
    if (!all) return
    const out: Record<string, string> = {}
    for (const [kind, c] of Object.entries(all)) {
      const data = c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data
      let any = false
      for (let i = 3; i < data.length; i += 4) if (data[i] > 127) { any = true; break }
      if (any) out[kind] = c.toDataURL('image/png')
    }
    setUrls(out)
  }, [])

  useEffect(() => {  // the masks, with what was saved painted on them
    if (!size || canvases.current) return
    const all: Record<string, HTMLCanvasElement> = {}
    for (const kind of KINDS) {
      const c = document.createElement('canvas')
      c.width = Math.max(1, Math.ceil(size.w / BRUSH_SCALE))
      c.height = Math.max(1, Math.ceil(size.h / BRUSH_SCALE))
      all[kind] = c
    }
    for (const { ring, hole, kind } of parts(saved)) {
      const c = all[kind] ?? all['1']
      if (!ring || ring.length < 3) continue
      const g = c.getContext('2d')!
      g.globalCompositeOperation = hole ? 'destination-out' : 'source-over'
      g.fillStyle = '#ffffff'
      g.beginPath()
      ring.forEach(([x, y], k) => (k ? g.lineTo(x / BRUSH_SCALE, y / BRUSH_SCALE) : g.moveTo(x / BRUSH_SCALE, y / BRUSH_SCALE)))
      g.closePath()
      g.fill()
      g.globalCompositeOperation = 'source-over'
    }
    canvases.current = all
    show()
  }, [size, saved, show])

  /** One dab of the brush: the chosen type painted (and taken off the others), or — clearing — every type erased and the
   *  part marked as fine. */
  const dab = (from: number[], to: number[], radius: number, kind: string, clear: boolean) => {
    const all = canvases.current
    if (!all) return
    for (const [k, c] of Object.entries(all)) {
      const g = c.getContext('2d')!
      const paint = clear ? k === INCLUDE : k === kind
      g.globalCompositeOperation = paint ? 'source-over' : 'destination-out'
      g.strokeStyle = '#ffffff'
      g.lineCap = 'round'
      g.lineWidth = Math.max(1, (2 * radius) / BRUSH_SCALE)
      g.beginPath()
      g.moveTo(from[0] / BRUSH_SCALE, from[1] / BRUSH_SCALE)
      g.lineTo(to[0] / BRUSH_SCALE + 0.01, to[1] / BRUSH_SCALE)
      g.stroke()
      g.globalCompositeOperation = 'source-over'
    }
  }

  /** A stroke starts at this field point (`radius` in field pixels): painting `kind`, or with `clear` marking as fine. */
  const begin = useCallback((x: number, y: number, radius: number, kind: string, clear: boolean) => {
    const all = canvases.current
    if (!all) return
    const before = Object.fromEntries(Object.entries(all).map(([k, c]) => [k, c.getContext('2d')!.getImageData(0, 0, c.width, c.height)]))
    stroke.current = { kind, clear, last: [x, y], radius, before }
    dab([x, y], [x, y], radius, kind, clear)
    show()
  }, [show])
  const frame = useRef(0)
  const extend = useCallback((x: number, y: number) => {
    const s = stroke.current
    if (!s) return
    dab(s.last, [x, y], s.radius, s.kind, s.clear)
    s.last = [x, y]
    if (!frame.current) frame.current = requestAnimationFrame(() => { frame.current = 0; show() })  // at most once a frame
  }, [show])
  /** The stroke ends: the masks as they were before it (for Z), or null when no stroke was on. */
  const end = useCallback((): Record<string, ImageData> | null => {
    const s = stroke.current
    stroke.current = null
    if (s) show()
    return s ? s.before : null
  }, [show])
  const restore = useCallback((before: Record<string, ImageData>) => {
    const all = canvases.current
    if (!all) return
    for (const [k, data] of Object.entries(before)) all[k]?.getContext('2d')!.putImageData(data, 0, 0)
    show()
  }, [show])
  /** What Save sends: one PNG per type, "" for a type with nothing painted (no type painted at all: the mark goes). */
  const pngs = useCallback(() => Object.fromEntries(KINDS.map((k) => [k, urls[k] ? canvases.current![k].toDataURL('image/png') : ''])), [urls])
  return { urls, painted: Object.keys(urls).length > 0, painting: () => !!stroke.current, begin, extend, end, restore, pngs }
}
