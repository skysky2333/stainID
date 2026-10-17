/* The exclusion brush of the field tool (key E): an edit of the excluded areas, painted on masks at a quarter of the field
   picture's size, one mask per type (`EXCLUDE_TYPES`: 1 fold or tissue damage, 2 artifact, 3 no tissue, 4 out of focus
   or bad staining; the keys 1–4 choose it while the brush is on). A drag paints the chosen type, Shift-drag clears: the
   part is counted, whatever detection or anyone earlier said. Each stroke is one step Z takes back.

   The masks hold only what was painted since the last Save: the page draws the field's part of the resolved layer from
   the server (detection's areas with every saved edit applied, the same pixels as the Cores view and the analysis) and
   these strokes over it, with the server picture cut away under them (`cover`), so the page shows what the layer will
   be. Save sends the masks as PNGs (labels/exclusions.py turns them into one new edit); then they are emptied (`reset`)
   once the server's layer has the edit. */
import { useCallback, useEffect, useRef, useState } from 'react'

export const BRUSH_SCALE = 4  // the masks are a quarter of the field picture in each direction
export const BRUSH_UM = { min: 5, max: 400, start: 40 }  // brush diameter, µm
export const INCLUDE = 'include'
/** The types of excluded area, with their colours (the same as the layer's, store_view.EXCLUDED_KINDS). */
export const EXCLUDE_TYPES: { key: string; name: string; colour: string }[] = [
  { key: '1', name: 'fold or tissue damage', colour: '#ffb000' },
  { key: '2', name: 'artifact (scratch, bubble, debris, linear)', colour: '#a855f7' },
  { key: '3', name: 'no tissue / empty', colour: '#9ca3af' },
  { key: '4', name: 'out of focus or bad staining', colour: '#00e5ff' },
]
/** Clearing (Shift-drag): only the brush cursor shows this colour; a cleared part simply shows no hatch. */
export const CLEARED = { key: INCLUDE, name: 'clear: counted after all', colour: '#22c55e' }
const KINDS = [...EXCLUDE_TYPES.map((t) => t.key), INCLUDE]

function painted(c: HTMLCanvasElement): boolean {
  const data = c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data
  for (let i = 3; i < data.length; i += 4) if (data[i] > 127) return true
  return false
}

export function useBrush(size: { w: number; h: number } | null) {
  const canvases = useRef<Record<string, HTMLCanvasElement> | null>(null)
  const stroke = useRef<{ kind: string; clear: boolean; last: number[]; before: Record<string, ImageData>; radius: number } | null>(null)
  const [urls, setUrls] = useState<Record<string, string>>({})  // per type (and include) with anything painted
  const [cover, setCover] = useState<string | null>(null)  // black wherever anything was painted or cleared: cut from the server's picture

  const show = useCallback(() => {
    const all = canvases.current
    if (!all) return
    const out: Record<string, string> = {}
    let union: HTMLCanvasElement | null = null
    for (const [kind, c] of Object.entries(all)) {
      if (!painted(c)) continue
      out[kind] = c.toDataURL('image/png')
      if (!union) {
        union = document.createElement('canvas')
        union.width = c.width
        union.height = c.height
      }
      union.getContext('2d')!.drawImage(c, 0, 0)
    }
    if (union) {
      const g = union.getContext('2d')!
      g.globalCompositeOperation = 'source-in'
      g.fillStyle = '#000000'
      g.fillRect(0, 0, union.width, union.height)
    }
    setUrls(out)
    setCover(union ? union.toDataURL('image/png') : null)
  }, [])

  useEffect(() => {  // empty masks the size of the field
    if (!size || canvases.current) return
    const all: Record<string, HTMLCanvasElement> = {}
    for (const kind of KINDS) {
      const c = document.createElement('canvas')
      c.width = Math.max(1, Math.ceil(size.w / BRUSH_SCALE))
      c.height = Math.max(1, Math.ceil(size.h / BRUSH_SCALE))
      all[kind] = c
    }
    canvases.current = all
    show()
  }, [size, show])

  /** One dab of the brush: the chosen type painted (and taken off the others), or — clearing — every type erased and the
   *  part marked as counted. */
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

  /** A stroke starts at this field point (`radius` in field pixels): painting `kind`, or with `clear` clearing. */
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
  /** Empty masks (after Save, once the server's layer has the edit). */
  const reset = useCallback(() => {
    for (const c of Object.values(canvases.current ?? {})) c.getContext('2d')!.clearRect(0, 0, c.width, c.height)
    show()
  }, [show])
  /** What Save sends: one PNG per type, "" for a type with nothing painted. */
  const pngs = useCallback(() => Object.fromEntries(KINDS.map((k) => [k, urls[k] ? canvases.current![k].toDataURL('image/png') : ''])), [urls])
  return { urls, cover, painted: Object.keys(urls).length > 0, painting: () => !!stroke.current, begin, extend, end, restore, reset, pngs }
}
