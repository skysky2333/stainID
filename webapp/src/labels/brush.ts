/* The brush of the field tool (key B): people change the existing bad areas, they do not draw separate new ones.

   For each type (`EXCLUDE_TYPES`: 1 fold or tissue damage, 2 artifact, 3 no tissue, 4 out of focus or bad staining;
   the keys 1–4 choose it while the brush is on, and the class keys rest) the current area is detection's area of that
   type with everyone's saved edits merged in (GET …/fields/{tile}/areas, `exclusions.final_say`). A drag adds the
   brushed pixels to the chosen type's area (a union: one region, however many strokes; pixels of another type are
   given to the chosen one); Shift-drag takes them out of the chosen type's area, detection's included. While the brush
   is on, the chosen type's current area is drawn as one merged region (the others faintly). Tab makes the brush
   smaller, Shift+Tab bigger (the size shows in the panel and as the ring under the mouse); the wheel always zooms.
   Each stroke is one step Z takes back.

   The masks are a quarter of the field picture in each direction: per type what was added and what was taken out since
   the last Save, over the type's current area (`base`). Save sends them as PNGs ("1"–"4" added, "-1"–"-4" taken out;
   labels/exclusions.py turns them into one edit in the label log, resolved per type into masks by
   `stainid.exclusions`); then they are emptied (`reset`) once the server's current areas have the edit. People's
   areas teach and test bad-area detection and leave the objects in them out of training and scoring; the analysis
   never reads them (it keeps detection's own, drawn hatched as "Excluded from analysis"). */
import { useCallback, useEffect, useRef, useState } from 'react'
import type { FieldAreas } from '../api'

export const BRUSH_SCALE = 4  // the masks are a quarter of the field picture in each direction
export const BRUSH_UM = { min: 5, max: 400, start: 40 }  // brush diameter, µm
export const BRUSH_STEP = 1.2  // Tab / Shift+Tab: the diameter this much smaller / bigger
export const INCLUDE = 'include'
/** The types of excluded area, with their colours (the same as the layer's, store_view.EXCLUDED_KINDS). */
export const EXCLUDE_TYPES: { key: string; name: string; colour: string }[] = [
  { key: '1', name: 'fold or tissue damage', colour: '#ffb000' },
  { key: '2', name: 'artifact (scratch, bubble, debris, linear)', colour: '#a855f7' },
  { key: '3', name: 'no tissue / empty', colour: '#9ca3af' },
  { key: '4', name: 'out of focus or bad staining', colour: '#00e5ff' },
]
/** Taking out (Shift-drag): the brush cursor's colour; parts people marked fine elsewhere ("Mark bad areas") too. */
export const CLEARED = { key: INCLUDE, name: 'marked fine', colour: '#22c55e' }
const TYPES = EXCLUDE_TYPES.map((t) => t.key)

/** The brush diameter after Tab (smaller) or Shift+Tab (bigger), within `BRUSH_UM`. */
export function brushStep(um: number, bigger: boolean): number {
  return Math.round(Math.min(BRUSH_UM.max, Math.max(BRUSH_UM.min, bigger ? um * BRUSH_STEP : um / BRUSH_STEP)))
}

function painted(c: HTMLCanvasElement): boolean {
  const data = c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data
  for (let i = 3; i < data.length; i += 4) if (data[i] > 127) return true
  return false
}

type Masks = Record<string, HTMLCanvasElement>  // "1"–"4": added since Save; "-1"–"-4": taken out since Save
type Snapshot = Record<string, ImageData>

export function useBrush(size: { w: number; h: number } | null, areas: FieldAreas | null) {
  const masks = useRef<Masks | null>(null)
  const base = useRef<Masks | null>(null)  // each type's current area as the server has it
  const scratch = useRef<HTMLCanvasElement | null>(null)
  const stroke = useRef<{ kind: string; erase: boolean; last: number[]; before: Snapshot; radius: number } | null>(null)
  const [current, setCurrent] = useState<Record<string, string>>({})  // per type: its area now (data URL), with the strokes since Save
  const [dirty, setDirty] = useState(false)

  const canvas = (w: number, h: number) => { const c = document.createElement('canvas'); c.width = w; c.height = h; return c }
  const show = useCallback(() => {
    const m = masks.current, b = base.current
    if (!m || !b) return
    const any = Object.values(m).some(painted)
    const out: Record<string, string> = {}
    for (const k of TYPES) {
      const c = scratch.current ?? (scratch.current = canvas(m[k].width, m[k].height))
      const g = c.getContext('2d')!
      g.globalCompositeOperation = 'source-over'
      g.clearRect(0, 0, c.width, c.height)
      g.drawImage(b[k], 0, 0)
      g.drawImage(m[k], 0, 0)  // added to it
      g.globalCompositeOperation = 'destination-out'
      g.drawImage(m[`-${k}`], 0, 0)  // taken out of it
      for (const j of TYPES) if (j !== k) g.drawImage(m[j], 0, 0)  // given to another type
      g.globalCompositeOperation = 'source-over'
      if (painted(c)) out[k] = c.toDataURL('image/png')
    }
    setCurrent(out)
    setDirty(any)
  }, [])

  useEffect(() => {  // empty masks the size of the field
    if (!size || masks.current) return
    const w = Math.max(1, Math.ceil(size.w / BRUSH_SCALE)), h = Math.max(1, Math.ceil(size.h / BRUSH_SCALE))
    masks.current = Object.fromEntries([...TYPES, ...TYPES.map((k) => `-${k}`)].map((k) => [k, canvas(w, h)]))
    base.current = Object.fromEntries(TYPES.map((k) => [k, canvas(w, h)]))
    show()
  }, [size, show])

  useEffect(() => {  // the current areas from the server, drawn onto the masks' grid
    const b = base.current
    if (!b || !areas) return
    let live = true
    const jobs = TYPES.map((k) => new Promise<void>((done) => {
      const c = b[k]
      const g = c.getContext('2d')!
      g.clearRect(0, 0, c.width, c.height)
      const url = areas.kinds?.[k]
      if (!url || areas.status !== 'done') { done(); return }
      const image = new Image()
      image.onload = () => {
        if (live) {
          g.imageSmoothingEnabled = false
          g.drawImage(image, (areas.x ?? 0) / BRUSH_SCALE, (areas.y ?? 0) / BRUSH_SCALE, (areas.width ?? 0) / BRUSH_SCALE, (areas.height ?? 0) / BRUSH_SCALE)
        }
        done()
      }
      image.onerror = () => done()
      image.src = url
    }))
    void Promise.all(jobs).then(() => { if (live) show() })
    return () => { live = false }
  }, [areas, size, show])

  /** One dab: painting `kind` adds to it (and takes the pixels off the other types and off what was taken out of
   *  it); erasing takes the pixels out of `kind` (also out of what was added to it). */
  const dab = (from: number[], to: number[], radius: number, kind: string, erase: boolean) => {
    const m = masks.current
    if (!m) return
    const draw = (k: string, on: boolean) => {
      const g = m[k].getContext('2d')!
      g.globalCompositeOperation = on ? 'source-over' : 'destination-out'
      g.strokeStyle = '#ffffff'
      g.lineCap = 'round'
      g.lineWidth = Math.max(1, (2 * radius) / BRUSH_SCALE)
      g.beginPath()
      g.moveTo(from[0] / BRUSH_SCALE, from[1] / BRUSH_SCALE)
      g.lineTo(to[0] / BRUSH_SCALE + 0.01, to[1] / BRUSH_SCALE)
      g.stroke()
      g.globalCompositeOperation = 'source-over'
    }
    if (erase) { draw(`-${kind}`, true); draw(kind, false); return }
    draw(kind, true)
    draw(`-${kind}`, false)
    for (const j of TYPES) if (j !== kind) draw(j, false)
  }

  const snapshot = (): Snapshot => Object.fromEntries(Object.entries(masks.current ?? {}).map(([k, c]) => [k, c.getContext('2d')!.getImageData(0, 0, c.width, c.height)]))
  /** A stroke starts at this field point (`radius` in field pixels): adding to `kind`, or with `erase` taking out of it. */
  const begin = useCallback((x: number, y: number, radius: number, kind: string, erase: boolean) => {
    if (!masks.current) return
    stroke.current = { kind, erase, last: [x, y], radius, before: snapshot() }
    dab([x, y], [x, y], radius, kind, erase)
    show()
  }, [show])
  const frame = useRef(0)
  const extend = useCallback((x: number, y: number) => {
    const s = stroke.current
    if (!s) return
    dab(s.last, [x, y], s.radius, s.kind, s.erase)
    s.last = [x, y]
    if (!frame.current) frame.current = requestAnimationFrame(() => { frame.current = 0; show() })  // at most once a frame
  }, [show])
  /** The stroke ends: the masks as they were before it (for Z), or null when no stroke was on. */
  const end = useCallback((): Snapshot | null => {
    const s = stroke.current
    stroke.current = null
    if (s) show()
    return s ? s.before : null
  }, [show])
  const restore = useCallback((before: Snapshot) => {
    const m = masks.current
    if (!m) return
    for (const [k, data] of Object.entries(before)) m[k]?.getContext('2d')!.putImageData(data, 0, 0)
    show()
  }, [show])
  /** Empty masks (after Save, once the server's current areas have the edit; or "Reset field"). */
  const reset = useCallback(() => {
    for (const c of Object.values(masks.current ?? {})) c.getContext('2d')!.clearRect(0, 0, c.width, c.height)
    show()
  }, [show])
  /** What Save sends: a PNG per type with something added ("1"–"4") or taken out ("-1"–"-4"). */
  const pngs = useCallback(() => {
    const m = masks.current ?? {}
    return Object.fromEntries(Object.entries(m).filter(([, c]) => painted(c)).map(([k, c]) => [k, c.toDataURL('image/png')]))
  }, [])
  return { current, dirty, painting: () => !!stroke.current, begin, extend, end, restore, reset, pngs }
}
