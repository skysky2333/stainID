/* The exclusion brush of the field tool (key E): a person paints the bad areas of a field (folds, smudges, empty glass…)
   on a mask at a quarter of the field picture's size. Left-drag paints, right-drag (or Shift-drag) erases; each stroke
   is one step Z takes back. The mask is sent as a PNG with the field's other changes on Save, and the server keeps it
   as rings in field pixels (labels/exclusions.py), which come back here when the field opens again. Objects whose
   centre lies in the painted area are left out of scoring and training. */
import { useCallback, useEffect, useRef, useState } from 'react'

export const BRUSH_SCALE = 4  // the mask is a quarter of the field picture in each direction
export const BRUSH_UM = { min: 5, max: 400, start: 40 }  // brush diameter, µm

/** A saved mark: its rings in field pixels and which are holes (`value` "holes:0,1"). */
export interface MarkedArea { points: number[][][] | null; value: string }

export function useBrush(size: { w: number; h: number } | null, saved: MarkedArea | null) {
  const canvas = useRef<HTMLCanvasElement | null>(null)
  const stroke = useRef<{ erase: boolean; last: number[]; before: ImageData; radius: number } | null>(null)
  const [url, setUrl] = useState('')
  const [painted, setPainted] = useState(false)

  const show = useCallback(() => {
    const c = canvas.current
    if (!c) return
    const data = c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data
    let any = false
    for (let i = 3; i < data.length; i += 4) if (data[i] > 127) { any = true; break }
    setPainted(any)
    setUrl(any ? c.toDataURL('image/png') : '')
  }, [])

  useEffect(() => {  // the mask, with what was saved painted on it
    if (!size || canvas.current) return
    const c = document.createElement('canvas')
    c.width = Math.max(1, Math.ceil(size.w / BRUSH_SCALE))
    c.height = Math.max(1, Math.ceil(size.h / BRUSH_SCALE))
    canvas.current = c
    const g = c.getContext('2d')!
    const rings = saved?.points ?? []
    const holes = saved?.value?.startsWith('holes:') ? saved.value.slice(6).split(',').map((h) => h === '1') : []
    rings.forEach((ring, i) => {
      if (!ring || ring.length < 3) return
      g.globalCompositeOperation = holes[i] ? 'destination-out' : 'source-over'
      g.fillStyle = '#ffffff'
      g.beginPath()
      ring.forEach(([x, y], k) => (k ? g.lineTo(x / BRUSH_SCALE, y / BRUSH_SCALE) : g.moveTo(x / BRUSH_SCALE, y / BRUSH_SCALE)))
      g.closePath()
      g.fill()
    })
    g.globalCompositeOperation = 'source-over'
    show()
  }, [size, saved, show])

  const dab = (from: number[], to: number[], radius: number, erase: boolean) => {
    const g = canvas.current?.getContext('2d')
    if (!g) return
    g.globalCompositeOperation = erase ? 'destination-out' : 'source-over'
    g.strokeStyle = g.fillStyle = '#ffffff'
    g.lineCap = 'round'
    g.lineWidth = Math.max(1, (2 * radius) / BRUSH_SCALE)
    g.beginPath()
    g.moveTo(from[0] / BRUSH_SCALE, from[1] / BRUSH_SCALE)
    g.lineTo(to[0] / BRUSH_SCALE + 0.01, to[1] / BRUSH_SCALE)
    g.stroke()
    g.globalCompositeOperation = 'source-over'
  }

  /** A stroke starts at this field point (`radius` in field pixels). */
  const begin = useCallback((x: number, y: number, radius: number, erase: boolean) => {
    const c = canvas.current
    if (!c) return
    stroke.current = { erase, last: [x, y], radius, before: c.getContext('2d')!.getImageData(0, 0, c.width, c.height) }
    dab([x, y], [x, y], radius, erase)
    show()
  }, [show])
  const frame = useRef(0)
  const extend = useCallback((x: number, y: number) => {
    const s = stroke.current
    if (!s) return
    dab(s.last, [x, y], s.radius, s.erase)
    s.last = [x, y]
    if (!frame.current) frame.current = requestAnimationFrame(() => { frame.current = 0; show() })  // at most once a frame
  }, [show])
  /** The stroke ends: the mask as it was before it (for Z), or null when no stroke was on. */
  const end = useCallback((): ImageData | null => {
    const s = stroke.current
    stroke.current = null
    if (s) show()
    return s ? s.before : null
  }, [show])
  const restore = useCallback((before: ImageData) => {
    canvas.current?.getContext('2d')!.putImageData(before, 0, 0)
    show()
  }, [show])
  /** What Save sends: the mask as a PNG, or "" when nothing is painted (the mark goes). */
  const png = useCallback(() => (painted && canvas.current ? canvas.current.toDataURL('image/png') : ''), [painted])
  return { url, painted, painting: () => !!stroke.current, begin, extend, end, restore, png }
}
