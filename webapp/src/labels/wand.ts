/* The neurite brush of "Fix the field (simple)" (simplefield.tsx): over the thread segments of an AT8 field (stage 2's
   lines at one step of the thread ladder, field px, flat [x0, y0, x1, y1, …]), a brush stroke takes every segment it
   touches. Plain functions: tests/js/wand.test.ts runs them. */

export interface Segment { points: number[][]; box: [number, number, number, number] }

export function segmentsOf(lines: number[][]): Segment[] {
  return lines.map((flat) => {
    const points: number[][] = []
    for (let i = 0; i + 1 < flat.length; i += 2) points.push([flat[i], flat[i + 1]])
    const xs = points.map((p) => p[0]), ys = points.map((p) => p[1])
    return { points, box: [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)] }
  })
}

/** The shortest distance from a point to the line between a and b. */
function toLine(px: number, py: number, a: number[], b: number[]): number {
  const dx = b[0] - a[0], dy = b[1] - a[1]
  const length = dx * dx + dy * dy
  const t = length ? Math.max(0, Math.min(1, ((px - a[0]) * dx + (py - a[1]) * dy) / length)) : 0
  return Math.hypot(px - (a[0] + t * dx), py - (a[1] + t * dy))
}

/** The stroke's points close enough together that a segment passing between two of them is not missed (at most
 *  half the radius apart). */
export function densify(stroke: number[][], radius: number): number[][] {
  const step = Math.max(1, radius / 2)
  const out: number[][] = []
  stroke.forEach((p, i) => {
    if (i === 0) { out.push(p); return }
    const q = stroke[i - 1]
    const n = Math.ceil(Math.hypot(p[0] - q[0], p[1] - q[1]) / step)
    for (let k = 1; k <= n; k++) out.push([q[0] + ((p[0] - q[0]) * k) / n, q[1] + ((p[1] - q[1]) * k) / n])
  })
  return out
}

/** Every segment the brush (`radius` px around each point of `stroke`) touches: a segment is under the stroke when
 *  some part of it is within the radius. The stroke's points go in a grid first, so a field of thousands of segments
 *  answers at once. */
export function underStroke(segments: Segment[], stroke: number[][], radius: number): number[] {
  if (!stroke.length || radius <= 0) return []
  const dabs = densify(stroke, radius)
  const cell = Math.max(radius, 8)
  const grid = new Map<string, number[][]>()
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity
  for (const p of dabs) {
    const key = `${Math.floor(p[0] / cell)}:${Math.floor(p[1] / cell)}`
    const list = grid.get(key)
    if (list) list.push(p); else grid.set(key, [p])
    x0 = Math.min(x0, p[0]); y0 = Math.min(y0, p[1]); x1 = Math.max(x1, p[0]); y1 = Math.max(y1, p[1])
  }
  const near = (x: number, y: number) => {
    const cx = Math.floor(x / cell), cy = Math.floor(y / cell)
    for (let i = cx - 1; i <= cx + 1; i++) for (let j = cy - 1; j <= cy + 1; j++) {
      for (const p of grid.get(`${i}:${j}`) ?? []) if (Math.hypot(p[0] - x, p[1] - y) <= radius) return true
    }
    return false
  }
  const out: number[] = []
  segments.forEach((s, i) => {
    if (s.box[0] > x1 + radius || s.box[2] < x0 - radius || s.box[1] > y1 + radius || s.box[3] < y0 - radius) return
    const pts = s.points
    if (pts.length === 1) { if (near(pts[0][0], pts[0][1])) out.push(i); return }
    for (let k = 1; k < pts.length; k++) {
      const a = pts[k - 1], b = pts[k]
      const n = Math.max(1, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) / Math.max(1, radius / 2)))
      let hit = false
      for (let m = 0; m <= n && !hit; m++) hit = near(a[0] + ((b[0] - a[0]) * m) / n, a[1] + ((b[1] - a[1]) * m) / n)
      if (hit) { out.push(i); return }
    }
  })
  return out
}

/** Whether a point is within `most` px of a segment (its pieces between points). */
export function nearSegment(s: Segment, x: number, y: number, most: number): boolean {
  if (x < s.box[0] - most || x > s.box[2] + most || y < s.box[1] - most || y > s.box[3] + most) return false
  if (s.points.length === 1) return Math.hypot(s.points[0][0] - x, s.points[0][1] - y) <= most
  for (let k = 1; k < s.points.length; k++) if (toLine(x, y, s.points[k - 1], s.points[k]) <= most) return true
  return false
}
