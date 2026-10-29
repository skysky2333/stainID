/* The neurite brush of "Fix the field (simple)" (simplefield.tsx): over the thread segments of an AT8 field (stage 2's
   lines at one step of the thread ladder, field px, flat [x0, y0, x1, y1, …]), a brush stroke touches every segment it
   reaches. Every segment at the slider's level starts chosen; the strokes (`Edit`) take out or put back the segments
   they touch, in the order painted, at whatever level the slider is at (`chosenAt`): a place taken out stays out when
   the slider moves. Plain functions: tests/js/wand.test.ts runs them. */

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

/** One stroke of the neurite brush (field px): `erase` takes out what it touches, else it puts it back. */
export interface Edit { erase: boolean; points: number[][]; radius: number }

/** The segments chosen at one level: all of them, then each stroke in turn takes out (or puts back) those it touches;
 *  the last stroke over a segment decides. */
export function chosenAt(segments: Segment[], edits: Edit[]): number[] {
  const on = segments.map(() => true)
  for (const e of edits) for (const i of underStroke(segments, e.points, e.radius)) on[i] = !e.erase
  return on.flatMap((x, i) => (x ? [i] : []))
}

/** The points of a segment more than `end` px (along it) from both its ends, one every px: the inside it does not share
 *  with the segments it meets; its middle when it is shorter. */
function inner(points: number[][], end: number): number[][] {
  const dense = densify(points, 2)
  if (dense.length < 2) return dense
  const along = [0]
  for (let k = 1; k < dense.length; k++) along.push(along[k - 1] + Math.hypot(dense[k][0] - dense[k - 1][0], dense[k][1] - dense[k - 1][1]))
  const total = along[along.length - 1]
  const kept = dense.filter((_, k) => along[k] > end && along[k] < total - end)
  if (kept.length) return kept
  const middle = along.findIndex((a) => a >= total / 2)
  return [dense[Math.max(0, middle)]]
}

/** A selection saved as pieces of one level (before the strokes were kept) as strokes: a thin one along the inside of
 *  each segment of that level that is not chosen, so the same places stay out at any level. */
export function outStrokes(segments: Segment[], chosen: Set<number>, radius = 0.75, end = 2): Edit[] {
  return segments.flatMap((s, i) => (chosen.has(i) ? [] : [{ erase: true, points: inner(s.points, end), radius }]))
}

/** A stroke's points with those closer than `step` px to the last kept dropped (what is saved). */
export function thinned(points: number[][], step: number): number[][] {
  const out: number[][] = []
  for (const p of points) {
    const q = out[out.length - 1]
    if (!q || Math.hypot(p[0] - q[0], p[1] - q[1]) >= step) out.push([Math.round(p[0] * 10) / 10, Math.round(p[1] * 10) / 10])
  }
  const last = points[points.length - 1]
  if (last && out.length > 1 && (out[out.length - 1][0] !== last[0] || out[out.length - 1][1] !== last[1])) out.push([Math.round(last[0] * 10) / 10, Math.round(last[1] * 10) / 10])
  return out
}
