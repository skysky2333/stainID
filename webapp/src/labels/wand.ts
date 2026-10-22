/* The magic wand of "Fix the field (simple)" (simplefield.tsx): over the thread segments detection found on an AT8
   field (stage 2's lines, field px, flat [x0, y0, x1, y1, …]), a click takes the segment under it and every segment
   reached from it in steps of at most `reach` px. Plain functions: tests/js/wand.test.ts runs them. */

export interface Segment { points: number[][]; box: [number, number, number, number] }

export function segmentsOf(lines: number[][]): Segment[] {
  return lines.map((flat) => {
    const points: number[][] = []
    for (let i = 0; i + 1 < flat.length; i += 2) points.push([flat[i], flat[i + 1]])
    const xs = points.map((p) => p[0]), ys = points.map((p) => p[1])
    return { points, box: [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)] }
  })
}

/** The shortest distance between two segments' points (px; long ones sampled), Infinity when their boxes are further
 *  apart than `most`. */
export function gap(a: Segment, b: Segment, most: number): number {
  if (a.box[0] - most > b.box[2] || b.box[0] - most > a.box[2] || a.box[1] - most > b.box[3] || b.box[1] - most > a.box[3]) return Infinity
  let best = Infinity
  const stepA = Math.max(1, Math.floor(a.points.length / 40)), stepB = Math.max(1, Math.floor(b.points.length / 40))
  for (let i = 0; i < a.points.length; i += stepA) {
    for (let j = 0; j < b.points.length; j += stepB) {
      const d = Math.hypot(a.points[i][0] - b.points[j][0], a.points[i][1] - b.points[j][1])
      if (d < best) best = d
    }
  }
  return best
}

/** The segment nearest a point, within `most` px (else -1). */
export function nearestSegment(segments: Segment[], x: number, y: number, most: number): number {
  let found = -1, best = most
  segments.forEach((s, i) => {
    if (x < s.box[0] - most || x > s.box[2] + most || y < s.box[1] - most || y > s.box[3] + most) return
    for (const p of s.points) { const d = Math.hypot(p[0] - x, p[1] - y); if (d <= best) { best = d; found = i } }
  })
  return found
}

/** This segment and every one reached from it in steps of at most `reach` px (at least 2 px: touching ones). The
 *  segments are put in a grid first, so each step looks only at its neighbours (a field has thousands of segments). */
export function wandGroup(segments: Segment[], start: number, reach: number): number[] {
  const most = Math.max(2, reach)
  const cell = Math.max(most, 24)
  const grid = new Map<string, number[]>()
  const cells = (box: [number, number, number, number], pad: number) => {
    const out: string[] = []
    for (let cx = Math.floor((box[0] - pad) / cell); cx <= Math.floor((box[2] + pad) / cell); cx++) {
      for (let cy = Math.floor((box[1] - pad) / cell); cy <= Math.floor((box[3] + pad) / cell); cy++) out.push(`${cx}:${cy}`)
    }
    return out
  }
  segments.forEach((s, i) => { for (const c of cells(s.box, 0)) { const list = grid.get(c); if (list) list.push(i); else grid.set(c, [i]) } })
  const seen = new Set([start])
  const queue = [start]
  while (queue.length) {
    const at = queue.shift()!
    const near = new Set<number>()
    for (const c of cells(segments[at].box, most)) for (const i of grid.get(c) ?? []) near.add(i)
    for (const i of near) if (!seen.has(i) && gap(segments[at], segments[i], most) <= most) { seen.add(i); queue.push(i) }
  }
  return [...seen].sort((a, b) => a - b)
}
