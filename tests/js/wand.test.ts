// The neurite brush of "Fix the field (simple)" (webapp/src/labels/wand.ts). Run by tests/test_simple_mode.py:
// node --experimental-strip-types tests/js/wand.test.ts
import assert from 'node:assert/strict'
import { densify, nearSegment, segmentsOf, underStroke } from '../../webapp/src/labels/wand.ts'

// three pieces in a row, 5 px apart, and one far away
const lines = [[0, 0, 10, 0, 20, 0], [25, 0, 35, 0], [40, 0, 50, 0], [200, 200, 210, 210]]
const segments = segmentsOf(lines)
assert.equal(segments.length, 4)
assert.deepEqual(segments[0].box, [0, 0, 20, 0])

// a dab on a piece takes it alone; a stroke along the row takes the row; a stroke beside it, out of reach, nothing
assert.deepEqual(underStroke(segments, [[30, 3]], 4), [1])
assert.deepEqual(underStroke(segments, [[0, 2], [50, 2]], 4), [0, 1, 2])
assert.deepEqual(underStroke(segments, [[0, 20], [50, 20]], 4), [])
assert.deepEqual(underStroke(segments, [[0, 20], [50, 20]], 25), [0, 1, 2])  // a bigger brush reaches it
assert.deepEqual(underStroke(segments, [], 10), [])
// a fast stroke (two points far apart) still takes what lies between them
assert.deepEqual(underStroke(segments, [[0, -5], [210, 205]], 6).includes(3), true)
assert.ok(densify([[0, 0], [100, 0]], 10).length >= 20)
// a segment crossed between two of its points (the line, not only the points)
assert.deepEqual(underStroke(segmentsOf([[0, 0, 100, 0]]), [[50, 3]], 4), [0])
assert.ok(nearSegment(segments[0], 15, 2, 3) && !nearSegment(segments[0], 15, 9, 3))
console.log('brush ok')

// thousands of segments (a real AT8 field): a long stroke answers at once
const many: number[][] = []
for (let i = 0; i < 4000; i++) { const x = (i % 50) * 40, y = Math.floor(i / 50) * 40; many.push([x, y, x + 12, y + 3, x + 20, y + 8]) }
const big = segmentsOf(many)
const stroke: number[][] = []
for (let x = 0; x <= 2000; x += 15) stroke.push([x, 4 + 40 * 3])
const t0 = performance.now()
const taken = underStroke(big, stroke, 12)
assert.ok(performance.now() - t0 < 300, 'the brush is fast')
assert.equal(taken.length, 50)  // one row of 50
assert.ok(taken.every((i) => Math.floor(i / 50) === 3))
console.log('brush fast ok')

// every piece starts chosen; strokes take out (erase) or put back, the last stroke over a piece decides, at any level
{
  const { chosenAt, outStrokes, thinned } = await import('../../webapp/src/labels/wand.ts')
  const level = segmentsOf([[0, 0, 10, 0], [10, 0, 20, 0], [20, 0, 30, 0]])
  assert.deepEqual(chosenAt(level, []), [0, 1, 2])
  const out = { erase: true, points: [[15, 2]], radius: 3 }
  assert.deepEqual(chosenAt(level, [out]), [0, 2])
  assert.deepEqual(chosenAt(level, [out, { erase: false, points: [[15, 2]], radius: 3 }]), [0, 1, 2])  // put back
  const finer = segmentsOf([[0, 0, 5, 0], [5, 0, 12, 0], [12, 0, 18, 0], [18, 0, 30, 0]])  // another level: the same place stays out
  assert.deepEqual(chosenAt(finer, [out]), [0, 1, 3])
  // a selection saved as pieces: strokes along the inside of the pieces left out, not touching their neighbours
  assert.deepEqual(chosenAt(level, outStrokes(level, new Set([0, 2]))), [0, 2])
  assert.ok(thinned([[0, 0], [0.2, 0], [0.4, 0], [3, 0], [3.1, 0]], 1).length === 3)
  console.log('neurite pass ok')
}
