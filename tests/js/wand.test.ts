// The magic wand of "Fix the field (simple)" (webapp/src/labels/wand.ts). Run by tests/test_simple_mode.py:
// node --experimental-strip-types tests/js/wand.test.ts
import assert from 'node:assert/strict'
import { nearestSegment, segmentsOf, wandGroup } from '../../webapp/src/labels/wand.ts'

// three pieces in a row, 5 px apart, and one far away
const lines = [[0, 0, 10, 0, 20, 0], [25, 0, 35, 0], [40, 0, 50, 0], [200, 200, 210, 210]]
const segments = segmentsOf(lines)
assert.equal(segments.length, 4)
assert.deepEqual(segments[0].box, [0, 0, 20, 0])

// a click near a segment finds it; far from every one, none
assert.equal(nearestSegment(segments, 30, 3, 8), 1)
assert.equal(nearestSegment(segments, 100, 100, 8), -1)

// touching only (reach 2 px): the piece alone; 6 px: the whole row; never the far one
assert.deepEqual(wandGroup(segments, 1, 2), [1])
assert.deepEqual(wandGroup(segments, 1, 6), [0, 1, 2])
assert.deepEqual(wandGroup(segments, 0, 6), [0, 1, 2])  // reached step by step, not only from the first
assert.deepEqual(wandGroup(segments, 3, 40), [3])
assert.deepEqual(wandGroup(segments, 0, 400), [0, 1, 2, 3])
console.log('wand ok')

// thousands of segments (a real AT8 field): one click answers at once
const many: number[][] = []
for (let i = 0; i < 2000; i++) { const x = (i % 50) * 40, y = Math.floor(i / 50) * 40; many.push([x, y, x + 12, y + 3, x + 20, y + 8]) }
const big = segmentsOf(many)
const t0 = performance.now()
const reached = wandGroup(big, 0, 39)
assert.ok(performance.now() - t0 < 500, 'the wand is fast')
assert.equal(reached.length, 2000)  // 20 px apart along a row, about 38 px to the next row: 39 px reaches every one
assert.equal(wandGroup(big, 0, 30).length, 50)  // 30 px: its row only
assert.deepEqual(wandGroup(big, 0, 2), [0])
console.log('wand fast ok')
