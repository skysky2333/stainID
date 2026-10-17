// Step 1 of "Fix the field" (webapp/src/labels/fieldStart.ts): the acceptance threshold, where objects start at it,
// and its lock. Run by tests/test_annotation_rounds.py: node --experimental-strip-types tests/js/field_start.test.ts
import assert from 'node:assert/strict'
import { acceptedAt, lock, moveTo, slider, sliderText, startPick, unlock } from '../../webapp/src/labels/fieldStart.ts'

const REJECT = 'not a plaque'
const accepted = { ai_value: 'plaque', ai_subtype: 'compact', ai_probability: 0.8, if_accepted: ['plaque', 'compact'] as [string, string] }
const below = { ai_value: REJECT, ai_subtype: '', ai_probability: 0.42, if_accepted: ['plaque', 'diffuse'] as [string, string] }
const ruled = { ai_value: REJECT, ai_subtype: '', ai_probability: 0.9, if_accepted: null }  // a rule rejected it (a speck)
const objects = [accepted, below, ruled]
const positive = (identity: string) => identity === 'plaque'

// At detection's own threshold every object starts as detection called it.
for (const o of objects) assert.deepEqual(startPick(o, 0.5, 0.5, REJECT), { identity: o.ai_value, subtype: o.ai_subtype || null })
// Lower: the model's reject above the new threshold starts accepted, in the type detection would give it.
assert.deepEqual(startPick(below, 0.4, 0.5, REJECT), { identity: 'plaque', subtype: 'diffuse' })
// Higher: an accepted object below it starts rejected.
assert.deepEqual(startPick(accepted, 0.85, 0.5, REJECT), { identity: REJECT, subtype: null })
// A rule's reject stays rejected whatever the threshold.
assert.deepEqual(startPick(ruled, 0.1, 0.5, REJECT), { identity: REJECT, subtype: null })
assert.equal(acceptedAt(objects, 0.5, 0.5, REJECT, positive), 1)
assert.equal(acceptedAt(objects, 0.4, 0.5, REJECT, positive), 2)
assert.equal(acceptedAt(objects, 0.9, 0.5, REJECT, positive), 0)
assert.equal(sliderText(2, 0.4, 1, 0.5), '2 accepted at 0.40 (detection: 1 at 0.50)')

// It starts unlocked at detection's threshold (locked at the value saved with the field, when there is one).
let s = slider(0.5)
assert.deepEqual(s, { value: 0.5, locked: false, detection: 0.5 })
assert.deepEqual(slider(0.5, 0.37), { value: 0.37, locked: true, detection: 0.5 })
s = moveTo(s, 0.433)
assert.ok(Math.abs(s.value - 0.43) < 1e-9 && !s.locked)  // on the step
assert.equal(moveTo(s, 7).value, 1)
assert.equal(moveTo(s, -1).value, 0)
// The first edit (or Start editing) locks it; locked, it does not move.
s = lock(moveTo(s, 0.43))
assert.ok(s.locked)
assert.equal(moveTo(s, 0.9).value, s.value)
assert.equal(lock(s), s)
// Reset field unlocks it (the page discards the field's unsaved edits), at the value it had.
s = unlock(s)
assert.ok(!s.locked && Math.abs(s.value - 0.43) < 1e-9)
assert.ok(Math.abs(moveTo(s, 0.6).value - 0.6) < 1e-9)
console.log('field start ok')
