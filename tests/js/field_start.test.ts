// Step 1 of "Fix the field" (webapp/src/labels/fieldStart.ts): the acceptance threshold, where objects start at it,
// and its lock. Run by tests/test_annotation_rounds.py: node --experimental-strip-types tests/js/field_start.test.ts
import assert from 'node:assert/strict'
import { acceptedAt, lock, moveTo, slider, sliderText, startPick, unlock } from '../../webapp/src/labels/fieldStart.ts'
import { reasonText, whyFound, whyStart } from '../../webapp/src/labels/reasons.ts'
import { isAiLabel, peopleChecks } from '../../webapp/src/labels/aiLabels.ts'

const REJECT = 'not a plaque'
const accepted = { ai_value: 'plaque', ai_subtype: 'compact', ai_probability: 0.8, if_accepted: ['plaque', 'compact'] as [string, string] }
const below = { ai_value: REJECT, ai_subtype: '', ai_probability: 0.42, if_accepted: ['plaque', 'diffuse'] as [string, string] }
const excluded = { ai_value: REJECT, ai_subtype: '', ai_probability: 0.9, if_accepted: null }  // in an excluded area
const speck = { ai_value: REJECT, ai_subtype: '', ai_probability: 0.7, if_accepted: null }  // a rule decided it: detection's call at every value
const vesselIfPassed = { ai_value: REJECT, ai_subtype: '', ai_probability: 0.3, if_accepted: ['CAA', ''] as [string, string] }  // a rule after the model, once it passes
const unscored = { ai_value: REJECT, ai_subtype: '', ai_probability: null, if_accepted: ['plaque', 'diffuse'] as [string, string] }
const zero = { ai_value: REJECT, ai_subtype: '', ai_probability: 0, if_accepted: ['plaque', 'diffuse'] as [string, string] }
const top = { ai_value: 'plaque', ai_subtype: 'compact', ai_probability: 1, if_accepted: ['plaque', 'compact'] as [string, string] }
const objects = [accepted, below, excluded]
const positive = (identity: string) => identity === 'plaque'

// At detection's own threshold every object starts as detection called it.
for (const o of objects) assert.deepEqual(startPick(o, 0.5, 0.5, REJECT), { identity: o.ai_value, subtype: o.ai_subtype || null })
// Lower: the model's reject above the new threshold starts accepted, in the type detection would give it.
assert.deepEqual(startPick(below, 0.4, 0.5, REJECT), { identity: 'plaque', subtype: 'diffuse' })
// Higher: an accepted object below it starts rejected.
assert.deepEqual(startPick(accepted, 0.85, 0.5, REJECT), { identity: REJECT, subtype: null })
// In an excluded area it stays as detection called it whatever the threshold.
assert.deepEqual(startPick(excluded, 0.1, 0.5, REJECT), { identity: REJECT, subtype: null })
// p >= threshold passes the model (a missing score is 0) and then gets stage 4's call; a rule's call stays at every
// value: at 0 every object no rule decided counts, at 1 only p = 1, and the count never grows with the threshold.
assert.deepEqual(startPick(speck, 0.1, 0.5, REJECT), { identity: REJECT, subtype: null })
assert.deepEqual(startPick(vesselIfPassed, 0.2, 0.5, REJECT), { identity: 'CAA', subtype: null })
const all = [accepted, below, speck, unscored, zero, top, vesselIfPassed]
for (const o of [accepted, below, unscored, zero, top]) assert.equal(startPick(o, 0, 0.5, REJECT).identity, 'plaque')
assert.equal(acceptedAt(all, 0, 0.5, REJECT, positive), 5)
assert.equal(acceptedAt(all, 1, 0.5, REJECT, positive), 1)
let last = Infinity
for (let step = 0; step <= 100; step += 1) {
  const n = acceptedAt(all, moveTo(slider(0.5), step / 100).value, 0.5, REJECT, positive)
  assert.ok(n <= last, `more counted at ${step / 100} than below it`)
  last = n
}
assert.equal(startPick(top, 1, 0.5, REJECT).identity, 'plaque')
assert.equal(startPick({ ...below, ai_probability: 0.29 }, moveTo(slider(0.5), 0.29).value, 0.5, REJECT).identity, 'plaque')
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

// A rule call that names a class is that class: a deposit detection's vessel rule calls CAA stays CAA at every value
// (never the reject), and the Why row says so; an AT8 object the model does not count, lying on threads, is a thread
// below the threshold and a tau+ neuron above it; a rule that only rejects says so.
const caa = { ai_value: 'CAA', ai_subtype: '', ai_probability: 0.7, if_accepted: null, reason: 'rule:vascular_or_edge', rule: 'rule:vascular_or_edge' }
for (let t = 0; t <= 100; t += 1) assert.deepEqual(startPick(caa, t / 100, 0.5, REJECT), { identity: 'CAA', subtype: null })
assert.match(whyStart(caa, 0.2, 0.5), /^set by detection's vessel rule.*CAA at every threshold$/)
assert.match(whyStart(caa, null, null), /vessel rule/)
const thread = { ai_value: 'thread', ai_subtype: '', ai_probability: 0.3, if_accepted: ['tau+ neuron', 'pretangle'] as [string, string],
  if_rejected: ['thread', ''] as [string, string], reason: 'rejected_by_model', rule: '' }
assert.deepEqual(startPick(thread, 0.5, 0.5, 'not tau'), { identity: 'thread', subtype: null })
assert.deepEqual(startPick(thread, 0.6, 0.5, 'not tau'), { identity: 'thread', subtype: null })
assert.deepEqual(startPick(thread, 0.2, 0.5, 'not tau'), { identity: 'tau+ neuron', subtype: 'pretangle' })
assert.match(whyStart(thread, 0.5, 0.5), /thread rule/)
assert.equal(whyStart(thread, 0.2, 0.5), '')
const tangleOnThreads = { ai_value: 'tau+ neuron', ai_subtype: 'pretangle', ai_probability: 0.8, if_accepted: ['tau+ neuron', 'pretangle'] as [string, string],
  if_rejected: ['thread', ''] as [string, string], reason: 'accepted' }
assert.deepEqual(startPick(tangleOnThreads, 0.9, 0.5, 'not tau'), { identity: 'thread', subtype: null })
assert.match(whyStart({ ...speck, reason: 'rule:speck', rule: 'rule:speck' }, 0.1, 0.5), /a speck.*detection's rule, whatever the threshold/)
assert.match(whyFound('label:thread/', 'rejected_by_model'), /thread rule/)
assert.match(whyFound('label:CAA/', 'rule:vascular_or_edge'), /vessel rule/)
assert.equal(whyFound('rejected', 'rejected_by_model'), reasonText('rejected_by_model'))
// a shape rule left it out but its stain lies along threads: a thread at every value (a duplicate stays a reject)
const smallThread = { ai_value: 'thread', ai_subtype: '', ai_probability: null, if_accepted: null, reason: 'rule:small', rule: 'rule:small' }
for (const t of [0, 0.5, 1]) assert.deepEqual(startPick(smallThread, t, 0.5, 'not tau'), { identity: 'thread', subtype: null })
assert.match(whyStart(smallThread, 0.2, 0.5), /^too small for a tau\+ neuron, and its stain lies along threads: set by detection's thread rule; a thread at every threshold$/)
assert.match(whyStart({ ...smallThread, ai_value: 'not tau', reason: 'rule:duplicate', rule: 'rule:duplicate' }, 0.2, 0.5), /whatever the threshold/)
// AI labels never show while annotating: an object only the AI checked starts as detection made it.
for (const s of ['import: reference_6e10', 'AI reviewer', 'ai:round4']) assert.ok(isAiLabel(s), s)
for (const s of ['Ann', '', 'aileen']) assert.ok(!isAiLabel(s), s)
const check = (by: string[]) => ({ by, identity: 'plaque', subtype: '', time: 1 })
assert.deepEqual(peopleChecks({ a: check(['import: x']), b: check(['ai:r4', 'Ann']), c: check(['Ben']) }), { b: check(['Ann']), c: check(['Ben']) })
console.log('field start ok')
