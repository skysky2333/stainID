// The keys of the quick check and the field page (webapp/src/labels/classKeys.ts). Run by tests/test_label_classes.py:
// node --experimental-strip-types tests/js/class_keys.test.ts
import assert from 'node:assert/strict'
import { expire, pressKey, resolveClass } from '../../webapp/src/labels/classKeys.ts'
import type { KeyMap, Vocabulary } from '../../webapp/src/labels/classKeys.ts'

const keys: KeyMap = {  // as Vocabulary.keymap gives it for AT8
  direct: { 1: ['tau+ neuron', null], 2: ['neuritic plaque', null], 4: ['thread', null], q: ['tau+ neuron', 'pretangle'], w: ['tau+ neuron', 'mature tangle'] },
  then: { 'tau+ neuron': { 1: 'pretangle', 2: 'mature tangle', 3: 'ghost tangle' } },
}
const WAIT = 1000

// A class without types answers at once.
assert.deepEqual(pressKey(keys, null, '4', 0, WAIT), { answers: [{ identity: 'thread', subtype: null }], waiting: null, handled: true })

// A class with types waits; a number within the time picks the type by its place.
let step = pressKey(keys, null, '1', 0, WAIT)
assert.deepEqual(step.answers, [])
assert.equal(step.waiting?.identity, 'tau+ neuron')
assert.deepEqual(pressKey(keys, step.waiting, '2', 600, WAIT).answers, [{ identity: 'tau+ neuron', subtype: 'mature tangle' }])

// 0 or Enter: no type, at once. Escape and other keys mean nothing: the class keeps waiting.
assert.deepEqual(pressKey(keys, step.waiting, '0', 100, WAIT).answers, [{ identity: 'tau+ neuron', subtype: null }])
assert.deepEqual(pressKey(keys, step.waiting, 'Enter', 100, WAIT).answers, [{ identity: 'tau+ neuron', subtype: null }])
const other = pressKey(keys, step.waiting, 'Shift', 100, WAIT)
assert.equal(other.handled, false)
assert.equal(other.waiting?.identity, 'tau+ neuron')

// Time up: the class stands without a type (the page's timer asks `expire`), and a late number is a new class.
assert.equal(expire(step.waiting, 500, WAIT), null)
assert.deepEqual(expire(step.waiting, 1200, WAIT), { identity: 'tau+ neuron', subtype: null })
step = pressKey(keys, step.waiting, '2', 1500, WAIT)
assert.deepEqual(step.answers, [{ identity: 'tau+ neuron', subtype: null }, { identity: 'neuritic plaque', subtype: null }])

// A number that is no type of the waiting class is another class: the waiting one stands without a type.
step = pressKey(keys, null, '1', 0, WAIT)
assert.deepEqual(pressKey(keys, step.waiting, '4', 100, WAIT).answers, [{ identity: 'tau+ neuron', subtype: null }, { identity: 'thread', subtype: null }])

// A type's own key answers class and type in one press.
assert.deepEqual(pressKey(keys, null, 'W', 0, WAIT).answers, [{ identity: 'tau+ neuron', subtype: 'mature tangle' }])

// No time limit (0): the class waits until a type or "no type" is chosen; chosen with the mouse (sticky) likewise.
step = pressKey(keys, null, '1', 0, 0)
assert.equal(expire(step.waiting, 60_000, 0), null)
assert.deepEqual(pressKey(keys, step.waiting, '3', 60_000, 0).answers, [{ identity: 'tau+ neuron', subtype: 'ghost tangle' }])
assert.equal(expire({ identity: 'tau+ neuron', at: 0, sticky: true }, 60_000, WAIT), null)

// Without types asked, a class key always answers at once.
assert.deepEqual(pressKey({ direct: keys.direct, then: {} }, null, '1', 0, WAIT).answers, [{ identity: 'tau+ neuron', subtype: null }])

// Stored labels and AI calls read as identity and type.
const words: Vocabulary = {
  stain: 'AT8', subtype_wait_s: 1, untyped: 'type not recorded',
  aliases: { 'tau+ neuron: tangle': ['tau+ neuron', 'mature tangle'], 'not a tau+ neuron': ['not tau', ''] },
  classes: [
    { id: 'tau+ neuron', name: 'tau+ neuron', role: 'positive', description: '', key: '1', example: '', hidden: false,
      subtypes: [{ id: 'mature tangle', name: 'mature tangle', description: '', key: 'w', example: '', hidden: false }] },
    { id: 'not tau', name: 'not tau', role: 'negative', description: '', key: '5', example: '', hidden: false, subtypes: [] },
  ],
}
assert.deepEqual(resolveClass(words, 'tau+ neuron: tangle'), { identity: 'tau+ neuron', subtype: 'mature tangle' })
assert.deepEqual(resolveClass(words, 'not a tau+ neuron'), { identity: 'not tau', subtype: null })
assert.deepEqual(resolveClass(words, 'tau+ neuron', 'mature tangle'), { identity: 'tau+ neuron', subtype: 'mature tangle' })
assert.deepEqual(resolveClass(words, 'Tau+ neuron / Mature tangle'), { identity: 'tau+ neuron', subtype: 'mature tangle' })
console.log('class keys: ok')
