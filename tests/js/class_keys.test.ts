// The keys of the labelling pages (webapp/src/labels/classKeys.ts). Run by tests/test_label_classes.py:
// node --experimental-strip-types tests/js/class_keys.test.ts
import assert from 'node:assert/strict'
import { byKey, keyGroups, keyLabel, keyName, keySummary, pressKey, resolveClass } from '../../webapp/src/labels/classKeys.ts'
import type { KeyMap, Vocabulary } from '../../webapp/src/labels/classKeys.ts'

const keys: KeyMap = {  // as Vocabulary.keymap gives it for AT8
  direct: {
    1: ['tau+ neuron', 'pretangle'], 2: ['tau+ neuron', 'mature tangle'], 3: ['tau+ neuron', 'ghost tangle'],
    q: ['neuritic plaque', null], w: ['astroglial tau', null], e: ['thread', null], delete: ['not tau', null], '`': ['unsure', null],
  },
  aliases: { 0: '`', backspace: 'delete' }, brush: 'b',
}

// One key picks a type at once: no waiting, no second key.
assert.deepEqual(pressKey(keys, '2'), { identity: 'tau+ neuron', subtype: 'mature tangle' })
assert.deepEqual(pressKey(keys, 'E'), { identity: 'thread', subtype: null })  // letters in either case
assert.deepEqual(pressKey(keys, 'Delete'), { identity: 'not tau', subtype: null })  // the single reject
assert.deepEqual(pressKey(keys, 'Backspace'), { identity: 'not tau', subtype: null })  // a Mac's delete key
assert.deepEqual(pressKey(keys, '`'), { identity: 'unsure', subtype: null })
assert.deepEqual(pressKey(keys, '0'), { identity: 'unsure', subtype: null })  // the quiet alias of `
assert.equal(keyName(keys, '0'), '`')
for (const other of ['Enter', ' ', 'Escape', 'Shift', 'Tab', 'ArrowLeft', 'b', 'z', 's', '9']) assert.equal(pressKey(keys, other), null, other)

// The groups and the line the panel, the key list and the tutorial show.
const words: Vocabulary = {
  stain: 'AT8', subtype_wait_s: 1, untyped: 'type not recorded', reject: 'not tau', unsure: 'unsure',
  aliases: { 'tau+ neuron: tangle': ['tau+ neuron', 'mature tangle'], 'not a tau+ neuron': ['not tau', ''], 'artifact or unclear': ['not tau', ''] },
  classes: [
    { id: 'tau+ neuron', name: 'tau+ neuron', role: 'positive', description: '', key: '1', example: '', hidden: false,
      subtypes: [{ id: 'pretangle', name: 'pretangle', description: '', key: '1', example: '', hidden: false },
        { id: 'mature tangle', name: 'mature tangle', description: '', key: '2', example: '', hidden: false },
        { id: 'ghost tangle', name: 'ghost tangle', description: '', key: '3', example: '', hidden: false }] },
    { id: 'unsure', name: 'unsure', role: 'skip', description: '', key: '`', example: '', hidden: false, subtypes: [] },
    { id: 'thread', name: 'thread', role: 'negative', description: '', key: 'e', example: '', hidden: false, subtypes: [] },
    { id: 'neuritic plaque', name: 'neuritic crown', role: 'negative', description: '', key: 'q', example: '', hidden: false, subtypes: [] },
    { id: 'astroglial tau', name: 'tau in an astrocyte', role: 'negative', description: '', key: 'w', example: '', hidden: false, subtypes: [] },
    { id: 'not tau', name: 'not tau', role: 'negative', description: '', key: 'delete', example: '', hidden: false, subtypes: [] },
  ],
}
const groups = keyGroups(words, keys)
assert.deepEqual(groups.map((g) => [g.group, g.keys]), [['counted', '1–3'], ['lookalike', 'Q–E'], ['reject', 'Del'], ['unsure', '`']])
assert.deepEqual(byKey(words, keys).map((c) => c.id), ['tau+ neuron', 'neuritic plaque', 'astroglial tau', 'thread', 'not tau', 'unsure'])
assert.equal(keyLabel('delete'), 'Del')
const server = { ...keys, summary: 'Tau+ neurons 1–3 · Not tau+ neurons Q–E · Del not tau · ` unsure · B brush' }
assert.equal(keySummary(words, server), server.summary)
assert.equal(keySummary(words, server, false), 'Tau+ neurons 1–3 · Not tau+ neurons Q–E · Del not tau · ` unsure')  // pages without the brush

// Stored labels and AI calls read as identity and type; old "artifact or unclear" labels are the reject now.
assert.deepEqual(resolveClass(words, 'tau+ neuron: tangle'), { identity: 'tau+ neuron', subtype: 'mature tangle' })
assert.deepEqual(resolveClass(words, 'not a tau+ neuron'), { identity: 'not tau', subtype: null })
assert.deepEqual(resolveClass(words, 'artifact or unclear'), { identity: 'not tau', subtype: null })
assert.deepEqual(resolveClass(words, 'tau+ neuron', 'mature tangle'), { identity: 'tau+ neuron', subtype: 'mature tangle' })
assert.deepEqual(resolveClass(words, 'tau+ neuron'), { identity: 'tau+ neuron', subtype: null })  // an identity with no type stays valid
assert.deepEqual(resolveClass(words, 'Tau+ neuron / Mature tangle'), { identity: 'tau+ neuron', subtype: 'mature tangle' })
console.log('class keys: ok')
