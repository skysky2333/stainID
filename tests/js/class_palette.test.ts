// One colour per counted type and named look-alike everywhere (webapp/src/classPalette.ts). Run by tests/test_label_classes.py:
// node --experimental-strip-types tests/js/class_palette.test.ts
import assert from 'node:assert/strict'
import { CLASS_COLOURS, INACTIVE_GREY, MODEL_CLASSES, UNSURE_AMBER, classColour, classSlot, slotColour } from '../../webapp/src/classPalette.ts'
import type { Vocabulary } from '../../webapp/src/labels/classKeys.ts'

const cls = (id: string, role: 'positive' | 'negative' | 'skip', subtypes: string[] = [], hidden = false) => ({
  id, name: id, role, description: '', key: '', example: '', hidden, subtypes: subtypes.map((s) => ({ id: s, name: s, description: '', key: '', example: '', hidden: false })),
})
const at8: Vocabulary = {
  stain: 'AT8', subtype_wait_s: 1, aliases: {}, untyped: 'type not recorded', reject: 'not tau', unsure: 'unsure',
  classes: [cls('neuritic plaque', 'negative'), cls('tau+ neuron', 'positive', ['pretangle', 'mature tangle']), cls('thread', 'negative'),
    cls('not tau', 'negative'), cls('astroglial tau', 'negative', [], true), cls('unsure', 'skip')],
}

// Worked out as the server's Vocabulary.colour_slots: the positive identity, each of its types, then the others (hidden
// ones keep their place); the reject and "unsure" have none.
assert.equal(classSlot(at8, 'tau+ neuron'), 0)
assert.equal(classSlot(at8, 'tau+ neuron', 'pretangle'), 1)
assert.equal(classSlot(at8, 'tau+ neuron', 'mature tangle'), 2)
assert.equal(classSlot(at8, 'neuritic plaque'), 3)
assert.equal(classSlot(at8, 'astroglial tau'), 5)
assert.equal(classColour(at8, 'tau+ neuron', 'pretangle'), CLASS_COLOURS[1])
assert.equal(classColour(at8, 'tau+ neuron', 'no such type'), classColour(at8, 'tau+ neuron'))  // a type it does not know: the identity's
assert.equal(classColour(at8, 'not tau'), INACTIVE_GREY)  // the reject: as detection's own rejects
assert.equal(classColour(at8, 'unsure'), UNSURE_AMBER)
assert.equal(slotColour(-2), INACTIVE_GREY)
assert.equal(slotColour(-3), UNSURE_AMBER)

// The server's slots win when it sends them.
assert.equal(classColour({ ...at8, slots: { 'tau+ neuron': 4, 'tau+ neuron/pretangle': 7 } }, 'tau+ neuron', 'pretangle'), CLASS_COLOURS[7])

// Every type and look-alike of a stain has its own colour.
const colours = [classColour(at8, 'tau+ neuron'), classColour(at8, 'tau+ neuron', 'pretangle'), classColour(at8, 'tau+ neuron', 'mature tangle'),
  classColour(at8, 'neuritic plaque'), classColour(at8, 'thread')]
assert.equal(new Set(colours).size, colours.length)

// Detection's class names land on the label classes, so the Cores view and the annotation pages agree.
const plaques: Vocabulary = { ...at8, stain: '6E10', reject: 'not a plaque', classes: [cls('plaque', 'positive', ['compact', 'diffuse']), cls('not a plaque', 'negative'), cls('unsure', 'skip')] }
assert.equal(classColour(plaques, ...MODEL_CLASSES.compact), classColour(plaques, 'plaque', 'compact'))
assert.equal(classColour(at8, MODEL_CLASSES.tau_neuron_ring[0]), classColour(at8, 'tau+ neuron'))

// Something that is no class of the vocabulary gets the neutral colour, never a class's.
assert.ok(!CLASS_COLOURS.includes(classColour(at8, 'no such class')))
console.log('class palette: ok')
