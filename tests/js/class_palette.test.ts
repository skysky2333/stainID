// One colour per label class everywhere (webapp/src/classPalette.ts). Run by tests/test_label_classes.py:
// node --experimental-strip-types tests/js/class_palette.test.ts
import assert from 'node:assert/strict'
import { CLASS_COLOURS, MODEL_CLASSES, classColour, classSlot, slotColour } from '../../webapp/src/classPalette.ts'
import type { Vocabulary } from '../../webapp/src/labels/classKeys.ts'

const cls = (id: string, role: 'positive' | 'negative' | 'skip', subtypes: string[] = [], hidden = false) => ({
  id, name: id, role, description: '', key: '', example: '', hidden, subtypes: subtypes.map((s) => ({ id: s, name: s, description: '', key: '', example: '', hidden: false })),
})
const at8: Vocabulary = {
  stain: 'AT8', subtype_wait_s: 1, aliases: {}, untyped: 'type not recorded',
  classes: [cls('neuritic plaque', 'negative'), cls('tau+ neuron', 'positive', ['pretangle', 'mature tangle']), cls('thread', 'negative'),
    cls('not tau', 'negative'), cls('astroglial tau', 'negative', [], true), cls('unsure', 'skip')],
}

// The positive class comes first, then the others in the vocabulary's order (hidden ones keep their place): the order of
// the server's class_groups, so its colour slots are these colours.
assert.equal(classSlot(at8, 'tau+ neuron'), 0)
assert.equal(classSlot(at8, 'neuritic plaque'), 1)
assert.equal(classSlot(at8, 'astroglial tau'), 4)
assert.equal(classColour(at8, 'tau+ neuron'), CLASS_COLOURS[0])
assert.equal(classColour(at8, 'thread'), slotColour(classSlot(at8, 'thread')))

// Every class of a stain has its own colour.
const colours = at8.classes.map((c) => classColour(at8, c.id))
assert.equal(new Set(colours).size, colours.length)

// Detection's class names land on the label classes, so the Cores view and the annotation pages agree.
assert.equal(classColour(at8, MODEL_CLASSES.tau_neuron_dense[0]), classColour(at8, 'tau+ neuron'))
assert.equal(classColour(at8, MODEL_CLASSES.tau_neuron_ring[0]), classColour(at8, 'tau+ neuron'))

// Something that is no class of the vocabulary gets the neutral colour, never a class's.
assert.ok(!CLASS_COLOURS.includes(classColour(at8, 'no such class')))
console.log('class palette: ok')
