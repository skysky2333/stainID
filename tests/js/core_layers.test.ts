// The Cores view's objects (webapp/src/pages/coreLayers.ts): accepted ones at every zoom, rejected candidates once the
// view spans REJECTED_VIEW_UM or less, by whichever way it got there. Run by tests/test_core_outline_choice.py.
import assert from 'node:assert/strict'
import { REJECTED_VIEW_UM, closeEnough, drawnObjects, rejectedNote, viewSpanUm } from '../../webapp/src/pages/coreLayers.ts'

const PX = 0.2738  // the study's µm per pixel
const kept = { cls: 'label:plaque/compact', x: 100, y: 100, outline: { rings: [] } }
const rejectedHere = { cls: 'rejected', x: 120, y: 130 }
const rejectedFar = { cls: 'rejected', x: 15000, y: 15000, outline: { rings: [] } }
const objects = [kept, rejectedHere, rejectedFar]
const on = { objects: true, rejected: true }

// the span: the view's larger side in µm, from the image's pixel size (else 0.5 µm), Infinity before any view
const whole = { x0: 0, y0: 0, x1: 16000, y1: 16000 }
const field = { x0: 0, y0: 0, x1: 2304, y1: 2000 }
assert.equal(viewSpanUm(null, PX), Infinity)
assert.equal(viewSpanUm(field, PX), 2304 * PX)
assert.equal(viewSpanUm(field, null), 2304 * 0.5)
assert.equal(viewSpanUm(field, Number.NaN), 2304 * 0.5)
assert.equal(viewSpanUm({ x0: 0, y0: 0, x1: 1000, y1: 3000 }, 1), 3000)  // a tall view: its height
assert.ok(!closeEnough(viewSpanUm(whole, PX)) && closeEnough(viewSpanUm(field, PX)))
assert.ok(closeEnough(REJECTED_VIEW_UM) && !closeEnough(REJECTED_VIEW_UM + 1))
// the threshold is about two labelling fields of the study, never a sliver nor the whole core
assert.ok(REJECTED_VIEW_UM / PX > 2 * 2304 * 0.8 && REJECTED_VIEW_UM / PX < 16000 / 2)

// zoomed out: the kept objects only, and the legend says to zoom in
assert.deepEqual(drawnObjects(objects, on, false, true), [kept])
assert.equal(rejectedNote(objects, on, false, whole), 'Zoom in to see rejected candidates')
// zoomed in: the rejected ones too, and how many are in view; none here: said so (they are elsewhere in the core)
assert.deepEqual(drawnObjects(objects, on, true, true), objects)
assert.equal(rejectedNote(objects, on, true, field), '1 rejected in view')
assert.equal(rejectedNote(objects, on, true, { x0: 5000, y0: 5000, x1: 6000, y1: 6000 }), 'No rejected candidates in this part of the core')
// the layer turned off (before or after zooming): none, and nothing to say; turned on again: back
const off = { objects: true, rejected: false }
assert.deepEqual(drawnObjects(objects, off, true, true), [kept])
assert.equal(rejectedNote(objects, off, true, field), null)
assert.deepEqual(drawnObjects(objects, on, true, true), objects)
// the objects layer off: only outlined objects (outlines shown), rejected ones still only close in
assert.deepEqual(drawnObjects(objects, { objects: false, rejected: true }, true, true), [kept, rejectedFar])
assert.deepEqual(drawnObjects(objects, { objects: false, rejected: true }, false, true), [kept])
// a core without rejected candidates: no note
assert.equal(rejectedNote([kept], on, false, whole), null)
console.log('core layers: ok')
