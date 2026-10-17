/* The colours of the label classes: one source for every page. Each class (identity) keeps one colour, from its place
   among the stain's classes, positive classes first (the same order as `class_groups` on the server, so the slot it
   sends is the same colour): the Cores view, the annotation pages, the labels views, legends and charts. A class's
   types share its colour. Detection's own class names (the Cores view's "tau_neuron_dense", "compact", …) are mapped
   onto the label classes first (`MODEL_CLASSES`), so a tau+ neuron is the same colour on every page. Dotted light grey
   is only for what is not active: rejected by detection, removed, or new without a class.
   The colours are saturated and clear of DAB brown and haematoxylin blue, so they read on the tissue; on the light
   pages they are drawn with a thin dark edge. No runtime imports: tests/js/class_palette.test.ts runs this file in Node. */
import type { Vocabulary } from './labels/classKeys'

export const CLASS_COLOURS = ['#ff2bd6', '#ffe600', '#00e5ff', '#7cff3a', '#b98cff', '#ff4d4d', '#ffa64d', '#9dffe0', '#ffffff']
/** Not active: rejected by detection, removed by a person, new without a class (always dotted). */
export const INACTIVE_GREY = '#d0d4da'
/** A value that is no class of the vocabulary (an old label, a class since removed). */
export const UNKNOWN_COLOUR = '#9aa1ab'

/** A colour slot's colour (the server's `class_groups` sends slots). */
export function slotColour(slot: number | null | undefined): string {
  return slot == null || slot < 0 ? UNKNOWN_COLOUR : CLASS_COLOURS[slot % CLASS_COLOURS.length]
}

/** A class's slot: its place among the stain's classes, positive ones first, hidden ones included (so it never moves). */
export function classSlot(vocabulary: Vocabulary | null | undefined, identity: string | null | undefined): number {
  if (!vocabulary || !identity) return -1
  const order = [...vocabulary.classes].sort((a, b) => Number(a.role !== 'positive') - Number(b.role !== 'positive'))
  return order.findIndex((c) => c.id === identity)
}

/** The colour of a label class (its types share it). */
export function classColour(vocabulary: Vocabulary | null | undefined, identity: string | null | undefined): string {
  return slotColour(classSlot(vocabulary, identity))
}

/** Detection's class names (as the Cores view gets them for the current outputs) as label classes: [identity, type]. */
export const MODEL_CLASSES: Record<string, [string, string]> = {
  neuron: ['NeuN+ neuron', ''],
  compact: ['plaque', 'compact'], diffuse: ['plaque', 'diffuse'], small_plaque: ['plaque', 'small'],
  tau_neuron_ring: ['tau+ neuron', ''], tau_neuron_dense: ['tau+ neuron', ''],
}
