/* The colours of the label classes: one source for every page. Each counted type (plaque · compact, tau+ neuron ·
   pretangle) and each named look-alike (CAA, thread) keeps one colour, its slot from the server
   (`Vocabulary.colour_slots`, sent as `vocabulary.slots`; the same slots `class_groups` sends): the Cores view, the
   annotation pages, the labels views, legends and charts. A positive identity without a type (detection's "plaque"
   with no type) has its own colour; a look-alike's types share its colour. Detection's own class names (the Cores
   view's "tau_neuron_dense", "compact", …) are mapped onto the label classes first (`MODEL_CLASSES`), so a compact
   plaque is the same colour on every page. Dotted light grey is the reject (Del) and detection's own rejects: not
   active; dotted amber with a "?" is "unsure".
   The colours are saturated and clear of DAB brown and haematoxylin blue, so they read on the tissue; on the light
   pages they are drawn with a thin dark edge. No runtime imports: tests/js/class_palette.test.ts runs this file in Node. */
import type { Vocabulary } from './labels/classKeys'

export const CLASS_COLOURS = ['#ff2bd6', '#ffe600', '#00e5ff', '#7cff3a', '#b98cff', '#ff4d4d', '#ffa64d', '#9dffe0', '#ffffff']
/** Not active: rejected (Del, or by detection), removed by a person, new without a class (always dotted). */
export const INACTIVE_GREY = '#d0d4da'
/** "Unsure" (` or 0): dotted, with a small "?". */
export const UNSURE_AMBER = '#ffb020'
/** A value that is no class of the vocabulary (an old label, a class since removed). */
export const UNKNOWN_COLOUR = '#9aa1ab'
export const REJECT_SLOT = -2
export const UNSURE_SLOT = -3

/** A colour slot's colour (the server's `class_groups` sends slots: -2 the reject, -3 "unsure"). */
export function slotColour(slot: number | null | undefined): string {
  if (slot === REJECT_SLOT) return INACTIVE_GREY
  if (slot === UNSURE_SLOT) return UNSURE_AMBER
  return slot == null || slot < 0 ? UNKNOWN_COLOUR : CLASS_COLOURS[slot % CLASS_COLOURS.length]
}

/** The slots of a vocabulary: the server's, else worked out the same way (positive identities and their types first,
 *  then the other classes; the reject and "unsure" none). */
function slotsOf(vocabulary: Vocabulary): Record<string, number> {
  if (vocabulary.slots) return vocabulary.slots
  const out: Record<string, number> = {}
  const special = new Set([vocabulary.reject, vocabulary.unsure].filter(Boolean))
  const order = [...vocabulary.classes].sort((a, b) => Number(a.role !== 'positive') - Number(b.role !== 'positive'))
  for (const c of order) {
    if (special.has(c.id)) continue
    out[c.id] = Object.keys(out).length
    if (c.role === 'positive') for (const s of c.subtypes) out[`${c.id}/${s.id}`] = Object.keys(out).length
  }
  return out
}

/** A class's slot (and its type's, when it has its own): `REJECT_SLOT`, `UNSURE_SLOT`, -1 for no class. */
export function classSlot(vocabulary: Vocabulary | null | undefined, identity: string | null | undefined, subtype?: string | null): number {
  if (!vocabulary || !identity) return -1
  if (identity === vocabulary.reject) return REJECT_SLOT
  if (identity === vocabulary.unsure || (!vocabulary.unsure && vocabulary.classes.find((c) => c.id === identity)?.role === 'skip')) return UNSURE_SLOT
  const slots = slotsOf(vocabulary)
  return (subtype ? slots[`${identity}/${subtype}`] : undefined) ?? slots[identity] ?? -1
}

/** The colour of a label class, of its type when given (each counted type its own colour). */
export function classColour(vocabulary: Vocabulary | null | undefined, identity: string | null | undefined, subtype?: string | null): string {
  return slotColour(classSlot(vocabulary, identity, subtype))
}

/** Detection's class names (as the Cores view gets them for the current outputs) as label classes: [identity, type]. */
export const MODEL_CLASSES: Record<string, [string, string]> = {
  neuron: ['NeuN+ neuron', ''],
  compact: ['plaque', 'compact'], diffuse: ['plaque', 'diffuse'], small_plaque: ['plaque', 'small'],
  tau_neuron_ring: ['tau+ neuron', ''], tau_neuron_dense: ['tau+ neuron', ''],
}
