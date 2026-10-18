/* The one palette (src/stainid/palette.json, the same file the server draws with and serves at /api/palette): every
   colour the pages draw with, all distinct (tests/test_palette.py). Label classes take their colour by slot
   (classPalette.ts); the rest are named here. No runtime imports beyond the JSON: tests/js run this file in Node. */
import colours from '../../src/stainid/palette.json' with { type: 'json' }

export const PALETTE = colours as {
  classes: string[]
  states: { inactive: string; unsure: string; unknown: string }
  areas: Record<'1' | '2' | '3' | '4' | 'taken_out' | 'edge_guard' | 'not_analysed', string>
  overlays: { unproposed: string; threads: string; dab: string; tiles: string; excluded: string }
  markers: { edit: string; others: string; reference: string; spot_added: string; spot_removed: string; box: string; split_a: string; split_b: string; merge: string; unsaved: string }
}
/** A bad-area type's colour ("1"–"4"; "include", an older edit's "taken out", the taken-out colour). */
export const areaColour = (kind: string): string => (kind === 'include' || kind === 'taken_out' ? PALETTE.areas.taken_out
  : (PALETTE.areas as Record<string, string>)[kind] ?? PALETTE.states.unknown)
