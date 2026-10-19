/* Step 1 of every field in "Fix the field": the acceptance threshold, and where each object starts at it.

   The slider starts at detection's own threshold. Moving it re-decides, live, which objects start accepted (solid, in
   the type detection would give them: `if_accepted`) and which start rejected (dotted grey), as stage 4 would at that
   threshold: an object passes where p >= the threshold (a missing score is 0) and then gets stage 4's call, its rules
   after the model included (a speck stays rejected, a vessel deposit is CAA). Objects a rule decided, or in an excluded
   area, keep detection's call at every value (`if_accepted` null). So raising it never counts more, and at 0 every
   object no rule decided counts; the side panel says which rule decides an object (reasons.ts, `whyStart`). It locks on "Start editing" (or Enter) and by itself on the first edit of any kind (a class
   key, Del, `, a mask, a precise edit, a split, a merge, a brush stroke); locked, it does not move. "Reset field"
   (after a confirmation) discards the field's unsaved edits and unlocks it. The locked value is saved with the field
   (a `threshold` label): a training and evaluation signal, never an analysis input (labels/starting.py has the same
   rule on the server, for the objects left as they started when the field is finished). */

export interface StartObject { ai_value: string; ai_subtype: string; ai_probability: number | null; if_accepted?: [string, string] | null; excluded?: boolean }
export interface Pick2 { identity: string; subtype: string | null }
export interface Slider { value: number; locked: boolean; detection: number }
export const SLIDER_STEP = 0.01

const close = (a: number, b: number) => Math.abs(a - b) < 1e-9

/** p >= threshold, the slider's one rule (to the step's float error: 0.29 is 29 × 0.01 = 0.29000000000000004). */
export const passes = (p: number, threshold: number) => p >= threshold - 1e-9

/** A candidate's score for the slider: a missing one is 0. */
export const scoreOf = (o: { ai_probability: number | null }) => (o.ai_probability == null || Number.isNaN(o.ai_probability) ? 0 : o.ai_probability)

/** Where an object starts at threshold `value`: detection's call at detection's own threshold and in an excluded area
 *  (`if_accepted` null); else its `if_accepted` call where its score is at least `value`, `reject` below. */
export function startPick(o: StartObject, value: number, detection: number, reject: string): Pick2 {
  const own: Pick2 = { identity: o.ai_value, subtype: o.ai_subtype || null }
  if (close(value, detection) || !o.if_accepted || o.excluded) return own  // an excluded area: as detection called it
  return passes(scoreOf(o), value) ? { identity: o.if_accepted[0], subtype: o.if_accepted[1] || null } : { identity: reject, subtype: null }
}

/** How many of these objects start accepted at `value` (rule-decided ones as detection called them). */
export function acceptedAt(objects: StartObject[], value: number, detection: number, reject: string, positive: (identity: string) => boolean): number {
  let n = 0
  for (const o of objects) if (positive(startPick(o, value, detection, reject).identity)) n += 1
  return n
}

export function slider(detection: number, saved?: number | null): Slider {
  return saved == null ? { value: detection, locked: false, detection } : { value: saved, locked: true, detection }
}

/** The slider moved (ignored once locked), kept within 0–1 on the step. */
export function moveTo(s: Slider, value: number): Slider {
  if (s.locked) return s
  return { ...s, value: Math.min(1, Math.max(0, Math.round(value / SLIDER_STEP) * SLIDER_STEP)) }
}

/** "Start editing", Enter or the first edit: locked at its value. */
export function lock(s: Slider): Slider {
  return s.locked ? s : { ...s, locked: true }
}

/** "Reset field" (confirmed): unlocked again, at the value it had. The page discards the field's unsaved edits. */
export function unlock(s: Slider): Slider {
  return { ...s, locked: false }
}

export function sliderText(accepted: number, value: number, detectionAccepted: number, detection: number): string {
  return `${accepted.toLocaleString()} accepted at ${value.toFixed(2)} (detection: ${detectionAccepted.toLocaleString()} at ${detection.toFixed(2)})`
}
