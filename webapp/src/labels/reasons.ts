/* Why detection kept or rejected a candidate (the v2 reason codes, stainid/v2/classify.py), in plain words: the Cores
   view's "Why" and, in Fix the field, why an object starts as it does at the acceptance threshold (fieldStart.ts). */

const REASON_TEXT: Record<string, string> = {
  accepted: 'kept by the model', rejected_by_model: 'the model judged it not an object', 'rule:speck': 'smaller than a plaque (a speck)',
  'rule:vascular_or_edge': "set by detection's vessel rule: amyloid on a vessel wall or tissue edge", 'rule:duplicate': 'a second detection of a nearby object',
  'rule:excluded': 'in an area left out of the analysis', 'rule:near_profile': 'a nucleus beside a stained neuron',
  'rule:pale_nucleus': 'a pale nucleus without stain', 'rule:shape': 'too small, too large or too long for a neuron',
  'rule:low_contrast': 'too faint against its surroundings', 'rule:small': 'too small for a tau+ neuron',
  'rule:artifact_shape': 'shaped like an artifact', 'rule:thread_like': "set by detection's thread rule: it lies on threads",
  'rule:edge': 'a brown rim along a tissue edge or hole', 'rule:irregular': 'too irregular for a cell body (thread tangle)',
  'rule:fold': 'on a tissue fold',
}

export const reasonText = (code: string) => REASON_TEXT[code] ?? code.replace(/^rule:/, '').replace(/_/g, ' ')

/** The rules whose call is a look-alike class of the vocabulary, not a reject (stainid/v2/classify.py CLASS_RULES):
 *  the object is that class (CAA is 6E10's Q, a thread AT8's E), drawn solid in its colour and counted as it. */
export const CLASS_RULES: Record<string, string> = { 'rule:vascular_or_edge': 'CAA', 'rule:thread_like': 'thread' }
/** Why an object the model does not count is a look-alike (stage 4 names it, `if_rejected`), by class. */
const BELOW_TEXT: Record<string, string> = { thread: "set by detection's thread rule: the model does not count it as a tau+ neuron and its stain lies along threads" }
/** The rules saying an object is no object of its own: whatever class stage 4 left on it, a reject
 *  (stainid/v2/classify.py REJECT_RULES). */
export const REJECT_RULES = new Set(['rule:duplicate', 'rule:excluded', 'rule:edge', 'rule:fold'])

/** The Cores view's "Why" of an object (`cls` its legend class, "label:<identity>/<subtype>"): a look-alike the model
 *  left out says the rule that names it (an AT8 thread), else the reason code in words. */
export function whyFound(cls: string, reason: string): string {
  const identity = cls.startsWith('label:') ? cls.slice(6).split('/')[0] : ''
  if (!BELOW_TEXT[identity] || CLASS_RULES[reason] || reason === 'accepted') return reasonText(reason)
  return reason === 'rejected_by_model' ? BELOW_TEXT[identity] : `${reasonText(reason)}, and its stain lies along threads: set by detection's thread rule`
}

export interface WhyObject {
  ai_probability: number | null; reason?: string; if_accepted?: [string, string] | null; if_rejected?: [string, string] | null
  rule?: string; excluded?: boolean; ai_value?: string
}

/** Why an object of Fix the field starts as it does at acceptance threshold `value` (detection's own at `detection`),
 *  when its score alone does not say: an excluded area, or the rule that left it out of detection's count. */
export function whyStart(o: WhyObject, value: number | null, detection: number | null): string {
  if (o.excluded || o.reason === 'rule:excluded') return 'in an excluded area: left out of the analysis and not asked about'
  const rule = o.rule ?? ''
  const atDetection = value == null || (detection != null && Math.abs(value - detection) < 1e-9)
  const score = o.ai_probability ?? 0
  const passing = atDetection ? o.reason !== 'rejected_by_model' : score >= (value ?? 0) - 1e-9
  if (!passing && o.if_rejected) return BELOW_TEXT[o.if_rejected[0]] ?? "set by detection's rule while the model does not count it"
  if (!rule) return ''
  if (!o.if_accepted) {  // a rule decided it: a class it names (CAA, a thread) or a reject, at every value
    if (CLASS_RULES[rule]) return `${reasonText(rule)}; ${CLASS_RULES[rule]} at every threshold`
    if (o.ai_value && BELOW_TEXT[o.ai_value] && !REJECT_RULES.has(rule)) return `${whyFound(`label:${o.ai_value}/`, rule)}; a ${o.ai_value} at every threshold`
    return `${reasonText(rule)}: detection's rule, whatever the threshold`
  }
  return atDetection || passing ? `${reasonText(rule)}: detection's rule once it passes the threshold` : ''
}
