/* Why detection kept or rejected a candidate (the v2 reason codes, stainid/v2/classify.py), in plain words: the Cores
   view's "Why" and, in Fix the field, why an object starts as it does at the acceptance threshold (fieldStart.ts). */

const REASON_TEXT: Record<string, string> = {
  accepted: 'kept by the model', rejected_by_model: 'the model judged it not an object', 'rule:speck': 'smaller than a plaque (a speck)',
  'rule:vascular_or_edge': 'on a vessel wall or tissue edge (CAA)', 'rule:duplicate': 'a second detection of a nearby object',
  'rule:excluded': 'in an area left out of the analysis', 'rule:near_profile': 'a nucleus beside a stained neuron',
  'rule:pale_nucleus': 'a pale nucleus without stain', 'rule:shape': 'too small, too large or too long for a neuron',
  'rule:low_contrast': 'too faint against its surroundings', 'rule:small': 'too small for a tau+ neuron',
  'rule:artifact_shape': 'shaped like an artifact', 'rule:thread_like': 'lies on a thread',
  'rule:edge': 'a brown rim along a tissue edge or hole', 'rule:irregular': 'too irregular for a cell body (thread tangle)',
  'rule:fold': 'on a tissue fold',
}

export const reasonText = (code: string) => REASON_TEXT[code] ?? code.replace(/^rule:/, '').replace(/_/g, ' ')

export interface WhyObject { ai_probability: number | null; reason?: string; if_accepted?: [string, string] | null; rule?: string; excluded?: boolean }

/** Why an object of Fix the field starts as it does at acceptance threshold `value` (detection's own at `detection`),
 *  when its score alone does not say: an excluded area, or the rule that left it out of detection's count. */
export function whyStart(o: WhyObject, value: number | null, detection: number | null): string {
  if (o.excluded || o.reason === 'rule:excluded') return 'in an excluded area: left out of the analysis and not asked about'
  const rule = o.rule ?? ''
  if (!rule) return ''
  if (!o.if_accepted) return `${reasonText(rule)}: detection's rule, whatever the threshold`
  const score = o.ai_probability ?? 0
  return value == null || score >= value - 1e-9 || (detection != null && Math.abs(value - detection) < 1e-9)
    ? `${reasonText(rule)}: detection's rule once it passes the threshold` : ''
}
