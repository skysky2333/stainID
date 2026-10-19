/* AI labels: the labels (objects and bad areas) the AI made, not a person: the imported reference sets ("import: …"),
   the earlier AI reviewer ("AI reviewer") and AI review sessions (accounts "ai:<name>"). The annotation pages (Fix the
   field, Mark bad areas, the brush, the tutorial) never show them nor start from them: people correct the pipeline's
   output, not the AI's. The Labels and Models pages and the Cores view's Annotations layer keep them, marked as AI.
   The server has the same rule (stainid/labels/store.py `is_ai_label`) and leaves them out of what those pages get. */

import type { RoundCheck } from '../api'

export const isAiLabel = (source: string | null | undefined): boolean => {
  const s = source ?? ''
  return s.startsWith('import:') || s === 'AI reviewer' || s.startsWith('ai:')
}

/** Who decided each candidate, people only: an entry only the AI's labels made is dropped (the server already leaves
 *  them out; this keeps the page to the rule whatever it is sent). */
export function peopleChecks(checked: Record<string, RoundCheck> | undefined): Record<string, RoundCheck> {
  const out: Record<string, RoundCheck> = {}
  for (const [id, c] of Object.entries(checked ?? {})) {
    const by = c.by.filter((w) => !isAiLabel(w))
    if (by.length) out[id] = by.length === c.by.length ? c : { ...c, by }
  }
  return out
}
