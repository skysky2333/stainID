/* How reproducible each measure is (replicate-core ICC) and how it differs between diagnosis groups, as the server
   gives them (GET /api/tables/<table>/stats, stainid/analysis/measure_stats.py), and the filters of the measure list. */

export interface Icc { icc: number; pairs: number; source: 'results' | 'study' | 'computed'; scale: 'log' | 'raw' | null; file?: string; unique_variance?: number }
export interface GroupTest {
  effect: number; p: number; fdr: number; n: number | null; n_first: number | null; n_second: number | null; terms: string[]; file?: string
}
export interface TestModel { key: string; terms: string[]; adjusted: string; source: 'study' | 'quick'; default: boolean }
export interface Comparison {
  key: string; first: string; second: string; within: string; models: TestModel[]; tests: Record<string, Record<string, GroupTest>>
}
export interface TableStats { icc: Record<string, Icc>; comparisons: Comparison[]; icc_files: string[]; contrast_files: string[] }

export const QUICK = 'quick'

/** What the measure list shows: everything (the default), or only measures that differ between two groups and/or whose
 *  replicate cores agree well enough. A measure with no agreement measured is never hidden by the second. */
export interface MeasureFilter {
  significant: boolean
  comparison: string  // Comparison.key; '' = the first one
  basis: 'fdr' | 'p'
  level: number
  model: string  // TestModel.key; '' = the comparison's default
  quick: boolean  // use a quick test where the study has none
  reliable: boolean  // a new name, so the switch starts off for everyone (the stored `reproducible` of before is ignored)
  minIcc: number
}
export const DEFAULT_FILTER: MeasureFilter = {
  significant: false, comparison: '', basis: 'fdr', level: 0.05, model: '', quick: true, reliable: false, minIcc: 0.5,
}

export const ICC_HELP = 'Replicate agreement (ICC): many donors have two cores from the same brain region. If the measure gives about the same value '
  + 'on both cores, the measure is reliable. 1 = the two cores always agree, 0 = they agree no better than two cores from different donors. '
  + 'Good from 0.75, fair 0.5–0.75, weak below 0.5.'
export const NO_ICC_HELP = 'Replicate agreement was not measured for this measure (too few donor-regions with two cores that both have a value). '
  + 'It is never hidden by “Only show reliable measures”.'
/** What a level means for someone reading the plot. */
export const ICC_MEANING: Record<'good' | 'fair' | 'weak', string> = {
  good: 'Two cores of the same donor and region give close values: differences between groups are not just which core was cut.',
  fair: 'Two cores of the same donor and region give fairly similar values.',
  weak: 'Two cores of the same donor and region often give quite different values, so one core says little about the donor-region: read differences between groups with care.',
}

export const iccLevel = (icc: number) => (icc >= 0.75 ? 'good' : icc >= 0.5 ? 'fair' : 'weak')

export function iccHover(icc: Icc) {
  const from = icc.source === 'results' ? `From the results' reliability table (${icc.file}).`
    : icc.source === 'study' ? `From the study's reliability table (${icc.file}).` : 'Computed from the core table.'
  const unique = icc.unique_variance != null ? ` Unique variance ${icc.unique_variance}.` : ''
  return `${ICC_HELP}\n${icc.pairs ? `${icc.pairs} donor-regions with two cores` : 'Donor-regions with two cores'}${icc.scale === 'log' ? ', on the log scale' : ''}.${unique} ${from}`
}

export function chosenComparison(stats: TableStats | null, filter: MeasureFilter): Comparison | undefined {
  return stats?.comparisons.find((c) => c.key === filter.comparison) ?? stats?.comparisons[0]
}

export function chosenModel(comparison: Comparison | undefined, filter: MeasureFilter): TestModel | undefined {
  return comparison?.models.find((m) => m.key === filter.model && m.source === 'study')
    ?? comparison?.models.find((m) => m.default) ?? comparison?.models[0]
}

/** The test of a measure for the chosen comparison and model, or the quick test when the study has none for it. */
export function testOf(comparison: Comparison | undefined, model: TestModel | undefined, column: string, filter: MeasureFilter):
  { test: GroupTest; quick: boolean } | null {
  const tests = comparison?.tests[column]
  if (!tests) return null
  if (model && model.key !== QUICK && tests[model.key]) return { test: tests[model.key], quick: false }
  if (tests[QUICK] && (filter.quick || model?.key === QUICK)) return { test: tests[QUICK], quick: true }
  return null
}

export const isSignificant = (test: GroupTest, filter: MeasureFilter) => (filter.basis === 'p' ? test.p : test.fdr) < filter.level

/** "+1.76 SD" */
export const formatEffect = (effect: number) => `${effect > 0 ? '+' : effect < 0 ? '−' : ''}${Math.abs(effect).toFixed(2)} SD`

/** "0.00067", "0.016", "0.42", "< 0.0001" */
export function formatP(p: number) {
  if (p < 0.0001) return '< 0.0001'
  return p >= 0.1 ? p.toFixed(2) : p.toPrecision(2)
}

/** How a test was adjusted, in words: "the study's analysis, adjusted for amyloid burden, age, brain region and TMA". */
export function testSource(model: TestModel | undefined, quick: boolean, terms: string[]) {
  if (quick) return `quick test (not the study's main analysis), adjusted for ${words(terms)}`
  return `the study's analysis, adjusted for ${model?.adjusted ?? words(terms)}`
}

const TERM_WORDS: Record<string, string> = { region: 'brain region', tma: 'TMA', la: 'amyloid burden', age: 'age', e4_count: 'APOE ε4' }
function words(terms: string[]) {
  const list = terms.map((t) => TERM_WORDS[t] ?? t)
  if (list.length < 2) return list[0] ?? 'nothing'
  return `${list.slice(0, -1).join(', ')} and ${list[list.length - 1]}`
}
