import { useProject } from '../project'
import { Advanced } from '../setup/parts'
import type { Comparison, Icc, MeasureFilter, TableStats } from './stats'
import { ICC_HELP, QUICK, chosenComparison, chosenModel, formatEffect, formatP, iccHover, iccLevel, isSignificant, testOf, testSource } from './stats'

/** "ICC 0.74", coloured good / fair / weak, with what it means on hover. */
export function IccChip({ icc, long = false }: { icc: Icc | undefined; long?: boolean }) {
  if (!icc) return long ? <span className="icc-chip icc-none" title={ICC_HELP}>ICC not available</span> : null
  const level = iccLevel(icc.icc)
  return <span className={`icc-chip icc-${level}`} title={iccHover(icc)}>ICC {icc.icc.toFixed(2)}{long && ` · ${level}`}</span>
}

export function useComparisonName() {
  const { groupLabel } = useProject()
  return (c: Comparison) => `${groupLabel(c.first)} vs ${groupLabel(c.second)}`
}

/** "AD vs ASYMAD: +1.76 SD, p 0.00067, FDR 0.0041 — the study's analysis, adjusted for …" for the chosen comparison. */
export function TestLine({ stats, filter, column }: { stats: TableStats; filter: MeasureFilter; column: string }) {
  const name = useComparisonName()
  const comparison = chosenComparison(stats, filter)
  if (!comparison) return null
  const model = chosenModel(comparison, filter)
  const found = testOf(comparison, model, column, filter)
  if (!found) {
    const why = comparison.tests[column]?.[QUICK] ? 'quick tests are turned off under Test options' : 'too few values in one of the groups'
    return <p className="small muted test-line">{name(comparison)}: no test for this measure ({why}).</p>
  }
  const { test, quick } = found
  const significant = isSignificant(test, filter)
  return (
    <p className="small test-line">
      <span className={`test-mark${significant ? ' significant' : ''}`}>{significant ? `Significant, ${filter.basis === 'p' ? 'p' : 'FDR'} < ${filter.level}` : 'Not significant'}</span>
      {' '}<b>{name(comparison)}: {formatEffect(test.effect)}</b>, p {formatP(test.p)}, FDR {formatP(test.fdr)}
      <span className="secondary"> — {testSource(model, quick, test.terms)}. {comparison.within}</span>
    </p>
  )
}

/** Every comparison of the measure in a small table. */
export function GroupDifferences({ stats, filter, column }: { stats: TableStats; filter: MeasureFilter; column: string }) {
  const name = useComparisonName()
  const chosen = chosenComparison(stats, filter)
  const rows = stats.comparisons.map((c) => {
    const model = chosenModel(c, c.key === chosen?.key ? filter : { ...filter, model: '' })
    return { comparison: c, model, found: testOf(c, model, column, filter) }
  }).filter((r) => r.found)
  if (!rows.length) return null
  return (
    <section className="results-block">
      <h3>Differences between groups</h3>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Comparison</th><th className="num">Difference</th><th className="num">p</th><th className="num">FDR</th><th className="num">Values</th><th>Test</th></tr></thead>
          <tbody>{rows.map(({ comparison, model, found }) => (
            <tr key={comparison.key} className={isSignificant(found!.test, filter) ? 'significant-row' : ''}>
              <td>{name(comparison)}</td>
              <td className="num"><b>{formatEffect(found!.test.effect)}</b></td>
              <td className="num">{formatP(found!.test.p)}</td>
              <td className="num">{formatP(found!.test.fdr)}</td>
              <td className="num">{found!.test.n ?? '—'}</td>
              <td className="wrap small secondary">{testSource(model, found!.quick, found!.test.terms)}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
      <p className="muted small chart-note">Difference: first group minus second, in standard deviations of the measure (densities on a log scale).
        FDR is the p value corrected for testing many measures. {rows[0].comparison.within}</p>
    </section>
  )
}

/** The "Significant only" and "Reproducible only" switches above the measure list, their options under Advanced. */
export function MeasureFilters({ state, filter, onChange, counts }: {
  state: { data: TableStats | null; error: string | null; retry: () => void }; filter: MeasureFilter; onChange: (f: MeasureFilter) => void
  counts: { shown: number; total: number; noTest: number; noIcc: number }
}) {
  const name = useComparisonName()
  const stats = state.data
  const set = (part: Partial<MeasureFilter>) => onChange({ ...filter, ...part })
  const comparison = chosenComparison(stats, filter)
  const model = chosenModel(comparison, filter)
  const studyModels = comparison?.models.filter((m) => m.source === 'study') ?? []
  const hasQuick = comparison?.models.some((m) => m.key === QUICK)
  const active = filter.significant || filter.reproducible
  return (
    <div className="measure-filters">
      <div className="measure-filter-row">
        <label className="layer-toggle"><input type="checkbox" checked={filter.significant} disabled={!comparison} onChange={() => set({ significant: !filter.significant })} />Significant only</label>
        {stats && stats.comparisons.length > 0 && (
          <select aria-label="Comparison" value={comparison?.key} onChange={(e) => set({ comparison: e.target.value, model: '' })}>
            {stats.comparisons.map((c) => <option key={c.key} value={c.key}>{name(c)}</option>)}
          </select>
        )}
      </div>
      <label className="layer-toggle" title={ICC_HELP}><input type="checkbox" checked={filter.reproducible} disabled={!stats} onChange={() => set({ reproducible: !filter.reproducible })} />
        Reproducible only (ICC &gt; {filter.minIcc})</label>
      {!stats && !state.error && <span className="muted small"><span className="spinner" aria-hidden="true" /> Loading ICC and group tests…</span>}
      {state.error && <span className="small">Could not load ICC and group tests. <button className="link-btn small" onClick={state.retry}>Try again</button></span>}
      {stats && active && (
        <p className="muted small filter-note">
          {counts.shown} of {counts.total} measures
          {filter.significant && comparison && <>: {name(comparison)}, {filter.basis === 'p' ? 'p' : 'FDR'} &lt; {filter.level}, {model?.source === 'quick' ? 'quick test' : `adjusted for ${model?.adjusted}`}{filter.quick && hasQuick && model?.source !== 'quick' ? ' (quick test where the study has none)' : ''}</>}
          {filter.reproducible && <>{filter.significant ? '; ' : ': '}ICC &gt; {filter.minIcc}</>}.
          {filter.significant && counts.noTest > 0 && <> {counts.noTest} hidden for having no test ({filter.quick || !hasQuick ? 'too few values in a group' : 'quick tests are off'}).</>}
          {filter.reproducible && counts.noIcc > 0 && <> {counts.noIcc} hidden for having no ICC.</>}
        </p>
      )}
      {stats && (
        <Advanced title="Test options">
          <label className="inline-field">Significant when
            <select value={filter.basis} onChange={(e) => set({ basis: e.target.value as MeasureFilter['basis'] })} aria-label="Significance by">
              <option value="fdr">FDR (corrected for many measures)</option><option value="p">p (uncorrected)</option>
            </select>
            &lt;
            <input type="number" min={0.0001} max={0.5} step={0.01} value={filter.level} aria-label="Significance threshold"
              onChange={(e) => { const v = Number(e.target.value); if (v > 0 && v <= 0.5) set({ level: v }) }} />
          </label>
          {studyModels.length > 1 && (
            <label className="inline-field">Study model, adjusted for
              <select value={model?.source === 'study' ? model.key : ''} onChange={(e) => set({ model: e.target.value })} aria-label="Study model">
                {studyModels.map((m) => <option key={m.key} value={m.key}>{m.adjusted}{m.default ? ' (main)' : ''}</option>)}
              </select>
            </label>
          )}
          {hasQuick && studyModels.length > 0 && (
            <label className="layer-toggle"><input type="checkbox" checked={filter.quick} onChange={() => set({ quick: !filter.quick })} />
              Quick test for measures the study did not test</label>
          )}
          <label className="inline-field">Reproducible when ICC &gt;
            <input type="number" min={0} max={0.99} step={0.05} value={filter.minIcc} aria-label="ICC threshold"
              onChange={(e) => { const v = Number(e.target.value); if (v >= 0 && v < 1) set({ minIcc: v }) }} />
          </label>
          <p className="small secondary" style={{ margin: 0 }}>Quick test: a linear model of the measure (densities on a log scale) with brain region and TMA,
            on the TMAs that hold both groups, donors counted once (clustered errors). It is a first look, not the study's main analysis.</p>
          {(stats.icc_files.length > 0 || stats.contrast_files.length > 0) && (
            <details className="small secondary source-files">
              <summary>Study tables used ({stats.contrast_files.length + stats.icc_files.length})</summary>
              <ul className="source-list">{[...stats.contrast_files, ...stats.icc_files].map((f) => <li key={f}><code>{f}</code></li>)}</ul>
              A study value is used only where its table holds the same numbers as the results; other values are computed from the results tables.
            </details>
          )}
        </Advanced>
      )}
    </div>
  )
}
