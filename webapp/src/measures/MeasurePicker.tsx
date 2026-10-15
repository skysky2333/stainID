import { useMemo, useState } from 'react'
import type { Measure } from './data'
import { measureHover } from './data'
import type { MeasureFilter, TableStats } from './stats'
import { chosenComparison, chosenModel, formatEffect, formatP, isSignificant, testOf } from './stats'
import { IccChip, MeasureFilters } from './StatsParts'

interface Section { title: string; measures: Measure[]; outlines: Measure[] }

/** Sections of the measure list: the main measures first, then one per topic (neurons, plaques, tau …). */
function sections(measures: Measure[], topics: Record<string, string>, search: string): Section[] {
  const words = search.toLowerCase().split(/\s+/).filter(Boolean)
  const shown = measures.filter((m) => words.every((w) => `${m.label} ${m.unit} ${m.description} ${m.column} ${topics[m.topic] ?? ''}`.toLowerCase().includes(w)))
  const out: Section[] = []
  const key = shown.filter((m) => m.key)
  if (key.length && !search) out.push({ title: 'Main measures', measures: key, outlines: [] })
  for (const [topic, title] of Object.entries(topics)) {
    const inTopic = shown.filter((m) => m.topic === topic)
    if (inTopic.length) out.push({ title, measures: inTopic.filter((m) => !m.outline), outlines: inTopic.filter((m) => m.outline) })
  }
  return out
}

/** The measures that pass the "Significant only" and "Reproducible only" filters, and how many each one hid. */
function filtered(measures: Measure[], stats: TableStats | null, filter: MeasureFilter) {
  if (!stats || !(filter.significant || filter.reproducible)) return { shown: measures, noTest: 0, noIcc: 0 }
  const comparison = chosenComparison(stats, filter)
  const model = chosenModel(comparison, filter)
  let noTest = 0
  let noIcc = 0
  const shown = measures.filter((m) => {
    if (filter.significant) {
      const found = testOf(comparison, model, m.column, filter)
      if (!found) { noTest += 1; return false }
      if (!isSignificant(found.test, filter)) return false
    }
    if (filter.reproducible) {
      const icc = stats.icc[m.column]
      if (!icc) { noIcc += 1; return false }
      if (!(icc.icc > filter.minIcc)) return false
    }
    return true
  })
  return { shown, noTest, noIcc }
}

export function MeasurePicker({ measures, topics, value, onChange, stats, filter, onFilter }: {
  measures: Measure[]; topics: Record<string, string>; value: string; onChange: (column: string) => void
  stats: { data: TableStats | null; error: string | null; retry: () => void }; filter: MeasureFilter; onFilter: (f: MeasureFilter) => void
}) {
  const [search, setSearch] = useState('')
  const result = useMemo(() => filtered(measures, stats.data, filter), [measures, stats.data, filter])
  const list = useMemo(() => sections(result.shown, topics, search), [result.shown, topics, search])
  const comparison = chosenComparison(stats.data, filter)
  const model = chosenModel(comparison, filter)
  const item = (m: Measure) => {
    const found = filter.significant ? testOf(comparison, model, m.column, filter) : null
    return (
      <button key={m.column} className={`measure-item${m.column === value ? ' selected' : ''}`} title={measureHover(m)}
        aria-pressed={m.column === value} onClick={() => onChange(m.column)}>
        <span className="measure-item-top"><span>{m.label}</span><IccChip icc={stats.data?.icc[m.column]} /></span>
        {m.unit && <span className="muted small">{m.unit}</span>}
        {found && <span className="small measure-test">{formatEffect(found.test.effect)} · p {formatP(found.test.p)}{found.quick ? ' · quick test' : ''}</span>}
      </button>
    )
  }
  const empty = search ? `No measure matches “${search}”.` : 'No measure passes these filters.'
  return (
    <div className="measure-picker">
      <MeasureFilters state={stats} filter={filter} onChange={onFilter}
        counts={{ shown: result.shown.length, total: measures.length, noTest: result.noTest, noIcc: result.noIcc }} />
      <input type="search" placeholder="Find a measure, e.g. plaque size" value={search} onChange={(e) => setSearch(e.target.value)} aria-label="Find a measure" />
      <nav className="measure-list" aria-label="Measures">
        {list.map((s) => (
          <div key={s.title} className="measure-section">
            <div className="measure-section-title">{s.title}</div>
            {s.measures.map(item)}
            {s.outlines.length > 0 && <><div className="measure-subtitle">Shape, from object outlines</div>{s.outlines.map(item)}</>}
          </div>
        ))}
        {!list.length && <p className="muted small" style={{ padding: 8 }}>{empty}</p>}
      </nav>
    </div>
  )
}
