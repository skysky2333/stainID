import { useMemo, useState } from 'react'
import type { Measure } from './data'
import { measureHover } from './data'

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

export function MeasurePicker({ measures, topics, value, onChange }: {
  measures: Measure[]; topics: Record<string, string>; value: string; onChange: (column: string) => void
}) {
  const [search, setSearch] = useState('')
  const list = useMemo(() => sections(measures, topics, search), [measures, topics, search])
  const item = (m: Measure) => (
    <button key={m.column} className={`measure-item${m.column === value ? ' selected' : ''}`} title={measureHover(m)}
      aria-pressed={m.column === value} onClick={() => onChange(m.column)}>
      <span>{m.label}</span>{m.unit && <span className="muted small">{m.unit}</span>}
    </button>
  )
  return (
    <div className="measure-picker">
      <input type="search" placeholder="Find a measure, e.g. plaque size" value={search} onChange={(e) => setSearch(e.target.value)} aria-label="Find a measure" />
      <nav className="measure-list" aria-label="Measures">
        {list.map((s) => (
          <div key={s.title} className="measure-section">
            <div className="measure-section-title">{s.title}</div>
            {s.measures.map(item)}
            {s.outlines.length > 0 && <><div className="measure-subtitle">Shape, from object outlines</div>{s.outlines.map(item)}</>}
          </div>
        ))}
        {!list.length && <p className="muted small" style={{ padding: 8 }}>No measure matches “{search}”.</p>}
      </nav>
    </div>
  )
}
