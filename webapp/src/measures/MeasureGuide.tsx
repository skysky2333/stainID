import { useState } from 'react'
import type { TableColumns } from './data'
import { ID_LABELS } from './data'

/** What every column of the table means, grouped by topic, with the column's name in the file (for R, Python, Excel). */
export function MeasureGuide({ info, onPick }: { info: TableColumns; onPick: (column: string) => void }) {
  const [search, setSearch] = useState('')
  const words = search.toLowerCase().split(/\s+/).filter(Boolean)
  const matches = (text: string) => words.every((w) => text.toLowerCase().includes(w))
  const ids = info.columns.filter((c) => !info.measures.some((m) => m.column === c) && matches(`${c} ${ID_LABELS[c] ?? ''} ${info.descriptions[c] ?? ''}`))
  const topics = Object.entries(info.topics)
    .map(([topic, title]) => ({ title, measures: info.measures.filter((m) => m.topic === topic && matches(`${m.label} ${m.unit} ${m.description} ${m.column}`)) }))
    .filter((t) => t.measures.length)
  return (
    <div className="measure-guide">
      <div className="row">
        <input type="search" placeholder="Find a measure or column" value={search} onChange={(e) => setSearch(e.target.value)} aria-label="Find a measure or column" />
        <span className="muted small">Densities are per mm² of usable tissue (folds, holes and artifacts are left out). Fractions run from 0 to 1.</span>
      </div>
      {topics.map((t) => (
        <section key={t.title} className="results-block">
          <h3>{t.title}</h3>
          <div className="table-wrap">
            <table className="guide-table">
              <colgroup><col style={{ width: '26%' }} /><col style={{ width: '14%' }} /><col style={{ width: '38%' }} /><col style={{ width: '22%' }} /></colgroup>
              <thead><tr><th>Measure</th><th>Unit</th><th>What it means</th><th>Column in the file</th></tr></thead>
              <tbody>{t.measures.map((m) => (
                <tr key={m.column}>
                  <td className="wrap"><button className="link-btn" title="Plot this measure" onClick={() => onPick(m.column)}>{m.label}</button></td>
                  <td className="wrap secondary">{m.unit || '—'}</td>
                  <td className="wrap">{m.description || <span className="muted">—</span>}</td>
                  <td><code className="muted">{m.column}</code></td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        </section>
      ))}
      {ids.length > 0 && (
        <section className="results-block">
          <h3>Names and groups</h3>
          <div className="table-wrap">
            <table className="guide-table">
              <colgroup><col style={{ width: '26%' }} /><col style={{ width: '52%' }} /><col style={{ width: '22%' }} /></colgroup>
              <thead><tr><th>Column</th><th>What it means</th><th>Column in the file</th></tr></thead>
              <tbody>{ids.map((c) => (
                <tr key={c}><td>{ID_LABELS[c] ?? c}</td><td className="wrap">{info.descriptions[c] || <span className="muted">—</span>}</td><td><code className="muted">{c}</code></td></tr>
              ))}</tbody>
            </table>
          </div>
        </section>
      )}
      {!topics.length && !ids.length && <p className="muted">Nothing matches “{search}”.</p>}
    </div>
  )
}
