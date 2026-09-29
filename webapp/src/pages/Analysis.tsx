import { useEffect, useMemo, useState } from 'react'
import type { Row, TableInfo } from '../api'
import { enc } from '../api'
import type { Point } from '../charts'
import { GroupStrip } from '../charts'
import { Empty, ErrorNote, PageHead, fmt } from '../components'
import { useFetch, useStored } from '../hooks'

export default function Analysis() {
  const tables = useFetch<TableInfo[]>('/api/tables')
  const [table, setTable] = useStored('stainid.analysis.table', 'fields_donor_region.csv')
  const [tab, setTab] = useState<'explore' | 'table'>('explore')
  const donorTables = useMemo(() => (tables.data ?? []).filter((t) => t.donor_region), [tables.data])
  const chosen = (tables.data ?? []).find((t) => t.name === table) ? table : donorTables[0]?.name

  return (
    <>
      <PageHead title="Analysis" subtitle="Explore any output table: feature distributions by group, split by region, or the raw rows.">
        <label className="field">Table
          <select value={chosen ?? ''} onChange={(e) => setTable(e.target.value)} style={{ maxWidth: 420 }}>
            <optgroup label="Donor-region / core tables">{donorTables.map((t) => <option key={t.name}>{t.name}</option>)}</optgroup>
            <optgroup label="Other tables">{(tables.data ?? []).filter((t) => !t.donor_region).map((t) => <option key={t.name}>{t.name}</option>)}</optgroup>
          </select>
        </label>
      </PageHead>
      <ErrorNote error={tables.error} />
      <div className="tabs">
        <button className={`tab${tab === 'explore' ? ' active' : ''}`} onClick={() => setTab('explore')}>Feature explorer</button>
        <button className={`tab${tab === 'table' ? ' active' : ''}`} onClick={() => setTab('table')}>Rows</button>
      </div>
      {chosen ? (tab === 'explore' ? <Explorer table={chosen} /> : <TableView table={chosen} />) : <Empty>No tables yet — run and aggregate a pipeline first.</Empty>}
    </>
  )
}

function Explorer({ table }: { table: string }) {
  const columns = useFetch<{ columns: string[]; numeric: string[]; groupable: string[] }>(`/api/tables/${enc(table)}/columns`)
  const [feature, setFeature] = useState('')
  const [facet, setFacet] = useState('region')
  const [log, setLog] = useState(false)
  const [filter, setFilter] = useState('')
  useEffect(() => {
    if (columns.data && !columns.data.numeric.includes(feature)) setFeature(columns.data.numeric.find((c) => c.includes('density')) ?? columns.data.numeric[0] ?? '')
  }, [columns.data, feature])
  const data = useFetch<{ points: Row[] }>(feature && columns.data?.groupable.includes('disease_group')
    ? `/api/tables/${enc(table)}/feature?column=${enc(feature)}&group=disease_group${facet && columns.data?.columns.includes(facet) ? `&facet=${facet}` : ''}` : null)
  const points: Point[] = (data.data?.points ?? []).map((p) => ({
    group: String(p.disease_group), value: Number(p[feature]), facet: facet && p[facet] != null ? String(p[facet]) : undefined, id: String(p.sample_region_id ?? p.core_id ?? ''),
  }))
  const numeric = (columns.data?.numeric ?? []).filter((c) => c.toLowerCase().includes(filter.toLowerCase()))

  if (columns.data && !columns.data.groupable.includes('disease_group')) return <Empty>This table has no disease_group column — use the Rows tab.</Empty>
  return (
    <div className="grid" style={{ gridTemplateColumns: '300px 1fr', alignItems: 'start' }}>
      <div className="card">
        <h3>Feature</h3>
        <input type="text" placeholder="filter features…" value={filter} onChange={(e) => setFilter(e.target.value)} style={{ width: '100%', marginBottom: 8 }} />
        <div className="table-wrap" style={{ maxHeight: 460 }}>
          <table><tbody>
            {numeric.map((c) => <tr key={c} className={`clickable${c === feature ? ' selected' : ''}`} onClick={() => setFeature(c)}><td>{c}</td></tr>)}
          </tbody></table>
        </div>
      </div>
      <div className="card">
        <div className="row" style={{ marginBottom: 10 }}>
          <h2 style={{ margin: 0 }}>{feature}</h2><span className="spacer" />
          <label className="field">Split by<select value={facet} onChange={(e) => setFacet(e.target.value)}><option value="">none</option><option value="region">region</option><option value="tma">TMA</option></select></label>
          <label className="layer-toggle"><input type="checkbox" checked={log} onChange={() => setLog(!log)} />log scale</label>
        </div>
        <ErrorNote error={data.error} />
        {points.length ? <GroupStrip points={points} label={feature} log={log} /> : <p className="muted">No values.</p>}
        <p className="muted small">Each point is one row of the table (donor-region or core). Boxes show median and interquartile range. Raw values — adjusted AD vs ASYMAD contrasts come from the study's mixed models.</p>
      </div>
    </div>
  )
}

function TableView({ table }: { table: string }) {
  const rows = useFetch<{ total: number; rows: Row[] }>(`/api/tables/${enc(table)}/rows?limit=1000`)
  const columns = rows.data?.rows.length ? Object.keys(rows.data.rows[0]) : []
  return (
    <div className="card">
      <div className="row"><h2 style={{ margin: 0 }}>{table}</h2><span className="muted small">{rows.data?.total ?? 0} rows{rows.data && rows.data.total > 1000 ? ' (first 1000 shown)' : ''}</span></div>
      <ErrorNote error={rows.error} />
      <div className="table-wrap" style={{ marginTop: 10 }}>
        <table>
          <thead><tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
          <tbody>{(rows.data?.rows ?? []).map((r, i) => <tr key={i}>{columns.map((c) => <td key={c} className={typeof r[c] === 'number' ? 'num' : ''}>{fmt(r[c], 4)}</td>)}</tr>)}</tbody>
        </table>
      </div>
    </div>
  )
}
