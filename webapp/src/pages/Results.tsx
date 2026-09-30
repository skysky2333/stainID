import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import type { Row, TableInfo } from '../api'
import { downloadUrl, enc } from '../api'
import type { Point } from '../charts'
import { GroupStrip } from '../charts'
import { Empty, ErrorNote, HelpBox, PageHead, PathLine, Term, fmt } from '../components'
import { useFetch, useStored } from '../hooks'

const PREFERRED = ['neun_profile_density_mm2', 'v2_plaque_density_mm2', 'tau_neuron_density_mm2']

interface Columns { columns: string[]; numeric: string[]; groupable: string[]; descriptions: Record<string, string> }

export default function Results() {
  const tables = useFetch<TableInfo[]>('/api/tables')
  const [table, setTable] = useStored('stainid.results.table', 'results_donor_region.csv')
  const [tab, setTab] = useState<'plot' | 'columns' | 'rows'>('plot')
  const list = tables.data ?? []
  const main = list.find((t) => t.name === 'results_donor_region.csv')
  const chosen = main ? list.find((t) => t.name === table) ?? main : undefined

  return (
    <>
      <PageHead title="Results" subtitle="The final tables, what every column means, and each measure plotted by diagnostic group." />
      <HelpBox id="results">
        <ul>
          <li>The main table has <b>one row per <Term>donor-region</Term></b> and one column per measure, for every stain. It opens in Excel, R or Python.</li>
          <li>Counts and areas are pooled over all fields and replicate cores before dividing, so every <Term>density</Term> is per mm² of usable tissue.</li>
          <li>The plots show raw values by group. They are for looking at the data; formal comparisons should adjust for age, TMA and other covariates.</li>
        </ul>
      </HelpBox>
      <ErrorNote error={tables.error} />
      {main ? (
        <div className="card main-result">
          <div className="row"><h2 style={{ margin: 0 }}>Main results table</h2><span className="spacer" />
            <a className="btn primary" href={downloadUrl(main.path)}>Download CSV</a></div>
          <p className="secondary small">{main.description} {main.columns} columns.</p>
          <PathLine path={main.path} exists />
        </div>
      ) : (
        <Empty>No results table yet. Run <Link to="/workflow#summarize">Make results tables</Link> once the detection steps are done.</Empty>
      )}

      {chosen && (
        <div className="card" style={{ marginTop: 16 }}>
          <div className="row">
            <label className="field">Table
              <select value={chosen.name} onChange={(e) => setTable(e.target.value)} style={{ maxWidth: 460 }}>
                <optgroup label="Results">{list.filter((t) => t.main).map((t) => <option key={t.name} value={t.name}>{t.name} — {t.description}</option>)}</optgroup>
                <optgroup label="Other tables in the project">{list.filter((t) => !t.main).map((t) => <option key={t.name}>{t.name}</option>)}</optgroup>
              </select>
            </label>
            <span className="spacer" />
            <a className="btn small" href={downloadUrl(chosen.path)}>Download this table</a>
          </div>
          <div className="tabs" style={{ marginTop: 12 }}>
            <button className={`tab${tab === 'plot' ? ' active' : ''}`} onClick={() => setTab('plot')}>Plot by group</button>
            <button className={`tab${tab === 'columns' ? ' active' : ''}`} onClick={() => setTab('columns')}>What the columns mean</button>
            <button className={`tab${tab === 'rows' ? ' active' : ''}`} onClick={() => setTab('rows')}>Rows</button>
          </div>
          {tab === 'plot' && <Explorer table={chosen.name} />}
          {tab === 'columns' && <Dictionary table={chosen.name} />}
          {tab === 'rows' && <TableView table={chosen.name} />}
        </div>
      )}
    </>
  )
}

function Explorer({ table }: { table: string }) {
  const columns = useFetch<Columns>(`/api/tables/${enc(table)}/columns`)
  const [feature, setFeature] = useState('')
  const [facet, setFacet] = useState('region')
  const [log, setLog] = useState(false)
  const [filter, setFilter] = useState('')
  useEffect(() => {
    if (columns.data && !columns.data.numeric.includes(feature)) setFeature(PREFERRED.find((c) => columns.data!.numeric.includes(c)) ?? columns.data.numeric.find((c) => c.includes('density')) ?? columns.data.numeric[0] ?? '')
  }, [columns.data, feature])
  const data = useFetch<{ points: Row[] }>(feature && columns.data?.groupable.includes('disease_group')
    ? `/api/tables/${enc(table)}/feature?column=${enc(feature)}&group=disease_group${facet && columns.data?.columns.includes(facet) ? `&facet=${facet}` : ''}` : null)
  const points: Point[] = (data.data?.points ?? []).map((p) => ({
    group: String(p.disease_group), value: Number(p[feature]), facet: facet && p[facet] != null ? String(p[facet]) : undefined, id: String(p.sample_region_id ?? p.core_id ?? ''),
  }))
  const descriptions = columns.data?.descriptions ?? {}
  const numeric = useMemo(() => (columns.data?.numeric ?? []).filter((c) => `${c} ${descriptions[c] ?? ''}`.toLowerCase().includes(filter.toLowerCase())), [columns.data, filter, descriptions])

  if (columns.data && !columns.data.groupable.includes('disease_group')) return <Empty>This table has no diagnostic group column — use the Rows tab.</Empty>
  return (
    <div className="grid" style={{ gridTemplateColumns: '320px 1fr', alignItems: 'start' }}>
      <div>
        <input type="text" placeholder="search measures…" value={filter} onChange={(e) => setFilter(e.target.value)} style={{ width: '100%', marginBottom: 8 }} />
        <div className="table-wrap" style={{ maxHeight: 520 }}>
          <table><tbody>
            {numeric.map((c) => (
              <tr key={c} className={`clickable${c === feature ? ' selected' : ''}`} onClick={() => setFeature(c)}>
                <td className="wrap"><div>{c}</div>{descriptions[c] && <div className="muted small">{descriptions[c]}</div>}</td>
              </tr>
            ))}
          </tbody></table>
        </div>
      </div>
      <div>
        <div className="row" style={{ marginBottom: 10 }}>
          <div><h2 style={{ margin: 0 }}>{feature}</h2>{descriptions[feature] && <p className="secondary small" style={{ margin: '2px 0 0' }}>{descriptions[feature]}</p>}</div>
          <span className="spacer" />
          <label className="field">Split by<select value={facet} onChange={(e) => setFacet(e.target.value)}><option value="">nothing</option><option value="region">brain region</option><option value="tma">TMA</option></select></label>
          <label className="layer-toggle"><input type="checkbox" checked={log} onChange={() => setLog(!log)} />log scale</label>
        </div>
        <ErrorNote error={data.error} />
        {points.length ? <GroupStrip points={points} label={feature} log={log} /> : <p className="muted">No values.</p>}
        <p className="muted small">Each dot is one row of the table. Boxes show the median and middle half of the values.</p>
      </div>
    </div>
  )
}

function Dictionary({ table }: { table: string }) {
  const columns = useFetch<Columns>(`/api/tables/${enc(table)}/columns`)
  const [filter, setFilter] = useState('')
  const list = (columns.data?.columns ?? []).filter((c) => `${c} ${columns.data?.descriptions[c] ?? ''}`.toLowerCase().includes(filter.toLowerCase()))
  return (
    <>
      <input type="text" placeholder="search columns…" value={filter} onChange={(e) => setFilter(e.target.value)} style={{ width: 320, marginBottom: 8 }} />
      <div className="table-wrap">
        <table>
          <thead><tr><th>Column</th><th>Meaning</th></tr></thead>
          <tbody>{list.map((c) => <tr key={c}><td><code>{c}</code></td><td className="wrap">{columns.data?.descriptions[c] || <span className="muted">—</span>}</td></tr>)}</tbody>
        </table>
      </div>
    </>
  )
}

function TableView({ table }: { table: string }) {
  const rows = useFetch<{ total: number; rows: Row[] }>(`/api/tables/${enc(table)}/rows?limit=1000`)
  const columns = rows.data?.rows.length ? Object.keys(rows.data.rows[0]) : []
  return (
    <>
      <p className="muted small">{rows.data?.total ?? 0} rows{rows.data && rows.data.total > 1000 ? ' (first 1000 shown — download the table for all)' : ''}</p>
      <ErrorNote error={rows.error} />
      <div className="table-wrap">
        <table>
          <thead><tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
          <tbody>{(rows.data?.rows ?? []).map((r, i) => <tr key={i}>{columns.map((c) => <td key={c} className={typeof r[c] === 'number' ? 'num' : ''}>{fmt(r[c], 4)}</td>)}</tr>)}</tbody>
        </table>
      </div>
    </>
  )
}
