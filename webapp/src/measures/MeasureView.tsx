import { useMemo, useState } from 'react'
import type { Row } from '../api'
import { enc } from '../api'
import type { Point } from '../charts'
import { GroupLegend, GroupStrip, quantile, useGroupOrder } from '../charts'
import { Empty, GroupBadge } from '../components'
import { useStored } from '../hooks'
import { Advanced } from '../setup/parts'
import type { Measure, TableColumns } from './data'
import { ID_LABELS, downloadCsv, formatNumber, measureTitle, tableUrl, useShared } from './data'
import { Choice, Pager, Waiting } from './parts'
import type { MeasureFilter, TableStats } from './stats'
import { GroupDifferences, IccChip, TestLine } from './StatsParts'

type Facet = 'region' | 'tma' | ''
const FACET_TEXT: Record<Facet, string> = { region: 'Brain region', tma: 'TMA', '': 'Nothing' }

/** One measure: the plot by diagnosis group, a summary per group and the value of every donor-region (or core). */
export function MeasureView({ table, info, measure, stats, filter }: {
  table: string; info: TableColumns; measure: Measure; stats: TableStats | null; filter: MeasureFilter
}) {
  const [facet, setFacet] = useStored<Facet>('stainid.results.facet', 'region')
  const [log, setLog] = useState(false)
  const split = facet && info.groupable.includes(facet) ? facet : ''
  const idColumn = info.columns.includes('core_id') ? 'core_id' : 'sample_region_id'
  const url = `${tableUrl(table, 'feature')}?column=${enc(measure.column)}&group=disease_group${split ? `&facet=${split}` : ''}`
  const points = useShared<{ points: Row[] }>(url)
  const canLog = !/0–1|1 = /.test(measure.unit)
  const facetOptions = (['region', 'tma', ''] as Facet[]).filter((f) => !f || info.groupable.includes(f)).map((f) => [f, FACET_TEXT[f]] as [Facet, string])

  return (
    <div className="measure-view">
      <div className="measure-head">
        <h2>{measure.label}</h2>
        {measure.unit && <span className="badge">{measure.unit}</span>}
        {stats ? <IccChip icc={stats.icc[measure.column]} long /> : <span className="muted small">Loading ICC…</span>}
      </div>
      {measure.description && <p className="secondary measure-meaning">{measure.description}</p>}
      {stats && <TestLine stats={stats} filter={filter} column={measure.column} />}
      <div className="row chart-controls">
        <span className="small secondary">Side by side by</span>
        <Choice label="Side by side by" value={split} options={facetOptions} onChange={setFacet} />
        <span className="spacer" />
        {canLog && <label className="layer-toggle"><input type="checkbox" checked={log} onChange={() => setLog(!log)} />Log scale</label>}
      </div>
      <Waiting state={points} label="Loading the values…">
        {(data) => {
          const list: Point[] = data.points.map((p) => ({
            group: String(p.disease_group ?? '—'), value: Number(p[measure.column]), facet: split && p[split] != null ? String(p[split]) : undefined,
            id: String(p[idColumn] ?? p.donor_id ?? ''),
          }))
          if (!list.length) return <Empty>No values for this measure yet.</Empty>
          return (
            <>
              <Legend points={list} />
              <GroupStrip points={list} yLabel={measureTitle(measure)} facetLabel={FACET_TEXT[split]} log={log && canLog} format={formatNumber} />
              <p className="muted small chart-note">Each dot is one {idColumn === 'core_id' ? 'core' : 'donor-region'}; hover a dot for its name.
                Boxes show the middle half of the values and the line the median. These are raw values, for looking at the data;
                the tests of the differences between groups are below.</p>
              {stats && <GroupDifferences stats={stats} filter={filter} column={measure.column} />}
              <GroupSummary points={list} measure={measure} facetLabel={split ? FACET_TEXT[split] : ''} />
              <ValueList rows={data.points} measure={measure} idColumn={idColumn} />
            </>
          )
        }}
      </Waiting>
      <Advanced title="Where this comes from">
        <p className="small secondary" style={{ margin: 0 }}>Column <code>{measure.column}</code> of <code>{table}</code>.
          {' '}The table has {info.rows.toLocaleString()} rows and {info.columns.length} columns.</p>
      </Advanced>
    </div>
  )
}

function Legend({ points }: { points: Point[] }) {
  const groups = useGroupOrder(points)
  return <GroupLegend groups={groups} />
}

/** Count, median and middle half per group (and per region). */
function GroupSummary({ points, measure, facetLabel }: { points: Point[]; measure: Measure; facetLabel: string }) {
  const groups = useGroupOrder(points)
  const facets = useMemo(() => Array.from(new Set(points.map((p) => p.facet ?? ''))).sort(), [points])
  const rows = facets.flatMap((facet) => groups.map((group) => {
    const sorted = points.filter((p) => p.group === group && (p.facet ?? '') === facet).map((p) => p.value).sort((a, b) => a - b)
    return { facet, group, n: sorted.length, median: quantile(sorted, 0.5), q1: quantile(sorted, 0.25), q3: quantile(sorted, 0.75), min: sorted[0], max: sorted[sorted.length - 1] }
  })).filter((r) => r.n > 0)
  const unit = measure.unit ? ` (${measure.unit})` : ''
  return (
    <section className="results-block">
      <h3>Summary by group</h3>
      <div className="table-wrap">
        <table>
          <thead><tr>{facetLabel && <th>{facetLabel}</th>}<th>Diagnosis group</th><th className="num">Values</th><th className="num">Median{unit}</th>
            <th className="num">Middle half</th><th className="num">Lowest – highest</th></tr></thead>
          <tbody>{rows.map((r) => (
            <tr key={`${r.facet}-${r.group}`}>
              {facetLabel && <td>{r.facet}</td>}<td><GroupBadge group={r.group} /></td><td className="num">{r.n}</td>
              <td className="num"><b>{formatNumber(r.median)}</b></td><td className="num">{formatNumber(r.q1)} – {formatNumber(r.q3)}</td>
              <td className="num">{formatNumber(r.min)} – {formatNumber(r.max)}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
    </section>
  )
}

const PAGE = 20

/** The value of every donor-region (or core), sortable and a page at a time, with a CSV download. */
function ValueList({ rows, measure, idColumn }: { rows: Row[]; measure: Measure; idColumn: string }) {
  const [sort, setSort] = useState<{ column: string; desc: boolean }>({ column: measure.column, desc: true })
  const [search, setSearch] = useState('')
  const [offset, setOffset] = useState(0)
  const columns = useMemo(() => [idColumn, 'donor_id', 'region', 'disease_group'].filter((c, i, all) => all.indexOf(c) === i && rows.some((r) => r[c] != null)), [rows, idColumn])
  const shown = useMemo(() => {
    const word = search.toLowerCase()
    const hits = word ? rows.filter((r) => columns.some((c) => String(r[c] ?? '').toLowerCase().includes(word))) : rows
    const key = sort.column === measure.column || columns.includes(sort.column) ? sort.column : measure.column
    return [...hits].sort((a, b) => {
      const [x, y] = [a[key], b[key]]
      const order = typeof x === 'number' && typeof y === 'number' ? x - y : String(x ?? '').localeCompare(String(y ?? ''), undefined, { numeric: true })
      return sort.desc ? -order : order
    })
  }, [rows, columns, search, sort, measure.column])
  const header = (column: string, text: string, numeric = false) => (
    <th className={numeric ? 'num' : ''} aria-sort={sort.column === column ? (sort.desc ? 'descending' : 'ascending') : 'none'}>
      <button className="sort-btn" onClick={() => { setSort({ column, desc: sort.column === column ? !sort.desc : numeric }); setOffset(0) }}>
        {text}{sort.column === column ? (sort.desc ? ' ↓' : ' ↑') : ''}
      </button>
    </th>
  )
  const what = idColumn === 'core_id' ? 'core' : 'donor-region'
  return (
    <section className="results-block">
      <div className="row">
        <h3 style={{ margin: 0 }}>Value per {what}</h3>
        <span className="spacer" />
        <input type="search" placeholder="Find a donor or region" value={search} onChange={(e) => { setSearch(e.target.value); setOffset(0) }} aria-label={`Find a ${what}`} />
        <button className="btn small" onClick={() => downloadCsv(`${measure.column}_by_${what}.csv`, [...columns, measure.column], shown)}>Download these values</button>
      </div>
      <div className="table-wrap" style={{ marginTop: 8 }}>
        <table>
          <thead><tr>{columns.map((c) => header(c, ID_LABELS[c] ?? c))}{header(measure.column, measureTitle(measure), true)}</tr></thead>
          <tbody>{shown.slice(offset, offset + PAGE).map((r, i) => (
            <tr key={`${r[idColumn]}-${i}`}>
              {columns.map((c) => <td key={c}>{c === 'disease_group' ? <GroupBadge group={String(r[c])} /> : String(r[c] ?? '—')}</td>)}
              <td className="num">{formatNumber(r[measure.column])}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
      {!shown.length && <p className="muted small">Nothing matches “{search}”.</p>}
      <Pager offset={offset} size={PAGE} total={shown.length} onChange={setOffset} />
    </section>
  )
}
