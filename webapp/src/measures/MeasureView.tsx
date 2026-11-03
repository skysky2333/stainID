import { useMemo, useState } from 'react'
import type { Row } from '../api'
import { enc } from '../api'
import type { NoValuePoint, Point } from '../charts'
import { GroupLegend, GroupStrip, quantile, useGroupOrder } from '../charts'
import { Empty, GroupBadge } from '../components'
import { useStored } from '../hooks'
import { useProject } from '../project'
import { Advanced } from '../setup/parts'
import type { Measure, TableColumns } from './data'
import { ID_LABELS, downloadCsv, formatNumber, measureTitle, reasonCounts, tableUrl, useShared } from './data'
import { Choice, NoValue, Pager, Waiting } from './parts'
import type { MeasureFilter, TableStats } from './stats'
import { GroupDifferences, IccChip, IccMeaning, TestLine } from './StatsParts'

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
  const points = useShared<{ points: Row[]; gaps?: Row[] | null }>(url)
  // a log axis cannot hold a value below zero: no log scale for such a measure, rather than drawing it somewhere it is not
  const canLog = !/0–1|1 = /.test(measure.unit) && !points.data?.points.some((p) => Number(p[measure.column]) < 0)
  const facetOptions = (['region', 'tma', ''] as Facet[]).filter((f) => !f || info.groupable.includes(f)).map((f) => [f, FACET_TEXT[f]] as [Facet, string])

  return (
    <div className="measure-view">
      <div className="measure-head">
        <h2>{measure.label}</h2>
        {measure.unit && <span className="badge">{measure.unit}</span>}
        {stats ? <IccChip icc={stats.icc[measure.column]} long /> : <span className="muted small">Loading replicate agreement…</span>}
      </div>
      {measure.description && <p className="secondary measure-meaning">{measure.description}</p>}
      {stats && <IccMeaning icc={stats.icc[measure.column]} />}
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
          const gaps: NoValuePoint[] | null = data.gaps ? data.gaps.map((p) => ({
            group: String(p.disease_group ?? '—'), facet: split && p[split] != null ? String(p[split]) : undefined, id: String(p[idColumn] ?? p.donor_id ?? ''),
            text: String(p.text), why: String(p.why),
          })) : null
          if (!list.length && !gaps?.length) return <Empty>No values for this measure yet.</Empty>
          const what = idColumn === 'core_id' ? 'core' : 'region'
          return (
            <>
              <Legend points={[...list, ...(gaps ?? [])]} />
              {gaps && <ShownLines points={list} gaps={gaps} what={what} />}
              <GroupStrip points={list} gaps={gaps} yLabel={measureTitle(measure)} facetLabel={FACET_TEXT[split]} log={log && canLog} format={formatNumber} />
              <p className="muted small chart-note">Each dot is one {idColumn === 'core_id' ? 'core' : 'donor-region'}; hover a dot for its name.
                Boxes show the middle half of the values and the line the median. These are raw values, for looking at the data;
                the tests of the differences between groups are below.
                {gaps && gaps.length > 0 && <> The hollow marks in the “no value” strip have no value for this measure, so they have no place on the axis and are in no box or test; hover one for why.</>}
                {log && canLog && <> The log scale is drawn as log(value + 1), so a true zero stays on the plot, at 0.</>}</p>
              {stats && <GroupDifferences stats={stats} filter={filter} column={measure.column} />}
              <GroupSummary points={list} gaps={gaps} measure={measure} facetLabel={split ? FACET_TEXT[split] : ''} />
              <ValueList rows={data.points} gaps={data.gaps ?? null} measure={measure} idColumn={idColumn} />
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

/** "Shown 4 of 74 control regions: 68 too few plaques, 2 no grey matter." for each diagnosis group. */
function ShownLines({ points, gaps, what }: { points: Point[]; gaps: NoValuePoint[]; what: string }) {
  const { groupLabel } = useProject()
  const groups = useGroupOrder([...points, ...gaps])
  return (
    <ul className="shown-lines small">
      {groups.map((group) => {
        const shown = points.filter((p) => p.group === group).length
        const counts = countReasons(gaps.filter((p) => p.group === group))
        const total = shown + Object.values(counts).reduce((a, b) => a + b, 0)
        const label = groupLabel(group)
        const name = /^[A-Z][a-z]/.test(label) ? label[0].toLowerCase() + label.slice(1) : label  // "control", but "AD"
        return <li key={group}>Shown <b>{shown.toLocaleString()} of {total.toLocaleString()}</b> {name} {what}{total === 1 ? '' : 's'}{total > shown ? `: ${reasonCounts(counts)}` : ''}.</li>
      })}
    </ul>
  )
}

function countReasons(gaps: { text: string }[]) {
  const counts: Record<string, number> = {}
  for (const gap of gaps) counts[gap.text] = (counts[gap.text] ?? 0) + 1
  return counts
}

function Legend({ points }: { points: { group: string }[] }) {
  const groups = useGroupOrder(points)
  return <GroupLegend groups={groups} />
}

/** Count, median and middle half per group (and per region). */
function GroupSummary({ points, gaps, measure, facetLabel }: { points: Point[]; gaps: NoValuePoint[] | null; measure: Measure; facetLabel: string }) {
  const every = useMemo(() => [...points, ...(gaps ?? [])], [points, gaps])
  const groups = useGroupOrder(every)
  const facets = useMemo(() => Array.from(new Set(every.map((p) => p.facet ?? ''))).sort(), [every])
  const rows = facets.flatMap((facet) => groups.map((group) => {
    const sorted = points.filter((p) => p.group === group && (p.facet ?? '') === facet).map((p) => p.value).sort((a, b) => a - b)
    const none = (gaps ?? []).filter((p) => p.group === group && (p.facet ?? '') === facet)
    return { facet, group, none, n: sorted.length, median: quantile(sorted, 0.5), q1: quantile(sorted, 0.25), q3: quantile(sorted, 0.75), min: sorted[0], max: sorted[sorted.length - 1] }
  })).filter((r) => r.n > 0 || r.none.length > 0)
  const unit = measure.unit ? ` (${measure.unit})` : ''
  return (
    <section className="results-block">
      <h3>Summary by group</h3>
      <div className="table-wrap">
        <table>
          <thead><tr>{facetLabel && <th>{facetLabel}</th>}<th>Diagnosis group</th><th className="num">Values</th><th className="num">Median{unit}</th>
            <th className="num">Middle half</th><th className="num">Lowest – highest</th>{gaps && <th>No value</th>}</tr></thead>
          <tbody>{rows.map((r) => (
            <tr key={`${r.facet}-${r.group}`}>
              {facetLabel && <td>{r.facet}</td>}<td><GroupBadge group={r.group} /></td><td className="num">{r.n}</td>
              <td className="num"><b>{formatNumber(r.median)}</b></td><td className="num">{r.n ? `${formatNumber(r.q1)} – ${formatNumber(r.q3)}` : '—'}</td>
              <td className="num">{r.n ? `${formatNumber(r.min)} – ${formatNumber(r.max)}` : '—'}</td>
              {gaps && <td className="wrap small secondary">{r.none.length ? reasonCounts(countReasons(r.none)) : 'none'}</td>}
            </tr>
          ))}</tbody>
        </table>
      </div>
    </section>
  )
}

const PAGE = 20

/** The value of every donor-region (or core), or why it has none, sortable and a page at a time, with a CSV download. */
function ValueList({ rows: values, gaps, measure, idColumn }: { rows: Row[]; gaps: Row[] | null; measure: Measure; idColumn: string }) {
  const rows = useMemo<Row[]>(() => [...values, ...(gaps ?? []).map((g) => ({ ...g, no_value_reason: g.text }))], [values, gaps])
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
      if (key === measure.column && (x == null || y == null)) return Number(x == null) - Number(y == null)  // no value: last either way
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
        <button className="btn small" onClick={() => downloadCsv(`${measure.column}_by_${what}.csv`, [...columns, measure.column, ...(gaps?.length ? ['no_value_reason'] : [])], shown)}>Download these values</button>
      </div>
      {gaps && gaps.length > 0 && <p className="muted small" style={{ margin: '6px 0 0' }}>{values.length.toLocaleString()} of {rows.length.toLocaleString()} have a value; the others say why instead.</p>}
      <div className="table-wrap" style={{ marginTop: 8 }}>
        <table>
          <thead><tr>{columns.map((c) => header(c, ID_LABELS[c] ?? c))}{header(measure.column, measureTitle(measure), true)}</tr></thead>
          <tbody>{shown.slice(offset, offset + PAGE).map((r, i) => (
            <tr key={`${r[idColumn]}-${i}`}>
              {columns.map((c) => <td key={c}>{c === 'disease_group' ? <GroupBadge group={String(r[c])} /> : String(r[c] ?? '—')}</td>)}
              <td className="num">{r.text != null ? <NoValue reason={{ text: String(r.text), why: String(r.why) }} /> : formatNumber(r[measure.column])}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
      {!shown.length && <p className="muted small">Nothing matches “{search}”.</p>}
      <Pager offset={offset} size={PAGE} total={shown.length} onChange={setOffset} />
    </section>
  )
}
