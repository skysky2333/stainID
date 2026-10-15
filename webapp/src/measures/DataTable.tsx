import { useEffect, useMemo, useState } from 'react'
import { GroupBadge } from '../components'
import { useStored } from '../hooks'
import type { Measure, RowsPage, TableColumns } from './data'
import { ID_LABELS, formatNumber, measureHover, tableUrl, useShared } from './data'
import { Pager, Waiting } from './parts'

const PAGE = 50
const ALL = '__all'
const KEY = '__key'

/** A results table a page at a time: the identifier columns and one chosen set of measures, readable headers with units,
 *  sorting by any column and finding rows by name. `plain` shows every column under its name in the file. */
export function DataTable({ table, info, plain = false }: { table: string; info: TableColumns; plain?: boolean }) {
  const [set, setSet] = useStored('stainid.results.columns', KEY)
  const [rawNames, setRawNames] = useStored('stainid.results.rawNames', false)
  const [sort, setSort] = useState<{ column: string; desc: boolean } | null>(null)
  const [search, setSearch] = useState('')
  const [query, setQuery] = useState('')
  const [offset, setOffset] = useState(0)
  useEffect(() => { const timer = setTimeout(() => { setQuery(search); setOffset(0) }, 300); return () => clearTimeout(timer) }, [search])
  useEffect(() => { setOffset(0); setSort(null) }, [table])

  const byColumn = useMemo(() => Object.fromEntries(info.measures.map((m) => [m.column, m])), [info.measures])
  const topics = Object.entries(info.topics).filter(([topic]) => info.measures.some((m) => m.topic === topic))
  const hasKey = info.measures.some((m) => m.key)
  const chosen = plain || set === ALL || (set === KEY && !hasKey) || (set !== KEY && !topics.some(([t]) => t === set)) ? ALL : set
  const ids = Object.keys(ID_LABELS).filter((c) => info.columns.includes(c) && !(c === 'tma' && chosen !== ALL))
  const columns = chosen === ALL ? info.columns : [...ids, ...info.measures.filter((m) => (chosen === KEY ? m.key : m.topic === chosen)).map((m) => m.column)]

  const params = new URLSearchParams({ offset: String(offset), limit: String(PAGE), q: query })
  if (sort) { params.set('sort', sort.column); params.set('desc', String(sort.desc)) }
  if (chosen !== ALL) params.set('columns', columns.join(','))
  const page = useShared<RowsPage>(`${tableUrl(table, 'rows')}?${params}`)
  const [last, setLast] = useState<RowsPage | null>(null)  // keep showing the previous page while the next one loads
  useEffect(() => { if (page.data) setLast(page.data) }, [page.data])
  const shown = page.data ?? last
  const raw = plain || rawNames

  const head = (column: string) => {
    const m: Measure | undefined = byColumn[column]
    const numeric = Boolean(m) || info.numeric.includes(column)
    const active = sort?.column === column
    return (
      <th key={column} className={numeric ? 'num' : ''} title={m ? measureHover(m) : column} aria-sort={active ? (sort!.desc ? 'descending' : 'ascending') : 'none'}>
        <button className="sort-btn" onClick={() => { setSort({ column, desc: active ? !sort!.desc : numeric }); setOffset(0) }}>
          {raw || !(m || ID_LABELS[column]) ? <code>{column}</code> : <span className="col-head"><span>{m?.label ?? ID_LABELS[column]}</span>{m?.unit && <span className="muted">{m.unit}</span>}</span>}
          {active ? (sort!.desc ? ' ↓' : ' ↑') : ''}
        </button>
      </th>
    )
  }

  return (
    <div className="data-table">
      <div className="row">
        {!plain && (
          <label className="inline-field">Columns
            <select value={chosen} onChange={(e) => { setSet(e.target.value); setOffset(0) }}>
              {hasKey && <option value={KEY}>Main measures</option>}
              {topics.map(([topic, title]) => <option key={topic} value={topic}>{title}</option>)}
              <option value={ALL}>Every column ({info.columns.length})</option>
            </select>
          </label>
        )}
        <input type="search" placeholder="Find a donor, region or core" value={search} onChange={(e) => setSearch(e.target.value)} aria-label="Find rows" />
        <span className="spacer" />
        {!plain && <label className="layer-toggle small" title="Show each column under its name in the CSV file"><input type="checkbox" checked={rawNames} onChange={() => setRawNames(!rawNames)} />Column names as in the file</label>}
      </div>
      <Waiting state={{ ...page, data: shown }} label="Loading the table…">
        {(data) => (
          <>
            <div className={`table-wrap${page.data ? '' : ' stale'}`} style={{ marginTop: 10 }}>
              <table>
                <thead><tr>{data.columns.map(head)}</tr></thead>
                <tbody>{data.rows.map((r, i) => (
                  <tr key={offset + i}>{data.columns.map((c) => (
                    <td key={c} className={info.numeric.includes(c) || typeof r[c] === 'number' ? 'num' : ''}>
                      {c === 'disease_group' && r[c] ? <GroupBadge group={String(r[c])} /> : formatNumber(r[c])}
                    </td>
                  ))}</tr>
                ))}</tbody>
              </table>
            </div>
            {!data.total && <p className="muted small">{query ? `No row matches “${query}”.` : 'This table has no rows.'}</p>}
            <Pager offset={offset} size={PAGE} total={data.total} onChange={setOffset} />
          </>
        )}
      </Waiting>
    </div>
  )
}
