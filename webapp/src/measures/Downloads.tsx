import { useState } from 'react'
import type { TableInfo } from '../api'
import { downloadUrl } from '../api'
import { PathLine, timeAgo } from '../components'
import { Advanced } from '../setup/parts'
import { Pager } from './parts'

const size = (kb: number) => (kb >= 1024 ? `${(kb / 1024).toFixed(1)} MB` : `${Math.max(1, Math.round(kb))} KB`)
const PAGE = 25

/** Every results table to download (CSV, opens in Excel, R or Python), and the project's other analysis tables. */
export function Downloads({ tables, current, onOpen }: { tables: TableInfo[]; current: string; onOpen: (name: string) => void }) {
  const main = tables.filter((t) => t.main)
  const others = tables.filter((t) => !t.main)
  return (
    <div className="downloads">
      <p className="secondary" style={{ marginTop: 0 }}>Each table is a CSV file: it opens in Excel, R or Python. The first one holds every measure;
        the others are the same numbers split by stain, or one row per core.</p>
      <ul className="table-list">
        {main.map((t) => <TableItem key={t.name} table={t} current={t.name === current} onOpen={onOpen} />)}
      </ul>
      {others.length > 0 && (
        <Advanced title="Other tables in the project" hint={`${others.length} analysis and quality-check tables`}>
          <OtherTables tables={others} current={current} onOpen={onOpen} />
        </Advanced>
      )}
      {main[0] && (
        <Advanced title="Where the files are">
          <PathLine path={main[0].path.replace(/[^/]+$/, '')} exists label="Results folder" />
        </Advanced>
      )}
    </div>
  )
}

function TableItem({ table, current, onOpen }: { table: TableInfo; current: boolean; onOpen: (name: string) => void }) {
  return (
    <li className={`table-item${current ? ' current' : ''}`}>
      <div className="table-item-text">
        <b>{table.main ? table.title : table.name}</b>
        {table.description && <span className="secondary small">{table.description}</span>}
        <span className="muted small" title={table.path}>{table.columns} columns · {size(table.size_kb)} · updated {timeAgo(table.modified)}</span>
      </div>
      <div className="row">
        {current ? <span className="muted small">Open now</span> : <button className="link-btn" onClick={() => onOpen(table.name)}>Show</button>}
        <a className="btn small" href={downloadUrl(table.path)} download>Download</a>
      </div>
    </li>
  )
}

function OtherTables({ tables, current, onOpen }: { tables: TableInfo[]; current: string; onOpen: (name: string) => void }) {
  const [search, setSearch] = useState('')
  const [offset, setOffset] = useState(0)
  const shown = tables.filter((t) => t.name.toLowerCase().includes(search.toLowerCase()))
  return (
    <>
      <input type="search" placeholder="Find a table by name" value={search} onChange={(e) => { setSearch(e.target.value); setOffset(0) }} aria-label="Find a table" />
      <ul className="table-list">
        {shown.slice(offset, offset + PAGE).map((t) => <TableItem key={t.name} table={t} current={t.name === current} onOpen={onOpen} />)}
      </ul>
      {!shown.length && <p className="muted small">No table matches “{search}”.</p>}
      <Pager offset={offset} size={PAGE} total={shown.length} onChange={setOffset} />
    </>
  )
}
