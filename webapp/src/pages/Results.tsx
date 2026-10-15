import { Link } from 'react-router-dom'
import type { TableInfo } from '../api'
import { downloadUrl } from '../api'
import { Empty, HelpBox, PageHead, Term } from '../components'
import { useStored } from '../hooks'
import { DataTable } from '../measures/DataTable'
import { Downloads } from '../measures/Downloads'
import { MeasureGuide } from '../measures/MeasureGuide'
import { MeasurePicker } from '../measures/MeasurePicker'
import { MeasureView } from '../measures/MeasureView'
import type { TableColumns } from '../measures/data'
import { CORE_TABLE, DONOR_REGION_TABLE, ResultsCache, tableUrl, useShared } from '../measures/data'
import { Choice, Waiting } from '../measures/parts'

type Tab = 'plot' | 'numbers' | 'guide' | 'downloads'
const TABS: [Tab, string][] = [['plot', 'Plot by group'], ['numbers', 'Table of numbers'], ['guide', 'What the measures mean'], ['downloads', 'Downloads']]

export default function Results() {
  return <ResultsCache><ResultsPage /></ResultsCache>
}

function ResultsPage() {
  const tables = useShared<TableInfo[]>('/api/tables')
  const main = tables.data?.find((t) => t.name === DONOR_REGION_TABLE)
  return (
    <>
      <PageHead title="Results" subtitle="Every measure of your study by diagnosis group and brain region, what it means, and the tables to download.">
        {main && <a className="btn primary" href={downloadUrl(main.path)} download>Download results table</a>}
      </PageHead>
      <HelpBox id="results">
        <ul>
          <li>Pick a measure on the left to see it plotted by diagnosis group, with a summary per group and the value of every <Term>donor-region</Term> below.</li>
          <li>Counts and areas are pooled over all fields and replicate cores before dividing, so every <Term>density</Term> is per mm² of usable tissue.</li>
          <li>The plots show raw values, for looking at the data. Formal comparisons should adjust for age, TMA and other covariates.</li>
        </ul>
      </HelpBox>
      <Waiting state={tables} label="Looking for results tables…">
        {(list) => list.some((t) => t.main) ? <ResultsBody tables={list} /> : (
          <Empty>
            <p style={{ marginTop: 0 }}>No results yet. They appear here once the run has made the results tables (its last step).</p>
            <Link className="btn primary" to="/workflow#summarize">Go to Run</Link>
          </Empty>
        )}
      </Waiting>
    </>
  )
}

function ResultsBody({ tables }: { tables: TableInfo[] }) {
  const [stored, setTable] = useStored('stainid.results.table', DONOR_REGION_TABLE)
  const [tab, setTab] = useStored<Tab>('stainid.results.tab', 'plot')
  const [measure, setMeasure] = useStored('stainid.results.measure', '')
  const fallback = tables.find((t) => t.name === DONOR_REGION_TABLE) ?? tables.find((t) => t.main)!
  const table = tables.find((t) => t.name === stored) ?? fallback
  const info = useShared<TableColumns>(tableUrl(table.name, 'columns'))
  const level = table.name === CORE_TABLE ? CORE_TABLE : DONOR_REGION_TABLE
  const levels: [string, string][] = [[DONOR_REGION_TABLE, 'Donor-region'], ...(tables.some((t) => t.name === CORE_TABLE) ? [[CORE_TABLE, 'Core'] as [string, string]] : [])]
  const open = (name: string) => { setTable(name); setTab('plot') }

  return (
    <div className="card results-card">
      <div className="row results-top">
        {table.name === DONOR_REGION_TABLE || table.name === CORE_TABLE ? (
          <>
            <span className="small secondary">One value per</span>
            <Choice label="One value per" value={level} options={levels} onChange={setTable} />
            <span className="muted small">{table.name === CORE_TABLE ? 'Each core on its own.' : 'Replicate cores of a donor and region pooled.'}</span>
          </>
        ) : (
          <>
            <span className="small">Showing <b>{table.main ? table.title : table.name}</b></span>
            <button className="link-btn small" onClick={() => setTable(DONOR_REGION_TABLE)}>Back to the main results</button>
          </>
        )}
      </div>
      <div className="tabs" role="tablist">
        {TABS.map(([key, text]) => <button key={key} role="tab" aria-selected={tab === key} className={`tab${tab === key ? ' active' : ''}`} onClick={() => setTab(key)}>{text}</button>)}
      </div>
      {tab === 'downloads' ? <Downloads tables={tables} current={table.name} onOpen={open} /> : (
        <Waiting state={info} label="Reading the table…">
          {(columns) => {
            if (tab === 'numbers') return <DataTable table={table.name} info={columns} plain={!table.main} />
            if (tab === 'guide') return <MeasureGuide info={columns} onPick={(c) => { setMeasure(c); setTab('plot') }} />
            return <Explore table={table.name} info={columns} measure={measure} onMeasure={setMeasure} onTable={() => setTab('numbers')} />
          }}
        </Waiting>
      )}
    </div>
  )
}

function Explore({ table, info, measure, onMeasure, onTable }: {
  table: string; info: TableColumns; measure: string; onMeasure: (column: string) => void; onTable: () => void
}) {
  if (!info.groupable.includes('disease_group') || !info.measures.length) {
    return (
      <Empty>
        <p style={{ marginTop: 0 }}>This table has {info.measures.length ? 'no diagnosis group column' : 'no measures'}, so it cannot be plotted by group.</p>
        <button className="btn" onClick={onTable}>See its numbers</button>
      </Empty>
    )
  }
  const chosen = info.measures.find((m) => m.column === measure) ?? info.measures.find((m) => m.key) ?? info.measures[0]
  return (
    <div className="explore">
      <MeasurePicker measures={info.measures} topics={info.topics} value={chosen.column} onChange={onMeasure} />
      <MeasureView key={chosen.column} table={table} info={info} measure={chosen} />
    </div>
  )
}
