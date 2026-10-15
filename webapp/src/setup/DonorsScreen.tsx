import { useEffect, useState } from 'react'
import type { DonorPreview, DonorTable, SheetTable } from '../api'
import { api, apiUrl } from '../api'
import { ErrorNote, Loading, PathLine, withError } from '../components'
import { useFetch } from '../hooks'
import { Advanced, Found, plural, ScreenData, ScreenFoot, UploadButton } from './parts'
import type { ScreenProps } from './Setup'

const SHOWN = ['donor_id', 'tma', 'disease_group', 'age', 'sex', 'pmd_hours', 'apoe', 'cerad', 'braak']

export default function DonorsScreen(props: ScreenProps) {
  const table = useFetch<DonorTable>('/api/setup/donors')
  return <ScreenData state={table} label="Loading the donor table…">{(data) => <Donors {...props} table={data} reload={table.reload} />}</ScreenData>
}

function Donors({ next, back, nextLabel, onChanged, table, reload }: ScreenProps & { table: DonorTable; reload: () => void }) {
  const [sheets, setSheets] = useState<SheetTable[]>([])
  const [sheet, setSheet] = useState(0)
  const [mapping, setMapping] = useState<Record<string, string>>({})
  const [keep, setKeep] = useState('study')
  const [mode, setMode] = useState('update')
  const [preview, setPreview] = useState<DonorPreview | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [reading, setReading] = useState(false)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const rows = sheets[sheet]?.rows

  useEffect(() => {
    if (!rows) return
    let live = true
    withError(async () => {
      const result = await api.post<DonorPreview>('/api/setup/donors/preview', { rows, mapping, keep })
      if (live) setPreview(result)
    }, (e) => { if (live) { setError(e); if (e) setPreview(null) } })
    return () => { live = false }
  }, [rows, mapping, keep])

  const read = (file: File | undefined) => file && withError(async () => {
    setReading(true)
    try {
      const result = await api.upload<{ tables: SheetTable[] }>('/api/setup/donors/read', file)
      setPreview(null); setSheets(result.tables); setMapping({}); setMessage(''); setKeep('study'); setMode('update')
      setSheet(result.tables.reduce((best, t, i) => (t.rows.length > result.tables[best].rows.length ? i : best), 0))
    } finally { setReading(false) }
  }, setError)
  const save = () => withError(async () => {
    setBusy(true)
    try {
      const result = await api.put<{ saved: string; count: number; added: number }>('/api/setup/donors', { rows, mapping, keep, mode })
      setMessage(`Saved ${plural(result.added, 'donor')}; the donor table now has ${result.count.toLocaleString()}.`)
      setSheets([]); setPreview(null); reload(); onChanged()
    } finally { setBusy(false) }
  }, setError)
  // the choices made for one sheet (which donors to keep, replace or update) never carry over to the next one
  const cancel = () => { setSheets([]); setPreview(null); setError(null); setMapping({}); setKeep('study'); setMode('update') }
  const fields = preview?.fields ?? table.fields
  const fieldName = (field: string) => field === 'skip' ? 'leave out' : field === '' ? 'keep as it is' : fields[field] ?? field
  const shown = preview ? SHOWN.filter((c) => preview.table_columns.includes(c)) : []
  const known = table.columns.filter((c) => c !== 'donor_id').map((c) => fields[c] ?? c.replace(/_/g, ' '))
  const looking = sheets.length > 0

  return (
    <div className="screen">
      <div className="card">
        {!looking && (
          <>
            {table.count > 0
              ? <Found>The donor table has <b>{plural(table.count, 'donor')}</b>{known.length > 0 && <>, with {known.slice(0, 6).join(', ')}{known.length > 6 ? ` and ${known.length - 6} more` : ''}</>}.
                  Add a spreadsheet to update it, or go on.</Found>
              : <p className="secondary">Your case spreadsheet with one row per donor (Excel or CSV). You can skip this step and add it later.</p>}
            <p className="small muted">Section rows such as “CONTROL &lt;65 (90 cases)” give the group, a repeated header row such as “JOG#” starts its own id series, and values
              like “~15” or “&gt;90” are read as numbers. You only check the result.</p>
            <div className="row">
              <label className={`btn${table.count ? '' : ' primary'}${reading ? ' disabled' : ''}`}>{reading ? 'Reading…' : table.count ? 'Add a newer spreadsheet…' : 'Add the case spreadsheet…'}
                <input type="file" accept=".xlsx,.xlsm,.csv,.tsv,.txt" hidden disabled={reading} onChange={(e) => { read(e.target.files?.[0]); e.target.value = '' }} /></label>
            </div>
          </>
        )}
        {message && <p className="small status-finished" role="status">{message}</p>}
        {looking && (
          <>
            <div className="row card-head">
              <b>Donors in {sheets[sheet]?.source || sheets[sheet]?.name}</b><span className="spacer" />
              {sheets.length > 1 && (
                <label className="field row-field">Sheet
                  <select value={sheet} onChange={(e) => { setSheet(Number(e.target.value)); setMapping({}); setPreview(null) }}>
                    {sheets.map((s, i) => <option key={i} value={i}>{s.name} ({plural(s.rows.length, 'row')})</option>)}</select></label>
              )}
            </div>
            {!preview ? (!error && <Loading label="Reading the donors…" />) : (
              <>
                <Found tone={preview.kept ? 'good' : 'warn'}>
                  Found <b>{plural(preview.total, 'donor')}</b>{preview.kept !== preview.total ? <>; <b>{preview.kept.toLocaleString()}</b> of them belong to this study</> : ''}.
                  {' '}Check the table below; which column is which can be changed under Advanced.
                  {preview.notes.length > 0 && <ul className="found-notes">{preview.notes.map((n) => <li key={n}>{n}</li>)}</ul>}
                </Found>
                <div className="table-wrap" style={{ maxHeight: 300 }}>
                  <table>
                    <thead><tr>{shown.map((c) => <th key={c}>{fields[c] ?? c}</th>)}</tr></thead>
                    <tbody>{preview.records.slice(0, 25).map((r) => <tr key={r.donor_id + r.sheet_row}>{shown.map((c) => <td key={c}>{r[c] || <span className="muted">—</span>}</td>)}</tr>)}</tbody>
                  </table>
                </div>
                {preview.records.length > 25 && <p className="small muted">The first 25 of {preview.kept.toLocaleString()} donors.</p>}
              </>
            )}
            <div className="row" style={{ marginTop: 10 }}>
              <button className="btn primary" disabled={!preview?.kept || busy} onClick={save}>{busy ? 'Saving…' : preview ? `Save ${plural(preview.kept, 'donor')}${mode === 'replace' && table.count ? ' (replaces the donor table)' : ''}` : 'Save'}</button>
              <button className="btn" onClick={cancel}>Cancel</button>
            </div>
          </>
        )}
        <ErrorNote error={error} />
      </div>

      <Advanced hint="column matches, which donors are kept, file location">
        {preview && (
          <>
            <b className="small">Which column is which</b>
            <p className="small secondary">Matched by the column names (header found on row {preview.header_row + 1}). Change a match if it is wrong; columns stainID does not know are kept under their own name.</p>
            <div className="table-wrap" style={{ maxHeight: 360 }}>
              <table className="compact">
                <thead><tr><th>Column in the sheet</th><th>Examples</th><th>stainID column</th></tr></thead>
                <tbody>{preview.columns.filter((c) => c.header || c.samples.length).map((c) => (
                  <tr key={c.index}>
                    <td>{c.header || <i className="muted">no name</i>}</td>
                    <td className="small secondary ellipsis">{c.samples.join(' · ')}</td>
                    <td><select value={mapping[c.index] ?? c.field} onChange={(e) => setMapping({ ...mapping, [c.index]: e.target.value })} aria-label={`stainID column for ${c.header}`}>
                      {[...Object.keys(preview.fields), '', 'skip'].map((f) => <option key={f} value={f}>{fieldName(f)}</option>)}</select></td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
            <div className="row">
              <label className="layer-toggle"><input type="checkbox" checked={keep === 'all'} onChange={(e) => setKeep(e.target.checked ? 'all' : 'study')} />
                Keep every donor of the sheet (not only those on the maps and TMAs of this study)</label>
              <label className="layer-toggle"><input type="checkbox" checked={mode === 'replace'} onChange={(e) => setMode(e.target.checked ? 'replace' : 'update')} />
                Replace the donor table (otherwise donors not in this sheet are kept)</label>
            </div>
          </>
        )}
        {!preview && looking && <p className="small muted">The column matches appear here once the sheet is read.</p>}
        <PathLine label="Donor table" path={table.path} exists={table.count > 0} wrap />
        {table.columns.length > 0 && <p className="small secondary">Columns now: <code>{table.columns.join(', ')}</code></p>}
        <p className="small secondary">Required columns: <code>donor_id</code>, <code>disease_group</code>; anything else (age, sex, APOE, PMI …) is used by your own statistics.
          A filled-in CSV can also be uploaded directly: <a href={apiUrl('/api/templates/donor_metadata.csv')}>template</a>.</p>
        <div className="row"><UploadButton label="Upload donor table (CSV)…" accept=".csv" onFile={async (f) => {
          await api.upload('/api/upload/donor_metadata', f); reload(); onChanged(); return 'Donor table uploaded.'
        }} /></div>
      </Advanced>
      <ScreenFoot back={back} next={next} nextLabel={table.count || looking ? nextLabel : 'Skip this step'} quiet={looking || !table.count} />
    </div>
  )
}
