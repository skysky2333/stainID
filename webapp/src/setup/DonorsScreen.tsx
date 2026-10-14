import { useEffect, useState } from 'react'
import type { DonorPreview, DonorTable, SheetTable } from '../api'
import { api, apiUrl } from '../api'
import { ErrorNote, PathLine, withError } from '../components'
import { useFetch } from '../hooks'
import { Advanced, Found, ScreenFoot } from './parts'
import type { ScreenProps } from './Setup'

const SHOWN = ['donor_id', 'tma', 'disease_group', 'age', 'sex', 'pmd_hours', 'apoe', 'cerad', 'braak']

export default function DonorsScreen({ next, back, onChanged }: ScreenProps) {
  const table = useFetch<DonorTable>('/api/setup/donors')
  const [sheets, setSheets] = useState<SheetTable[]>([])
  const [sheet, setSheet] = useState(0)
  const [mapping, setMapping] = useState<Record<string, string>>({})
  const [keep, setKeep] = useState('study')
  const [mode, setMode] = useState('update')
  const [preview, setPreview] = useState<DonorPreview | null>(null)
  const [error, setError] = useState<string | null>(null)
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
    const result = await api.upload<{ tables: SheetTable[] }>('/api/setup/donors/read', file)
    setSheets(result.tables); setMapping({}); setMessage('')
    setSheet(result.tables.reduce((best, t, i) => (t.rows.length > result.tables[best].rows.length ? i : best), 0))
  }, setError)
  const save = () => withError(async () => {
    setBusy(true)
    try {
      const result = await api.put<{ saved: string; count: number; added: number }>('/api/setup/donors', { rows, mapping, keep, mode })
      setMessage(`Saved ${result.added} donors; the donor table now has ${result.count}.`)
      setSheets([]); setPreview(null); table.reload(); onChanged()
    } finally { setBusy(false) }
  }, setError)
  const fieldName = (field: string) => field === 'skip' ? 'leave out' : field === '' ? 'keep as it is' : preview?.fields[field] ?? field
  const shown = preview ? SHOWN.filter((c) => preview.table_columns.includes(c)) : []

  return (
    <div className="screen">
      <div className="card">
        <h2>Donor information <span className="muted small">(optional)</span></h2>
        {table.data && table.data.count > 0 && !sheets.length && (
          <Found>The donor table has <b>{table.data.count} donors</b> ({table.data.columns.slice(0, 8).join(', ')}{table.data.columns.length > 8 ? ', …' : ''}).
            Add a spreadsheet to update it, or go on.</Found>
        )}
        {!sheets.length && (
          <>
            <p className="secondary small">Your case spreadsheet with one row per donor (Excel or CSV). Section rows such as “CONTROL &lt;65 (90 cases)” give the group, a repeated header row
              such as “JOG#” starts its own id series, and values like “~15” or “&gt;90” are read as numbers. You only check the result.</p>
            <div className="row">
              <label className="btn primary">Add the case spreadsheet…<input type="file" accept=".xlsx,.xlsm,.csv,.tsv,.txt" hidden onChange={(e) => read(e.target.files?.[0])} /></label>
              {next && <button className="btn" onClick={next}>Skip: no donor sheet</button>}
            </div>
          </>
        )}
        {message && <p className="small status-finished">{message}</p>}
        {sheets.length > 0 && (
          <>
            {sheets.length > 1 && (
              <label className="field row-field">Sheet
                <select value={sheet} onChange={(e) => { setSheet(Number(e.target.value)); setMapping({}) }}>
                  {sheets.map((s, i) => <option key={i} value={i}>{s.name} ({s.rows.length} rows)</option>)}</select></label>
            )}
            {preview && (
              <>
                <Found tone={preview.kept ? 'good' : 'warn'}>
                  Found <b>{preview.total} donors</b>{preview.kept !== preview.total ? <>, <b>{preview.kept}</b> of them belong to this study</> : ''}.
                  <ul className="found-notes">{preview.notes.map((n) => <li key={n}>{n}</li>)}</ul>
                </Found>
                <div className="table-wrap" style={{ maxHeight: 300 }}>
                  <table>
                    <thead><tr>{shown.map((c) => <th key={c}>{preview.fields[c] ?? c}</th>)}</tr></thead>
                    <tbody>{preview.records.slice(0, 25).map((r) => <tr key={r.donor_id + r.sheet_row}>{shown.map((c) => <td key={c}>{r[c] || <span className="muted">—</span>}</td>)}</tr>)}</tbody>
                  </table>
                </div>
                {preview.records.length > 25 && <p className="small muted">The first 25 of {preview.kept} donors.</p>}
              </>
            )}
            <div className="row" style={{ marginTop: 10 }}>
              <button className="btn primary" disabled={!preview?.kept || busy} onClick={save}>{busy ? 'Saving…' : `Save ${preview?.kept ?? ''} donors`}</button>
              <button className="btn" onClick={() => { setSheets([]); setPreview(null) }}>Cancel</button>
            </div>
          </>
        )}
        <ErrorNote error={error ?? table.error} />
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
        {table.data && <PathLine label="Donor table" path={table.data.path} exists={table.data.count > 0} wrap />}
        <p className="small secondary">Required columns: <code>donor_id</code>, <code>disease_group</code>; anything else (age, sex, APOE, PMI …) is used by your own statistics.
          A filled-in CSV can also be uploaded directly: <a href={apiUrl('/api/templates/donor_metadata.csv')}>template</a>.</p>
        <label className="btn small">Upload donor table (CSV)…<input type="file" accept=".csv" hidden onChange={(e) => e.target.files?.[0] && withError(async () => {
          await api.upload('/api/upload/donor_metadata', e.target.files![0]); table.reload(); onChanged(); setMessage('Donor table uploaded.')
        }, setError)} /></label>
      </Advanced>
      <ScreenFoot back={back} next={next} nextLabel="Next: checks" />
    </div>
  )
}
