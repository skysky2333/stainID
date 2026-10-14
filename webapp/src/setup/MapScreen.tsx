import { useEffect, useRef, useState } from 'react'
import type { FoundMap, MapOverview, MapTma, ParsedMap, SavedMap } from '../api'
import { api, apiUrl } from '../api'
import { Empty, ErrorNote, PathLine, withError } from '../components'
import { useFetch } from '../hooks'
import { Advanced, Found, MapGrid, MapSummaryLine, ScreenFoot, plural } from './parts'
import type { ScreenProps } from './Setup'

/** A map being looked at before it is saved: from a file, pasted, or typed in. */
interface Draft { key: string; name: string; source: string; tma: string; grid: string[][]; parsed: ParsedMap }

const byTma = (a: string, b: string) => a.length - b.length || a.localeCompare(b)

export default function MapScreen({ next, back, onChanged }: ScreenProps) {
  const overview = useFetch<MapOverview>('/api/setup/map')
  const [drafts, setDrafts] = useState<Draft[]>([])
  const [pasting, setPasting] = useState(false)
  const [text, setText] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState('')
  const [reading, setReading] = useState(false)
  const file = useRef<HTMLInputElement>(null)
  const tmas = overview.data?.tmas ?? []
  const missing = tmas.filter((t) => !t.saved && t.slides.length)

  const add = (found: FoundMap[]) => setDrafts((old) => [...old, ...found.map((m, i) => ({
    key: `${Date.now()}-${i}`, name: m.name, source: m.source, tma: m.tma, grid: m.grid, parsed: { cells: m.cells, summary: m.summary },
  }))])
  const readFiles = (files: FileList | null) => files && withError(async () => {
    setReading(true)
    try {
      for (const f of Array.from(files)) add((await api.upload<{ maps: FoundMap[] }>('/api/setup/map/read', f)).maps)
    } finally { setReading(false); if (file.current) file.current.value = '' }
  }, setError)
  const readText = () => withError(async () => {
    add((await api.post<{ maps: FoundMap[] }>('/api/setup/map/read-text', { text, name: 'Pasted map' })).maps)
    setText(''); setPasting(false)
  }, setError)
  const typeIn = (tma: MapTma) => withError(async () => {
    const saved = await api.get<SavedMap>(`/api/setup/map/${encodeURIComponent(tma.tma)}`)
    setDrafts((old) => [...old.filter((d) => d.tma !== tma.tma), { key: `edit-${tma.tma}-${Date.now()}`, name: `TMA ${tma.tma}`, source: saved.saved ? tma.sources.join(', ') : 'typed in',
      tma: tma.tma, grid: saved.grid, parsed: { cells: saved.cells, summary: saved.summary } }])
  }, setError)
  const saved = (draft: Draft, result: MapOverview) => {
    setDrafts((old) => old.filter((d) => d.key !== draft.key))
    overview.reload(); onChanged()
    const tma = result.tmas.find((t) => t.tma === draft.tma)
    setMessage(`Saved the map of TMA ${draft.tma}${tma?.saved ? `: ${tma.saved.cores} donor cores from ${tma.saved.donors} donors` : ''}.`)
  }

  if (!overview.data) return <ErrorNote error={overview.error} />
  return (
    <div className="screen">
      <div className="card">
        <h2>TMA maps</h2>
        {tmas.length ? (
          <table className="map-list">
            <tbody>{tmas.map((t) => (
              <tr key={t.tma}>
                <td style={{ width: 110 }}><b>TMA {t.tma}</b></td>
                <td className="small secondary">{t.slides.length ? t.slides.join(', ') : 'no slides'} · {t.size[0]} × {t.size[1]} positions</td>
                <td>{t.saved ? <span className="small"><span className="status-done">✓</span> {plural(t.saved.cores, 'donor core')}, {plural(t.saved.donors, 'donor')}{t.saved.controls ? `, ${plural(t.saved.controls, 'control')}` : ''}
                  {t.sources.length > 0 && <span className="muted"> · from {t.sources.join(', ')}</span>}</span>
                  : <span className="small status-outdated">No map yet</span>}</td>
                <td style={{ width: 120, textAlign: 'right' }}><button className="btn small" onClick={() => typeIn(t)}>{t.saved ? 'Edit' : 'Type it in'}</button></td>
              </tr>
            ))}</tbody>
          </table>
        ) : <Empty>Register the slides first: the maps are matched to their TMAs.</Empty>}
        {tmas.length > 0 && (missing.length ? <Found tone="plain">Still needed: the map of TMA {missing.map((t) => t.tma).join(', ')}. Add the files the lab keeps them in;
          stainID finds the grid in them and reads every cell (donor, region, group and pathology).</Found>
          : <Found>Every TMA with slides has a map.</Found>)}
        <div className="row" style={{ marginTop: 10 }}>
          <label className="btn primary">{reading ? 'Reading…' : 'Add map files (Excel, Word, CSV)…'}
            <input ref={file} type="file" multiple hidden accept=".xlsx,.xlsm,.docx,.csv,.tsv,.txt,.png,.jpg,.jpeg,.pdf" onChange={(e) => readFiles(e.target.files)} /></label>
          <button className="btn" onClick={() => setPasting(!pasting)}>Paste from a spreadsheet</button>
          <span className="small muted">Only a photo or a scan? Press “Type it in” next to its TMA and type each cell (core pictures help).</span>
        </div>
        {pasting && (
          <div className="panel">
            <span className="small secondary">Select the map’s cells in Excel or Word, copy, and paste here. Write the TMA number in the first line (e.g. “TMA 8”) or choose it below.</span>
            <textarea rows={6} value={text} onChange={(e) => setText(e.target.value)} placeholder={'TMA 8\n1673 FR\t1700 FR\tCBLM\nAD\tCT\t'} />
            <div><button className="btn small primary" disabled={!text.trim()} onClick={readText}>Read</button></div>
          </div>
        )}
        {message && <p className="small status-finished">{message}</p>}
        <ErrorNote error={error ?? overview.error} />
      </div>

      {drafts.map((d) => <DraftCard key={d.key} draft={d} tmas={tmas} onChange={(changed) => setDrafts((old) => old.map((x) => (x.key === d.key ? changed : x)))}
        onCancel={() => setDrafts((old) => old.filter((x) => x.key !== d.key))} onSaved={(result) => saved(d, result)} />)}

      <Advanced hint="TMA map file, CSV template, donor id style">
        <PathLine label="TMA map file" path={overview.data.path} exists={tmas.some((t) => t.saved)} wrap />
        <p className="small secondary">One row per core position: <code>tma, core_label, donor_id, region, disease_group</code> (and <code>tissue_control, cerad, braak, map_source</code>).
          Saving a map here replaces that TMA's rows and keeps the others. You can also fill the whole file in Excel:</p>
        <div className="row">
          <a className="btn small" href={apiUrl('/api/templates/tma_layout.csv')}>Download template (all positions filled in)</a>
          <label className="btn small">Upload the whole TMA map (CSV)…<input type="file" accept=".csv" hidden onChange={(e) => e.target.files?.[0] && withError(async () => {
            await api.upload('/api/upload/tma_layout', e.target.files![0]); overview.reload(); onChanged(); setMessage('TMA map uploaded.')
          }, setError)} /></label>
        </div>
        <IdStyle overview={overview.data} onSaved={() => { overview.reload(); setDrafts([]) }} />
      </Advanced>
      <ScreenFoot back={back} next={next} nextLabel={drafts.length ? 'Next (unsaved maps are dropped)' : 'Next: donor information'} />
    </div>
  )
}

function DraftCard({ draft, tmas, onChange, onCancel, onSaved }: {
  draft: Draft; tmas: MapTma[]; onChange: (d: Draft) => void; onCancel: () => void; onSaved: (result: MapOverview) => void
}) {
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const latest = useRef(draft)
  latest.current = draft
  const timer = useRef<number | undefined>(undefined)
  const target = tmas.find((t) => t.tma === draft.tma)
  useEffect(() => () => window.clearTimeout(timer.current), [])

  const edit = (row: number, column: number, value: string) => {
    const grid = latest.current.grid.map((r) => [...r])
    while (grid.length <= row) grid.push([])
    while (grid[row].length <= column) grid[row].push('')
    grid[row][column] = value
    const width = Math.max(...grid.map((r) => r.length))
    const changed = { ...latest.current, grid: grid.map((r) => [...r, ...Array(width - r.length).fill('')]) }
    latest.current = changed
    onChange(changed)
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => withError(async () => {
      const parsed = await api.post<ParsedMap>('/api/setup/map/parse', { grid: latest.current.grid })
      onChange({ ...latest.current, parsed })
    }, setError), 350)
  }
  const save = () => withError(async () => {
    setBusy(true)
    try {
      onSaved(await api.put<MapOverview>(`/api/setup/map/${encodeURIComponent(draft.tma)}`, { grid: latest.current.grid, source: draft.source }))
    } finally { setBusy(false) }
  }, setError)
  const shape = `${draft.grid.length} × ${Math.max(0, ...draft.grid.map((r) => r.length))}`
  const sizeDiffers = target && (draft.grid.length !== target.size[0] || Math.max(0, ...draft.grid.map((r) => r.length)) !== target.size[1])

  return (
    <div className="card draft-card">
      <div className="row">
        <h2 style={{ margin: 0 }}>{draft.name}</h2>
        {draft.source && <span className="muted small">from {draft.source}</span>}
        <span className="spacer" />
        <label className="field row-field">This is the map of
          <select value={draft.tma} onChange={(e) => onChange({ ...draft, tma: e.target.value })}>
            <option value="">choose a TMA…</option>
            {[...new Set([...tmas.map((t) => t.tma), draft.tma].filter(Boolean))].sort(byTma).map((t) => <option key={t} value={t}>TMA {t}</option>)}
          </select></label>
      </div>
      <Found tone={draft.parsed.summary.unreadable || sizeDiffers ? 'warn' : 'good'}>
        Read a {shape} map: <MapSummaryLine summary={draft.parsed.summary} />
        {sizeDiffers && <> The slides of TMA {draft.tma} have {target!.size[0]} × {target!.size[1]} positions: check that the map is the right one and lies the same way.</>}
        {draft.parsed.summary.unreadable > 0 && <> Cells with a yellow edge could not be read completely: correct them in the grid (e.g. <code>1673 FR AD</code>).</>}
      </Found>
      <p className="small muted">Each cell: donor id, region, group and (optional) pathology, e.g. <code>1673 FR</code> / <code>AD</code> / <code>C:C - B:6</code>. Control tissue such as <code>CBLM</code> is a control core. You can paste cells copied from Excel.</p>
      <MapGrid grid={draft.grid} cells={draft.parsed.cells} onEdit={edit} thumbs={draft.tma ? `/api/setup/thumbs/${encodeURIComponent(draft.tma)}` : undefined}
        positions={target?.positions.length ? target.positions : undefined} />
      <div className="row" style={{ marginTop: 10 }}>
        <button className="btn primary" disabled={!draft.tma || busy} onClick={save}>
          {busy ? 'Saving…' : target?.saved ? `Save as the map of TMA ${draft.tma} (replaces the saved one)` : draft.tma ? `Save as the map of TMA ${draft.tma}` : 'Choose the TMA first'}</button>
        <button className="btn" onClick={onCancel}>Cancel</button>
      </div>
      <ErrorNote error={error} />
    </div>
  )
}

function IdStyle({ overview, onSaved }: { overview: MapOverview; onSaved: () => void }) {
  const [digits, setDigits] = useState(String(overview.id_style.digits ?? 4))
  const [prefixed, setPrefixed] = useState(String(overview.id_style.prefixed_digits ?? 3))
  const [error, setError] = useState<string | null>(null)
  const save = () => withError(async () => { await api.put('/api/setup/id-style', { digits: Number(digits), prefixed_digits: Number(prefixed) }); onSaved() }, setError)
  return (
    <div className="panel">
      <b className="small">How donor ids are written</b>
      <span className="small secondary">The same rule is used for the maps and the donor sheet, so they match. Stray spaces, “#”, “_” and “-” in ids are ignored.</span>
      <div className="row">
        <label className="field">Plain numbers get at least … digits (0 = as typed)<input type="number" min={0} max={8} value={digits} style={{ width: 70 }} onChange={(e) => setDigits(e.target.value)} /></label>
        <label className="field">Numbers after a prefix get … digits<input type="number" min={0} max={8} value={prefixed} style={{ width: 70 }} onChange={(e) => setPrefixed(e.target.value)} /></label>
        <button className="btn small" onClick={save}>Save</button>
      </div>
      <span className="small muted">Now: 698 → {Number(digits) ? '698'.padStart(Number(digits), '0') : '698'}, JOG3 → JOG_{'3'.padStart(Number(prefixed) || 1, '0')}</span>
      <ErrorNote error={error} />
    </div>
  )
}
