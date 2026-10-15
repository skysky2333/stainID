import { useEffect, useRef, useState } from 'react'
import type { FoundMap, MapOverview, MapTma, ParsedMap, SavedMap } from '../api'
import { api, apiUrl } from '../api'
import { Empty, ErrorNote, PathLine, withError } from '../components'
import { useFetch } from '../hooks'
import { Advanced, Found, MapGrid, MapSummaryLine, plural, ScreenData, ScreenFoot } from './parts'
import type { ScreenProps } from './Setup'

/** A map being looked at before it is saved: from a file, pasted, or typed in. */
interface Draft { key: string; name: string; source: string; tma: string; grid: string[][]; parsed: ParsedMap }

const byTma = (a: string, b: string) => a.length - b.length || a.localeCompare(b)

export default function MapScreen(props: ScreenProps) {
  const overview = useFetch<MapOverview>('/api/setup/map')
  return <ScreenData state={overview} label="Loading the TMA maps…">{(data) => <Maps {...props} data={data} reload={overview.reload} />}</ScreenData>
}

function Maps({ next, back, nextLabel, go, onChanged, data, reload }: ScreenProps & { data: MapOverview; reload: () => void }) {
  const [drafts, setDrafts] = useState<Draft[]>([])
  const [pasting, setPasting] = useState(false)
  const [text, setText] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState('')
  const [reading, setReading] = useState(false)
  const [opening, setOpening] = useState('')
  const file = useRef<HTMLInputElement>(null)
  const tmas = data.tmas
  const missing = tmas.filter((t) => !t.saved && t.slides.length)
  const withSlides = tmas.filter((t) => t.slides.length)
  const add = (found: FoundMap[]) => setDrafts((old) => [...old, ...found.map((m, i) => ({
    key: `${Date.now()}-${i}`, name: m.name, source: m.source, tma: m.tma, grid: m.grid, parsed: { cells: m.cells, summary: m.summary },
  }))])
  const readFiles = (files: FileList | null) => files && withError(async () => {
    setReading(true); setMessage('')
    try {
      for (const f of Array.from(files)) add((await api.upload<{ maps: FoundMap[] }>('/api/setup/map/read', f)).maps)
    } finally { setReading(false); if (file.current) file.current.value = '' }
  }, setError)
  const readText = () => withError(async () => {
    add((await api.post<{ maps: FoundMap[] }>('/api/setup/map/read-text', { text, name: 'Pasted map' })).maps)
    setText(''); setPasting(false); setMessage('')
  }, setError)
  const typeIn = (tma: MapTma) => withError(async () => {
    setOpening(tma.tma)
    try {
      const saved = await api.get<SavedMap>(`/api/setup/map/${encodeURIComponent(tma.tma)}`)
      setDrafts((old) => [...old.filter((d) => d.tma !== tma.tma), { key: `edit-${tma.tma}-${Date.now()}`, name: `TMA ${tma.tma}`, source: saved.saved ? tma.sources.join(', ') : 'typed in',
        tma: tma.tma, grid: saved.grid, parsed: { cells: saved.cells, summary: saved.summary } }])
    } finally { setOpening('') }
  }, setError)
  const saved = (draft: Draft, result: MapOverview) => {
    setDrafts((old) => old.filter((d) => d.key !== draft.key))
    reload(); onChanged()
    const tma = result.tmas.find((t) => t.tma === draft.tma)
    setMessage(`Saved the map of TMA ${draft.tma}${tma?.saved ? `: ${plural(tma.saved.cores, 'donor core')} from ${plural(tma.saved.donors, 'donor')}` : ''}.`)
  }
  const needsAdding = missing.length > 0 && !drafts.length

  if (!tmas.length) {
    return (
      <div className="screen">
        <Empty><p>No TMAs yet: the maps are matched to the TMAs of the slides, so register the slides first.</p>
          <button className="btn primary" onClick={() => go('slides')}>Go to Slides</button></Empty>
        <ScreenFoot back={back} next={next} nextLabel={nextLabel} quiet />
      </div>
    )
  }
  return (
    <div className="screen">
      <div className="card">
        {missing.length ? <Found tone="plain"><b>{withSlides.length - missing.length} of {plural(withSlides.length, 'TMA')} have a map.</b> Still needed: the map of TMA {missing.map((t) => t.tma).join(', ')}.
          Add the files the lab keeps them in; stainID finds the grid in them and reads every cell (donor, region, group and pathology). You check the result before it is saved.</Found>
          : <Found>Every TMA with slides has a map. Add a file again, or press Edit, to change one.</Found>}
        <ul className="map-list">{tmas.map((t) => (
          <li key={t.tma}>
            <span className="map-list-name"><b>TMA {t.tma}</b>
              <span className="muted small" title={t.slides.join(', ')}>{t.slides.length ? plural(t.slides.length, 'slide') : 'no slides'} · {t.size[0]} × {t.size[1]}</span></span>
            <span className="map-list-state small">
              {t.saved ? <><span className="status-done">✓</span> {plural(t.saved.cores, 'donor core')}, {plural(t.saved.donors, 'donor')}{t.saved.controls ? `, ${plural(t.saved.controls, 'control')}` : ''}
                {t.sources.length > 0 && <span className="muted"> · from {t.sources.join(', ')}</span>}</>
                : <span className="status-outdated">No map yet</span>}
            </span>
            <button className="btn small" disabled={opening === t.tma} onClick={() => typeIn(t)}>{opening === t.tma ? 'Opening…' : t.saved ? 'Edit' : 'Type it in'}</button>
          </li>
        ))}</ul>
        <div className="row" style={{ marginTop: 12 }}>
          <label className={`btn${needsAdding ? ' primary' : ''}${reading ? ' disabled' : ''}`}>{reading ? 'Reading…' : 'Add map files (Excel, Word, CSV)…'}
            <input ref={file} type="file" multiple hidden disabled={reading} accept=".xlsx,.xlsm,.docx,.csv,.tsv,.txt,.png,.jpg,.jpeg,.pdf" onChange={(e) => readFiles(e.target.files)} /></label>
          <button className="btn" aria-expanded={pasting} onClick={() => setPasting(!pasting)}>Paste from a spreadsheet</button>
        </div>
        <p className="small muted">Only a photo or a scan of a map? Press <b>Type it in</b> next to its TMA and type each cell; the core pictures help.</p>
        {pasting && (
          <div className="panel">
            <span className="small secondary">Select the map's cells in Excel or Word, copy, and paste them here. Write the TMA number in the first line (e.g. “TMA 8”), or choose the TMA after reading.</span>
            <textarea rows={6} value={text} onChange={(e) => setText(e.target.value)} placeholder={'TMA 8\n1673 FR\t1700 FR\tCBLM\nAD\tCT\t'} aria-label="Pasted map" />
            <div className="row"><button className="btn small" disabled={!text.trim()} onClick={readText}>Read the pasted map</button>
              <button className="link-btn small" onClick={() => { setPasting(false); setText('') }}>Cancel</button></div>
          </div>
        )}
        {message && <p className="small status-finished" role="status">{message}</p>}
        <ErrorNote error={error} />
      </div>

      {drafts.map((d) => <DraftCard key={d.key} draft={d} tmas={tmas} onChange={(changed) => setDrafts((old) => old.map((x) => (x.key === d.key ? changed : x)))}
        onCancel={() => setDrafts((old) => old.filter((x) => x.key !== d.key))} onSaved={(result) => saved(d, result)} />)}

      <Advanced hint="TMA map file, CSV template, donor id style">
        <PathLine label="TMA map file" path={data.path} exists={tmas.some((t) => t.saved)} wrap />
        <p className="small secondary">One row per core position: <code>tma, core_label, donor_id, region, disease_group</code> (and <code>sample_region_id, tissue_control, cerad, braak, map_source</code>).
          Saving a map here replaces that TMA's rows and keeps the others. You can also fill the whole file in Excel:</p>
        <div className="row">
          <a className="btn small" href={apiUrl('/api/templates/tma_layout.csv')}>Download template (all positions filled in)</a>
          <label className="btn small">Upload the whole TMA map (CSV)…<input type="file" accept=".csv" hidden onChange={(e) => e.target.files?.[0] && withError(async () => {
            await api.upload('/api/upload/tma_layout', e.target.files![0]); reload(); onChanged(); setMessage('TMA map uploaded.')
          }, setError)} /></label>
        </div>
        <IdStyle overview={data} onSaved={() => { reload(); setDrafts([]) }} />
        <SampleIdRule overview={data} onSaved={() => { reload(); setDrafts([]) }} />
      </Advanced>
      <ScreenFoot back={back} next={next} nextLabel={drafts.length ? `${nextLabel} (unsaved maps are dropped)` : nextLabel} quiet={needsAdding || drafts.length > 0} />
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
        Read a map of {shape} positions: <MapSummaryLine summary={draft.parsed.summary} />
        {sizeDiffers && <> The slides of TMA {draft.tma} have {target!.size[0]} × {target!.size[1]} positions: check that the map is the right one and lies the same way.</>}
        {draft.parsed.cells.some((c) => c.sample_region_id) && <>{' '}Donor-regions are named like <code>{draft.parsed.cells.find((c) => c.sample_region_id)?.sample_region_id}</code>, as in the rest of the project (Advanced).</>}
        {draft.parsed.summary.unreadable > 0 && <> Cells with a yellow edge could not be read completely: correct them in the grid (e.g. <code>1673 FR AD</code>).</>}
      </Found>
      <p className="small muted">Each cell: donor id, region, group and (optional) pathology, e.g. <code>1673 FR</code> / <code>AD</code> / <code>C:C - B:6</code>. Control tissue such as <code>CBLM</code> is a control core. You can paste cells copied from Excel.</p>
      <MapGrid grid={draft.grid} cells={draft.parsed.cells} onEdit={edit} thumbs={draft.tma ? `/api/setup/thumbs/${encodeURIComponent(draft.tma)}` : undefined}
        positions={target?.positions.length ? target.positions : undefined} />
      <div className="row" style={{ marginTop: 10 }}>
        <button className="btn primary" disabled={!draft.tma || busy} onClick={save}>
          {busy ? 'Saving…' : draft.tma ? `Save the map of TMA ${draft.tma}` : 'Choose the TMA first'}</button>
        <button className="btn" onClick={onCancel}>Cancel</button>
        {target?.saved && <span className="small muted">Replaces the map of TMA {draft.tma} saved before.</span>}
      </div>
      <ErrorNote error={error} />
    </div>
  )
}

function SampleIdRule({ overview, onSaved }: { overview: MapOverview; onSaved: () => void }) {
  const rule = overview.sample_ids
  const setting = overview.sample_ids_setting
  const [plain, setPlain] = useState(setting.plain_prefix ?? '')
  const [prefixed, setPrefixed] = useState(setting.prefixed_prefix ?? '')
  const [region, setRegion] = useState(setting.region ?? 'name')
  const [error, setError] = useState<string | null>(null)
  const save = () => withError(async () => { await api.put('/api/setup/sample-ids', { plain_prefix: plain, prefixed_prefix: prefixed, region }); onSaved() }, setError)
  return (
    <div className="panel">
      <b className="small">How donor-regions are named (sample_region_id)</b>
      <span className="small secondary">
        Now: <code>{rule.examples.join(', ')}</code>{' '}
        {rule.source === 'layout'
          ? <>— learned from the maps already saved{Object.keys(rule.regions).length ? ` (${Object.entries(rule.regions).map(([r, c]) => `${r} → ${c}`).join(', ')})` : ''}, so new TMAs match the rest of the project. The setting below applies while no map is saved.</>
          : <>— from the setting below (no saved map shows another naming yet).</>}
      </span>
      <div className="row">
        <label className="field">Before plain donor ids (e.g. BRC_)<input type="text" value={plain} style={{ width: 90 }} onChange={(e) => setPlain(e.target.value)} /></label>
        <label className="field">Before prefixed ids (e.g. JOG_003)<input type="text" value={prefixed} style={{ width: 90 }} onChange={(e) => setPrefixed(e.target.value)} /></label>
        <label className="field">Region as<select value={region} onChange={(e) => setRegion(e.target.value)}><option value="name">name (frontal)</option><option value="code">code (FR)</option></select></label>
        <button className="btn small" onClick={save}>Save</button>
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
