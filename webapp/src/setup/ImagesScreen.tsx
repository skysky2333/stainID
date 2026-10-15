import { useState } from 'react'
import type { ImageRow, ImagesOverview } from '../api'
import { api } from '../api'
import { ErrorNote, FilePicker, PathLine, withError } from '../components'
import { useFetch } from '../hooks'
import ImagingCard from './ImagingCard'
import { Advanced, Found, plural, ScreenData, ScreenFoot, UploadButton } from './parts'
import type { ScreenProps } from './Setup'

const PAGE = 50
const EDITABLE = 500  // the page holds this many rows of the image list; longer lists are edited as a file

/** The Images (or Slides) screen of an image set or whole-slide study: which images, from which donor, in which stain,
and how they were imaged. Replaces the Slides, Cores and Map screens of a TMA study. */
export default function ImagesScreen(props: ScreenProps) {
  const info = useFetch<ImagesOverview>('/api/setup/images')
  return <ScreenData state={info} label="Reading the image list…">{(data) => <Images {...props} data={data} reload={info.reload} />}</ScreenData>
}

function Images({ next, nextLabel, onChanged, data, reload }: ScreenProps & { data: ImagesOverview; reload: () => void }) {
  const [adding, setAdding] = useState(data.total === 0)
  const thing = data.kind === 'slides' ? 'slide' : 'image'
  const changed = () => { reload(); onChanged() }
  const unknown = data.stains.filter((s) => s.images > 0 && !s.analysed_as && s.family !== 'ignore')
  return (
    <div className="screen">
      {data.total > 0 && (
        <Found tone={data.problems.length || unknown.length ? 'warn' : 'good'}>
          <b>{plural(data.total, thing)}</b> from {plural(data.donors, 'donor')}
          {data.stains.filter((s) => s.images).length > 0 && <> · {data.stains.filter((s) => s.images).map((s) => `${s.stain || 'no stain'} (${s.images})`).join(', ')}</>}
          {data.dataset.title && <> · from the dataset <b>{data.dataset.title}</b>{data.dataset.licence ? ` (${data.dataset.licence})` : ''}</>}
          {(data.problems.length > 0 || unknown.length > 0) && (
            <ul className="found-notes">
              {data.problems.map((p) => <li key={p}>{p}</li>)}
              {unknown.length > 0 && <li>Say below what {unknown.length === 1 ? `${unknown[0].stain || 'the unnamed stain'} is` : 'these stains are'}: {unknown.map((s) => s.stain || 'no stain').join(', ')}</li>}
            </ul>
          )}
        </Found>
      )}
      {adding ? <AddImages data={data} onSaved={() => { setAdding(false); changed() }} onCancel={data.total ? () => setAdding(false) : undefined} />
        : <ImageList data={data} onChanged={changed} onAdd={() => setAdding(true)} />}
      {data.total > 0 && <ImagingCard data={data} onSaved={changed} />}
      <Advanced hint="image list file, readers">
        <PathLine path={data.path} exists={data.total > 0} label="Image list" />
        <p className="small secondary">One row per {thing}: path, stain, donor_id and optionally image_id, region, slide_id, site, batch, pixel_size_um,
          disease_group and split. The files are read where they are; stainID never copies or changes them.</p>
        <p className="small secondary">Readers: {Object.entries(data.readers).map(([name, ok]) => `${name} ${ok ? 'installed' : 'not installed'}`).join(' · ')}.
          {data.kind === 'slides' && !data.readers.openslide && <> Most whole-slide formats need OpenSlide: <code>pip install "stainid[slides]"</code>.</>}</p>
      </Advanced>
      <ScreenFoot next={next} nextLabel={nextLabel} disabled={!data.total} />
    </div>
  )
}

function AddImages({ data, onSaved, onCancel }: { data: ImagesOverview; onSaved: () => void; onCancel?: () => void }) {
  const [picking, setPicking] = useState(false)
  const [found, setFound] = useState<ImageRow[] | null>(null)
  const [source, setSource] = useState('')
  const [stain, setStain] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const thing = data.kind === 'slides' ? 'slide' : 'image'
  const scan = (folder: string) => withError(async () => {
    setPicking(false); setBusy(true)
    try {
      const result = await api.post<{ rows: ImageRow[]; found: number; limit: number }>('/api/setup/images/scan', { folder })
      setFound(result.rows); setSource(folder)
    } finally { setBusy(false) }
  }, setError)
  const fromList = async (file: File) => {
    const result = await api.upload<{ rows: ImageRow[] }>('/api/setup/images/read', file)
    setFound(result.rows); setSource(file.name)
    return `${plural(result.rows.length, thing)} in ${file.name}`
  }
  const rows = (found ?? []).map((r) => (stain && !r.stain ? { ...r, stain } : r))
  const save = (mode: 'replace' | 'add') => withError(async () => {
    setBusy(true)
    try { await api.put('/api/setup/images', { rows, mode }); setFound(null); onSaved() } finally { setBusy(false) }
  }, setError)
  const noStain = rows.filter((r) => !r.stain).length
  const noDonor = rows.filter((r) => !r.donor_id).length

  return (
    <div className="card">
      <h3>{data.total ? `Add ${thing}s` : `Where are the ${thing}s?`}</h3>
      <p className="secondary small">Choose the folder that holds them (subfolders are looked in too: a folder per donor gives each {thing} its donor),
        or upload a list made in Excel with a column for the file, the stain and the donor.</p>
      <div className="row">
        <button className="btn primary" disabled={busy} onClick={() => setPicking(true)}>{busy && !found ? 'Looking…' : 'Choose a folder…'}</button>
        <UploadButton label="Upload a list (CSV or Excel)" accept=".csv,.tsv,.txt,.xlsx,.xls" onFile={fromList} />
        {onCancel && <button className="link-btn small" onClick={onCancel}>Cancel</button>}
      </div>
      {found && (
        <div className="images-found">
          <Found tone={noStain || noDonor ? 'warn' : 'good'}>
            {plural(found.length, thing)} in <b>{source}</b>
            {(noStain > 0 || noDonor > 0) && <ul className="found-notes">
              {noStain > 0 && <li>{plural(noStain, thing)} without a stain in the file name: give it below, or fix it in the list afterwards</li>}
              {noDonor > 0 && <li>{plural(noDonor, thing)} without a donor (put each donor's {thing}s in a folder of their own, or upload a list)</li>}
            </ul>}
          </Found>
          {noStain > 0 && <label className="inline-field small">Stain of the {thing}s without one
            <input type="text" value={stain} placeholder="e.g. 4G8, AT8, NeuN" onChange={(e) => setStain(e.target.value)} /></label>}
          <ImageTable rows={rows.slice(0, 12)} thing={thing} />
          {rows.length > 12 && <p className="small muted">… and {plural(rows.length - 12, `more ${thing}`)}</p>}
          <div className="row">
            <button className="btn primary" disabled={busy || !rows.length} onClick={() => save(data.total ? 'add' : 'replace')}>
              {busy ? 'Saving…' : data.total ? `Add these ${plural(rows.length, thing)}` : `Use these ${plural(rows.length, thing)}`}</button>
            {data.total > 0 && <button className="btn" disabled={busy || !rows.length} onClick={() => save('replace')}>Replace the list with these</button>}
          </div>
        </div>
      )}
      <ErrorNote error={error} />
      {picking && <FilePicker title={`Choose the folder with the ${thing}s`} onClose={() => setPicking(false)} onPick={scan} />}
    </div>
  )
}

function ImageTable({ rows, thing = 'image', onEdit }: { rows: ImageRow[]; thing?: string; onEdit?: (index: number, column: 'donor_id' | 'stain' | 'region', value: string) => void }) {
  const cell = (r: ImageRow, i: number, column: 'donor_id' | 'stain' | 'region') => onEdit
    ? <input type="text" value={r[column] ?? ''} aria-label={column} onChange={(e) => onEdit(i, column, e.target.value)} />
    : r[column] || <span className="muted">—</span>
  return (
    <div className="table-wrap images-table">
      <table className="compact small">
        <thead><tr><th>{thing === 'slide' ? 'Slide' : 'Image'}</th><th>Donor</th><th>Stain</th><th>Region</th></tr></thead>
        <tbody>{rows.map((r, i) => (
          <tr key={`${r.path}-${i}`}><td className="ellipsis" title={r.path}>{r.image_id || r.path.split(/[\\/]/).pop()}</td>
            <td>{cell(r, i, 'donor_id')}</td><td>{cell(r, i, 'stain')}</td><td>{cell(r, i, 'region')}</td></tr>
        ))}</tbody>
      </table>
    </div>
  )
}

function ImageList({ data, onChanged, onAdd }: { data: ImagesOverview; onChanged: () => void; onAdd: () => void }) {
  const [page, setPage] = useState(0)
  const [rows, setRows] = useState<ImageRow[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const editable = data.total <= EDITABLE
  const shown = rows ?? data.rows
  const pages = Math.ceil(shown.length / PAGE)
  const edit = (index: number, column: 'donor_id' | 'stain' | 'region', value: string) => {
    const copy = [...shown]
    copy[page * PAGE + index] = { ...copy[page * PAGE + index], [column]: value }
    setRows(copy)
  }
  const save = () => withError(async () => {
    setBusy(true)
    try { await api.put('/api/setup/images', { rows: shown, mode: 'replace' }); setRows(null); onChanged() } finally { setBusy(false) }
  }, setError)
  const thing = data.kind === 'slides' ? 'slide' : 'image'
  return (
    <div className="card">
      <div className="row card-head">
        <h3>The {thing}s</h3><span className="spacer" />
        <button className="btn small" onClick={onAdd}>Add {thing}s…</button>
      </div>
      <ImageTable rows={shown.slice(page * PAGE, (page + 1) * PAGE)} thing={thing} onEdit={editable ? edit : undefined} />
      <div className="row small">
        {pages > 1 && <span className="pager">
          <button className="btn small" disabled={page === 0} onClick={() => setPage(page - 1)}>← Previous</button>
          <span className="muted">{page * PAGE + 1}–{Math.min(shown.length, (page + 1) * PAGE)} of {shown.length.toLocaleString()}</span>
          <button className="btn small" disabled={page >= pages - 1} onClick={() => setPage(page + 1)}>Next →</button></span>}
        {!editable && <span className="muted">The first {EDITABLE.toLocaleString()} of {data.total.toLocaleString()} are shown; to change the others, edit the image list file (under Advanced).</span>}
        <span className="spacer" />
        {rows && <><button className="btn" onClick={() => setRows(null)}>Undo changes</button>
          <button className="btn primary" disabled={busy} onClick={save}>{busy ? 'Saving…' : 'Save changes'}</button></>}
      </div>
      <ErrorNote error={error} />
    </div>
  )
}
