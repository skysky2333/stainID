import { useEffect, useMemo, useState } from 'react'
import type { SlideRow, SlidesInfo } from '../api'
import { api } from '../api'
import { Empty, ErrorNote, FilePicker, PathLine, withError } from '../components'
import { useFetch } from '../hooks'
import { useProject } from '../project'
import { Advanced, Found, ScreenFoot } from './parts'
import type { ScreenProps } from './Setup'

const name = (path: string) => path.split(/[\\/]/).pop() ?? path

export default function SlidesScreen({ next, onChanged }: ScreenProps) {
  const { info: project, reload: reloadProject } = useProject()
  const info = useFetch<SlidesInfo>('/api/slides')
  const [rows, setRows] = useState<SlideRow[]>([])
  const [prefix, setPrefix] = useState('')
  const [picking, setPicking] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    if (!info.data) return
    setRows(info.data.rows)
    setPrefix(info.data.tma_prefix === 'TMA-' && info.data.prefix_guess ? info.data.prefix_guess : info.data.tma_prefix)
  }, [info.data])
  const edit = (i: number, key: keyof SlideRow, value: string) => setRows(rows.map((r, j) => (j === i ? { ...r, [key]: value } : r)))
  const tmas = useMemo(() => [...new Set(rows.map((r) => r.tma).filter(Boolean))].sort((a, b) => Number(a) - Number(b)), [rows])
  const stains = useMemo(() => [...new Set(rows.map((r) => r.stain).filter(Boolean))], [rows])
  const unsure = rows.filter((r) => !r.tma || !r.stain)
  const fresh = rows.filter((r) => r.new)
  const twice = rows.filter((r, i) => r.tma && r.stain && rows.findIndex((x) => x.tma === r.tma && x.stain === r.stain) !== i)

  const chooseFolder = (path: string) => withError(async () => {
    await api.put('/api/project/config', { changes: { inputs: { slides_dir: path } } })
    setPicking(false); reloadProject(); info.reload()
  }, setError)
  const save = () => withError(async () => {
    setBusy(true)
    try {
      await api.put('/api/slides', { rows })
      if (prefix && prefix !== info.data?.tma_prefix) { await api.put('/api/project/config', { changes: { tma: { prefix } } }); reloadProject() }
      onChanged()
      next?.()
    } finally { setBusy(false) }
  }, setError)

  if (!info.data) return <ErrorNote error={info.error} />
  const data = info.data
  return (
    <div className="screen">
      <div className="card">
        <div className="row">
          <h2 style={{ margin: 0 }}>Slide scans</h2><span className="spacer" />
          <button className="btn small" onClick={() => setPicking(true)}>{data.folder_exists ? 'Choose another folder…' : 'Choose the folder…'}</button>
        </div>
        {!data.folder_exists ? (
          <Empty>Choose the folder that holds your slide scans (.vsi, .svs, .ndpi, .czi, .tif …).<br />
            <button className="btn primary" style={{ marginTop: 10 }} onClick={() => setPicking(true)}>Choose the slides folder…</button></Empty>
        ) : !rows.length ? (
          <Empty>No slide scans in <code>{data.folder}</code>. Copy the scans there, or choose another folder.</Empty>
        ) : (
          <>
            <Found tone={unsure.length || twice.length ? 'warn' : 'good'}>
              Found <b>{rows.length} scans</b> in <code>{data.folder}</code>: {tmas.length} TMAs ({tmas.join(', ')}) × {stains.length} stains ({stains.join(', ')}).
              {fresh.length > 0 && data.saved && <> <b>{fresh.length} new</b> since last time.</>}
              {unsure.length > 0 && <> For {unsure.length} the TMA or stain could not be read from the file name: please fill them in below.</>}
              {twice.length > 0 && <> Some TMAs have the same stain twice: please correct.</>}
              {!unsure.length && !twice.length && <> The TMA and stain were read from each file name; check them and press Save.</>}
            </Found>
            <div className="table-wrap" style={{ maxHeight: 420 }}>
              <table>
                <thead><tr><th>Scan</th><th style={{ width: 110 }}>TMA</th><th style={{ width: 130 }}>Stain</th><th style={{ width: 60 }} /></tr></thead>
                <tbody>{rows.map((r, i) => (
                  <tr key={r.slide_path} className={!r.tma || !r.stain ? 'row-attention' : ''}>
                    <td title={r.slide_path}>{name(r.slide_path)}</td>
                    <td><input type="text" value={r.tma} style={{ width: 80 }} onChange={(e) => edit(i, 'tma', e.target.value)} aria-label={`TMA of ${name(r.slide_path)}`} /></td>
                    <td><select value={r.stain} onChange={(e) => edit(i, 'stain', e.target.value)} aria-label={`Stain of ${name(r.slide_path)}`}>
                      <option value="">choose…</option>{data.stains.map((s) => <option key={s}>{s}</option>)}</select></td>
                    <td>{r.new && data.saved && <span className="badge status-running">new</span>}</td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
            <div className="row naming">
              <label className="field">Core names start with
                <input type="text" value={prefix} style={{ width: 90 }} onChange={(e) => setPrefix(e.target.value)} />
              </label>
              <span className="small secondary">Cores will be named like <code>{prefix}{tmas[0] ?? '1'}_B-2</code> (TMA {tmas[0] ?? '1'}, column B, row 2).
                {data.prefix_guess && data.tma_prefix === 'TMA-' && prefix === data.prefix_guess && ' Suggested by the file names.'}</span>
            </div>
          </>
        )}
        <ErrorNote error={error ?? info.error} />
      </div>

      <Advanced hint="file locations, slides table, grid size">
        <PathLine label="Slides folder" path={data.folder_path} exists={data.folder_exists} wrap />
        <PathLine label="Slides table" path={data.table} exists={data.saved} wrap />
        {data.missing.length > 0 && <div className="warning-note small">These slides are in the slides table but their files are gone: {data.missing.map((m) => name(m.slide_path)).join(', ')}. They are dropped when you save.</div>}
        <p className="small secondary">The grid of cores on each slide is found automatically on the next screen; a slide can also be given its own rows and columns there.
          The size tried first is the project's: {project?.config.tma.rows} rows × {project?.config.tma.columns} columns (Settings → TMA layout).</p>
        {rows.some((r) => r.rows) && (
          <table className="compact small"><thead><tr><th>Scan</th><th>Rows</th><th>Columns</th><th>Set by</th><th>Turned</th></tr></thead>
            <tbody>{rows.filter((r) => r.rows).map((r) => <tr key={r.slide_path}><td>{name(r.slide_path)}</td><td>{r.rows}</td><td>{r.columns}</td>
              <td>{r.grid_source === 'auto' ? 'found automatically' : 'you'}</td><td>{r.orientation || 'as scanned'}</td></tr>)}</tbody></table>
        )}
      </Advanced>

      <ScreenFoot next={rows.length ? save : next} nextLabel={rows.length ? 'Save and continue' : 'Next'} busy={busy} disabled={rows.length > 0 && (unsure.length > 0 || twice.length > 0)} />
      {picking && <FilePicker title="Choose the folder with your slide scans" show="slides" start={data.folder_exists ? data.folder_path : undefined} onPick={chooseFolder} onClose={() => setPicking(false)} />}
    </div>
  )
}
