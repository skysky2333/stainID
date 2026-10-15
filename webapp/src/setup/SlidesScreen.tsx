import { useMemo, useState } from 'react'
import type { SlideRow, SlidesInfo } from '../api'
import { api } from '../api'
import { Empty, ErrorNote, FilePicker, PathLine, withError } from '../components'
import { useFetch } from '../hooks'
import { useProject } from '../project'
import { fileName as name, SlideTable, slideProblems } from '../slides'
import { Advanced, Found, plural, ScreenData, ScreenFoot } from './parts'
import type { ScreenProps } from './Setup'

const byNumber = (a: string, b: string) => Number(a) - Number(b) || a.localeCompare(b)
const folderName = (path: string) => path.replace(/[\\/]+$/, '').split(/[\\/]/).pop() || path
/** Scans that still need a choice first, then by TMA number and stain (file names sort TMA 10 before TMA 2). */
const inOrder = (rows: SlideRow[]) => [...rows].sort((a, b) => Number(Boolean(a.tma)) - Number(Boolean(b.tma)) || byNumber(a.tma, b.tma) || a.stain.localeCompare(b.stain))

export default function SlidesScreen(props: ScreenProps) {
  const info = useFetch<SlidesInfo>('/api/slides')
  return <ScreenData state={info} label="Looking for slide scans…">{(data) => <Slides {...props} data={data} reload={info.reload} />}</ScreenData>
}

function Slides({ next, nextLabel, onChanged, data, reload }: ScreenProps & { data: SlidesInfo; reload: () => void }) {
  const { info: project, reload: reloadProject } = useProject()
  const first = useMemo(() => inOrder(data.rows), [data.rows])
  // What people chose and have not saved yet, per scan. Kept apart from the scans the server lists, so looking at the
  // folder again (or choosing it again) keeps the choices for the scans that are still there.
  const [edits, setEdits] = useState<Record<string, Pick<SlideRow, 'tma' | 'stain'>>>({})
  const [typedPrefix, setPrefix] = useState<string | null>(null)
  const [picking, setPicking] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const rows = useMemo(() => first.map((r) => (edits[r.slide_path] ? { ...r, ...edits[r.slide_path] } : r)), [first, edits])
  const setRows = (changed: SlideRow[]) => setEdits(Object.fromEntries(changed.filter((r, i) => r.tma !== first[i]?.tma || r.stain !== first[i]?.stain)
    .map((r) => [r.slide_path, { tma: r.tma, stain: r.stain }])))
  const prefix = typedPrefix ?? (data.tma_prefix === 'TMA-' && data.prefix_guess ? data.prefix_guess : data.tma_prefix)
  const tmas = useMemo(() => [...new Set(rows.map((r) => r.tma).filter(Boolean))].sort(byNumber), [rows])
  const stains = useMemo(() => [...new Set(rows.map((r) => r.stain).filter(Boolean))], [rows])
  const problems = slideProblems(rows, data.stains)
  const fresh = rows.filter((r) => r.new)
  const fromName = rows.filter((r) => r.new && r.from_name?.tma && r.from_name?.stain).length
  const changed = !data.saved || fresh.length > 0 || data.missing.length > 0 || prefix !== data.tma_prefix
    || rows.some((r, i) => r.tma !== first[i]?.tma || r.stain !== first[i]?.stain)

  const chooseFolder = (path: string) => withError(async () => {
    await api.put('/api/project/config', { changes: { inputs: { slides_dir: path } } })
    setPicking(false); reloadProject(); reload(); onChanged()
  }, setError)
  const save = () => withError(async () => {
    setBusy(true)
    try {
      await api.put('/api/slides', { rows })
      if (prefix && prefix !== data.tma_prefix) { await api.put('/api/project/config', { changes: { tma: { prefix } } }); reloadProject() }
      setEdits({}); setPrefix(null)
      onChanged()
      next?.()
    } finally { setBusy(false) }
  }, setError)

  return (
    <div className="screen">
      <div className="card">
        {!data.folder_exists ? (
          <Empty>
            <p>First choose the folder that holds your slide scans (.vsi, .svs, .ndpi, .czi, .tif …).</p>
            <button className="btn primary" onClick={() => setPicking(true)}>Choose the slides folder…</button>
          </Empty>
        ) : !rows.length ? (
          <Empty>
            <p>There are no slide scans in the folder <b title={data.folder_path}>{folderName(data.folder)}</b>. Copy the scans there, or choose another folder.</p>
            <div className="row" style={{ justifyContent: 'center' }}>
              <button className="btn" onClick={reload}>Look again</button>
              <button className="btn primary" onClick={() => setPicking(true)}>Choose another folder…</button>
            </div>
          </Empty>
        ) : (
          <>
            <div className="row card-head">
              <span>
                <b>{plural(rows.length, 'scan')}</b> in the folder <b title={data.folder_path}>{folderName(data.folder)}</b>
                {tmas.length > 0 && <span className="secondary">: {plural(tmas.length, 'TMA')} × {plural(stains.length, 'stain')}</span>}
              </span>
              <span className="spacer" />
              <button className="btn small" onClick={() => setPicking(true)}>Choose another folder…</button>
            </div>
            <Found tone={problems.ok ? 'good' : 'warn'}>
              {!problems.ok ? <><b>{problems.text}.</b> Choose {problems.needTma.length && problems.needStain.length ? 'them' : 'it'} in the highlighted rows; the picture shows each slide's label.</>
                : fresh.length && data.saved ? <><b>{plural(fresh.length, 'new scan')}</b> since the last save{fromName ? ', read from the file names' : ''}. Check them and press <b>Save and continue</b>.</>
                : !data.saved ? <>{fromName === rows.length ? 'The TMA and the stain of every scan were read from its file name' : 'Every scan has a TMA and a stain'}. Check them against the labels and press <b>Save and continue</b>.</>
                : <>Every scan has its TMA and stain.{changed ? ' Press Save and continue to keep your changes.' : ''}</>}
            </Found>
            <SlideTable info={data} rows={rows} prefix={prefix} onChange={setRows} />
          </>
        )}
        <ErrorNote error={error} />
      </div>

      <Advanced hint="core names, file locations, grid size">
        {rows.length > 0 && (
          <div className="row naming">
            <label className="field">Core names start with
              <input type="text" value={prefix} style={{ width: 90 }} onChange={(e) => setPrefix(e.target.value)} />
            </label>
            <span className="small secondary">Cores are named like <code>{prefix}{tmas[0] ?? '1'}_B-2</code> (TMA {tmas[0] ?? '1'}, column B, row 2).
              {data.prefix_guess && data.tma_prefix === 'TMA-' && prefix === data.prefix_guess && ' Suggested by the file names.'}</span>
          </div>
        )}
        <PathLine label="Slides folder" path={data.folder_path} exists={data.folder_exists} wrap />
        <PathLine label="Slides table" path={data.table} exists={data.saved} wrap />
        {data.missing.length > 0 && <div className="warning-note small">These slides are in the slides table but their files are gone: {data.missing.map((m) => name(m.slide_path)).join(', ')}. They are dropped when you save.</div>}
        <p className="small secondary">The grid of cores on each slide is found automatically on the next screen; a slide can also be given its own rows and columns there.
          The size tried first is the project's: {project?.config.tma.rows} rows × {project?.config.tma.columns} columns (Settings → TMA layout).</p>
        {rows.some((r) => r.rows) && (
          <div className="table-wrap"><table className="compact small"><thead><tr><th>Scan</th><th className="num">Rows</th><th className="num">Columns</th><th>Set by</th><th>Turned</th></tr></thead>
            <tbody>{rows.filter((r) => r.rows).map((r) => <tr key={r.slide_path}><td>{name(r.slide_path)}</td><td className="num">{r.rows}</td><td className="num">{r.columns}</td>
              <td>{r.grid_source === 'auto' ? 'found automatically' : 'you'}</td><td>{r.orientation || 'as scanned'}</td></tr>)}</tbody></table></div>
        )}
      </Advanced>

      <ScreenFoot next={rows.length && changed ? save : next} nextLabel={rows.length && changed ? 'Save and continue' : nextLabel} busy={busy}
        disabled={rows.length > 0 && !problems.ok} quiet={!rows.length}
        extra={rows.length > 0 && !problems.ok && <span className="small cell-note">{problems.text}</span>} />
      {picking && <FilePicker title="Choose the folder with your slide scans" show="slides" start={data.folder_exists ? data.folder_path : undefined} onPick={chooseFolder} onClose={() => setPicking(false)} />}
    </div>
  )
}
