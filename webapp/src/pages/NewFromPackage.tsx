import { useState } from 'react'
import type { PackagePreview, ProjectEntry, ProjectsInfo, StainFamily } from '../api'
import { api, projectPath } from '../api'
import { ErrorNote, FilePicker, withError } from '../components'
import { Advanced, Found, plural } from '../setup/parts'

export const folderName = (name: string) => name.trim().replace(/[\\/:*?"<>|]+/g, '-').replace(/\s+/g, ' ')
const KIND_WORDS = { tma: 'a TMA study', images: 'an image set', slides: 'whole slides' }

/** "Where to keep it": the new study's folder, made inside the chosen one. */
export function StudyPlace({ folder, parent, onChoose }: { folder: string; parent: string; onChoose: () => void }) {
  return (
    <div className="field">Where to keep it
      <div className="row">
        <code className="path-box new-study-path" title={folder || parent}>{folder || (parent ? `${parent}/…` : 'no folder chosen yet')}</code>
        <button className="btn small" onClick={onChoose}>{parent ? 'Change…' : 'Choose…'}</button>
      </div>
      <span className="muted small">stainID makes a new folder with the study's name there and keeps everything for the study in it.</span>
    </div>
  )
}

/** New study → From a dataset package: choose the package, see what is in it, check what its stains and labels mean,
and make the study. The images stay where they are; nothing is copied. */
export function FromPackage({ info, onClose }: { info: ProjectsInfo; onClose: () => void }) {
  const [picking, setPicking] = useState<'package' | 'place' | null>(null)
  const [preview, setPreview] = useState<PackagePreview | null>(null)
  const [name, setName] = useState('')
  const [parent, setParent] = useState(info.projects_dir)
  const [classMap, setClassMap] = useState<Record<string, string>>({})
  const [families, setFamilies] = useState<Record<string, StainFamily>>({})
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState('')
  const read = (folder: string) => withError(async () => {
    setPicking(null); setBusy('reading')
    try {
      const found = await api.post<PackagePreview>('/api/projects/package-preview', { package: folder })
      setPreview(found); setName(found.title); setClassMap(found.class_map); setFamilies({})
    } finally { setBusy('') }
  }, setError)
  const folder = parent && folderName(name) ? `${parent.replace(/[\\/]+$/, '')}/${folderName(name)}` : ''
  const create = () => preview && withError(async () => {
    setBusy('creating')
    try {
      const stains = Object.fromEntries(Object.entries(families).map(([stain, family]) => [stain, { family }]))
      const entry = await api.post<ProjectEntry>('/api/projects/from-package', { package: preview.folder, path: folder, name: name.trim(), class_map: classMap, stains })
      location.assign(projectPath(entry.id, '/setup'))
    } finally { setBusy('') }
  }, setError)

  return (
    <div className="new-study-form">
      <div className="field">The dataset package
        <div className="row">
          <code className="path-box new-study-path">{preview?.folder ?? 'no package chosen yet'}</code>
          <button className={`btn small${preview ? '' : ' primary'}`} disabled={busy === 'reading'} onClick={() => setPicking('package')}>
            {busy === 'reading' ? 'Reading…' : preview ? 'Change…' : 'Choose…'}</button>
        </div>
        <span className="muted small">The folder with dataset.yaml (or the dataset folder that holds package/).</span>
      </div>
      {preview && <PackageSummary preview={preview} families={families} onFamily={(stain, family) => setFamilies({ ...families, [stain]: family })}
        classMap={classMap} onClass={(label, value) => setClassMap({ ...classMap, [label]: value })} />}
      {preview && <>
        <label className="field">Study name<input type="text" value={name} onChange={(e) => setName(e.target.value)} /></label>
        <StudyPlace folder={folder} parent={parent} onChoose={() => setPicking('place')} />
      </>}
      <div className="row">
        <button className={`btn${preview ? ' primary' : ''}`} disabled={!preview || !folder || Boolean(busy)} onClick={create}>{busy === 'creating' ? 'Creating…' : 'Create and set up'}</button>
        <button className="btn" onClick={onClose}>Cancel</button>
        {preview && !folder && <span className="small muted">{!name.trim() ? 'Give the study a name' : 'Choose where to keep it'}</span>}
      </div>
      <ErrorNote error={error} />
      {picking === 'package' && <FilePicker title="Choose the dataset package" start={preview?.folder} onClose={() => setPicking(null)} onPick={read} />}
      {picking === 'place' && <FilePicker title="Choose where to keep the new study" start={parent || undefined} onClose={() => setPicking(null)}
        onPick={(path) => { setParent(path); setPicking(null) }} />}
    </div>
  )
}

function PackageSummary({ preview, families, onFamily, classMap, onClass }: {
  preview: PackagePreview; families: Record<string, StainFamily>; onFamily: (stain: string, family: StainFamily) => void
  classMap: Record<string, string>; onClass: (label: string, value: string) => void
}) {
  const raters = Object.entries(preview.raters)
  const missingFiles = preview.files_checked - preview.files_found
  const unknown = preview.stains.filter((s) => !(families[s.stain] ?? s.family))
  const unmapped = Object.keys(preview.labels).filter((l) => l && !classMap[l])
  const classes = [...new Set(Object.values(preview.classes).flat())]
  return (
    <>
      <Found tone={missingFiles || unknown.length ? 'warn' : 'good'}>
        <b>{preview.title}</b> — {KIND_WORDS[preview.kind]}: {plural(preview.images, preview.kind === 'slides' ? 'slide' : 'image')} from {plural(preview.donors, 'donor')}
        {preview.annotations > 0 && <>, {plural(preview.annotations, 'label')} by {raters.length === 1 ? raters[0][0] : plural(raters.length, 'rater')}</>}
        {preview.pixel_size_um && <> · {preview.pixel_size_um} µm per pixel</>}
        <ul className="found-notes">
          {preview.licence && <li>Licence: {preview.licence}</li>}
          {preview.citation && <li>Please cite: {preview.citation}</li>}
          {missingFiles > 0 && <li>{missingFiles} of the first {preview.files_checked} image files are not where the package says (is the drive connected?)</li>}
          {raters.length > 1 && <li>Each rater's labels become a labelling session of their own{preview.raters.consensus ? '' : ', and stainID adds their consensus (what most of them marked)'}.</li>}
        </ul>
      </Found>
      <div className="field">The stains and what stainID reads them as
        <div className="table-wrap"><table className="compact small">
          <thead><tr><th>Stain</th><th>Colour</th><th className="num">Images</th><th>Read as</th></tr></thead>
          <tbody>{preview.stains.map((s) => (
            <tr key={s.stain}><td>{s.stain}{s.antibody && s.antibody !== s.stain ? ` (${s.antibody})` : ''}</td>
              <td>{s.chromogen}{s.others.length ? ` + ${s.others.join(', ')} (left out)` : ''}</td><td className="num">{s.images.toLocaleString()}</td>
              <td><select value={families[s.stain] ?? s.family} aria-label={`What ${s.stain} is`} onChange={(e) => onFamily(s.stain, e.target.value as StainFamily)}>
                <option value="">choose…</option>
                {Object.entries(preview.families).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
              </select></td></tr>
          ))}</tbody>
        </table></div>
      </div>
      {Object.keys(preview.labels).length > 0 && (
        <Advanced title="What the labels mean" hint={unmapped.length ? `${plural(unmapped.length, "label")} without a stainID class` : `${plural(Object.keys(preview.labels).length, "label")}, mapped by the package`}>
          <p className="small secondary">Each label of the dataset is one of stainID's classes, or left out. The package says most of this; change it here, or later on
            the Labels page.</p>
          <div className="table-wrap"><table className="compact small">
            <thead><tr><th>Label in the dataset</th><th className="num">Times used</th><th>stainID class</th></tr></thead>
            <tbody>{Object.entries(preview.labels).filter(([label]) => label).map(([label, n]) => (
              <tr key={label}><td>{label}</td><td className="num">{n.toLocaleString()}</td>
                <td><select value={classMap[label] ?? ''} aria-label={`Class of ${label}`} onChange={(e) => onClass(label, e.target.value)}>
                  <option value="">choose…</option>
                  {classes.map((c) => <option key={c} value={c}>{c}</option>)}
                  <option value="ignore">leave out</option>
                </select></td></tr>
            ))}</tbody>
          </table></div>
        </Advanced>
      )}
    </>
  )
}
