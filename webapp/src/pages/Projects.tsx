import { useState } from 'react'
import type { ProjectEntry, ProjectsInfo } from '../api'
import { api, projectPath } from '../api'
import { Empty, ErrorNote, FilePicker, HelpBox, PageHead, timeAgo, withError } from '../components'
import { useFetch } from '../hooks'
import { useSession } from '../session'
import { Advanced } from '../setup/parts'

const STATE_CLASS: Record<string, string> = { running: 'status-running', results: 'status-finished', set_up: 'status-done', not_set_up: '', missing: 'status-failed' }

/** The start screen: every study this server knows, a new study, or a project folder to open. */
export default function Projects() {
  const { reload: reloadSession } = useSession()
  const list = useFetch<ProjectsInfo>('/api/projects', 10000)
  const [creating, setCreating] = useState(false)
  const [opening, setOpening] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const open = (path: string) => withError(async () => {
    const entry = await api.post<ProjectEntry>('/api/projects/open', { path })
    location.assign(projectPath(entry.id))
  }, setError)
  const remove = (p: ProjectEntry) => {
    if (!window.confirm(`Remove “${p.name}” from this list? Nothing is deleted: the folder ${p.path} stays as it is, and you can open it again later.`)) return
    withError(async () => { await api.del(`/api/projects/${p.id}`); list.reload() }, setError)
  }
  const data = list.data

  return (
    <div className="projects-page">
      <header className="row projects-top">
        <div className="brand" style={{ margin: 0 }}><img src="/favicon.svg" alt="" />stainID</div>
        <span className="spacer" />
        <button className="link-btn" onClick={async () => { await api.del('/api/session/admin'); reloadSession() }}>sign out</button>
      </header>
      <main className="main projects-main">
        <PageHead title="Projects" subtitle="Each study lives in its own project folder. Pick one to work on, or start a new study.">
          <button className="btn" onClick={() => setOpening(true)}>Open existing folder…</button>
          <button className="btn primary" onClick={() => setCreating(!creating)}>New study</button>
        </PageHead>
        <HelpBox id="projects" title="Several studies, several people">
          Everyone who uses this computer's stainID sees this list. Several people can work at the same time, each in their own study: what one
          person opens or changes does not change anyone else's page. Steps that need a lot of computing (finding nuclei, detection, outlines)
          take turns across all studies, at most two at a time on this computer; a step that waits says which study it is waiting for.
        </HelpBox>
        {creating && data && <NewStudy info={data} onClose={() => setCreating(false)} />}
        <ErrorNote error={error ?? list.error} />
        {data && (data.projects.length ? (
          <div className="card project-list">
            <table>
              <thead><tr><th>Study</th><th>Status</th><th>Opened</th><th /></tr></thead>
              <tbody>{data.projects.map((p) => (
                <tr key={p.id}>
                  <td>
                    {p.exists ? <a href={projectPath(p.id)} className="project-name">{p.name}</a> : <span className="project-name muted">{p.name}</span>}
                    <div className="muted small wrap" title={p.path}>{p.path}</div>
                  </td>
                  <td><span className={`badge ${STATE_CLASS[p.status.state] ?? ''}`}>{p.status.state === 'running' && <span className="pulse" />}{p.status.text}</span></td>
                  <td className="small secondary">{p.last_opened ? timeAgo(p.last_opened) : p.in_projects_dir ? 'in the projects folder' : '—'}</td>
                  <td><div className="row" style={{ justifyContent: 'flex-end', flexWrap: 'nowrap' }}>
                    {p.exists && <a className="btn small primary" href={projectPath(p.id)}>Open</a>}
                    <button className="link-btn small" onClick={() => remove(p)}>Remove from list</button>
                  </div></td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        ) : !creating && (
          <Empty>No studies yet.<br />
            <button className="btn primary" style={{ marginTop: 10 }} onClick={() => setCreating(true)}>Start a new study</button>{' '}
            <button className="btn" style={{ marginTop: 10 }} onClick={() => setOpening(true)}>Open an existing project folder…</button></Empty>
        ))}
        {data && <ProjectsFolder info={data} onSaved={list.reload} />}
      </main>
      {opening && <FilePicker title="Choose a project folder (the one that contains stainid.yaml)" onClose={() => setOpening(false)}
        onPick={(path) => { setOpening(false); open(path) }} />}
    </div>
  )
}

const folderName = (name: string) => name.trim().replace(/[\\/:*?"<>|]+/g, '-').replace(/\s+/g, ' ')

function NewStudy({ info, onClose }: { info: ProjectsInfo; onClose: () => void }) {
  const [name, setName] = useState('')
  const [parent, setParent] = useState(info.projects_dir)
  const [picking, setPicking] = useState(false)
  const [source, setSource] = useState('')
  const [parts, setParts] = useState<string[]>(Object.keys(info.copyable))
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const folder = parent && folderName(name) ? `${parent.replace(/[\\/]+$/, '')}/${folderName(name)}` : ''
  const create = () => withError(async () => {
    setBusy(true)
    try {
      const entry = await api.post<ProjectEntry>('/api/projects/create', { path: folder, name: name.trim(), copy_from: source, settings: parts })
      location.assign(projectPath(entry.id, '/setup'))
    } finally { setBusy(false) }
  }, setError)
  const others = info.projects.filter((p) => p.exists)

  return (
    <div className="card new-study">
      <h2 style={{ marginTop: 0 }}>New study</h2>
      <div className="grid" style={{ gap: 10, maxWidth: 640 }}>
        <label className="field">Study name<input type="text" value={name} placeholder="e.g. Alzheimer TMA 2026" autoFocus onChange={(e) => setName(e.target.value)} /></label>
        <div className="field">Where to keep it
          <div className="row">
            <code className="path-box" style={{ flex: 1 }}>{folder || (parent ? `${parent}/…` : 'choose a folder')}</code>
            <button className="btn small" onClick={() => setPicking(true)}>{parent ? 'Change…' : 'Choose…'}</button>
          </div>
          <span className="muted small">stainID makes a new folder with the study's name there and keeps everything for the study in it.</span>
        </div>
        {others.length > 0 && (
          <div className="field">Start with the settings of
            <select value={source} onChange={(e) => setSource(e.target.value)}>
              <option value="">no other study (start from the defaults)</option>
              {others.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
            </select>
            {source && <div className="row small" style={{ marginTop: 4 }}>
              {Object.entries(info.copyable).map(([key, label]) => (
                <label key={key} className="layer-toggle"><input type="checkbox" checked={parts.includes(key)}
                  onChange={() => setParts(parts.includes(key) ? parts.filter((k) => k !== key) : [...parts, key])} />{label}</label>
              ))}
            </div>}
            {source && <span className="muted small">Slides, maps, donors and results are never copied: the new study starts empty. Models are shared, not copied.</span>}
          </div>
        )}
        <div className="row">
          <button className="btn primary" disabled={!folder || busy} onClick={create}>{busy ? 'Creating…' : 'Create and set up'}</button>
          <button className="btn" onClick={onClose}>Cancel</button>
        </div>
        <ErrorNote error={error} />
      </div>
      {picking && <FilePicker title="Choose where to keep the new study" start={parent || undefined} onClose={() => setPicking(false)}
        onPick={(path) => { setParent(path); setPicking(false) }} />}
    </div>
  )
}

function ProjectsFolder({ info, onSaved }: { info: ProjectsInfo; onSaved: () => void }) {
  const [picking, setPicking] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const save = (path: string) => withError(async () => { await api.put('/api/projects/settings', { path }); onSaved() }, setError)
  return (
    <Advanced hint="projects folder">
      <div className="field">Projects folder
        <div className="row">
          <code className="path-box" style={{ flex: 1 }}>{info.projects_dir || 'none'}</code>
          <button className="btn small" onClick={() => setPicking(true)}>Choose…</button>
          {info.projects_dir && info.projects_dir_from === 'setting' && <button className="btn small" onClick={() => save('')}>Clear</button>}
        </div>
      </div>
      <p className="muted small">Every folder inside the projects folder that holds a <code>stainid.yaml</code> is listed here, and new studies go there
        by default. {info.projects_dir_from === 'command line' && <>It is set for this run with <code>stainid serve --projects-dir</code>, which wins over the
        folder chosen here.</>} Each project's address is <code>/p/&lt;project&gt;/</code>, so a link to a page also opens the right study.</p>
      <ErrorNote error={error} />
      {picking && <FilePicker title="Choose the projects folder" start={info.projects_dir || undefined} onClose={() => setPicking(false)}
        onPick={(path) => { setPicking(false); save(path) }} />}
    </Advanced>
  )
}
