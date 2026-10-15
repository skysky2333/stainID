import { useState } from 'react'
import type { ProjectEntry, ProjectsInfo, ProjectState } from '../api'
import { api, projectPath } from '../api'
import { Empty, ErrorNote, FilePicker, HelpBox, Modal, PageHead, timeAgo, withError } from '../components'
import { useFetch } from '../hooks'
import { useSession } from '../session'
import { Advanced, ScreenData } from '../setup/parts'

const STATE: Record<ProjectState, { word: string; tone: string }> = {
  running: { word: 'Running', tone: 'status-running' }, results: { word: 'Results ready', tone: 'status-finished' }, set_up: { word: 'Set up', tone: 'status-done' },
  not_set_up: { word: 'Not set up yet', tone: '' }, missing: { word: 'Folder not found', tone: 'status-failed' },
}

/** The start screen: every study this server knows, a new study, or a project folder to open. */
export default function Projects() {
  const { reload: reloadSession } = useSession()
  const list = useFetch<ProjectsInfo>('/api/projects', 10000)
  const [creating, setCreating] = useState(false)
  const [opening, setOpening] = useState(false)
  const [removing, setRemoving] = useState<ProjectEntry | null>(null)
  const [message, setMessage] = useState('')
  const [error, setError] = useState<string | null>(null)
  const open = (path: string) => withError(async () => {
    const entry = await api.post<ProjectEntry>('/api/projects/open', { path })
    location.assign(projectPath(entry.id))
  }, setError)
  const removed = (p: ProjectEntry, stillRunning: boolean) => {
    setRemoving(null); list.reload()
    setMessage(`“${p.name}” is off the list; its folder is untouched.${stillRunning ? ' Its running steps carry on.' : ''} Open the folder again to bring it back.`)
  }

  return (
    <div className="projects-page">
      <header className="row projects-top">
        <div className="brand" style={{ margin: 0 }}><img src="/favicon.svg" alt="" />stainID</div>
        <span className="spacer" />
        <button className="link-btn" onClick={async () => { await api.del('/api/session/admin'); reloadSession() }}>Sign out</button>
      </header>
      <main className="main projects-main">
        <PageHead title="Projects" subtitle="Each study lives in its own project folder. Pick one to work on, or start a new study.">
          <button className="btn" onClick={() => setOpening(true)}>Open a folder…</button>
          {!creating && <button className="btn primary" onClick={() => setCreating(true)}>New study</button>}
        </PageHead>
        <HelpBox id="projects" title="Several studies, several people">
          Everyone who uses this computer's stainID sees this list. Several people can work at the same time, each in their own study: what one
          person opens or changes does not change anyone else's page. Steps that need a lot of computing (finding nuclei, detection, outlines)
          take turns across all studies, at most two at a time on this computer; a step that waits says which study it is waiting for.
        </HelpBox>
        {message && <p className="small status-finished" role="status">{message}</p>}
        <ErrorNote error={error} />
        <ScreenData state={list} label="Loading the studies…">
          {(data) => (
            <>
              {creating && <NewStudy info={data} onClose={() => setCreating(false)} />}
              {data.projects.length ? (
                <ul className="project-list">{data.projects.map((p) => <ProjectRow key={p.id} project={p} onRemove={() => setRemoving(p)} />)}</ul>
              ) : !creating && (
                <Empty>
                  <p>No studies yet. Start a new one, or open a project folder made before.</p>
                  <div className="row" style={{ justifyContent: 'center' }}>
                    <button className="btn" onClick={() => setOpening(true)}>Open a folder…</button>
                    <button className="btn primary" onClick={() => setCreating(true)}>Start a new study</button>
                  </div>
                </Empty>
              )}
              <ProjectsFolder info={data} onSaved={list.reload} />
            </>
          )}
        </ScreenData>
      </main>
      {opening && <FilePicker title="Choose a project folder (the one that contains stainid.yaml)" onClose={() => setOpening(false)}
        onPick={(path) => { setOpening(false); open(path) }} />}
      {removing && <RemoveProject project={removing} onClose={() => setRemoving(null)} onRemoved={removed} />}
    </div>
  )
}

function ProjectRow({ project: p, onRemove }: { project: ProjectEntry; onRemove: () => void }) {
  const state = STATE[p.status.state] ?? { word: p.status.text, tone: '' }
  const detail = p.status.state === 'running' ? [...new Set(p.status.running)].join(', ') : p.status.state === 'missing' ? p.status.text : ''
  return (
    <li className="card project-row">
      <div className="project-what">
        {p.exists ? <a href={projectPath(p.id)} className="project-name" title={p.path}>{p.name}</a> : <span className="project-name muted" title={p.path}>{p.name}</span>}
        <div className="row small project-status">
          <span className={`badge ${state.tone}`}>{p.status.state === 'running' && <span className="pulse" />}{state.word}</span>
          {detail && <span className="secondary ellipsis" title={detail}>{detail}</span>}
          <span className="muted">{p.last_opened ? `opened ${timeAgo(p.last_opened)}` : p.in_projects_dir ? 'in the projects folder' : ''}</span>
        </div>
      </div>
      <div className="row project-actions">
        {p.exists && <a className="btn" href={projectPath(p.id)}>Open</a>}
        <button className="link-btn small" onClick={onRemove}>Remove from list</button>
      </div>
    </li>
  )
}

function RemoveProject({ project, onClose, onRemoved }: { project: ProjectEntry; onClose: () => void; onRemoved: (p: ProjectEntry, stillRunning: boolean) => void }) {
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const remove = () => withError(async () => {
    setBusy(true)
    try { onRemoved(project, (await api.del<{ still_running: boolean }>(`/api/projects/${project.id}`)).still_running) } finally { setBusy(false) }
  }, setError)
  return (
    <Modal title={`Remove “${project.name}” from the list?`} onClose={onClose}>
      <p className="secondary">Nothing is deleted: the folder and everything in it stay as they are, and steps that are running carry on.
        You can bring the study back at any time with <b>Open a folder…</b>.</p>
      <code className="path-box">{project.path}</code>
      <ErrorNote error={error} />
      <div className="row"><span className="spacer" /><button className="btn" onClick={onClose}>Cancel</button>
        <button className="btn primary" disabled={busy} onClick={remove}>{busy ? 'Removing…' : 'Remove from list'}</button></div>
    </Modal>
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
  const missing = !name.trim() ? 'Give the study a name' : !parent ? 'Choose where to keep it' : ''

  return (
    <div className="card new-study">
      <h2>New study</h2>
      <div className="new-study-form">
        <label className="field">Study name<input type="text" value={name} placeholder="e.g. Alzheimer TMA 2026" autoFocus onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && folder && !busy) create() }} /></label>
        <div className="field">Where to keep it
          <div className="row">
            <code className="path-box new-study-path" title={folder || parent}>{folder || (parent ? `${parent}/…` : 'no folder chosen yet')}</code>
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
            {source && <fieldset className="copy-parts"><legend className="small secondary">Copy these settings</legend>
              {Object.entries(info.copyable).map(([key, label]) => (
                <label key={key} className="layer-toggle small"><input type="checkbox" checked={parts.includes(key)}
                  onChange={() => setParts(parts.includes(key) ? parts.filter((k) => k !== key) : [...parts, key])} />{label}</label>
              ))}
            </fieldset>}
            {source && <span className="muted small">Slides, maps, donors and results are never copied: the new study starts empty. Models are shared, not copied.</span>}
          </div>
        )}
        <div className="row">
          <button className="btn primary" disabled={!folder || busy} onClick={create}>{busy ? 'Creating…' : 'Create and set up'}</button>
          <button className="btn" onClick={onClose}>Cancel</button>
          {missing && <span className="small muted">{missing}</span>}
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
    <Advanced hint="projects folder, project locations">
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
      {info.projects.length > 0 && (
        <div className="table-wrap"><table className="compact small">
          <thead><tr><th>Study</th><th>Folder</th></tr></thead>
          <tbody>{info.projects.map((p) => <tr key={p.id}><td>{p.name}</td><td><code>{p.path}</code></td></tr>)}</tbody>
        </table></div>
      )}
      <ErrorNote error={error} />
      {picking && <FilePicker title="Choose the projects folder" start={info.projects_dir || undefined} onClose={() => setPicking(false)}
        onPick={(path) => { setPicking(false); save(path) }} />}
    </Advanced>
  )
}
