import { useState } from 'react'
import { api } from '../api'
import { ErrorNote, FilePicker, withError } from '../components'
import { useProject } from '../project'

/** Shown when the app is not pointed at a project yet: create a new one or open an existing one. */
export default function Welcome() {
  const { info, reload } = useProject()
  const [picking, setPicking] = useState<'create' | 'open' | null>(null)
  const [folder, setFolder] = useState('')
  const [name, setName] = useState('')
  const [error, setError] = useState<string | null>(null)

  const create = () => withError(async () => { await api.post('/api/project/create', { path: folder, name }); reload() }, setError)
  const open = (path: string) => withError(async () => { await api.post('/api/project/open', { path }); reload() }, setError)

  return (
    <div className="welcome">
      <div className="welcome-card card">
        <div className="brand" style={{ padding: 0 }}><img src="/favicon.svg" alt="" />stainID</div>
        <h1>Welcome</h1>
        <p className="secondary">
          stainID measures neurons (NeuN), amyloid plaques (6E10) and tau pathology (AT8) on tissue microarray slides.
          Everything for one study — settings, core images, results — lives in one <b>project folder</b>.
        </p>

        <h2>Start a new study</h2>
        <div className="grid" style={{ gap: 8 }}>
          <label className="field">Study name<input type="text" value={name} placeholder="e.g. Alzheimer TMA 2026" onChange={(e) => setName(e.target.value)} /></label>
          <label className="field">Project folder (an empty folder where stainID will keep everything)
            <div className="row"><input type="text" value={folder} style={{ flex: 1 }} onChange={(e) => setFolder(e.target.value)} placeholder="/Users/you/Studies/my-tma" />
              <button className="btn" onClick={() => setPicking('create')}>Choose…</button></div>
          </label>
          <div><button className="btn primary" disabled={!folder} onClick={create}>Create project</button></div>
        </div>

        <h2 style={{ marginTop: 24 }}>Open an existing study</h2>
        <div className="row">
          <button className="btn" onClick={() => setPicking('open')}>Choose project folder…</button>
          {info && !info.configured && <span className="muted small">The current folder ({info.root}) has no stainid.yaml.</span>}
        </div>
        {info && info.recent.length > 0 && (
          <div className="grid" style={{ gap: 6, marginTop: 10 }}>
            <span className="muted small">Recent projects</span>
            {info.recent.map((p) => <button key={p} className="picker-item" onClick={() => open(p)}>📁 {p}</button>)}
          </div>
        )}
        <ErrorNote error={error} />
      </div>
      {picking && (
        <FilePicker title={picking === 'create' ? 'Choose a folder for the new project' : 'Choose a project folder (it contains stainid.yaml)'}
          onClose={() => setPicking(null)}
          onPick={(path) => { setPicking(null); if (picking === 'create') setFolder(path); else open(path) }} />
      )}
    </div>
  )
}
