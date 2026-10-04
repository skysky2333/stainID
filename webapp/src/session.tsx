import { createContext, useContext, useState } from 'react'
import type { Session } from './api'
import { api } from './api'
import { ErrorNote, withError } from './components'

export const SessionContext = createContext<{ session: Session; reload: () => void } | null>(null)

export function useSession() {
  const value = useContext(SessionContext)
  if (!value) throw new Error('useSession must be used inside SessionContext')
  return value
}

export function AdminSignIn() {
  const { session, reload } = useSession()
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (await withError(() => api.post('/api/session/admin', { password }), setError)) reload()
  }
  return (
    <main className="main" style={{ display: 'grid', placeItems: 'center', minHeight: '80vh' }}>
      <form className="card" style={{ width: 380 }} onSubmit={submit}>
        <div className="brand" style={{ marginBottom: 12 }}><img src="/favicon.svg" alt="" />stainID</div>
        <label className="field">Admin password<input type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoFocus /></label>
        <button className="btn primary" style={{ marginTop: 12, width: '100%' }} disabled={!password}>Sign in</button>
        {session.default_password && session.local && <p className="muted small">The password is <b>admin</b> until you change it in Settings.</p>}
        <p className="muted small">Here to annotate? Go to <a href="/annotate">the annotation page</a>.</p>
        <ErrorNote error={error} />
      </form>
    </main>
  )
}

export function Register() {
  const { reload } = useSession()
  const [name, setName] = useState('')
  const [error, setError] = useState<string | null>(null)
  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (await withError(() => api.post('/api/session/annotator', { name }), setError)) reload()
  }
  return (
    <form className="card" style={{ maxWidth: 420 }} onSubmit={submit}>
      <h2>Your name</h2>
      <p className="muted small">Every annotation you make is saved under this name, so several people can annotate the same fields independently.
        Use the same name each time to continue where you left off.</p>
      <div className="row">
        <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Jane Doe" autoFocus style={{ flex: 1 }} />
        <button className="btn primary" disabled={!name.trim()}>Start annotating</button>
      </div>
      <ErrorNote error={error} />
    </form>
  )
}

export function Annotator() {
  const { session, reload } = useSession()
  if (!session.annotator) return null
  return (
    <span className="row small">
      <span className="muted">Annotating as</span><b>{session.annotator}</b>
      <button className="link-btn" onClick={async () => { await api.del('/api/session/annotator'); reload() }}>change name</button>
    </span>
  )
}
