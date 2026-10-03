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

export function SignIn() {
  const { session, reload } = useSession()
  const [name, setName] = useState('')
  const [code, setCode] = useState('')
  const [error, setError] = useState<string | null>(null)
  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (await withError(() => api.post('/api/session', { name, code }), setError)) reload()
  }
  return (
    <form className="card" style={{ maxWidth: 420 }} onSubmit={submit}>
      <h2>Sign in to annotate</h2>
      <p className="muted small">Your name is stored with every annotation you make, so several people can annotate the same fields independently.
        Use the same name each time to continue where you left off.</p>
      <div className="grid" style={{ gap: 10 }}>
        <label className="field">Your name<input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Jane Doe" autoFocus /></label>
        {session.code_required && <label className="field">Access code<input type="text" value={code} onChange={(e) => setCode(e.target.value)} placeholder="from the person who invited you" /></label>}
        <button className="btn primary" disabled={!name.trim()}>Sign in</button>
      </div>
      <ErrorNote error={error} />
    </form>
  )
}

/** Who is signed in: in the page header of the full app, in the top bar of the shared server. */
export function SignedIn({ topBar = false }: { topBar?: boolean }) {
  const { session, reload } = useSession()
  if (!session.annotator || topBar !== (session.mode === 'annotate')) return null
  return (
    <span className="row small">
      <span className="muted">Signed in as</span><b>{session.annotator}</b>
      <button className="link-btn" onClick={async () => { await api.del('/api/session'); reload() }}>sign out</button>
    </span>
  )
}
