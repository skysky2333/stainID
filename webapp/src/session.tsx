import { createContext, useContext, useState } from 'react'
import type { Session } from './api'
import { api, chooseServer, cleanServer, configuredServer, remote, savedServer, serverName } from './api'
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
        <ServerSetting />
      </form>
    </main>
  )
}

/** Which stainID server this page uses, folded away: it is almost always the one that served the page. */
export function ServerSetting({ open = false }: { open?: boolean }) {
  return (
    <details className="server-setting" open={open}>
      <summary className="muted small">Server: {new URL(serverName).host}{remote ? '' : ' · connect to another server'}</summary>
      <ServerChoice />
    </details>
  )
}

export function ServerChoice() {
  const [address, setAddress] = useState(savedServer)
  const [error, setError] = useState<string | null>(null)
  const [checking, setChecking] = useState(false)
  const connect = async () => {
    const target = cleanServer(address)
    setChecking(true)
    setError(null)
    try {
      const response = await fetch(`${target}/api/session`)
      if (!response.ok) throw new Error(String(response.status))
      chooseServer(target)
    } catch {
      setError(`No stainID server answered at ${target}. Check the address, and that the server was started with ` +
        `“stainid serve --allow-origin ${location.origin}” so it accepts this page.`)
      setChecking(false)
    }
  }
  return (
    <div className="small" style={{ marginTop: 6 }}>
      <p className="muted" style={{ margin: '4px 0' }}>This page uses the stainID server at <b>{serverName}</b>
        {savedServer ? ' (chosen in this browser)' : configuredServer ? ' (set in config.js)' : ' (the one that served this page)'}.</p>
      <div className="row">
        <input type="text" value={address} placeholder="e.g. http://lab-pc:8765" onChange={(e) => setAddress(e.target.value)} style={{ flex: 1 }}
          onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); if (address.trim()) connect() } }} />
        <button type="button" className="btn small" disabled={!address.trim() || checking} onClick={connect}>{checking ? 'Checking…' : 'Connect'}</button>
      </div>
      {savedServer && <button type="button" className="link-btn" onClick={() => chooseServer('')}>
        Back to {configuredServer || 'the server that served this page'}</button>}
      <p className="muted" style={{ margin: '4px 0' }}>A server on another computer only answers pages it was told about: start it with
        {' '}<code>stainid serve --allow-origin {location.origin}</code>.</p>
      <ErrorNote error={error} />
    </div>
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
