import { createContext, useContext, useState } from 'react'
import type { Session } from './api'
import { api, appPath, chooseServer, cleanServer, configuredServer, remote, savedServer, serverName } from './api'
import { ErrorNote, withError } from './components'

export const SessionContext = createContext<{ session: Session; reload: () => void } | null>(null)

export function useSession() {
  const value = useContext(SessionContext)
  if (!value) throw new Error('useSession must be used inside SessionContext')
  return value
}

export const MIN_PASSWORD = 8

/** The one sign-in of stainID: admins and annotators alike, by name and password; new people make an account here
 *  (an annotator account; admins are made in Settings → People). */
export function SignInForm({ intro, start = 'in' }: { intro?: React.ReactNode; start?: 'in' | 'up' }) {
  const { reload } = useSession()
  const [mode, setMode] = useState<'in' | 'up'>(start)
  const [name, setName] = useState('')
  const [password, setPassword] = useState('')
  const [repeat, setRepeat] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const up = mode === 'up'
  const ready = name.trim() && (up ? password.length >= MIN_PASSWORD && password === repeat : password)
  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    setBusy(true)
    const done = await withError(() => api.post(up ? '/api/session/register' : '/api/session/login', { name, password }), setError)
    setBusy(false)
    if (done) reload()
  }
  return (
    <form className="card register-card sign-in-card" onSubmit={submit}>
      <div className="segmented" role="tablist">
        <button type="button" role="tab" aria-selected={!up} className={!up ? 'active' : ''} onClick={() => { setMode('in'); setError(null) }}>Sign in</button>
        <button type="button" role="tab" aria-selected={up} className={up ? 'active' : ''} onClick={() => { setMode('up'); setError(null) }}>New here? Make an account</button>
      </div>
      {intro}
      <label className="field">Your name<input type="text" value={name} onChange={(e) => setName(e.target.value)} autoComplete="username"
        placeholder={up ? 'e.g. Jane Doe — others see this name' : ''} autoFocus /></label>
      <label className="field">{up ? `Choose a password (at least ${MIN_PASSWORD} characters)` : 'Password'}
        <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete={up ? 'new-password' : 'current-password'} /></label>
      {up && <label className="field">The password again<input type="password" value={repeat} onChange={(e) => setRepeat(e.target.value)} autoComplete="new-password" /></label>}
      {up && repeat !== '' && repeat !== password && <span className="level-warning small">The two passwords differ.</span>}
      <button className="btn primary" disabled={!ready || busy}>{busy ? '…' : up ? 'Make my account' : 'Sign in'}</button>
      {up ? <p className="muted small" style={{ margin: 0 }}>Your labels are kept under your account. Use the same name and password next time, on any computer.</p>
        : <p className="muted small" style={{ margin: 0 }}>Forgot your password? Ask the person who runs the study to reset it.</p>}
      <ErrorNote error={error} />
    </form>
  )
}

/** The sign-in page (everything but the tutorial needs an account). */
export function SignIn() {
  const { session } = useSession()
  return (
    <main className="main" style={{ display: 'grid', placeItems: 'center', minHeight: '80vh' }}>
      <div style={{ width: 420, maxWidth: '100%' }}>
        <div className="brand" style={{ marginBottom: 12 }}><img src="/favicon.svg" alt="" />stainID</div>
        <SignInForm />
        {session.default_password && <p className="muted small">First time? Sign in as <b>admin</b> with the password <b>admin</b>, then change it in Settings.</p>}
        <p className="muted small">Here to label images? Sign in or make an account above, then add the set you were given a key for.
          {' '}You can also <a href={appPath('/annotate/tutorial')}>try the tutorial</a> without an account.</p>
        <ServerSetting />
      </div>
    </main>
  )
}

/** The old name for the sign-in form, used by the annotation pages. */
export function Register() {
  return <SignInForm start="up" />
}

/** After an admin reset someone's password: they choose a new one before anything else. */
export function ChoosePassword({ title = 'Choose a new password', current = false, onDone }: { title?: string; current?: boolean; onDone?: () => void }) {
  const { reload } = useSession()
  const [form, setForm] = useState({ current: '', new: '', repeat: '' })
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState(false)
  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSaved(false)
    if (await withError(() => api.put('/api/session/password', { current: form.current, new: form.new }), setError)) {
      setForm({ current: '', new: '', repeat: '' })
      setSaved(true)
      reload()
      onDone?.()
    }
  }
  return (
    <form className="card register-card" onSubmit={submit}>
      <h2 style={{ margin: 0 }}>{title}</h2>
      {current && <label className="field">Current password<input type="password" autoComplete="current-password" value={form.current} onChange={(e) => setForm({ ...form, current: e.target.value })} /></label>}
      <label className="field">New password (at least {MIN_PASSWORD} characters)<input type="password" autoComplete="new-password" value={form.new} onChange={(e) => setForm({ ...form, new: e.target.value })} autoFocus={!current} /></label>
      <label className="field">New password again<input type="password" autoComplete="new-password" value={form.repeat} onChange={(e) => setForm({ ...form, repeat: e.target.value })} /></label>
      {form.repeat !== '' && form.repeat !== form.new && <span className="level-warning small">The two new passwords differ.</span>}
      <div className="row">
        <button className="btn primary" disabled={form.new.length < MIN_PASSWORD || form.new !== form.repeat || (current && !form.current)}>Change password</button>
        {saved && <span className="status-finished small">✓ Password changed.</span>}
      </div>
      <ErrorNote error={error} />
    </form>
  )
}

export async function signOut(reload: () => void) {
  await api.post('/api/session/logout', {})
  reload()
}

/** Who is signed in, in the header of the annotation pages. */
export function Annotator() {
  const { session, reload } = useSession()
  if (!session.account) return null
  return (
    <span className="row small">
      <span className="muted">Signed in as</span><b>{session.account.name}</b>
      <a className="link-btn" href={appPath('/annotate/account')}>Account</a>
      <button className="link-btn" onClick={() => signOut(reload)}>Sign out</button>
    </span>
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
