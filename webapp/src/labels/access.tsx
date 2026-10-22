import { useState } from 'react'
import { Link } from 'react-router-dom'
import type { SessionAccess } from '../api'
import { api } from '../api'
import { ErrorNote, Flash, Loaded, timeAgo, withError } from '../components'
import { useFetch, useFlash } from '../hooks'
import { ConfirmDialog, CopyButton } from '../labelling'
import { useSession } from '../session'

/** "Add to my home page" / "Remove from my home page" (admins): an admin's home page lists only the sessions they
 *  added. `pinned`: known already (the Labels page asks once for all its sessions), else asked here. */
export function HomePin({ name, pinned, onChanged }: { name: string; pinned?: boolean; onChanged?: () => void }) {
  const { session } = useSession()
  const own = useFetch<{ sessions: string[] }>(session.admin && pinned === undefined ? '/api/labels/home' : null)
  const [now, setNow] = useState<boolean | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [said, say] = useFlash()
  const known = pinned ?? own.data?.sessions.includes(name)
  if (!session.admin || known === undefined) return null
  const on = now ?? known
  const toggle = async () => {
    setBusy(true)
    setNow(!on)
    const ok = await withError(async () => { await api.put(`/api/labels/sessions/${name}/home`, { on: !on }); return true }, setError)
    setBusy(false)
    if (ok) say(on ? 'Taken off your home page' : 'Added to your home page')
    else setNow(null)
    own.reload()
    onChanged?.()
  }
  return (
    <span className="row small home-pin">
      <button className="btn small" disabled={busy} onClick={() => void toggle()} aria-pressed={on}
        title="Your home page lists only the sessions you add to it">{on ? 'Remove from my home page' : 'Add to my home page'}</button>
      <Flash text={said} />
      <ErrorNote error={error} />
    </span>
  )
}

/** Who can label a session (Labels → a session → More): its join key, and the people who joined. Annotators add a set
 *  with its key; a new key stops the old one, and the people who joined keep their access. Admins open every set. */
export function SessionAccessCard({ name }: { name: string }) {
  const access = useFetch<SessionAccess>(`/api/labels/sessions/${name}/access`)
  const [error, setError] = useState<string | null>(null)
  const [confirm, setConfirm] = useState(false)
  const [busy, setBusy] = useState<string | null>(null)  // the action under way: its control waits, shown at once
  const [joining, setJoining] = useState<boolean | null>(null)
  const [gone, setGone] = useState<Set<string>>(new Set())
  const [said, say] = useFlash()
  const act = async (what: string, call: () => Promise<unknown>, done: string) => {
    setBusy(what)
    const ok = await withError(async () => { await call(); return true }, setError)
    setBusy(null)
    if (ok) say(done)
    access.reload()
    return !!ok
  }
  const setEnabled = async (on: boolean) => {
    setJoining(on)
    await act('joining', () => api.put(`/api/labels/sessions/${name}/access`, { enabled: on }), on ? 'Joining is on' : 'Joining is off')
    setJoining(null)
  }
  const removeMember = async (id: string, who: string) => {
    setGone((g) => new Set(g).add(id))
    if (!await act(`member:${id}`, () => api.del(`/api/labels/sessions/${name}/access/members/${id}`), `${who} taken out of this set`)) {
      setGone((g) => { const next = new Set(g); next.delete(id); return next })
    }
  }
  return (
    <div className="session-access">
      <h3 style={{ margin: '12px 0 4px' }}>Who can label</h3>
      <Loaded state={access} label="Loading…">
        {(found) => { const a = { ...found, enabled: joining ?? found.enabled, members: found.members.filter((m) => !gone.has(m.id)) }; return <>
          <div className="row">
            <span className="secondary small">Key</span>
            <code className={`path-box join-key${a.enabled ? '' : ' muted'}`} style={{ fontSize: 15, letterSpacing: 1 }}>{a.key}</code>
            {a.enabled && a.key && <CopyButton text={a.key} label="Copy" />}
            <button className="btn small" onClick={() => setConfirm(true)} title="A new key: the old one stops working; people who joined keep their access">New key</button>
            <label className="small row" style={{ gap: 4 }}><input type="checkbox" checked={a.enabled} disabled={busy === 'joining'}
              onChange={(e) => void setEnabled(e.target.checked)} />Joining on</label>
            <Flash text={said} />
          </div>
          <p className="muted small" style={{ margin: '4px 0' }}>{a.enabled
            ? 'Send this key to the people who should label this set: they sign in on the labelling page and add it under “Add a set”.'
            : 'Joining is off: the key adds nobody. The people below keep their access.'}</p>
          {a.members.length ? (
            <table className="compact"><tbody>{a.members.map((m) => (
              <tr key={m.id}><td>{m.name}{m.handle !== m.name && <span className="muted"> (labels as {m.handle})</span>}</td><td className="muted">joined {timeAgo(m.joined)}</td>
                <td><button className="btn small danger-outline" disabled={busy === `member:${m.id}`} onClick={() => void removeMember(m.id, m.name)}>Remove</button></td></tr>
            ))}</tbody></table>
          ) : <p className="muted small" style={{ margin: 0 }}>Nobody has joined yet.</p>}
          {a.project && <p className="small" style={{ margin: '4px 0 0' }}><Link to={`/users?project=${encodeURIComponent(a.project)}&set=${encodeURIComponent(name)}`}>Manage members →</Link>
            <span className="muted"> add or take out anyone, on the Users page</span></p>}
        </> }}
      </Loaded>
      <ErrorNote error={error} />
      {confirm && (
        <ConfirmDialog title="Make a new key?" confirm="New key" onClose={() => setConfirm(false)}
          onConfirm={() => act('key', () => api.post(`/api/labels/sessions/${name}/access/key`, {}), 'New key made: the old one no longer works')}>
          <p>The current key stops working at once. People who already joined keep their access.</p>
        </ConfirmDialog>
      )}
    </div>
  )
}
