import { useState } from 'react'
import { Link } from 'react-router-dom'
import type { SessionAccess } from '../api'
import { api } from '../api'
import { ErrorNote, Loaded, timeAgo, withError } from '../components'
import { useFetch } from '../hooks'
import { ConfirmDialog, CopyButton } from '../labelling'

/** Who can label a session (Labels → a session → More): its join key, and the people who joined. Annotators add a set
 *  with its key; a new key stops the old one, and the people who joined keep their access. Admins open every set. */
export function SessionAccessCard({ name }: { name: string }) {
  const access = useFetch<SessionAccess>(`/api/labels/sessions/${name}/access`)
  const [error, setError] = useState<string | null>(null)
  const [confirm, setConfirm] = useState(false)
  const act = async (call: () => Promise<unknown>) => { if (await withError(call, setError)) access.reload() }
  return (
    <div className="session-access">
      <h3 style={{ margin: '12px 0 4px' }}>Who can label</h3>
      <Loaded state={access} label="Loading…">
        {(a) => <>
          <div className="row">
            <span className="secondary small">Key</span>
            <code className={`path-box join-key${a.enabled ? '' : ' muted'}`} style={{ fontSize: 15, letterSpacing: 1 }}>{a.key}</code>
            {a.enabled && a.key && <CopyButton text={a.key} label="Copy" />}
            <button className="btn small" onClick={() => setConfirm(true)} title="A new key: the old one stops working; people who joined keep their access">New key</button>
            <label className="small row" style={{ gap: 4 }}><input type="checkbox" checked={a.enabled}
              onChange={(e) => act(() => api.put(`/api/labels/sessions/${name}/access`, { enabled: e.target.checked }))} />Joining on</label>
          </div>
          <p className="muted small" style={{ margin: '4px 0' }}>{a.enabled
            ? 'Send this key to the people who should label this set: they sign in on the labelling page and add it under “Add a set”.'
            : 'Joining is off: the key adds nobody. The people below keep their access.'}</p>
          {a.members.length ? (
            <table className="compact"><tbody>{a.members.map((m) => (
              <tr key={m.id}><td>{m.name}{m.handle !== m.name && <span className="muted"> (labels as {m.handle})</span>}</td><td className="muted">joined {timeAgo(m.joined)}</td>
                <td><button className="btn small danger-outline" onClick={() => act(() => api.del(`/api/labels/sessions/${name}/access/members/${m.id}`))}>Remove</button></td></tr>
            ))}</tbody></table>
          ) : <p className="muted small" style={{ margin: 0 }}>Nobody has joined yet.</p>}
          {a.project && <p className="small" style={{ margin: '4px 0 0' }}><Link to={`/users?project=${encodeURIComponent(a.project)}&set=${encodeURIComponent(name)}`}>Manage members →</Link>
            <span className="muted"> add or take out anyone, on the Users page</span></p>}
        </>}
      </Loaded>
      <ErrorNote error={error} />
      {confirm && (
        <ConfirmDialog title="Make a new key?" confirm="New key" onClose={() => setConfirm(false)}
          onConfirm={() => withError(async () => { await api.post(`/api/labels/sessions/${name}/access/key`, {}); access.reload() }, setError)}>
          <p>The current key stops working at once. People who already joined keep their access.</p>
        </ConfirmDialog>
      )}
    </div>
  )
}
