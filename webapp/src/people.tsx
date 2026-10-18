import { useState } from 'react'
import type { Account, ApiToken, People } from './api'
import { api } from './api'
import { ErrorNote, Loaded, timeAgo, withError } from './components'
import { useFetch } from './hooks'
import { ConfirmDialog, CopyButton } from './labelling'
import { MIN_PASSWORD, useSession } from './session'

/** Settings → People: everyone with an account on this server. Admins see every page; annotators only the labelling
 *  sets they joined. There is always at least one admin. */
export function PeopleSettings() {
  const { session } = useSession()
  const people = useFetch<People>('/api/session/people')
  const [error, setError] = useState<string | null>(null)
  const [temporary, setTemporary] = useState<{ name: string; password: string } | null>(null)
  const [removing, setRemoving] = useState<Account | null>(null)
  const [adding, setAdding] = useState(false)
  const change = async (a: Account, body: { role?: string; past_name?: string }) => {
    if (await withError(() => api.put(`/api/session/people/${a.id}`, body), setError)) people.reload()
  }
  const reset = async (a: Account) => {
    const made = await withError(() => api.post<{ password: string }>(`/api/session/people/${a.id}/reset`, {}), setError)
    if (made) setTemporary({ name: a.name, password: made.password })
  }
  return (
    <section className="card" id="people">
      <div className="row"><h2 style={{ margin: 0 }}>People</h2><span className="spacer" />
        <button className="btn small" onClick={() => setAdding(!adding)}>{adding ? 'Cancel' : 'Add a person'}</button></div>
      <p className="muted small">Everyone signs in on the same page. <b>Admins</b> see all of stainID and every labelling set; <b>annotators</b> only the
        sets they added with a key (Labels → a session → More → Who can label). New people make their own annotator account.</p>
      {adding && <AddPerson onAdded={() => { setAdding(false); people.reload() }} />}
      <Loaded state={people} label="Loading people…">
        {(data) => {
          const admins = data.people.filter((p) => p.role === 'admin').length
          const free = data.past_names.filter((n) => !n.account).map((n) => n.name)
          return (
            <div className="table-scroll">
              <table className="compact">
                <thead><tr><th>Name</th><th>Role</th><th>Past labels as</th><th>Since</th><th /></tr></thead>
                <tbody>{data.people.map((p) => {
                  const last = p.role === 'admin' && admins <= 1
                  return (
                    <tr key={p.id}>
                      <td><b>{p.name}</b>{p.id === session.account?.id && <span className="muted"> (you)</span>}{p.must_change && <span className="badge">password reset</span>}</td>
                      <td><select value={p.role} disabled={last} title={last ? 'The only admin: make someone else an admin first' : ''}
                        onChange={(e) => change(p, { role: e.target.value })}>
                        <option value="annotator">annotator</option><option value="admin">admin</option></select></td>
                      <td><select value={p.past_name} onChange={(e) => change(p, { past_name: e.target.value })}
                        title="Labels made before accounts existed, under a name: link the account that is this person">
                        <option value="">—</option>
                        {[...(p.past_name ? [p.past_name.toLowerCase()] : []), ...free].map((n) => <option key={n} value={n}>{n}</option>)}
                      </select></td>
                      <td className="muted">{timeAgo(p.created)}</td>
                      <td className="row" style={{ gap: 6 }}>
                        <button className="btn small" onClick={() => reset(p)}>Reset password</button>
                        <button className="btn small danger-outline" disabled={last} onClick={() => setRemoving(p)}>Remove</button>
                      </td>
                    </tr>
                  )
                })}</tbody>
              </table>
              {free.length > 0 && <p className="muted small">Names in past labels that no account is yet: {free.join(', ')}. Nobody can sign up under
                them; link an account to one to give it that history.</p>}
            </div>
          )
        }}
      </Loaded>
      {temporary && (
        <div className="info-note small">One-time password for <b>{temporary.name}</b>: <code>{temporary.password}</code> <CopyButton text={temporary.password} label="Copy" />
          {' '}Give it to them; they choose their own when they sign in with it. <button className="link-btn" onClick={() => setTemporary(null)}>Hide</button></div>
      )}
      <ErrorNote error={error} />
      {removing && (
        <ConfirmDialog title={`Remove ${removing.name}?`} confirm="Remove" danger onClose={() => setRemoving(null)}
          onConfirm={() => withError(async () => { await api.del(`/api/session/people/${removing.id}`); people.reload() }, setError)}>
          <p>They can no longer sign in, and their API tokens stop working. <b>Their labels are kept</b> (under their name).</p>
        </ConfirmDialog>
      )}
    </section>
  )
}

function AddPerson({ onAdded }: { onAdded: () => void }) {
  const [form, setForm] = useState({ name: '', password: '', role: 'annotator' })
  const [error, setError] = useState<string | null>(null)
  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (await withError(() => api.post('/api/session/people', form), setError)) onAdded()
  }
  return (
    <form className="row" onSubmit={submit} style={{ margin: '8px 0', flexWrap: 'wrap' }}>
      <input type="text" placeholder="Name" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
      <input type="text" placeholder={`First password (${MIN_PASSWORD}+ characters)`} value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} />
      <select value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value })}><option value="annotator">annotator</option><option value="admin">admin</option></select>
      <button className="btn small primary" disabled={!form.name.trim() || form.password.length < MIN_PASSWORD}>Add</button>
      <ErrorNote error={error} />
    </form>
  )
}

/** API tokens of the signed-in account, for scripts and agents (`Authorization: Bearer <token>`). */
export function ApiTokens() {
  const tokens = useFetch<ApiToken[]>('/api/session/tokens')
  const [label, setLabel] = useState('')
  const [made, setMade] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const make = async () => {
    const found = await withError(() => api.post<{ token: string }>('/api/session/tokens', { label }), setError)
    if (found) { setMade(found.token); setLabel(''); tokens.reload() }
  }
  return (
    <div>
      <p className="muted small" style={{ marginTop: 0 }}>A token lets a script or agent use this server as you: send it as <code>Authorization: Bearer &lt;token&gt;</code>
        {' '}(or <code>?access_token=</code> on downloads). Also on the command line: <code>stainid accounts token NAME</code>.</p>
      <div className="row">
        <input type="text" placeholder="What it is for, e.g. nightly export" value={label} onChange={(e) => setLabel(e.target.value)} style={{ flex: 1 }} />
        <button className="btn small" onClick={make}>New token</button>
      </div>
      {made && <div className="info-note small">Your new token (shown only now): <code>{made}</code> <CopyButton text={made} label="Copy" /></div>}
      <Loaded state={tokens} label="Loading tokens…">
        {(list) => list.length ? (
          <table className="compact"><tbody>{list.map((t) => (
            <tr key={t.id}><td>{t.label || <span className="muted">no label</span>}</td><td className="muted">made {timeAgo(t.created)}</td>
              <td className="muted">{t.last_used ? `used ${timeAgo(t.last_used)}` : 'never used'}</td>
              <td><button className="btn small danger-outline" onClick={() => withError(async () => { await api.del(`/api/session/tokens/${t.id}`); tokens.reload() }, setError)}>Revoke</button></td></tr>
          ))}</tbody></table>
        ) : <p className="muted small">No tokens yet.</p>}
      </Loaded>
      <ErrorNote error={error} />
    </div>
  )
}
