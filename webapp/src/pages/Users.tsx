/* The admin's Users page (/users): everyone with an account on this server, of every project (`/api/admin/users`,
   api/routes/users.py). Add people (a one-time password, shown once), reset passwords (the same), change roles,
   disable or enable, remove (typed confirmation; labels stay), add to or take out of any labelling set, API tokens,
   the past name an account labels under, and the audit log of every admin action. `?project=&set=` filters to the
   members of one set (Labels → a session → Who can label → Manage members). */
import { useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import type { Account, ApiToken } from '../api'
import { api } from '../api'
import { ErrorNote, Flash, Loaded, Modal, PageHead, timeAgo, withError } from '../components'
import { useFetch, useFlash } from '../hooks'
import { CopyButton } from '../labelling'

interface UserSet { project: string; project_name: string; session: string; title: string; joined?: number; exists?: boolean; stain?: string }
interface UserRow extends Account { disabled: boolean; last_sign_in: number | null; sets: UserSet[]; labels: number; tokens: number }
interface AuditEntry { id: number; time: number; actor_name: string; action: string; target_name: string; detail: string }
interface UsersData { users: UserRow[]; sets: UserSet[]; audit: AuditEntry[]; me: string; past_names: { name: string; account: string | null }[] }
type Sort = 'name' | 'role' | 'created' | 'last_sign_in' | 'labels'

const setKey = (s: { project: string; session: string }) => `${s.project}/${s.session}`
const setName = (s: UserSet, many: boolean) => (many ? `${s.project_name} · ${s.title}` : s.title)

/** A one-time secret, shown only now. */
function Secret({ what, who, value, onHide }: { what: string; who: string; value: string; onHide: () => void }) {
  return (
    <div className="info-note small users-secret" role="status">
      {what} for <b>{who}</b>: <code>{value}</code> <CopyButton text={value} label="Copy" />
      {' '}Shown only now: it is kept hashed and cannot be seen again. <button className="link-btn" onClick={onHide}>Hide</button>
    </div>
  )
}

export default function Users() {
  const data = useFetch<UsersData>('/api/admin/users')
  const [params, setParams] = useSearchParams()
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState<Sort>('name')
  const [error, setError] = useState<string | null>(null)
  const [secret, setSecret] = useState<{ what: string; who: string; value: string } | null>(null)
  const [adding, setAdding] = useState(false)
  const [open, setOpen] = useState<string | null>(null)
  const [removing, setRemoving] = useState<UserRow | null>(null)
  const filter = params.get('project') && params.get('set') ? `${params.get('project')}/${params.get('set')}` : ''
  const act = async (call: () => Promise<unknown>) => {  // the list again right after the answer (also when the answer is empty)
    let ok = false
    const made = await withError(async () => { const r = await call(); ok = true; return r }, setError)
    if (ok) data.reload()
    return ok ? made ?? true : undefined
  }

  return (
    <>
      <PageHead title="Users" subtitle="Everyone with an account on this server: roles, passwords, labelling sets and API tokens.">
        <button className="btn primary" onClick={() => setAdding(!adding)}>{adding ? 'Cancel' : 'Add a user'}</button>
      </PageHead>
      <ErrorNote error={error} />
      {secret && <Secret {...secret} onHide={() => setSecret(null)} />}
      <Loaded state={data} label="Loading the users…">
        {(d) => {
          const many = new Set(d.sets.map((s) => s.project)).size > 1
          const filtered = d.sets.find((s) => setKey(s) === filter)
          const words = query.toLowerCase().split(/\s+/).filter(Boolean)
          const rows = d.users
            .filter((u) => !filter || u.sets.some((s) => setKey(s) === filter))
            .filter((u) => words.every((w) => `${u.name} ${u.past_name} ${u.role} ${u.sets.map((s) => s.title).join(' ')}`.toLowerCase().includes(w)))
            .sort((a, b) => sort === 'name' ? a.name.localeCompare(b.name) : sort === 'role' ? a.role.localeCompare(b.role) || a.name.localeCompare(b.name)
              : sort === 'labels' ? b.labels - a.labels : ((b[sort] ?? 0) as number) - ((a[sort] ?? 0) as number))
          const admins = d.users.filter((u) => u.role === 'admin' && !u.disabled).length
          const free = d.past_names.filter((n) => !n.account).map((n) => n.name)
          return <>
            {adding && <AddUser sets={d.sets} many={many} preset={filter} onAdded={(name, password) => { setAdding(false); setSecret({ what: 'One-time password', who: name, value: password }); data.reload() }} />}
            <section className="card">
              <div className="row users-tools" style={{ flexWrap: 'wrap' }}>
                <input type="search" placeholder="Search names, roles, sets" value={query} onChange={(e) => setQuery(e.target.value)} aria-label="Search users" style={{ flex: '1 1 220px' }} />
                <label className="small row" style={{ gap: 4 }}>Sort
                  <select value={sort} onChange={(e) => setSort(e.target.value as Sort)} aria-label="Sort by">
                    <option value="name">name</option><option value="role">role</option><option value="created">newest</option>
                    <option value="last_sign_in">last sign-in</option><option value="labels">labels made</option>
                  </select></label>
                {filter && <span className="badge">members of {filtered ? setName(filtered, many) : filter}
                  <button className="link-btn small" onClick={() => setParams({})} aria-label="Show everyone"> ×</button></span>}
                <span className="spacer" /><span className="muted small">{rows.length} of {d.users.length}</span>
              </div>
              <div className="table-scroll"><table className="compact users-table">
                <thead><tr><th>Name</th><th>Role</th><th>Created</th><th>Last sign-in</th><th>Sets</th><th>Labels</th><th /></tr></thead>
                <tbody>{rows.map((u) => {
                  const me = u.id === d.me
                  const last = u.role === 'admin' && !u.disabled && admins <= 1
                  const guard = me ? 'Not your own account: ask another admin' : last ? 'The only admin: make someone else an admin first' : ''
                  return [
                    <tr key={u.id} className={u.disabled ? 'muted users-disabled' : ''} data-user={u.name}>
                      <td><b>{u.name}</b>{me && <span className="muted"> (you)</span>}{u.past_name && <span className="muted small"> labels as {u.past_name}</span>}
                        {u.disabled && <span className="badge status-failed"> disabled</span>}{u.must_change && <span className="badge"> password reset</span>}</td>
                      <td><select value={u.role} disabled={(me || last) && u.role === 'admin'} title={u.role === 'admin' ? guard : ''} aria-label={`Role of ${u.name}`}
                        onChange={(e) => act(() => api.put(`/api/admin/users/${u.id}/role`, { role: e.target.value }))}>
                        <option value="annotator">annotator</option><option value="admin">admin</option></select></td>
                      <td className="muted small">{timeAgo(u.created)}</td>
                      <td className="muted small">{u.last_sign_in ? timeAgo(u.last_sign_in) : 'never'}</td>
                      <td className="small">{u.sets.length ? u.sets.map((s) => <span key={setKey(s)} className="users-set">{setName(s, many)}{!s.exists && <span className="muted"> (gone)</span>}</span>) : <span className="muted">none</span>}</td>
                      <td>{u.labels.toLocaleString()}</td>
                      <td className="row" style={{ gap: 6, flexWrap: 'nowrap' }}>
                        <button className="btn small" onClick={() => setOpen(open === u.id ? null : u.id)} aria-expanded={open === u.id}>{open === u.id ? 'Close' : 'Manage'}</button>
                      </td>
                    </tr>,
                    open === u.id && <tr key={`${u.id}-more`} className="users-more"><td colSpan={7}>
                      <Manage user={u} me={me} guard={guard} sets={d.sets} many={many} free={free} act={act}
                        onSecret={(what, value) => setSecret({ what, who: u.name, value })} onRemove={() => setRemoving(u)} />
                    </td></tr>,
                  ]
                })}{!rows.length && <tr><td colSpan={7} className="muted">Nobody matches.</td></tr>}</tbody>
              </table></div>
            </section>
            <section className="card">
              <h2 style={{ marginTop: 0 }}>Audit log</h2>
              <p className="muted small">Every admin action on accounts and set memberships: who did what, to whom, when (the latest {d.audit.length}).</p>
              {d.audit.length ? <div className="table-scroll"><table className="compact small audit-table">
                <thead><tr><th>When</th><th>Who</th><th>What</th><th>Whom</th><th>Detail</th></tr></thead>
                <tbody>{d.audit.map((e) => <tr key={e.id}><td className="muted" title={new Date(e.time * 1000).toLocaleString()}>{timeAgo(e.time)}</td>
                  <td>{e.actor_name}</td><td>{e.action}</td><td>{e.target_name || '—'}</td><td className="muted">{e.detail}</td></tr>)}</tbody>
              </table></div> : <p className="muted small">Nothing yet.</p>}
            </section>
          </>
        }}
      </Loaded>
      {removing && <RemoveUser user={removing} onClose={() => setRemoving(null)} onRemoved={() => { setRemoving(null); setOpen(null); data.reload() }} />}
    </>
  )
}

function AddUser({ sets, many, preset, onAdded }: { sets: UserSet[]; many: boolean; preset: string; onAdded: (name: string, password: string) => void }) {
  const [form, setForm] = useState({ name: '', role: 'annotator', set: preset })
  const [error, setError] = useState<string | null>(null)
  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    const chosen = sets.find((s) => setKey(s) === form.set)
    const made = await withError(() => api.post<{ user: Account; password: string }>('/api/admin/users',
      { name: form.name, role: form.role, project: chosen?.project ?? '', session: chosen?.session ?? '' }), setError)
    if (made) onAdded(made.user.name, made.password)
  }
  return (
    <form className="card row users-add" onSubmit={submit} style={{ flexWrap: 'wrap' }}>
      <input type="text" placeholder="Name" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} aria-label="Name" autoFocus />
      <select value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value })} aria-label="Role"><option value="annotator">annotator</option><option value="admin">admin</option></select>
      <select value={form.set} onChange={(e) => setForm({ ...form, set: e.target.value })} aria-label="Labelling set">
        <option value="">no labelling set yet</option>
        {sets.map((s) => <option key={setKey(s)} value={setKey(s)}>{setName(s, many)}</option>)}
      </select>
      <button className="btn primary" disabled={!form.name.trim()}>Add</button>
      <span className="muted small">They get a one-time password (shown once) and choose their own when they first sign in.</span>
      <ErrorNote error={error} />
    </form>
  )
}

function Manage({ user, me, guard, sets, many, free, act, onSecret, onRemove }: {
  user: UserRow; me: boolean; guard: string; sets: UserSet[]; many: boolean; free: string[]
  act: (call: () => Promise<unknown>) => Promise<unknown>; onSecret: (what: string, value: string) => void; onRemove: () => void
}) {
  const tokens = useFetch<ApiToken[]>(`/api/admin/users/${user.id}/tokens`)
  const [adding, setAdding] = useState('')
  const [label, setLabel] = useState('')
  const joined = useMemo(() => new Set(user.sets.map(setKey)), [user.sets])
  const [busy, setBusy] = useState<string | null>(null)  // the membership being added or taken out
  const [said, say] = useFlash()
  const member = async (key: string, call: () => Promise<unknown>, done: string) => {
    setBusy(key)
    if (await act(call) !== undefined) say(done)
    setBusy(null)
  }
  const reset = async () => {
    const made = await act(() => api.post<{ password: string }>(`/api/admin/users/${user.id}/reset`, {})) as { password: string } | undefined
    if (made) onSecret('One-time password', made.password)
  }
  const token = async () => {
    const made = await act(() => api.post<{ token: string }>(`/api/admin/users/${user.id}/tokens`, { label })) as { token: string } | undefined
    if (made) { onSecret('API token', made.token); setLabel(''); tokens.reload() }
  }
  return (
    <div className="users-manage">
      <div className="row" style={{ flexWrap: 'wrap', gap: 6 }}>
        <button className="btn small" onClick={reset} disabled={me} title={me ? 'Change your own on your Account page' : 'A one-time password: they choose their own when signing in with it'}>Reset password</button>
        <button className="btn small" disabled={!user.disabled && !!guard} title={user.disabled ? '' : guard || 'They cannot sign in, and their sessions and tokens stop working'}
          onClick={() => act(() => api.put(`/api/admin/users/${user.id}/disabled`, { disabled: !user.disabled }))}>{user.disabled ? 'Enable' : 'Disable'}</button>
        <button className="btn small danger-outline" disabled={!!guard} title={guard || 'Delete the account; their labels are kept'} onClick={onRemove}>Remove…</button>
        <label className="small row" style={{ gap: 4 }} title="Labels made before accounts existed, under a name: link the account that is this person">Labels as
          <select value={user.past_name} onChange={(e) => act(() => api.put(`/api/admin/users/${user.id}/past-name`, { past_name: e.target.value }))}>
            <option value="">— its own name</option>
            {[...(user.past_name ? [user.past_name.toLowerCase()] : []), ...free].map((n) => <option key={n} value={n}>{n}</option>)}
          </select></label>
      </div>
      <h4>Labelling sets</h4>
      {user.sets.length ? <ul className="users-sets small">{user.sets.map((s) => (
        <li key={setKey(s)}>{setName(s, many)} <span className="muted">· joined {s.joined ? timeAgo(s.joined) : ''}</span>{' '}
          <button className="link-btn small" disabled={busy === setKey(s)} onClick={() => void member(setKey(s),
            () => api.del(`/api/admin/users/${user.id}/sets/${encodeURIComponent(s.project)}/${encodeURIComponent(s.session)}`), `Taken out of ${s.title}`)}>
            {busy === setKey(s) ? 'removing…' : 'remove'}</button></li>
      ))}</ul> : <p className="muted small">In no labelling set.</p>}
      <div className="row small">
        <select value={adding} onChange={(e) => setAdding(e.target.value)} aria-label={`Add ${user.name} to a set`}>
          <option value="">Add to a set…</option>
          {sets.filter((s) => !joined.has(setKey(s))).map((s) => <option key={setKey(s)} value={setKey(s)}>{setName(s, many)}</option>)}
        </select>
        <button className="btn small" disabled={!adding || busy === 'add'} onClick={async () => {
          const s = sets.find((x) => setKey(x) === adding)
          if (s) { await member('add', () => api.post(`/api/admin/users/${user.id}/sets`, { project: s.project, session: s.session }), `Added to ${s.title}`); setAdding('') }
        }}>{busy === 'add' ? 'Adding…' : 'Add'}</button>
        <Flash text={said} />
      </div>
      <h4>API tokens</h4>
      <Loaded state={tokens} label="Loading tokens…">
        {(list) => list.length ? <table className="compact small"><tbody>{list.map((t) => (
          <tr key={t.id}><td>{t.label || <span className="muted">no label</span>}</td><td className="muted">made {timeAgo(t.created)}</td>
            <td className="muted">{t.last_used ? `used ${timeAgo(t.last_used)}` : 'never used'}</td>
            <td><button className="btn small danger-outline" onClick={async () => { await act(() => api.del(`/api/admin/users/${user.id}/tokens/${t.id}`)); tokens.reload() }}>Revoke</button></td></tr>
        ))}</tbody></table> : <p className="muted small">No tokens.</p>}
      </Loaded>
      <div className="row small">
        <input type="text" placeholder="What it is for" value={label} onChange={(e) => setLabel(e.target.value)} aria-label="Token label" />
        <button className="btn small" onClick={token}>New token</button>
      </div>
    </div>
  )
}

function RemoveUser({ user, onClose, onRemoved }: { user: UserRow; onClose: () => void; onRemoved: () => void }) {
  const [typed, setTyped] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const ok = typed.trim().toLowerCase() === user.name.toLowerCase()
  const remove = async () => {
    setBusy(true)
    const done = await withError(() => api.del(`/api/admin/users/${user.id}?confirm=${encodeURIComponent(typed)}`), setError)
    setBusy(false)
    if (done !== undefined) onRemoved()
  }
  return (
    <Modal title={`Remove ${user.name}?`} onClose={onClose}>
      <p className="secondary">The account is deleted: they can no longer sign in, and their sets and API tokens go with it.
        {' '}<b>Their {user.labels.toLocaleString()} labels are kept</b>, with their name. To only stop them for now, disable the account instead.</p>
      <label className="field">Type <b>{user.name}</b> to confirm<input type="text" value={typed} onChange={(e) => setTyped(e.target.value)} autoFocus aria-label="Type the name to confirm" /></label>
      <ErrorNote error={error} />
      <div className="row"><span className="spacer" /><button className="btn" onClick={onClose}>Cancel</button>
        <button className="btn danger" disabled={!ok || busy} onClick={remove}>{busy ? 'Removing…' : 'Remove'}</button></div>
    </Modal>
  )
}

/** Settings → People: where people are managed now. */
export function UsersLink() {
  return (
    <section className="card" id="people">
      <h2 style={{ margin: 0 }}>People</h2>
      <p className="muted small">Accounts, roles, passwords, labelling sets and API tokens of everyone on this server are on the <Link to="/users">Users</Link> page.</p>
    </section>
  )
}
