/* Settings: the signed-in person's own API tokens (people are managed on the Users page, pages/Users.tsx). */
import { useState } from 'react'
import type { ApiToken } from './api'
import { api } from './api'
import { ErrorNote, Loaded, timeAgo, withError } from './components'
import { useFetch } from './hooks'
import { CopyButton } from './labelling'

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
