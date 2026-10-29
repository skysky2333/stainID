import { useState } from 'react'
import { Link } from 'react-router-dom'
import type { MySet } from '../api'
import { api, projectId } from '../api'
import { Empty, ErrorNote, Loaded, PageHead, Progress, withError } from '../components'
import { useFetch } from '../hooks'
import { kindText } from '../labelling'
import { AREAS_TEXT, AREAS_TITLE } from '../labels/markareas'
import { MATTER_TEXT, MATTER_TITLE } from '../labels/matter'
import { ROUND_TEXT, ROUND_TITLE } from '../labels/rounds'
import { ChoosePassword, useSession } from '../session'
import { TutorialCards } from '../labels/tutorial'

/** Everyone's home after signing in (annotators; admins see it first on their Home, the study's status after it): the
 *  person's labelling sets — of the project the page is in (an admin's Home), else of every project on this server —,
 *  "Add a set" with a key, and the tutorials. A set is in someone's sets once they entered its key or an admin put it on
 *  their home page ("Add to my home page"): one collection per person. Closed sets are listed folded under "Closed". */
export default function AnnotateHome() {
  return (
    <div className="annotate-home">
      <PageHead title="My labelling sets" subtitle="You can stop and come back at any time: sign in again to continue where you left off." />
      <LabellingHome />
    </div>
  )
}

/** The labelling part of a home: my sets, "Add a set", the tutorials. */
export function LabellingHome({ admin = false }: { admin?: boolean }) {
  const list = useFetch<MySet[]>(`/api/session/sets${projectId ? `?here=${encodeURIComponent(projectId)}` : ''}`, 15000)
  return (
    <>
      <AddSet onAdded={list.reload} first={!admin && !list.data?.length} />
      <MySets list={list} admin={admin} />
      <h2 className="annotate-older">Tutorials</h2>
      <TutorialCards sets={list.data} />
    </>
  )
}

function MySets({ list, admin }: { list: ReturnType<typeof useFetch<MySet[]>>; admin: boolean }) {
  return (
      <Loaded state={list} label="Loading your sets…">
        {(sets) => sets.length ? (() => {
          const rank = (s: MySet) => (s.round === 'areas' ? 0.5 : s.round === 'matter' ? 0.4 : s.round ?? 0)  // the field tool first, then calls, then bad areas, grey / white matter
          const projects = [...new Set(sets.map((s) => s.project))]
          return projects.map((p) => {
            const all = sets.filter((s) => s.project === p)
            const mine = all.filter((s) => s.open !== false)
            const closed = all.filter((s) => s.open === false)  // closed for everyone: they open read-only
            const rounds = mine.filter((s) => s.round).sort((a, b) => rank(b) - rank(a))
            const older = mine.filter((s) => !s.round)
            return (
              <section key={p}>
                {projects.length > 1 && <h2 className="annotate-older">{all[0].project_name}</h2>}
                {rounds.length > 0 && <div className="grid grid-2">{rounds.map((s) => <SessionCard key={s.name} s={s} />)}</div>}
                {older.length > 0 && <>
                  {rounds.length > 0 && <h2 className="annotate-older">Other kinds of session</h2>}
                  <div className="grid grid-2">{older.map((s) => <SessionCard key={s.name} s={s} />)}</div>
                </>}
                {closed.length > 0 && <details className="closed-sets">
                  <summary><h2 className="annotate-older">Closed <span className="muted small">· {closed.length} set{closed.length === 1 ? '' : 's'}, read-only</span></h2></summary>
                  <div className="grid grid-2">{closed.map((s) => <SessionCard key={s.name} s={s} />)}</div>
                </details>}
              </section>
            )
          })
        })() : admin ? (
          <p className="small muted">No session on your home page yet: add one with “Add to my home page” on the <Link to="/labels">Labels</Link> page or on the session’s page,
            or with its key above.</p>
        ) : (
          <Empty>
            <p style={{ marginTop: 0 }}>No labelling set yet. Enter the key you were given above: the set then appears here.</p>
            <p className="small muted">Meanwhile, the tutorials below show how labelling works.</p>
          </Empty>
        )}
      </Loaded>
  )
}

function AddSet({ onAdded, first }: { onAdded: () => void; first: boolean }) {
  const [key, setKey] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [added, setAdded] = useState<string | null>(null)
  const [open, setOpen] = useState(false)
  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    const made = await withError(() => api.post<MySet>('/api/session/sets', { key }), setError)
    if (made) {
      setAdded(made.title ?? made.name)
      setKey('')
      onAdded()
    }
  }
  if (!first && !open) {
    return <p className="small"><button className="link-btn" onClick={() => setOpen(true)}>+ Add a set with a key</button>{added && <span className="status-finished"> ✓ “{added}” added.</span>}</p>
  }
  return (
    <form className="card add-set" onSubmit={submit} style={{ marginBottom: 16 }}>
      <label className="field" htmlFor="set-key"><b>Add a set</b></label>
      <p className="muted small" style={{ margin: '0 0 6px' }}>Type the key the person running the study gave you (like <code>k7mq2-xr9ta</code>).</p>
      <div className="row">
        <input id="set-key" type="text" value={key} onChange={(e) => setKey(e.target.value)} placeholder="the key" autoComplete="off" spellCheck={false}
          style={{ flex: 1, minWidth: 180, fontFamily: 'var(--mono, monospace)' }} autoFocus={first || open} />
        <button className="btn primary" disabled={!key.trim()}>Add</button>
      </div>
      {added && <p className="status-finished small">✓ “{added}” is now in your sets.</p>}
      <ErrorNote error={error} />
    </form>
  )
}

function SessionCard({ s }: { s: MySet }) {
  const started = s.done_by_me > 0 || s.my_labels > 0
  const progress = s.round ? s.progress : undefined
  const finished = progress ? progress.total > 0 && progress.decided >= progress.total : s.kind !== 'queue' && s.fields > 0 && s.done_by_me >= s.fields
  return (
    <div className="card annotate-session">
      <h2 style={{ margin: 0 }}>{s.title}</h2>
      <p className="muted small" style={{ margin: '4px 0 10px' }}>{s.round === 'areas' ? `Every stain · ${AREAS_TITLE}` : s.round === 'matter' ? `Every stain · ${MATTER_TITLE}`
        : `${s.stain} · ${s.round ? ROUND_TITLE[s.round] : kindText(s.kind, s.kind_title)}`}</p>
      {s.instructions && <p className="small secondary">{s.instructions}</p>}
      {progress ? <>
        <Progress done={progress.decided} total={progress.total} />
        <p className="small secondary">{finished ? 'Everything here has been checked — thank you!'
          : s.round === 'areas' || s.round === 'matter' ? `${progress.decided.toLocaleString()} of ${progress.total.toLocaleString()} cores ${s.round === 'areas' ? 'checked' : 'done'} by everyone`
            : `${progress.decided.toLocaleString()} of ${progress.total.toLocaleString()} fields done by everyone${s.my_labels ? ` · ${s.my_labels.toLocaleString()} answers by you` : ''}`}</p>
        <p className="small muted" style={{ marginTop: 0 }}>{s.round === 'areas' ? AREAS_TEXT : s.round === 'matter' ? MATTER_TEXT : s.round === 1 ? ROUND_TEXT[1] : ROUND_TEXT[2]}</p>
      </> : s.kind !== 'queue' ? <>
        <Progress done={s.done_by_me} total={s.fields} />
        <p className="small secondary">{finished ? 'You have finished every field — thank you!' : `${s.done_by_me} of ${s.fields} fields finished by you`}</p>
      </> : <p className="small secondary">{s.my_labels ? `${s.my_labels.toLocaleString()} objects labelled by you` : 'Pick what each object is with one key press.'}</p>}
      {s.open === false ? <><p className="small muted" style={{ margin: '0 0 6px' }}>Closed: it opens to look at; nothing in it can be changed.</p>
        <a className="btn" href={s.link}>Look (read-only)</a></>
        : !finished && <a className="btn primary big" href={s.link}>{started ? 'Continue' : 'Start'}</a>}
    </div>
  )
}

/** The person's own account: their name and their password. */
export function AccountPage() {
  const { session } = useSession()
  if (!session.account) return null
  return (
    <div className="annotate-home">
      <PageHead title="Your account" subtitle={`Signed in as ${session.account.name}${session.account.past_name ? ` (labels as “${session.account.past_name}”)` : ''}.`} />
      <ChoosePassword title="Change your password" current />
      <SignOutElsewhere />
      <p className="small" style={{ marginTop: 12 }}><Link to="/">← Home</Link></p>
    </div>
  )
}

/** End the person's other sign-ins (other browsers and computers); this one stays. */
function SignOutElsewhere() {
  const { reload } = useSession()
  const [error, setError] = useState<string | null>(null)
  const [done, setDone] = useState(false)
  const run = async () => {
    setDone(false)
    if (await withError(() => api.post('/api/session/sign-out-elsewhere', {}), setError)) { setDone(true); reload() }
  }
  return (
    <div className="card register-card">
      <h2 style={{ margin: 0 }}>Other sign-ins</h2>
      <p className="small secondary" style={{ margin: 0 }}>Signed in on a computer you no longer use? Sign out everywhere else; this browser stays signed in.
        API tokens are not sign-ins: they keep working until revoked.</p>
      <div className="row"><button className="btn" onClick={run}>Sign out of other sessions</button>
        {done && <span className="status-finished small">✓ Every other sign-in has ended.</span>}</div>
      <ErrorNote error={error} />
    </div>
  )
}
