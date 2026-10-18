import { useState } from 'react'
import { Link } from 'react-router-dom'
import type { MySet } from '../api'
import { api, projectId } from '../api'
import { Empty, ErrorNote, Loaded, PageHead, Progress, withError } from '../components'
import { useFetch } from '../hooks'
import { KIND_TEXT } from '../labelling'
import { AREAS_TEXT, AREAS_TITLE } from '../labels/markareas'
import { ROUND_TEXT, ROUND_TITLE } from '../labels/rounds'
import { ChoosePassword, SignInForm, useSession } from '../session'
import { useFirstTutorial } from '../labels/tutorial'

/** The person's labelling sets ("my sets", of every project on this server), and "Add a set" with a key. */
export default function AnnotateHome() {
  const { session } = useSession()
  const list = useFetch<MySet[]>(session.account ? `/api/session/sets${projectId ? `?here=${encodeURIComponent(projectId)}` : ''}` : null, 15000)
  useFirstTutorial()
  if (!session.account) return <Welcome />
  return (
    <div className="annotate-home">
      <PageHead title="My labelling sets" subtitle="You can stop and come back at any time: sign in again to continue where you left off." />
      <AddSet onAdded={list.reload} first={!list.data?.length} />
      <Loaded state={list} label="Loading your sets…">
        {(sets) => sets.length ? (() => {
          const rank = (s: MySet) => (s.round === 'areas' ? 0.5 : s.round ?? 0)  // the field tool first, then calls, then bad areas
          const projects = [...new Set(sets.map((s) => s.project))]
          return projects.map((p) => {
            const mine = sets.filter((s) => s.project === p)
            const rounds = mine.filter((s) => s.round).sort((a, b) => rank(b) - rank(a))
            const older = mine.filter((s) => !s.round)
            return (
              <section key={p}>
                {projects.length > 1 && <h2 className="annotate-older">{mine[0].project_name}</h2>}
                {rounds.length > 0 && <div className="grid grid-2">{rounds.map((s) => <SessionCard key={s.name} s={s} />)}</div>}
                {older.length > 0 && <>
                  {rounds.length > 0 && <h2 className="annotate-older">Other kinds of session</h2>}
                  <div className="grid grid-2">{older.map((s) => <SessionCard key={s.name} s={s} />)}</div>
                </>}
              </section>
            )
          })
        })() : (
          <Empty>
            <p style={{ marginTop: 0 }}>No labelling set yet. Enter the key you were given above: the set then appears here.</p>
            <p className="small muted">Meanwhile, the <Link to="/annotate/tutorial">tutorial</Link> shows how labelling works.</p>
          </Empty>
        )}
      </Loaded>
    </div>
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
      <p className="muted small" style={{ margin: '4px 0 10px' }}>{s.round === 'areas' ? `Every stain · ${AREAS_TITLE}`
        : `${s.stain} · ${s.round ? `Round ${s.round} · ${ROUND_TITLE[s.round]}` : KIND_TEXT[s.kind as keyof typeof KIND_TEXT]}`}</p>
      {s.instructions && <p className="small secondary">{s.instructions}</p>}
      {progress ? <>
        <Progress done={progress.decided} total={progress.total} />
        <p className="small secondary">{finished ? 'Everything here has been checked — thank you!'
          : s.round === 'areas' ? `${progress.decided.toLocaleString()} of ${progress.total.toLocaleString()} cores checked by everyone`
            : `${progress.decided.toLocaleString()} of ${progress.total.toLocaleString()} fields done by everyone${s.my_labels ? ` · ${s.my_labels.toLocaleString()} answers by you` : ''}`}</p>
        <p className="small muted" style={{ marginTop: 0 }}>{s.round === 'areas' ? AREAS_TEXT : s.round === 1 ? ROUND_TEXT[1] : ROUND_TEXT[2]}</p>
      </> : s.kind !== 'queue' ? <>
        <Progress done={s.done_by_me} total={s.fields} />
        <p className="small secondary">{finished ? 'You have finished every field — thank you!' : `${s.done_by_me} of ${s.fields} fields finished by you`}</p>
      </> : <p className="small secondary">{s.my_labels ? `${s.my_labels.toLocaleString()} objects labelled by you` : 'Pick what each object is with one key press.'}</p>}
      {!finished && <a className="btn primary big" href={s.link}>{started ? 'Continue' : 'Start'}</a>}
    </div>
  )
}

function Welcome() {
  return (
    <div className="annotate-welcome">
      <PageHead title="Help check tissue images" subtitle="You were invited to label objects in brain tissue images for a study." />
      <ol className="plain-steps big">
        <li><b>Sign in</b>, or make an account with your name and a password. Your labels are kept under it; sign in again next time to continue.</li>
        <li><b>Add a set</b> with the key you were given. Each set asks you to look at images for one stain.</li>
        <li><b>Correct the computer’s guesses</b> by clicking and pressing number keys. Short instructions are on every page.</li>
      </ol>
      <SignInForm start="up" />
      <p className="small muted">Just curious? <Link to="/annotate/tutorial">Try the tutorial</Link>: no account needed, and nothing is kept.</p>
    </div>
  )
}

/** The person's own account: their name and their password. */
export function AccountPage() {
  const { session } = useSession()
  if (!session.account) return <Welcome />
  return (
    <div className="annotate-home">
      <PageHead title="Your account" subtitle={`Signed in as ${session.account.name}${session.account.past_name ? ` (labels as “${session.account.past_name}”)` : ''}.`} />
      <ChoosePassword title="Change your password" current />
      <p className="small" style={{ marginTop: 12 }}><Link to="/annotate">← My labelling sets</Link></p>
    </div>
  )
}
