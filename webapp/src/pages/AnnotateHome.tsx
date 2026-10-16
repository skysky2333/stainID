import { Link } from 'react-router-dom'
import type { OpenSession } from '../api'
import { Empty, Loaded, PageHead, Progress } from '../components'
import { useFetch } from '../hooks'
import { KIND_TEXT } from '../labelling'
import { ROUND_TEXT, ROUND_TITLE } from '../labels/rounds'
import { Register, useSession } from '../session'
import { useFirstTutorial } from '../labels/tutorial'

export default function AnnotateHome() {
  const { session } = useSession()
  const list = useFetch<OpenSession[]>(session.annotator ? '/api/annotate' : null, 15000)
  useFirstTutorial()
  if (!session.annotator) return <Welcome />
  return (
    <div className="annotate-home">
      <PageHead title="Choose a session" subtitle="Everything you do is saved as you go, so you can stop and come back at any time." />
      <Loaded state={list} label="Loading sessions…">
        {(sessions) => sessions.length ? (
          <div className="grid grid-2">{sessions.map((s) => <SessionCard key={s.name} s={s} />)}</div>
        ) : (
          <Empty>
            <p style={{ marginTop: 0 }}>No session is open right now. The person who sent you the link opens one when it is ready.</p>
            <button className="btn" onClick={list.reload}>Check again</button>
          </Empty>
        )}
      </Loaded>
    </div>
  )
}

function SessionCard({ s }: { s: OpenSession }) {
  const started = s.done_by_me > 0 || s.my_labels > 0
  const progress = s.round ? s.progress : undefined
  const finished = progress ? progress.total > 0 && progress.decided >= progress.total : s.kind !== 'queue' && s.fields > 0 && s.done_by_me >= s.fields
  return (
    <div className="card annotate-session">
      <h2 style={{ margin: 0 }}>{s.title}</h2>
      <p className="muted small" style={{ margin: '4px 0 10px' }}>{s.stain} · {s.round ? `Round ${s.round} · ${ROUND_TITLE[s.round]}` : KIND_TEXT[s.kind]}</p>
      {s.instructions && <p className="small secondary">{s.instructions}</p>}
      {progress ? <>
        <Progress done={progress.decided} total={progress.total} />
        <p className="small secondary">{finished ? 'Everything here has been checked — thank you!'
          : `${progress.decided.toLocaleString()} of ${progress.total.toLocaleString()} ${s.round === 1 ? 'objects' : 'fields'} done by everyone${s.my_labels ? ` · ${s.my_labels.toLocaleString()} answers by you` : ''}`}</p>
        <p className="small muted" style={{ marginTop: 0 }}>{s.round === 1 ? ROUND_TEXT[1] : ROUND_TEXT[2]}</p>
      </> : s.kind !== 'queue' ? <>
        <Progress done={s.done_by_me} total={s.fields} />
        <p className="small secondary">{finished ? 'You have finished every field — thank you!' : `${s.done_by_me} of ${s.fields} fields finished by you`}</p>
      </> : <p className="small secondary">{s.my_labels ? `${s.my_labels.toLocaleString()} objects labelled by you` : 'Pick what each object is with one key press.'}</p>}
      {!finished && <Link className="btn primary big" to={`/annotate/${s.name}`}>{started ? 'Continue' : 'Start'}</Link>}
    </div>
  )
}

function Welcome() {
  return (
    <div className="annotate-welcome">
      <PageHead title="Help check tissue images" subtitle="You were sent this link to label objects in brain tissue images for a study." />
      <ol className="plain-steps big">
        <li><b>Type your name</b> below. Your labels are saved under it; use the same name next time to continue where you left off.</li>
        <li><b>Pick a session</b>. Each one asks you to look at a set of images for one stain.</li>
        <li><b>Correct the computer’s guesses</b> by clicking and pressing number keys. Short instructions are on every page.</li>
      </ol>
      <Register />
    </div>
  )
}
