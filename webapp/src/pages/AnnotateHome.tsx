import { Link } from 'react-router-dom'
import type { OpenSession } from '../api'
import { Empty, ErrorNote, PageHead, Progress } from '../components'
import { useFetch } from '../hooks'
import { Register, useSession } from '../session'

export default function AnnotateHome() {
  const { session } = useSession()
  const list = useFetch<OpenSession[]>(session.annotator ? '/api/annotate' : null, 15000)
  if (!session.annotator) return <><PageHead title="Label tissue images" subtitle="Register with your name to start." /><Register /></>
  return (
    <>
      <PageHead title="Sessions" subtitle="Pick a session. Everything is saved as you go, so you can stop and come back any time." />
      <ErrorNote error={list.error} />
      {list.data && (list.data.length ? (
        <div className="grid grid-3">
          {list.data.map((s) => (
            <div key={s.name} className="card">
              <h3 style={{ marginTop: 0 }}>{s.title}</h3>
              {s.kind === 'fields' ? <>
                <p className="muted small">{s.stain} · review whole fields · {s.fields} fields</p>
                <Progress done={s.done_by_me} total={s.fields} />
                <p className="muted small">{s.done_by_me} of {s.fields} finished by you</p>
              </> : <p className="muted small">{s.stain} · one object at a time · {s.my_labels} labelled by you</p>}
              <Link className="btn primary" to={`/annotate/${s.name}`}>{s.done_by_me || s.my_labels ? 'Continue' : 'Start'}</Link>
            </div>
          ))}
        </div>
      ) : <Empty>No session is running right now. Check back later or ask the person who sent you the link.</Empty>)}
    </>
  )
}
