import type { AnnotationListItem } from '../api'
import { Link } from 'react-router-dom'
import { Empty, ErrorNote, PageHead, Progress } from '../components'
import { useFetch } from '../hooks'
import { Register, useSession } from '../session'

export default function AnnotateHome() {
  const { session } = useSession()
  const list = useFetch<AnnotationListItem[]>(session.annotator ? '/api/annotate' : null, 15000)
  if (!session.annotator) return <><PageHead title="Annotate" subtitle="Help label tissue images. Register with your name to start." /><Register /></>
  return (
    <>
      <PageHead title="Annotation sessions" subtitle="Pick a session. Your work is saved as you go, so you can stop and come back any time." />
      <ErrorNote error={list.error} />
      {list.data && (list.data.length ? (
        <div className="grid grid-3">
          {list.data.map((s) => (
            <div key={s.name} className="card">
              <h3 style={{ marginTop: 0 }}>{s.title}</h3>
              <p className="muted small">{s.stain} · {s.fields} fields</p>
              <Progress done={s.done_by_me} total={s.fields} />
              <p className="muted small">{s.done_by_me} of {s.fields} finished by you</p>
              <Link className="btn primary" to={`/annotate/${s.name}`}>{s.done_by_me ? 'Continue' : 'Start'}</Link>
            </div>
          ))}
        </div>
      ) : <Empty>No annotation session is running right now. Check back later or ask the person who sent you the link.</Empty>)}
    </>
  )
}
