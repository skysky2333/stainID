import type { MySet } from '../api'
import { PageHead } from '../components'
import { useFetch } from '../hooks'
import { TutorialCards } from '../labels/tutorial'
import { useSession } from '../session'

/** Both tutorials, for anyone (no account needed: nothing is kept). */
export default function Tutorials() {
  const { session } = useSession()
  const sets = useFetch<MySet[]>(session.account ? '/api/session/sets' : null)
  return (
    <div className="annotate-home">
      <PageHead title="Tutorials" subtitle="Practise the two labelling tasks on real images of the study. Nothing you do here is saved." />
      <TutorialCards sets={sets.data} />
    </div>
  )
}
