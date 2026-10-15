import { Link, Navigate, useNavigate, useParams } from 'react-router-dom'
import type { SetupOverview } from '../api'
import { Loading, PageHead } from '../components'
import { useFetch } from '../hooks'
import ChecksScreen from './ChecksScreen'
import CoresScreen from './CoresScreen'
import DonorsScreen from './DonorsScreen'
import MapScreen from './MapScreen'
import { plainError, Problem, ScreenMark } from './parts'
import RunScreen from './RunScreen'
import SlidesScreen from './SlidesScreen'

export const SCREENS = ['slides', 'cores', 'map', 'donors', 'checks', 'run'] as const
export type ScreenId = (typeof SCREENS)[number]

/** Each screen's name and the one sentence saying what it is for (shown at once, before anything has loaded). */
const SCREEN: Record<ScreenId, { title: string; intro: string }> = {
  slides: { title: 'Slides', intro: 'Tell stainID which scan is which TMA and stain. Most are read from the file names; choose the rest from the lists.' },
  cores: { title: 'Cores', intro: 'stainID finds every core on each slide by itself. Look at each picture: every circle should sit on a core.' },
  map: { title: 'Map', intro: 'Tell stainID who is on each core: add the TMA maps as Excel, Word or CSV files, paste them, or type them in.' },
  donors: { title: 'Donors', intro: 'Optional: age, sex, APOE and other details per donor, from your case spreadsheet. The columns are matched for you.' },
  checks: { title: 'Checks', intro: 'The maps, the donor table and the slides compared with each other. Most problems can be fixed with one click.' },
  run: { title: 'Run', intro: 'Start the analysis. The steps run in the right order by themselves and keep running when you close the browser.' },
}

const STATE_WORD: Record<string, string> = { done: 'done', todo: 'not done yet', attention: 'needs attention', warning: 'please look', optional: 'optional', running: 'running' }

/** The guided setup: a few short screens from slide scans to a study that runs. Each screen shows what was found
automatically, asks only for what could not be worked out, and keeps every technical detail under "Advanced". */
export default function Setup() {
  const { screen = '' } = useParams()
  const navigate = useNavigate()
  const overview = useFetch<SetupOverview>('/api/setup', 6000)
  const known = SCREENS.includes(screen as ScreenId)
  if (!known && overview.data) {
    const first = overview.data.screens.find((s) => s.state !== 'done' && s.state !== 'optional')
    return <Navigate to={`/setup/${first?.id ?? 'run'}`} replace />
  }
  const id = known ? (screen as ScreenId) : null
  const index = id ? SCREENS.indexOf(id) : -1
  const go = (to: ScreenId) => { overview.reload(); navigate(`/setup/${to}`) }
  const nextId = index >= 0 && index < SCREENS.length - 1 ? SCREENS[index + 1] : undefined
  const props: ScreenProps = {
    next: nextId && (() => go(nextId)), back: index > 0 ? () => go(SCREENS[index - 1]) : undefined,
    nextLabel: nextId ? `Next: ${SCREEN[nextId].title}` : undefined, go, onChanged: overview.reload, overview: overview.data,
  }

  return (
    <>
      <PageHead title="Set up"
        subtitle={<>{overview.data?.set_up ? 'Add new slides, or change what was set up.' : 'Six short steps from slide scans to a study that runs.'} You can leave at any time and come back; nothing is lost.</>} />
      <Stepper current={id} overview={overview.data} onGo={go} />
      {overview.error && <Problem text={`The state of each step could not be loaded: ${plainError(overview.error)}`} onRetry={overview.reload} />}
      {!id ? (!overview.error && <Loading label="Finding where you left off…" />) : (
        <section className="setup-screen" aria-labelledby="screen-title">
          <div className="screen-head">
            <h2 id="screen-title"><span className="muted">Step {index + 1} of {SCREENS.length} · </span>{SCREEN[id].title}</h2>
            <p className="secondary">{SCREEN[id].intro}</p>
          </div>
          {id === 'slides' && <SlidesScreen {...props} />}
          {id === 'cores' && <CoresScreen {...props} />}
          {id === 'map' && <MapScreen {...props} />}
          {id === 'donors' && <DonorsScreen {...props} />}
          {id === 'checks' && <ChecksScreen {...props} />}
          {id === 'run' && <RunScreen {...props} />}
        </section>
      )}
      <p className="small muted setup-more">Every step with all its options: <Link to="/workflow">the full list of steps</Link>.</p>
    </>
  )
}

/** The six steps across the top: where you are, which are done, and one line on each once the overview is in. */
function Stepper({ current, overview, onGo }: { current: ScreenId | null; overview: SetupOverview | null; onGo: (screen: ScreenId) => void }) {
  return (
    <nav className="stepper" aria-label="Setup steps">
      <ol>
        {SCREENS.map((s, i) => {
          const info = overview?.screens.find((x) => x.id === s)
          const state = info?.state ?? 'todo'
          return (
            <li key={s}>
              <button className={`stepper-item state-${state}${s === current ? ' current' : ''}`} onClick={() => onGo(s)} aria-current={s === current ? 'step' : undefined}
                title={info ? `${SCREEN[s].title}: ${info.summary}` : SCREEN[s].title} aria-label={`Step ${i + 1}, ${SCREEN[s].title}${info ? `, ${STATE_WORD[state] ?? state}` : ''}`}>
                <ScreenMark state={state} number={i + 1} />
                <span className="stepper-text">
                  <b>{SCREEN[s].title}</b>
                  <span className="muted small stepper-summary">{info ? info.summary : <span className="skeleton-line" aria-hidden="true" />}</span>
                </span>
              </button>
            </li>
          )
        })}
      </ol>
    </nav>
  )
}

export interface ScreenProps {
  next?: () => void
  back?: () => void
  /** "Next: <the next screen>" */
  nextLabel?: string
  go: (screen: ScreenId) => void
  onChanged: () => void
  overview: SetupOverview | null
}
