import { Link, Navigate, useNavigate, useParams } from 'react-router-dom'
import type { SetupOverview } from '../api'
import { ErrorNote, PageHead } from '../components'
import { useFetch } from '../hooks'
import ChecksScreen from './ChecksScreen'
import CoresScreen from './CoresScreen'
import DonorsScreen from './DonorsScreen'
import MapScreen from './MapScreen'
import { ScreenMark } from './parts'
import RunScreen from './RunScreen'
import SlidesScreen from './SlidesScreen'

export const SCREENS = ['slides', 'cores', 'map', 'donors', 'checks', 'run'] as const
export type ScreenId = (typeof SCREENS)[number]

const INTRO: Record<ScreenId, string> = {
  slides: 'Which scan is which TMA and stain. stainID reads it from the file names; check the table and correct anything it got wrong.',
  cores: 'stainID finds every core on each slide and works out the grid size by itself. Look at each picture: every circle should sit on a core.',
  map: 'Who is on each core. Add the TMA maps as Excel, Word or CSV files, paste them from a spreadsheet, or type them into the grid.',
  donors: 'Optional: age, sex, APOE and other information per donor, from your case spreadsheet. The columns are matched for you.',
  checks: 'The map, the donor table and the slides compared with each other. Most things can be fixed with one click.',
  run: 'Everything is ready: run all the analysis steps in one go.',
}

/** The guided setup: a few short screens from slide scans to a study that runs. Each screen shows what was found
automatically, asks only for what could not be worked out, and keeps every technical detail under "Advanced". */
export default function Setup() {
  const { screen = '' } = useParams()
  const navigate = useNavigate()
  const overview = useFetch<SetupOverview>('/api/setup', 6000)
  if (!SCREENS.includes(screen as ScreenId)) {
    if (!overview.data) return <ErrorNote error={overview.error} />
    const first = overview.data.screens.find((s) => s.state !== 'done' && s.state !== 'optional')
    return <Navigate to={`/setup/${first?.id ?? 'run'}`} replace />
  }
  const id = screen as ScreenId
  const index = SCREENS.indexOf(id)
  const go = (to: ScreenId) => { overview.reload(); navigate(`/setup/${to}`) }
  const next = index < SCREENS.length - 1 ? () => go(SCREENS[index + 1]) : undefined
  const back = index > 0 ? () => go(SCREENS[index - 1]) : undefined
  const props = { next, back, go, onChanged: overview.reload, overview: overview.data }
  const setUp = overview.data?.set_up

  return (
    <>
      <PageHead title={setUp ? 'Add slides or change the setup' : 'Set up your study'}
        subtitle={<>Six short steps from slide scans to results. You can leave at any time and come back; nothing is lost. <Link to="/workflow">All steps in detail</Link></>} />
      <nav className="stepper" aria-label="Setup steps">
        {SCREENS.map((s, i) => {
          const info = overview.data?.screens.find((x) => x.id === s)
          return (
            <button key={s} className={`stepper-item${s === id ? ' current' : ''}`} onClick={() => go(s)} aria-current={s === id ? 'step' : undefined}>
              <ScreenMark state={info?.state ?? 'todo'} number={i + 1} />
              <span><b>{info?.title ?? s}</b><span className="muted small stepper-summary">{info?.summary}</span></span>
            </button>
          )
        })}
      </nav>
      <p className="secondary screen-intro">{INTRO[id]}</p>
      <ErrorNote error={overview.error} />
      {id === 'slides' && <SlidesScreen {...props} />}
      {id === 'cores' && <CoresScreen {...props} />}
      {id === 'map' && <MapScreen {...props} />}
      {id === 'donors' && <DonorsScreen {...props} />}
      {id === 'checks' && <ChecksScreen {...props} />}
      {id === 'run' && <RunScreen {...props} />}
    </>
  )
}

export interface ScreenProps {
  next?: () => void
  back?: () => void
  go: (screen: ScreenId) => void
  onChanged: () => void
  overview: SetupOverview | null
}
