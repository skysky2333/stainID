import { Link, Navigate, useNavigate, useParams } from 'react-router-dom'
import type { ProjectKind, SetupOverview } from '../api'
import { Loading, PageHead } from '../components'
import { useFetch } from '../hooks'
import ChecksScreen from './ChecksScreen'
import CoresScreen from './CoresScreen'
import DonorsScreen from './DonorsScreen'
import ImagesScreen from './ImagesScreen'
import MapScreen from './MapScreen'
import { plainError, Problem, ScreenMark } from './parts'
import RunScreen from './RunScreen'
import SlidesScreen from './SlidesScreen'

/** The screens of each kind of study: a TMA study finds cores on slides and needs its TMA map; an image set or a set of
whole slides has no map and no core grid, so it starts from its images. */
const KIND_SCREENS = {
  tma: ['slides', 'cores', 'map', 'donors', 'checks', 'run'],
  images: ['images', 'donors', 'checks', 'run'],
  slides: ['images', 'donors', 'checks', 'run'],
} as const
export const SCREENS = ['slides', 'cores', 'map', 'images', 'donors', 'checks', 'run'] as const
export type ScreenId = (typeof SCREENS)[number]

/** Each screen's name and the one sentence saying what it is for (shown at once, before anything has loaded). */
const SCREEN: Record<ScreenId, { title: string; intro: string }> = {
  slides: { title: 'Slides', intro: 'Tell stainID which scan is which TMA and stain. Most are read from the file names; choose the rest from the lists.' },
  cores: { title: 'Cores', intro: 'stainID finds every core on each slide by itself. Look at each picture: every circle should sit on a core.' },
  map: { title: 'Map', intro: 'Tell stainID who is on each core: add the TMA maps as Excel, Word or CSV files, paste them, or type them in.' },
  images: { title: 'Images', intro: 'The images of the study, each with its donor and stain, and how they were imaged. Each image is analysed the way a TMA core is.' },
  donors: { title: 'Donors', intro: 'Optional: age, sex, APOE and other details per donor, from your case spreadsheet. The columns are matched for you.' },
  checks: { title: 'Checks', intro: 'What was set up, compared with itself. Most problems can be fixed with one click.' },
  run: { title: 'Run', intro: 'Start the analysis. The steps run in the right order by themselves and keep running when you close the browser.' },
}
const SLIDES_IMAGES = { title: 'Slides', intro: 'The whole-slide scans of the study, each with its donor and stain. stainID finds the tissue on each slide and analyses it region by region, the way a TMA core is.' }
const SUBTITLE: Record<ProjectKind, string> = {
  tma: 'Six short steps from slide scans to a study that runs.',
  images: 'Four short steps from a folder of images to a study that runs. There is no TMA map: each image is analysed like a core.',
  slides: 'Four short steps from whole-slide scans to a study that runs. There is no TMA map: the tissue of each slide is analysed region by region.',
}

const STATE_WORD: Record<string, string> = { done: 'done', todo: 'not done yet', attention: 'needs attention', warning: 'please look', optional: 'optional', running: 'running' }

const screenInfo = (id: ScreenId, kind: ProjectKind) => (id === 'images' && kind === 'slides' ? SLIDES_IMAGES : SCREEN[id])

/** The guided setup: a few short screens from slide scans (or images) to a study that runs. Each screen shows what was
found automatically, asks only for what could not be worked out, and keeps every technical detail under "Advanced". */
export default function Setup() {
  const { screen = '' } = useParams()
  const navigate = useNavigate()
  const overview = useFetch<SetupOverview>('/api/setup', 6000)
  const kind: ProjectKind = overview.data?.kind ?? 'tma'
  const screens: readonly ScreenId[] = KIND_SCREENS[kind]
  const known = screens.includes(screen as ScreenId)
  if (!known && overview.data) {
    const first = overview.data.screens.find((s) => s.state !== 'done' && s.state !== 'optional')
    return <Navigate to={`/setup/${first?.id ?? 'run'}`} replace />
  }
  const id = known && overview.data ? (screen as ScreenId) : null
  const index = id ? screens.indexOf(id) : -1
  const go = (to: ScreenId) => { overview.reload(); navigate(`/setup/${to}`) }
  const nextId = index >= 0 && index < screens.length - 1 ? screens[index + 1] : undefined
  const props: ScreenProps = {
    next: nextId && (() => go(nextId)), back: index > 0 ? () => go(screens[index - 1]) : undefined,
    nextLabel: nextId ? `Next: ${screenInfo(nextId, kind).title}` : undefined, go, onChanged: overview.reload, overview: overview.data,
  }

  return (
    <>
      <PageHead title="Set up"
        subtitle={<>{overview.data?.set_up ? `Add new ${kind === 'tma' ? 'slides' : kind === 'slides' ? 'slides' : 'images'}, or change what was set up.` : SUBTITLE[kind]} You can leave at any time and come back; nothing is lost.</>} />
      <Stepper screens={overview.data ? screens : []} kind={kind} current={id} overview={overview.data} onGo={go} />
      {overview.error && <Problem text={`The state of each step could not be loaded: ${plainError(overview.error)}`} onRetry={overview.reload} />}
      {!id ? (!overview.error && <Loading label="Finding where you left off…" />) : (
        <section className="setup-screen" aria-labelledby="screen-title">
          <div className="screen-head">
            <h2 id="screen-title"><span className="muted">Step {index + 1} of {screens.length} · </span>{screenInfo(id, kind).title}</h2>
            <p className="secondary">{screenInfo(id, kind).intro}</p>
          </div>
          {id === 'slides' && <SlidesScreen {...props} />}
          {id === 'cores' && <CoresScreen {...props} />}
          {id === 'map' && <MapScreen {...props} />}
          {id === 'images' && <ImagesScreen {...props} />}
          {id === 'donors' && <DonorsScreen {...props} />}
          {id === 'checks' && <ChecksScreen {...props} />}
          {id === 'run' && <RunScreen {...props} />}
        </section>
      )}
      <p className="small muted setup-more">Every step with all its options: <Link to="/workflow">the full list of steps</Link>.</p>
    </>
  )
}

/** The steps across the top: where you are, which are done, and one line on each once the overview is in. */
function Stepper({ screens, kind, current, overview, onGo }: {
  screens: readonly ScreenId[]; kind: ProjectKind; current: ScreenId | null; overview: SetupOverview | null; onGo: (screen: ScreenId) => void
}) {
  return (
    <nav className="stepper" aria-label="Setup steps">
      <ol>
        {screens.map((s, i) => {
          const info = overview?.screens.find((x) => x.id === s)
          const state = info?.state ?? 'todo'
          const title = screenInfo(s, kind).title
          return (
            <li key={s}>
              <button className={`stepper-item state-${state}${s === current ? ' current' : ''}`} onClick={() => onGo(s)} aria-current={s === current ? 'step' : undefined}
                title={info ? `${title}: ${info.summary}` : title} aria-label={`Step ${i + 1}, ${title}${info ? `, ${STATE_WORD[state] ?? state}` : ''}`}>
                <ScreenMark state={state} number={i + 1} />
                <span className="stepper-text">
                  <b>{title}</b>
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
