import { useEffect } from 'react'
import { Link, NavLink, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import type { Job, ProjectsInfo, Session } from './api'
import { api, projectId, projectPath, remote } from './api'
import { ActivityBar, ErrorNote, Loading } from './components'
import { useFetch, useTheme } from './hooks'
import { ProjectProvider, ProjectSwitcher, useProject } from './project'
import { AdminSignIn, Annotator, ServerSetting, SessionContext, useSession } from './session'
import AnnotateField from './pages/AnnotateField'
import AnnotateHome from './pages/AnnotateHome'
import { Tutorial, TutorialLink } from './labels/tutorial'
import Calibration from './pages/Calibration'
import Cohort from './pages/Cohort'
import CoreView from './pages/CoreView'
import Grids from './pages/Grids'
import Help from './pages/Help'
import Home from './pages/Home'
import Models from './pages/Models'
import Projects from './pages/Projects'
import Results from './pages/Results'
import Labels from './pages/Labels'
import Settings from './pages/Settings'
import Setup from './setup/Setup'
import Workflow from './pages/Workflow'

/** A sidebar link: [address, label, what the page is for, other addresses that belong to the page]. */
type NavItem = [string, string, string, string[]?]
const NAV: { section: string; links: NavItem[] }[] = [
  { section: '', links: [
    ['/', 'Home', 'Where the study stands and what to do next'],
    ['/setup', 'Set up', 'Add slides, find the cores, read the TMA maps'],
    ['/workflow', 'Run', 'Run the analysis and follow its progress'],
  ] },
  { section: 'Check', links: [
    ['/cohort', 'Cores', 'Look at every core and what was found in it', ['/cores']],
    ['/calibration', 'Stain thresholds', 'The stain level counted as positive on each slide'],
  ] },
  { section: '', links: [['/results', 'Results', 'Measures by diagnosis group, and the tables to download']] },
  { section: 'Improve accuracy', links: [
    ['/labels', 'Labels', 'Check and correct what detection found'],
    ['/models', 'Models', 'Get, train and choose the detection models'],
  ] },
]
const FOOT_NAV: NavItem[] = [['/settings', 'Settings', 'Study settings, admin password, appearance'], ['/help', 'Help', 'Getting started, common questions, glossary']]

const belongs = (pathname: string, to: string) => (to === '/' ? pathname === '/' : pathname === to || pathname.startsWith(`${to}/`))

/** The label of the page at this address, for the browser tab. */
function pageLabel(pathname: string) {
  const item = [...NAV.flatMap((g) => g.links), ...FOOT_NAV].find(([to, , , more = []]) => [to, ...more].some((m) => belongs(pathname, m)))
  return item?.[1]
}

export default function App() {
  const session = useFetch<Session>('/api/session')
  const { pathname } = useLocation()
  useTheme()
  if (!session.data) return <main className="main">{session.error ? <><ErrorNote error={session.error} /><ServerSetting open={remote} /></> : <Loading />}</main>
  const annotating = pathname === '/annotate' || pathname.startsWith('/annotate/')
  const admin = session.data.admin
  let page
  if (annotating) page = <AnnotatorShell />  // in a project, or (old links) the project opened last
  else if (!admin) page = <AdminSignIn />
  else if (projectId) page = <ProjectProvider><Shell /></ProjectProvider>
  else if (pathname === '/' || pathname === '/projects') page = <Projects />
  else page = <ToLastProject />
  return <SessionContext.Provider value={{ session: session.data, reload: session.reload }}><ActivityBar />{page}</SessionContext.Provider>
}

/** An address of the previous release (/workflow, /cohort …): the same page of the project opened last. */
function ToLastProject() {
  const { pathname, search, hash } = useLocation()
  const projects = useFetch<ProjectsInfo>('/api/projects')
  const target = projects.data ? (projects.data.default ? projectPath(projects.data.default, `${pathname}${search}${hash}`) : '/') : null
  useEffect(() => { if (target) location.replace(target) }, [target])
  return <main className="main">{projects.error ? <ErrorNote error={projects.error} /> : <Loading />}</main>
}

/** The public annotation pages (/annotate): anyone with the link, for sessions the admin has started. */
function AnnotatorShell() {
  return (
    <div className="annotator-shell">
      <header className="row" style={{ padding: '10px 24px', borderBottom: '1px solid var(--border)', background: 'var(--surface-2)' }}>
        <Link to="/annotate" className="brand" style={{ margin: 0 }}><img src="/favicon.svg" alt="" />stainID annotation</Link>
        <span className="spacer" />
        <TutorialLink />
        <Annotator />
      </header>
      <main className="main">
        <Routes>
          <Route path="/annotate" element={<AnnotateHome />} />
          <Route path="/annotate/tutorial" element={<Tutorial />} />
          <Route path="/annotate/:name" element={<AnnotateField />} />
          <Route path="/annotate/:name/:tile" element={<AnnotateField />} />
        </Routes>
      </main>
    </div>
  )
}

function Shell() {
  const { info, error } = useProject()
  const { reload } = useSession()
  const { pathname } = useLocation()
  useEffect(() => { document.title = [pageLabel(pathname), info?.name, 'stainID'].filter(Boolean).join(' · ') }, [pathname, info?.name])
  if (error && !info) {
    return <main className="main"><div className="card" style={{ maxWidth: 560 }}><h2>This project cannot be opened</h2><ErrorNote error={error} />
      <a className="btn primary" href="/">Go to the projects</a></div></main>
  }
  if (info && !info.configured) {
    return <main className="main"><div className="card" style={{ maxWidth: 560 }}><h2>Not a stainID project</h2>
      <p className="secondary">The folder <code>{info.root}</code> has no <code>stainid.yaml</code>.</p>
      <a className="btn primary" href="/">Choose or create a project</a></div></main>
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand"><img src="/favicon.svg" alt="" />stainID</div>
        <ProjectSwitcher />
        <JobsIndicator />
        <nav className="nav" aria-label="Pages">
          {NAV.map((group, i) => (
            <div key={i} className="nav-group">
              {group.section && <div className="nav-section">{group.section}</div>}
              {group.links.map((item) => <SideLink key={item[0]} item={item} />)}
            </div>
          ))}
        </nav>
        <div className="sidebar-foot">
          <nav className="nav-group" aria-label="Settings and help">{FOOT_NAV.map((item) => <SideLink key={item[0]} item={item} />)}</nav>
          <span className="row signed-in">Signed in as admin<span className="spacer" />
            <button className="link-btn" onClick={async () => { await api.del('/api/session/admin'); reload() }}>Sign out</button></span>
        </div>
      </aside>
      <main className="main">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/workflow" element={<Workflow />} />
          <Route path="/setup/grids" element={<Grids />} />
          <Route path="/setup" element={<Setup />} />
          <Route path="/setup/:screen" element={<Setup />} />
          <Route path="/cohort" element={<Cohort />} />
          <Route path="/cores/:coreId" element={<CoreView />} />
          <Route path="/calibration" element={<Calibration />} />
          <Route path="/labels" element={<Labels />} />
          <Route path="/models" element={<Models />} />
          <Route path="/results" element={<Results />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/help" element={<Help />} />
          <Route path="*" element={<Navigate to="/" />} />
        </Routes>
      </main>
    </div>
  )
}

/** A sidebar link, shown as the current page also on the addresses that belong to it (a core belongs to Cores). */
function SideLink({ item: [to, label, purpose, more = []] }: { item: NavItem }) {
  const { pathname } = useLocation()
  const inside = more.some((m) => belongs(pathname, m))
  return (
    <NavLink to={to} end={to === '/'} title={purpose} aria-current={inside ? 'page' : undefined}
      className={({ isActive }) => `nav-link${isActive || inside ? ' active' : ''}`}>{label}</NavLink>
  )
}

function JobsIndicator() {
  const jobs = useFetch<Job[]>('/api/jobs', 4000)
  const running = (jobs.data ?? []).filter((j) => j.status === 'running')
  const queued = (jobs.data ?? []).filter((j) => j.status === 'queued')
  if (!running.length && !queued.length) return null
  const counts = [running.length && `${running.length} ${running.length === 1 ? 'step' : 'steps'} running`, queued.length && `${queued.length} waiting`]
  const parts = running.reduce<Record<string, number>>((count, j) => ({ ...count, [j.title]: (count[j.title] ?? 0) + 1 }), {})
  const titles = Object.entries(parts).map(([title, n]) => (n > 1 ? `${title} (${n} parts)` : title))
  return (
    <Link to="/workflow#running" className="jobs-chip" title={titles.join('\n')}>
      <span className="pulse" />{counts.filter(Boolean).join(' · ')}
      {titles.length > 0 && <div className="jobs-chip-titles ellipsis">{titles.join(', ')}</div>}
    </Link>
  )
}
