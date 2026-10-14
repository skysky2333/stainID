import { useEffect } from 'react'
import { Link, NavLink, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import type { Job, ProjectsInfo, Session } from './api'
import { api, projectId, projectPath, remote } from './api'
import { ErrorNote } from './components'
import { useFetch, useStored } from './hooks'
import { ProjectProvider, ProjectSwitcher, useProject } from './project'
import { AdminSignIn, Annotator, ServerSetting, SessionContext, useSession } from './session'
import AnnotateField from './pages/AnnotateField'
import AnnotateHome from './pages/AnnotateHome'
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

const NAV: { section: string; links: [string, string][] }[] = [
  { section: '', links: [['/', 'Home'], ['/setup', 'Guided setup'], ['/workflow', 'Workflow']] },
  { section: 'Look at the data', links: [['/cohort', 'Cohort & cores'], ['/calibration', 'Stain thresholds']] },
  { section: 'Improve accuracy', links: [['/labels', 'Labels'], ['/models', 'Models']] },
  { section: 'Results', links: [['/results', 'Results']] },
  { section: '', links: [['/settings', 'Settings'], ['/help', 'Help']] },
]

export default function App() {
  const session = useFetch<Session>('/api/session')
  const { pathname } = useLocation()
  const [theme] = useStored<'system' | 'light' | 'dark'>('stainid.theme', 'system')
  if (theme === 'system') document.documentElement.removeAttribute('data-theme')
  else document.documentElement.setAttribute('data-theme', theme)
  if (!session.data) return <main className="main"><ErrorNote error={session.error} />{session.error && <ServerSetting open={remote} />}</main>
  const annotating = pathname === '/annotate' || pathname.startsWith('/annotate/')
  const admin = session.data.admin
  let page
  if (annotating) page = <AnnotatorShell />  // in a project, or (old links) the project opened last
  else if (!admin) page = <AdminSignIn />
  else if (projectId) page = <ProjectProvider><Shell /></ProjectProvider>
  else if (pathname === '/' || pathname === '/projects') page = <Projects />
  else page = <ToLastProject />
  return <SessionContext.Provider value={{ session: session.data, reload: session.reload }}>{page}</SessionContext.Provider>
}

/** An address of the previous release (/workflow, /cohort …): the same page of the project opened last. */
function ToLastProject() {
  const { pathname, search, hash } = useLocation()
  const projects = useFetch<ProjectsInfo>('/api/projects')
  const target = projects.data ? (projects.data.default ? projectPath(projects.data.default, `${pathname}${search}${hash}`) : '/') : null
  useEffect(() => { if (target) location.replace(target) }, [target])
  return <main className="main"><ErrorNote error={projects.error} /></main>
}

/** The public annotation pages (/annotate): anyone with the link, for sessions the admin has started. */
function AnnotatorShell() {
  return (
    <div>
      <header className="row" style={{ padding: '10px 24px', borderBottom: '1px solid var(--border)', background: 'var(--surface-2)' }}>
        <Link to="/annotate" className="brand" style={{ margin: 0 }}><img src="/favicon.svg" alt="" />stainID annotation</Link>
        <span className="spacer" />
        <Annotator />
      </header>
      <main className="main">
        <Routes>
          <Route path="/annotate" element={<AnnotateHome />} />
          <Route path="/annotate/:name" element={<AnnotateField />} />
          <Route path="/annotate/:name/:tile" element={<AnnotateField />} />
        </Routes>
      </main>
    </div>
  )
}

function Shell() {
  const { info, error } = useProject()
  const { session, reload } = useSession()
  const [theme, setTheme] = useStored<'system' | 'light' | 'dark'>('stainid.theme', 'system')
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
        {NAV.map((group, i) => (
          <div key={i}>
            {group.section && <div className="nav-section">{group.section}</div>}
            {group.links.map(([to, label]) => (
              <NavLink key={to} to={to} end={to === '/'} className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>{label}</NavLink>
            ))}
          </div>
        ))}
        <div className="sidebar-foot">
          <label className="field">
            Theme
            <select value={theme} onChange={(e) => setTheme(e.target.value as 'system' | 'light' | 'dark')}>
              <option value="system">System</option>
              <option value="light">Light</option>
              <option value="dark">Dark</option>
            </select>
          </label>
          <span className="row">stainID v2<span className="spacer" /><button className="link-btn" onClick={async () => { await api.del('/api/session/admin'); reload() }}>sign out</button></span>
        </div>
      </aside>
      <main className="main">
        {session.default_password && <div className="instructions">The admin password is still the default (<b>admin</b>). <Link to="/settings#password">Change it in Settings</Link> before letting others on your network use stainID.</div>}
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

function JobsIndicator() {
  const jobs = useFetch<Job[]>('/api/jobs', 4000)
  const running = (jobs.data ?? []).filter((j) => j.status === 'running')
  const queued = (jobs.data ?? []).filter((j) => j.status === 'queued')
  if (!running.length && !queued.length) return null
  return (
    <Link to="/workflow#running" className="jobs-chip">
      <span className="pulse" />
      {running.length} running{queued.length ? ` · ${queued.length} waiting` : ''}
      <div className="muted small ellipsis">{running.map((j) => j.title).join(', ')}</div>
    </Link>
  )
}
