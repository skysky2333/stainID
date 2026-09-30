import { Link, NavLink, Navigate, Route, Routes } from 'react-router-dom'
import type { Job } from './api'
import { useFetch, useStored } from './hooks'
import { ProjectProvider, useProject } from './project'
import Calibration from './pages/Calibration'
import Cohort from './pages/Cohort'
import CoreView from './pages/CoreView'
import Grids from './pages/Grids'
import Help from './pages/Help'
import Home from './pages/Home'
import Models from './pages/Models'
import Results from './pages/Results'
import ReviewLabel from './pages/ReviewLabel'
import Reviews from './pages/Reviews'
import Settings from './pages/Settings'
import Welcome from './pages/Welcome'
import Workflow from './pages/Workflow'

const NAV: { section: string; links: [string, string][] }[] = [
  { section: '', links: [['/', 'Home'], ['/workflow', 'Workflow']] },
  { section: 'Look at the data', links: [['/cohort', 'Cohort & cores'], ['/calibration', 'Stain thresholds']] },
  { section: 'Improve accuracy', links: [['/label', 'Label & check'], ['/models', 'Models']] },
  { section: 'Results', links: [['/results', 'Results']] },
  { section: '', links: [['/settings', 'Settings'], ['/help', 'Help']] },
]

export default function App() {
  return (
    <ProjectProvider>
      <Shell />
    </ProjectProvider>
  )
}

function Shell() {
  const { info } = useProject()
  const [theme, setTheme] = useStored<'system' | 'light' | 'dark'>('stainid.theme', 'system')
  if (theme === 'system') document.documentElement.removeAttribute('data-theme')
  else document.documentElement.setAttribute('data-theme', theme)
  if (info && !info.configured) return <main className="main"><Welcome /></main>

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand"><img src="/favicon.svg" alt="" />stainID</div>
        {info && <Link to="/settings" className="project-chip" title={info.root}><span className="muted small">Project</span>{info.name}</Link>}
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
          <span>stainID v2</span>
        </div>
      </aside>
      <main className="main">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/workflow" element={<Workflow />} />
          <Route path="/setup/grids" element={<Grids />} />
          <Route path="/cohort" element={<Cohort />} />
          <Route path="/cores/:coreId" element={<CoreView />} />
          <Route path="/calibration" element={<Calibration />} />
          <Route path="/label" element={<Reviews />} />
          <Route path="/label/*" element={<ReviewLabel />} />
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
    <Link to="/workflow#jobs" className="jobs-chip">
      <span className="pulse" />
      {running.length} running{queued.length ? ` · ${queued.length} waiting` : ''}
      <div className="muted small ellipsis">{running.map((j) => j.title).join(', ')}</div>
    </Link>
  )
}
