import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import { useStored } from './hooks'
import Analysis from './pages/Analysis'
import Calibration from './pages/Calibration'
import Cohort from './pages/Cohort'
import CoreView from './pages/CoreView'
import Models from './pages/Models'
import Overview from './pages/Overview'
import Pipelines from './pages/Pipelines'
import Project from './pages/Project'
import ReviewLabel from './pages/ReviewLabel'
import Reviews from './pages/Reviews'

const NAV = [
  { section: 'Data', links: [['/', 'Overview'], ['/cohort', 'Cohort & cores'], ['/calibration', 'Calibration']] },
  { section: 'Run', links: [['/pipelines', 'Pipelines & jobs']] },
  { section: 'Validate', links: [['/reviews', 'Review & annotate'], ['/models', 'Models']] },
  { section: 'Results', links: [['/analysis', 'Analysis']] },
  { section: 'Settings', links: [['/project', 'Project']] },
]

export default function App() {
  const [theme, setTheme] = useStored<'system' | 'light' | 'dark'>('stainid.theme', 'system')
  if (theme === 'system') document.documentElement.removeAttribute('data-theme')
  else document.documentElement.setAttribute('data-theme', theme)

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand"><img src="/favicon.svg" alt="" />stainID</div>
        {NAV.map((group) => (
          <div key={group.section}>
            <div className="nav-section">{group.section}</div>
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
          <Route path="/" element={<Overview />} />
          <Route path="/cohort" element={<Cohort />} />
          <Route path="/cores/:coreId" element={<CoreView />} />
          <Route path="/calibration" element={<Calibration />} />
          <Route path="/pipelines" element={<Pipelines />} />
          <Route path="/reviews" element={<Reviews />} />
          <Route path="/reviews/label/*" element={<ReviewLabel />} />
          <Route path="/models" element={<Models />} />
          <Route path="/analysis" element={<Analysis />} />
          <Route path="/project" element={<Project />} />
          <Route path="*" element={<Navigate to="/" />} />
        </Routes>
      </main>
    </div>
  )
}
