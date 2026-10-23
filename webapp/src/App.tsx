import { useEffect, useState } from 'react'
import { Link, NavLink, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import type { Job, ProjectsInfo, Session } from './api'
import { appBase, projectId, projectPath, remote } from './api'
import { ActivityBar, ErrorNote, Loading } from './components'
import { useFetch, useTheme } from './hooks'
import { ProjectProvider, ProjectSwitcher, useProject } from './project'
import { AppHeader, BrandLink, ChoosePassword, ServerSetting, SessionContext, SignIn, useSession } from './session'
import { AccountPage } from './pages/AnnotateHome'
import AnnotateField from './pages/AnnotateField'
import AnnotateHome from './pages/AnnotateHome'
import { Tutorial } from './labels/tutorial'
import Tutorials from './pages/Tutorials'
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
import Users from './pages/Users'
import Setup from './setup/Setup'
import Workflow from './pages/Workflow'
import { NavIcon } from './navicons'

/** A sidebar link: [address, label, what the page is for, other addresses that belong to the page]. */
type NavItem = [string, string, string, string[]?]
/* One menu for everyone: the labelling items for all, the study's pages only for admins (`ADMIN_NAV`). */
const LABELLING_NAV: NavItem[] = [
  ['/', 'Home', 'Your labelling sets (and, for admins, where the study stands)', ['/annotate']],
  ['/tutorials', 'Tutorials', 'Practise Fix the field and Mark bad areas: nothing is saved', ['/tutorial']],
]
const ADMIN_NAV: { section: string; links: NavItem[] }[] = [
  { section: '', links: [
    ['/setup', 'Set up', 'Add slides, find the cores, read the TMA maps'],
    ['/workflow', 'Run', 'Run the analysis and follow its progress'],
  ] },
  { section: 'Check', links: [
    ['/cohort', 'Cores', 'Look at every core and what was found in it', ['/cores', '/slides']],
    ['/calibration', 'Stain thresholds', 'The stain level counted as positive on each slide'],
  ] },
  { section: '', links: [['/results', 'Results', 'Measures by diagnosis group, and the tables to download']] },
  { section: 'Improve accuracy', links: [
    ['/labels', 'Labels', 'Check and correct what detection found'],
    ['/models', 'Models', 'Get, train and choose the detection models'],
  ] },
]
const ADMIN_FOOT: NavItem[] = [['/users', 'Users', 'Everyone with an account: roles, passwords, labelling sets, tokens'],
  ['/settings', 'Settings', 'Study settings, label classes, appearance']]
const FOOT_NAV: NavItem[] = [['/help', 'Help', 'The tutorials, common questions, glossary'], ['/account', 'Account', 'Your name and password']]

const belongs = (pathname: string, to: string) => (to === '/' ? pathname === '/' : pathname === to || pathname.startsWith(`${to}/`))

/** The label of the page at this address, for the browser tab. */
function pageLabel(pathname: string) {
  const item = [...LABELLING_NAV, ...ADMIN_NAV.flatMap((g) => g.links), ...ADMIN_FOOT, ...FOOT_NAV]
    .find(([to, , , more = []]) => [to, ...more].some((m) => belongs(pathname, m)))
  return item?.[1]
}

/** The addresses of the separate annotation pages of earlier releases, now pages of the one app: /annotate is home,
 *  /annotate/tutorial and /annotate/account the tutorial and account pages; a session (/annotate/<name>) stays where it is. */
function oldAddress(pathname: string, search: string) {
  if (pathname === '/annotate' || pathname === '/annotate/') return `/${search}`
  if (pathname === '/annotate/tutorial') return `/tutorial${search}`
  if (pathname === '/annotate/account') return '/account'
  return null
}

/** What anyone may see without signing in: the tutorials (nothing is kept). */
const PUBLIC = (pathname: string) => pathname === '/tutorial' || pathname === '/tutorials'

export default function App() {
  const session = useFetch<Session>('/api/session')
  const { pathname, search } = useLocation()
  useTheme()
  if (!session.data) return <main className="main">{session.error ? <><ErrorNote error={session.error} /><ServerSetting open={remote} /></> : <Loading />}</main>
  const admin = session.data.admin
  const account = session.data.account
  const moved = oldAddress(pathname, search)
  let page
  if (moved) page = <Navigate to={moved} replace />
  else if (account?.must_change) page = <NewPassword />  // an admin reset the password: a new one first
  else if (!account) page = PUBLIC(pathname) ? <GuestShell /> : <ToSignIn />  // one sign-in page for everyone: the main page
  else if (afterSignIn(search)) page = <ToNext />
  else if (admin && !projectId && pathname === '/projects') page = <Projects />
  else if (admin && !projectId) page = <ToLastProject />
  else page = <ProjectProvider fetch={admin && !!projectId}><Shell admin={admin} /></ProjectProvider>
  return <SessionContext.Provider value={{ session: session.data, reload: session.reload }}><ActivityBar />{page}</SessionContext.Provider>
}

/** The page someone asked for before signing in (`?next=` on the main page), when it is an address of this app. */
function afterSignIn(search: string) {
  const next = new URLSearchParams(search).get('next') ?? ''
  return !projectId && next.startsWith('/') && !next.startsWith('//') ? next : null
}

function ToNext() {
  const { search } = useLocation()
  const next = afterSignIn(search)
  useEffect(() => { if (next) location.replace(next) }, [next])
  return <main className="main"><Loading /></main>
}

/** Not signed in: the sign-in on the main page (the app's root), and back to the page asked for afterwards. */
function ToSignIn() {
  const { pathname, search, hash } = useLocation()
  const atRoot = !projectId && pathname === '/'
  const asked = `${appBase}${pathname}${search}${hash}`
  useEffect(() => { if (!atRoot) location.replace(`/?next=${encodeURIComponent(asked)}`) }, [atRoot, asked])
  return atRoot ? <SignIn /> : <main className="main"><Loading /></main>
}

/** The tutorials without an account. */
function GuestShell() {
  return (
    <div className="annotator-shell">
      <AppHeader />
      <main className="main">
        <Routes>
          <Route path="/tutorial" element={<Tutorial />} />
          <Route path="/tutorials" element={<Tutorials />} />
        </Routes>
      </main>
    </div>
  )
}

/** An admin on an address without a project (the main page after signing in, or an address of the previous release such
 *  as /workflow): the same page of the project opened last; no project yet: the projects. */
function ToLastProject() {
  const { pathname, search, hash } = useLocation()
  const projects = useFetch<ProjectsInfo>('/api/projects')
  const target = projects.data?.default ? projectPath(projects.data.default, `${pathname}${search}${hash}`) : null
  useEffect(() => { if (target) location.replace(target) }, [target])
  if (projects.data && !target) return <Projects />
  return <main className="main">{projects.error ? <ErrorNote error={projects.error} /> : <Loading />}</main>
}

function NewPassword() {
  const { session } = useSession()
  return (
    <main className="main" style={{ display: 'grid', placeItems: 'center', minHeight: '80vh' }}>
      <div style={{ width: 420, maxWidth: '100%' }}>
        <p className="secondary">Welcome back, <b>{session.account?.name}</b>. Your password was reset: choose your own to carry on.</p>
        <ChoosePassword />
      </div>
    </main>
  )
}

const SIDEBAR_KEY = 'stainid.sidebar.collapsed'

/** The left menu hidden (a thin strip with its button) or shown, remembered in this browser. */
function useCollapsedSidebar(): [boolean, (on: boolean) => void] {
  const [collapsed, setCollapsed] = useState(() => { try { return window.localStorage.getItem(SIDEBAR_KEY) === '1' } catch { return false } })
  const set = (on: boolean) => {
    setCollapsed(on)
    try { window.localStorage.setItem(SIDEBAR_KEY, on ? '1' : '0') } catch { /* not remembered: still works */ }
  }
  return [collapsed, set]
}

/** The one app, for everyone signed in: the labelling pages (home with my sets, the tutorials, the labelling itself,
 *  help, the account) and, for admins, the study's pages. An annotator's pages need no project in the address (their
 *  sets are of every project); a session's link names its project. */
function Shell({ admin }: { admin: boolean }) {
  const { info, error } = useProject()
  const [collapsed, setCollapsed] = useCollapsedSidebar()
  const { pathname } = useLocation()
  useEffect(() => { document.title = [pageLabel(pathname), info?.name, 'stainID'].filter(Boolean).join(' · ') }, [pathname, info?.name])
  if (admin && error && !info) {
    return <main className="main"><div className="card" style={{ maxWidth: 560 }}><h2>This project cannot be opened</h2><ErrorNote error={error} />
      <a className="btn primary" href="/projects">Go to the projects</a></div></main>
  }
  if (admin && info && !info.configured) {
    return <main className="main"><div className="card" style={{ maxWidth: 560 }}><h2>Not a stainID project</h2>
      <p className="secondary">The folder <code>{info.root}</code> has no <code>stainid.yaml</code>.</p>
      <a className="btn primary" href="/projects">Choose or create a project</a></div></main>
  }

  return (
    <div className={`layout${collapsed ? ' nav-collapsed' : ''}`}>
      <aside className="sidebar" aria-label="Navigation">
        <button className="btn small sidebar-toggle" onClick={() => setCollapsed(!collapsed)} aria-expanded={!collapsed}
          title={collapsed ? 'Show the menu' : 'Hide the menu: more room for the picture'} aria-label={collapsed ? 'Show the menu' : 'Hide the menu'}>
          {collapsed ? '»' : '«'}
        </button>
        {collapsed && (  /* folded: one icon per page, the current one lit; the project's icon unfolds the menu */
          <nav className="nav-strip" aria-label="Pages">
            {admin && <button className="nav-icon-link" onClick={() => setCollapsed(false)} title={info?.name ? `Project: ${info.name} (show the menu to switch)` : 'Project'}
              aria-label="Show the menu to switch the project"><NavIcon of="project" /></button>}
            {[...LABELLING_NAV, ...(admin ? ADMIN_NAV.flatMap((g) => g.links) : [])].map((item) => <StripLink key={item[0]} item={item} />)}
            <span className="nav-strip-gap" />
            {[...(admin ? ADMIN_FOOT : []), ...FOOT_NAV].map((item) => <StripLink key={item[0]} item={item} />)}
          </nav>
        )}
        <div className="sidebar-body">
        <BrandLink />
        {admin && <ProjectSwitcher />}
        {admin && <JobsIndicator />}
        <nav className="nav" aria-label="Pages">
          <div className="nav-group">{LABELLING_NAV.map((item) => <SideLink key={item[0]} item={item} />)}</div>
          {admin && ADMIN_NAV.map((group, i) => (
            <div key={i} className="nav-group">
              {group.section && <div className="nav-section">{group.section}</div>}
              {group.links.map((item) => <SideLink key={item[0]} item={item} />)}
            </div>
          ))}
        </nav>
        <div className="sidebar-foot">
          <nav className="nav-group" aria-label="Settings and help">{[...(admin ? ADMIN_FOOT : []), ...FOOT_NAV].map((item) => <SideLink key={item[0]} item={item} />)}</nav>
        </div>
        </div>
      </aside>
      <main className="main">
        <AppHeader brand={false} />
        <Routes>
          <Route path="/" element={admin ? <Home /> : <AnnotateHome />} />
          <Route path="/tutorial" element={<Tutorial />} />
          <Route path="/tutorials" element={<Tutorials />} />
          <Route path="/account" element={<AccountPage />} />
          <Route path="/help" element={<Help />} />
          <Route path="/annotate/:name" element={<AnnotateField />} />
          <Route path="/annotate/:name/:tile" element={<AnnotateField />} />
          {admin && <>
            <Route path="/workflow" element={<Workflow />} />
            <Route path="/setup/grids" element={<Grids />} />
            <Route path="/setup" element={<Setup />} />
            <Route path="/setup/:screen" element={<Setup />} />
            <Route path="/cohort" element={<Cohort />} />
            <Route path="/cores/:coreId" element={<CoreView />} />
            <Route path="/slides/:slideId" element={<CoreView unit="slide" />} />
            <Route path="/calibration" element={<Calibration />} />
            <Route path="/labels" element={<Labels />} />
            <Route path="/models" element={<Models />} />
            <Route path="/results" element={<Results />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="/users" element={<Users />} />
          </>}
          <Route path="*" element={<Navigate to="/" />} />
        </Routes>
      </main>
    </div>
  )
}

/** A page's icon in the folded menu: its name as the tooltip, lit on the current page (as `SideLink`). */
function StripLink({ item: [to, label, , more = []] }: { item: NavItem }) {
  const { pathname } = useLocation()
  const inside = more.some((m) => belongs(pathname, m))
  return (
    <NavLink to={to} end={to === '/'} title={label} aria-label={label} aria-current={inside ? 'page' : undefined}
      className={({ isActive }) => `nav-icon-link${isActive || inside ? ' active' : ''}`}><NavIcon of={to} /></NavLink>
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
