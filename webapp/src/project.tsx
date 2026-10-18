import { createContext, useContext, useMemo } from 'react'
import type { ReactNode } from 'react'
import { useLocation } from 'react-router-dom'
import type { ProjectInfo, ProjectsInfo } from './api'
import { projectId, projectPath } from './api'
import { useFetch } from './hooks'

interface ProjectContextValue {
  info: ProjectInfo | null
  error: string | null
  reload: () => void
  groups: string[]
  groupLabel: (code: string) => string
  groupColor: (code: string) => string
  tmaName: (tma: string | number) => string
}

const Context = createContext<ProjectContextValue | null>(null)

/** The page's project (`fetch` false: none to read — an annotator's pages, which need only the labelling routes). */
export function ProjectProvider({ children, fetch = true }: { children: ReactNode; fetch?: boolean }) {
  const { data, error, reload } = useFetch<ProjectInfo>(fetch ? '/api/project' : null)
  const value = useMemo<ProjectContextValue>(() => {
    const groups = data?.groups ?? []
    const byCode = Object.fromEntries(groups.map((g, i) => [g.code, { ...g, slot: g.color ?? i + 1 }]))
    return {
      info: data,
      error,
      reload,
      groups: groups.map((g) => g.code),
      groupLabel: (code) => byCode[code]?.label ?? code,
      groupColor: (code) => (byCode[code] ? `var(--series-${byCode[code].slot})` : 'var(--text-muted)'),
      tmaName: (tma) => `${data?.tma_prefix ?? ''}${tma}`,
    }
  }, [data, error, reload])
  return <Context.Provider value={value}>{children}</Context.Provider>
}

export function useProject() {
  const value = useContext(Context)
  if (!value) throw new Error('useProject must be used inside ProjectProvider')
  return value
}

const ALL = '__all'

/** The page to show in another project: the same section (a core or a setup screen of this project has no twin there). */
function samePage(pathname: string) {
  const section = pathname.split('/')[1] ?? ''
  return section === 'cores' ? '/cohort' : section ? `/${section}` : '/'
}

/** The current project, and a menu to switch to another one or to the list of all projects. */
export function ProjectSwitcher() {
  const { info } = useProject()
  const list = useFetch<ProjectsInfo>('/api/projects')
  const { pathname } = useLocation()
  const projects = list.data?.projects.filter((p) => p.exists) ?? []
  const others = projects.filter((p) => p.id !== projectId)
  const go = (id: string) => { if (id) location.assign(id === ALL ? '/projects' : projectPath(id, samePage(pathname))) }
  return (
    <div className="project-chip" title={info?.root}>
      <span className="muted small">Project</span>
      <span className="project-current">{info?.name ?? '…'}</span>
      <select className="project-select" value="" aria-label="Switch to another project" onChange={(e) => go(e.target.value)}>
        <option value="">Switch project…</option>
        {others.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
        <option value={ALL}>All projects…</option>
      </select>
    </div>
  )
}
