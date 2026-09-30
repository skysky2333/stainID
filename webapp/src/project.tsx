import { createContext, useContext, useMemo } from 'react'
import type { ReactNode } from 'react'
import type { ProjectInfo } from './api'
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

export function ProjectProvider({ children }: { children: ReactNode }) {
  const { data, error, reload } = useFetch<ProjectInfo>('/api/project')
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
