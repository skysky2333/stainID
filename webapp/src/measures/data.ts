import { createContext, createElement, useCallback, useContext, useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import type { Row } from '../api'
import { api, enc } from '../api'

/** One measure (numeric column) of a results table, as the server describes it (stainid/analysis/dictionary.py). */
export interface Measure { column: string; label: string; unit: string; topic: string; description: string; key: boolean; outline: boolean }
export interface TableColumns {
  columns: string[]; numeric: string[]; groupable: string[]; descriptions: Record<string, string>
  measures: Measure[]; topics: Record<string, string>; rows: number
}
export interface RowsPage { total: number; offset: number; columns: string[]; rows: Row[] }

export const DONOR_REGION_TABLE = 'results_donor_region.csv'
export const CORE_TABLE = 'results_core.csv'

/** What the identifier columns are called on screen. */
export const ID_LABELS: Record<string, string> = {
  sample_region_id: 'Donor-region', core_id: 'Core', donor_id: 'Donor', region: 'Brain region', disease_group: 'Diagnosis group', tma: 'TMA',
}

export const tableUrl = (table: string, part: 'columns' | 'rows' | 'feature' | 'stats') => `/api/tables/${enc(table)}/${part}`

/** "Neuron density (neurons per mm²)" */
export const measureTitle = (m: Measure) => (!m.unit ? m.label : m.label.endsWith(')') ? `${m.label}, ${m.unit}` : `${m.label} (${m.unit})`)
/** Hover text for a measure: its column name in the table and what it means. */
export const measureHover = (m: Measure) => `${m.column}${m.description ? ` — ${m.description}` : ''}`

/** Numbers as people read them: thousands separators, about three significant digits. */
export function formatNumber(value: unknown): string {
  if (value === null || value === undefined || value === '') return '—'
  if (typeof value !== 'number') return String(value)
  if (!Number.isFinite(value)) return '—'
  const size = Math.abs(value)
  if (Number.isInteger(value) || size >= 100) return Math.round(value).toLocaleString()
  if (size !== 0 && size < 0.0001) return value.toExponential(1)
  return value.toLocaleString(undefined, { maximumSignificantDigits: 3 })
}

/* Requests the Results page shares: each address is asked for once while the page is open (the measure list, the plot
   and the numbers all use the same table description), and again only after "Try again" or reopening the page. */
const Cache = createContext<Map<string, Promise<unknown>> | null>(null)

export function ResultsCache({ children }: { children: ReactNode }) {
  const [cache] = useState(() => new Map<string, Promise<unknown>>())
  return createElement(Cache.Provider, { value: cache }, children)
}

export function useShared<T>(url: string | null) {
  const cache = useContext(Cache)
  const [state, setState] = useState<{ url: string | null; data: T | null; error: string | null }>({ url: null, data: null, error: null })
  const [attempt, setAttempt] = useState(0)
  useEffect(() => {
    if (!url) return
    let active = true
    let request = cache?.get(url) as Promise<T> | undefined
    if (!request) {
      request = api.get<T>(url)
      cache?.set(url, request)
      request.catch(() => cache?.delete(url))
    }
    request.then((data) => active && setState({ url, data, error: null }), (e: Error) => active && setState({ url, data: null, error: e.message }))
    return () => { active = false }
  }, [url, attempt, cache])
  const retry = useCallback(() => { if (url) cache?.delete(url); setAttempt((a) => a + 1) }, [url, cache])
  const current = state.url === url
  return { data: current ? state.data : null, error: current ? state.error : null, retry }
}

/** Save rows as a CSV file (for the numbers shown on screen). */
export function downloadCsv(name: string, columns: string[], rows: Row[]) {
  const cell = (v: unknown) => {
    const text = v === null || v === undefined ? '' : String(v)
    return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text
  }
  const text = [columns.map(cell).join(','), ...rows.map((r) => columns.map((c) => cell(r[c])).join(','))].join('\n')
  const link = document.createElement('a')
  link.href = URL.createObjectURL(new Blob([text], { type: 'text/csv' }))
  link.download = name
  link.click()
  setTimeout(() => URL.revokeObjectURL(link.href), 1000)
}
