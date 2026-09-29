import { useCallback, useEffect, useState } from 'react'
import { api } from './api'

export function useFetch<T>(url: string | null, refreshMs?: number) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [tick, setTick] = useState(0)
  const reload = useCallback(() => setTick((t) => t + 1), [])

  useEffect(() => {
    if (!url) return
    let active = true
    setLoading(true)
    api.get<T>(url)
      .then((value) => { if (active) { setData(value); setError(null) } })
      .catch((reason: Error) => { if (active) setError(reason.message) })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [url, tick])

  useEffect(() => {
    if (!refreshMs) return
    const timer = setInterval(reload, refreshMs)
    return () => clearInterval(timer)
  }, [refreshMs, reload])

  return { data, error, loading, reload }
}

export function useStored<T>(key: string, initial: T): [T, (value: T) => void] {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(key)
      return raw === null ? initial : (JSON.parse(raw) as T)
    } catch {
      return initial
    }
  })
  const update = (next: T) => {
    setValue(next)
    try { localStorage.setItem(key, JSON.stringify(next)) } catch { /* storage unavailable */ }
  }
  return [value, update]
}
