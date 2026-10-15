import { useEffect, useState } from 'react'
import { api } from './api'
import { ErrorNote, Loaded, Progress, timeAgo, withError } from './components'
import { useFetch } from './hooks'
import { ConfirmDialog } from './labelling'

/** What `/api/project/cache` answers (see `stainid.disk_cache.usage`). */
interface CacheUsage {
  used_bytes: number
  files: number
  limit_bytes: number
  limit_gb: number | null  // this project's own limit; null: the server's default
  default_bytes: number
  kinds: { kind: string; label: string; bytes: number; files: number }[]
  counted: boolean
  busy: 'clearing' | null
  last_cleanup: { when: number; files: number; bytes: number } | null
  in_use_minutes: number
}

const GB = 1e9

/** Bytes in plain units: 512 KB, 48 MB, 2.4 GB. */
export function bytes(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(1)} TB`
  if (n >= GB) return `${(n / GB).toFixed(n >= 10 * GB ? 0 : 1)} GB`
  if (n >= 1e6) return `${Math.round(n / 1e6)} MB`
  return `${Math.max(n ? 1 : 0, Math.round(n / 1e3))} KB`
}

/** Settings → Advanced: the space the pictures the app makes for viewing take, their limit, and Clear all. */
export function CacheSpace() {
  const [polling, setPolling] = useState(false)
  const usage = useFetch<CacheUsage>('/api/project/cache', polling ? 2000 : undefined)
  const busy = usage.data?.busy === 'clearing'
  useEffect(() => { setPolling(busy) }, [busy])
  return (
    <Loaded state={usage} label="Counting the pictures…">
      {(data) => <CacheSpaceCard data={data} onChanged={usage.reload} onClearing={() => setPolling(true)} />}
    </Loaded>
  )
}

function CacheSpaceCard({ data, onChanged, onClearing }: { data: CacheUsage; onChanged: () => void; onClearing: () => void }) {
  const [own, setOwn] = useState(data.limit_gb !== null)
  const [value, setValue] = useState(String(data.limit_gb ?? Math.round(data.default_bytes / GB)))
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [confirm, setConfirm] = useState(false)
  useEffect(() => {  // saved here or by someone else
    setOwn(data.limit_gb !== null)
    setValue(String(data.limit_gb ?? Math.round(data.default_bytes / GB)))
  }, [data.limit_gb, data.default_bytes])
  const wanted = own ? Number(value) : null
  const invalid = own && !(Number(value) >= 0.5)
  const changed = wanted !== data.limit_gb
  const save = async () => {
    setSaving(true)
    if (await withError(() => api.put('/api/project/cache', { limit_gb: wanted }), setError)) onChanged()
    setSaving(false)
  }
  const clear = () => withError(async () => { await api.post('/api/project/cache/clear', {}); onClearing(); onChanged() }, setError)
  const share = data.limit_bytes ? data.used_bytes / data.limit_bytes : 0

  return (
    <div className="cache-space">
      <p className="muted small">
        Zoomable cores, thumbnails, tile and labelling pictures are kept once made, so they open at once the next time. They can always be made
        again from the study, so when they need more than the space below, the ones not looked at for longest are deleted and made again
        when someone opens them. Study data, labels, results and models are never deleted.
      </p>
      <div className="cache-space-total">
        <b>{bytes(data.used_bytes)}</b> <span className="secondary">of {bytes(data.limit_bytes)} used</span>
        <span className="muted small">{data.files.toLocaleString()} pictures and files{data.counted ? '' : ' (still counting)'}</span>
      </div>
      <Progress done={Math.round(share * 1000)} total={1000} />
      {data.last_cleanup && (
        <p className="muted small">Last cleaned up {timeAgo(data.last_cleanup.when)}: {data.last_cleanup.files.toLocaleString()} deleted ({bytes(data.last_cleanup.bytes)}).</p>
      )}

      <fieldset className="radio-list cache-space-limit">
        <legend>Space they may take</legend>
        <label><input type="radio" name="cache-limit" checked={!own} onChange={() => setOwn(false)} />
          The server’s default ({bytes(data.default_bytes)})</label>
        <label><input type="radio" name="cache-limit" checked={own} onChange={() => setOwn(true)} />
          For this study:
          <input type="number" min="0.5" step="1" value={value} disabled={!own} aria-label="Space in GB" style={{ width: 90 }}
            onChange={(e) => setValue(e.target.value)} /> GB</label>
      </fieldset>
      <div className="row">
        <button className="btn" disabled={!changed || invalid || saving} onClick={save}>{saving ? 'Saving…' : 'Save space'}</button>
        {invalid && <span className="level-warning small">At least 0.5 GB.</span>}
        <span className="spacer" />
        <button className="btn danger-outline" disabled={data.busy === 'clearing' || !data.files} onClick={() => setConfirm(true)}>
          {data.busy === 'clearing' ? 'Clearing…' : 'Clear all…'}</button>
      </div>
      <ErrorNote error={error} />

      {data.kinds.length > 0 && (
        <div className="table-wrap" style={{ maxHeight: 'none' }}>
          <table className="cache-space-kinds">
            <thead><tr><th>Kind</th><th className="num">Files</th><th className="num">Space</th></tr></thead>
            <tbody>{data.kinds.map((k) => (
              <tr key={k.kind}><td title={k.kind}>{k.label}</td><td className="num">{k.files.toLocaleString()}</td><td className="num">{bytes(k.bytes)}</td></tr>
            ))}</tbody>
          </table>
        </div>
      )}
      <p className="muted small">Also settable in <code>stainid.yaml</code> (<code>cache_limit_gb</code>); the server’s default with{' '}
        <code>stainid serve --cache-limit 100G</code>. The pictures are in <code>data/cache/</code>.</p>

      {confirm && (
        <ConfirmDialog title="Delete all pictures made for viewing?" confirm="Clear all" danger onClose={() => setConfirm(false)} onConfirm={clear}>
          <p>This frees {bytes(data.used_bytes)}. Each picture is made again when someone opens it, which takes a few seconds per core the
            first time. Pictures someone has open right now are kept. Study data, labels, results and models are not touched.</p>
        </ConfirmDialog>
      )}
    </div>
  )
}
