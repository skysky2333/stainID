import { useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import type { FsListing, StepOption } from './api'
import { downloadUrl, reveal } from './api'
import { GLOSSARY } from './glossary'
import { useFetch, useStored } from './hooks'
import { useProject } from './project'

export function PageHead({ title, subtitle, children }: { title: string; subtitle?: ReactNode; children?: ReactNode }) {
  return (
    <div className="page-head">
      <div>
        <h1>{title}</h1>
        {subtitle && <p className="subtitle">{subtitle}</p>}
      </div>
      {children && <div className="row">{children}</div>}
    </div>
  )
}

/** "How this page works" panel; remembers whether the reader closed it. */
export function HelpBox({ id, title = 'How this works', children }: { id: string; title?: string; children: ReactNode }) {
  const [open, setOpen] = useStored(`stainid.help.${id}`, true)
  return (
    <div className={`help-box${open ? '' : ' closed'}`}>
      <button className="help-toggle" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className="help-icon">?</span>{title}<span className="spacer" /><span className="muted small">{open ? 'hide' : 'show'}</span>
      </button>
      {open && <div className="help-body">{children}</div>}
    </div>
  )
}

/** A word with a hover definition from the glossary. */
export function Term({ children, term }: { children: ReactNode; term?: string }) {
  const key = term ?? String(children)
  return <span className="term" title={GLOSSARY[key] ?? ''}>{children}</span>
}

export function Stat({ value, label }: { value: ReactNode; label: ReactNode }) {
  return (
    <div className="card stat">
      <div className="value">{value}</div>
      <div className="label">{label}</div>
    </div>
  )
}

export function Progress({ done, total }: { done: number; total: number }) {
  const pct = total ? Math.min(100, (100 * done) / total) : 0
  return (
    <div className="progress" title={`${done} / ${total}`}>
      <div style={{ width: `${pct}%` }} />
    </div>
  )
}

export function GroupBadge({ group }: { group: string | null | undefined }) {
  const { groupColor, groupLabel } = useProject()
  if (!group) return <span className="badge">control core</span>
  return (
    <span className="badge">
      <span className="dot" style={{ background: groupColor(group) }} />
      {groupLabel(group)}
    </span>
  )
}

const STATUS_TEXT: Record<string, string> = {
  done: 'Done', partial: 'Partly done', todo: 'Not started', running: 'Running', queued: 'Waiting to start', failed: 'Failed',
  finished: 'Finished', cancelled: 'Stopped', interrupted: 'Interrupted', blocked: 'Needs earlier steps', outdated: 'Made with an older model',
}

export function StatusBadge({ status }: { status: string }) {
  return <span className={`badge status-${status}`}>{STATUS_TEXT[status] ?? status}</span>
}

export function ErrorNote({ error }: { error: string | null }) {
  return error ? <div className="error-note" role="alert">{error.replace(/^\d{3}: /, '')}</div> : null
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty-state">{children}</div>
}

export function Modal({ title, onClose, children, wide = false }: { title: string; onClose: () => void; children: ReactNode; wide?: boolean }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className={`modal${wide ? ' wide' : ''}`} role="dialog" aria-label={title} onClick={(e) => e.stopPropagation()}>
        <div className="row"><h2 style={{ margin: 0 }}>{title}</h2><span className="spacer" /><button className="btn small" onClick={onClose}>Close</button></div>
        {children}
      </div>
    </div>
  )
}

/** Browse folders on this computer and pick a folder (or a file). */
export function FilePicker({ title, start, pickFiles = false, show = '', onPick, onClose }: {
  title: string; start?: string; pickFiles?: boolean; show?: string; onPick: (path: string) => void; onClose: () => void
}) {
  const [path, setPath] = useState(start ?? '')
  const listing = useFetch<FsListing>(`/api/fs?path=${encodeURIComponent(path)}&show=${show}`)
  const data = listing.data
  const join = (name: string) => `${data?.path.replace(/\/$/, '')}/${name}`
  return (
    <Modal title={title} onClose={onClose} wide>
      <div className="row small">
        {data?.shortcuts.map((s) => <button key={s.path} className="btn small" onClick={() => setPath(s.path)}>{s.label}</button>)}
      </div>
      <div className="row"><code className="path-box">{data?.path}</code></div>
      <ErrorNote error={listing.error} />
      <div className="picker-list">
        {data?.parent && <button className="picker-item" onClick={() => setPath(data.parent!)}>⬑ Up one folder</button>}
        {data?.dirs.map((d) => (
          <button key={d} className="picker-item" onDoubleClick={() => setPath(join(d))} onClick={() => setPath(join(d))}>📁 {d}</button>
        ))}
        {data?.files.map((f) => (
          <button key={f} className="picker-item file" disabled={!pickFiles} onClick={() => pickFiles && onPick(join(f))}>📄 {f}</button>
        ))}
        {data && !data.dirs.length && !data.files.length && <p className="muted small">This folder is empty.</p>}
      </div>
      {!pickFiles && <div className="row"><span className="spacer" /><button className="btn primary" onClick={() => data && onPick(data.path)}>Use this folder</button></div>}
    </Modal>
  )
}

/** A file or folder location with "Show in Finder" and (for files) "Download". */
export function PathLine({ path, exists, label, wrap = false }: { path: string; exists: boolean; label?: ReactNode; wrap?: boolean }) {
  const isFile = /\.[a-z0-9]+$/i.test(path)
  return (
    <div className={`path-line${wrap ? ' wrap-path' : ''}`}>
      <span className={`exists-dot ${exists ? 'yes' : 'no'}`} title={exists ? 'exists' : 'not there yet'} />
      {label && <span className="path-label">{label}</span>}
      <code title={path}>{path}</code>
      {exists && <button className="link-btn" onClick={() => reveal(path)}>Show in Finder</button>}
      {exists && isFile && <a className="link-btn" href={downloadUrl(path)}>Download</a>}
    </div>
  )
}

export function OptionField({ option, value, onChange }: { option: StepOption; value: unknown; onChange: (value: unknown) => void }) {
  const label = <span className="label-line">{option.label}{option.help && <span className="hint" title={option.help}>?</span>}</span>
  if (option.kind === 'bool') {
    return <label className="layer-toggle" title={option.help}><input type="checkbox" checked={Boolean(value)} onChange={(e) => onChange(e.target.checked)} />{label}</label>
  }
  if (option.kind === 'stains') {
    const list = (value as string[]) ?? []
    return (
      <fieldset className="field"><legend>{label}</legend>
        <div className="row">{option.choices.map((c) => (
          <label key={c} className="layer-toggle"><input type="checkbox" checked={list.includes(c)}
            onChange={() => onChange(list.includes(c) ? list.filter((x) => x !== c) : [...list, c])} />{CHOICE_TEXT[c] ?? c}</label>
        ))}</div>
      </fieldset>
    )
  }
  if (option.kind === 'select') {
    return <label className="field">{label}<select value={String(value)} onChange={(e) => onChange(e.target.value)}>
      {option.choices.map((c) => <option key={c} value={c}>{CHOICE_TEXT[c] ?? c}</option>)}</select></label>
  }
  return <label className="field">{label}<input type={option.kind === 'number' ? 'number' : 'text'} value={String(value ?? '')} style={{ width: option.kind === 'number' ? 90 : 200 }}
    onChange={(e) => onChange(option.kind === 'number' ? Number(e.target.value) : e.target.value)} /></label>
}

const CHOICE_TEXT: Record<string, string> = {
  sample_region_id: 'donor-region', core_id: 'core', cellpose: 'Cellpose-SAM', sam: 'Segment Anything', huggingface_home: 'Phikon',
  uncertain: 'include uncertain ones', random: 'random', cpu: 'CPU', mps: 'Apple GPU (mps)',
}

export function fmt(value: unknown, digits = 3): string {
  if (value === null || value === undefined || value === '') return '—'
  if (typeof value === 'number') {
    if (Number.isInteger(value)) return value.toLocaleString()
    return Math.abs(value) >= 1000 ? value.toFixed(0) : value.toPrecision(digits)
  }
  return String(value)
}

export function timeAgo(seconds: number): string {
  const s = Date.now() / 1000 - seconds
  if (s < 60) return 'just now'
  if (s < 3600) return `${Math.round(s / 60)} min ago`
  if (s < 86400) return `${Math.round(s / 3600)} h ago`
  return new Date(seconds * 1000).toLocaleDateString()
}

export async function withError<T>(action: () => Promise<T>, setError: (e: string | null) => void): Promise<T | undefined> {
  setError(null)
  try {
    return await action()
  } catch (e) {
    setError((e as Error).message)
    return undefined
  }
}

