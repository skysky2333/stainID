import type { ReactNode } from 'react'

export const GROUP_ORDER = ['CT', 'ASYMP', 'AD']
export const GROUP_LABEL: Record<string, string> = { CT: 'Control', ASYMP: 'ASYMAD', AD: 'AD' }
export const GROUP_COLOR: Record<string, string> = { CT: 'var(--series-3)', ASYMP: 'var(--series-1)', AD: 'var(--series-2)' }

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

export function Stat({ value, label }: { value: ReactNode; label: string }) {
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
  if (!group) return <span className="badge">control core</span>
  return (
    <span className="badge">
      <span className="dot" style={{ background: GROUP_COLOR[group] ?? 'var(--text-muted)' }} />
      {GROUP_LABEL[group] ?? group}
    </span>
  )
}

export function StatusBadge({ status }: { status: string }) {
  return <span className={`badge status-${status}`}>{status}</span>
}

export function ErrorNote({ error }: { error: string | null }) {
  return error ? <div className="card" style={{ borderColor: 'var(--critical)', color: 'var(--critical)' }}>{error}</div> : null
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty-state">{children}</div>
}

export function fmt(value: unknown, digits = 3): string {
  if (value === null || value === undefined || value === '') return '—'
  if (typeof value === 'number') {
    if (Number.isInteger(value)) return value.toLocaleString()
    return Math.abs(value) >= 1000 ? value.toFixed(0) : value.toPrecision(digits)
  }
  return String(value)
}
