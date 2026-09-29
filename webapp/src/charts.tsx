import { useMemo, useState } from 'react'
import { GROUP_COLOR, GROUP_LABEL, GROUP_ORDER, fmt } from './components'

export interface Point { group: string; value: number; facet?: string; id?: string }

interface Tip { x: number; y: number; text: string }

function quantile(sorted: number[], q: number) {
  if (!sorted.length) return NaN
  const pos = (sorted.length - 1) * q
  const lo = Math.floor(pos)
  const hi = Math.ceil(pos)
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo)
}

function niceTicks(min: number, max: number, count = 5) {
  const span = max - min || 1
  const step = Math.pow(10, Math.floor(Math.log10(span / count)))
  const err = (count * step) / span
  const nice = step * (err <= 0.15 ? 10 : err <= 0.35 ? 5 : err <= 0.75 ? 2 : 1)
  const ticks = []
  for (let t = Math.ceil(min / nice) * nice; t <= max + 1e-9; t += nice) ticks.push(+t.toPrecision(12))
  return ticks
}

/** Strip + box plot of one feature by group (optionally split by facet, e.g. region). */
export function GroupStrip({ points, label, log = false, height = 300 }: { points: Point[]; label: string; log?: boolean; height?: number }) {
  const [tip, setTip] = useState<Tip | null>(null)
  const facets = useMemo(() => Array.from(new Set(points.map((p) => p.facet ?? ''))).sort(), [points])
  const groups = GROUP_ORDER.filter((g) => points.some((p) => p.group === g))
  const tx = (v: number) => (log ? Math.log10(v + 1) : v)
  const values = points.map((p) => tx(p.value))
  const lo = Math.min(...values, 0)
  const hi = Math.max(...values, lo + 1)
  const ticks = niceTicks(lo, hi)
  const width = Math.max(360, facets.length * groups.length * 90 + 70)
  const m = { l: 56, r: 12, t: 12, b: facets.length > 1 ? 44 : 30 }
  const y = (v: number) => m.t + (height - m.t - m.b) * (1 - (v - lo) / (ticks[ticks.length - 1] - lo || 1))
  const slot = (width - m.l - m.r) / (facets.length * groups.length)

  return (
    <div style={{ position: 'relative', overflowX: 'auto' }}>
      <svg width={width} height={height} role="img" aria-label={`${label} by group`}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={m.l} x2={width - m.r} y1={y(t)} y2={y(t)} stroke="var(--border)" strokeWidth={1} />
            <text x={m.l - 8} y={y(t) + 4} textAnchor="end" fontSize={11} fill="var(--text-muted)">{log ? fmt(10 ** t - 1, 2) : fmt(t, 3)}</text>
          </g>
        ))}
        <text x={14} y={height / 2} transform={`rotate(-90 14 ${height / 2})`} textAnchor="middle" fontSize={11.5} fill="var(--text-secondary)">
          {label}{log ? ' (log scale)' : ''}
        </text>
        {facets.map((facet, fi) =>
          groups.map((group, gi) => {
            const cx = m.l + slot * (fi * groups.length + gi) + slot / 2
            const vals = points.filter((p) => p.group === group && (p.facet ?? '') === facet)
            const sorted = vals.map((p) => tx(p.value)).sort((a, b) => a - b)
            const [q1, q2, q3] = [0.25, 0.5, 0.75].map((q) => quantile(sorted, q))
            return (
              <g key={`${facet}-${group}`}>
                {sorted.length > 2 && (
                  <>
                    <rect x={cx - 18} y={y(q3)} width={36} height={Math.max(1, y(q1) - y(q3))} fill="none" stroke="var(--text-muted)" strokeWidth={1.5} rx={4} />
                    <line x1={cx - 18} x2={cx + 18} y1={y(q2)} y2={y(q2)} stroke="var(--text-primary)" strokeWidth={2} />
                  </>
                )}
                {vals.map((p, i) => {
                  const jitter = ((i * 9301 + 49297) % 233280) / 233280 - 0.5
                  return (
                    <circle key={i} cx={cx + jitter * 30} cy={y(tx(p.value))} r={4} fill={GROUP_COLOR[group]} stroke="var(--surface-2)" strokeWidth={1.5}
                      onMouseMove={(e) => setTip({ x: e.clientX + 12, y: e.clientY + 12, text: `${p.id ?? ''} · ${GROUP_LABEL[group]} · ${fmt(p.value, 4)}` })}
                      onMouseLeave={() => setTip(null)} />
                  )
                })}
                <text x={cx} y={height - m.b + 16} textAnchor="middle" fontSize={11.5} fill="var(--text-secondary)">{GROUP_LABEL[group]} ({vals.length})</text>
                {gi === 0 && facet && (
                  <text x={m.l + slot * fi * groups.length + (slot * groups.length) / 2} y={height - 8} textAnchor="middle" fontSize={11.5} fontWeight={600} fill="var(--text-primary)">{facet}</text>
                )}
              </g>
            )
          }),
        )}
      </svg>
      {tip && <div className="chart-tooltip" style={{ left: tip.x, top: tip.y }}>{tip.text}</div>}
    </div>
  )
}

/** Horizontal bars (e.g. DAB thresholds per TMA), one series, labelled values. */
export function Bars({ items, unit = '' }: { items: { label: string; value: number; color?: string }[]; unit?: string }) {
  const max = Math.max(...items.map((i) => i.value), 1e-9)
  return (
    <div style={{ display: 'grid', gap: 6 }}>
      {items.map((item) => (
        <div key={item.label} style={{ display: 'grid', gridTemplateColumns: '90px 1fr 70px', alignItems: 'center', gap: 8, fontSize: 12.5 }}>
          <span className="secondary">{item.label}</span>
          <div style={{ height: 12, background: 'var(--surface-0)', borderRadius: 4 }}>
            <div style={{ width: `${(100 * item.value) / max}%`, height: '100%', background: item.color ?? 'var(--series-1)', borderRadius: 4 }} />
          </div>
          <span style={{ textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{fmt(item.value, 3)}{unit}</span>
        </div>
      ))}
    </div>
  )
}
