import { useEffect, useMemo, useRef, useState } from 'react'
import { fmt } from './components'
import { useProject } from './project'

export interface Point { group: string; value: number; facet?: string; id?: string }

interface Tip { x: number; y: number; lines: string[] }

export function quantile(sorted: number[], q: number) {
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
  for (let t = Math.floor(min / nice) * nice; t < max + nice - 1e-9; t += nice) ticks.push(+t.toPrecision(12))
  return ticks
}

/** Ticks for a log axis drawn as log10(value + 1): 0, 1, 3, 10, 30, 100 … until the largest value is covered. */
function logTicks(max: number) {
  const ticks = [0]
  for (let decade = 1; ; decade *= 10) {
    for (const v of [decade, 3 * decade]) {
      ticks.push(Math.log10(v + 1))
      if (ticks[ticks.length - 1] >= max) return ticks
    }
  }
}

/** The width of an element, following window and layout changes. */
function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [width, setWidth] = useState(0)
  useEffect(() => {
    if (!ref.current) return
    const observer = new ResizeObserver(([entry]) => setWidth(Math.floor(entry.contentRect.width)))
    observer.observe(ref.current)
    return () => observer.disconnect()
  }, [])
  return [ref, width] as const
}

/** The groups present in the points, in the project's order (groups not in the settings last). */
export function useGroupOrder(points: { group: string }[]) {
  const { groups: order } = useProject()
  return useMemo(() => {
    const present = new Set(points.map((p) => p.group))
    return [...order.filter((g) => present.has(g)), ...[...present].filter((g) => !order.includes(g)).sort()]
  }, [points, order])
}

/** Dot + box plot of one measure by diagnosis group, optionally side by side for each facet (e.g. brain region).
 *  Each dot is one value; the box is the middle half, the thick line the median, whiskers reach 1.5 box-lengths. */
export function GroupStrip({ points, yLabel, facetLabel, log = false, format = (v) => fmt(v, 3), height = 320 }: {
  points: Point[]; yLabel: string; facetLabel?: string; log?: boolean; format?: (v: number) => string; height?: number
}) {
  const [tip, setTip] = useState<Tip | null>(null)
  const [wrap, available] = useWidth<HTMLDivElement>()
  const { groupColor, groupLabel } = useProject()
  const groups = useGroupOrder(points)
  const facets = useMemo(() => Array.from(new Set(points.map((p) => p.facet ?? ''))).sort(), [points])
  const tx = (v: number) => (log ? Math.log10(Math.max(v, 0) + 1) : v)
  const values = points.map((p) => tx(p.value))
  const ticks = log ? logTicks(Math.max(...values, 1)) : niceTicks(Math.min(...values, 0), Math.max(...values, 1e-9))
  const lo = ticks[0]
  const hi = ticks[ticks.length - 1]
  const faceted = facets.length > 1 || facets[0] !== ''
  const slots = facets.length * groups.length
  const m = { l: 64, r: 12, t: 14, b: faceted ? 62 : 44 }
  const width = Math.max(available, m.l + m.r + slots * 64)
  const slot = (width - m.l - m.r) / Math.max(slots, 1)
  const box = Math.min(40, slot * 0.5)
  const y = (v: number) => m.t + (height - m.t - m.b) * (1 - (v - lo) / (hi - lo || 1))
  const tickText = (t: number) => (log ? format(Math.round(10 ** t - 1)) : format(t))
  const show = (e: React.MouseEvent, p: Point) => setTip({
    x: e.clientX > window.innerWidth - 220 ? e.clientX - 200 : e.clientX + 14, y: e.clientY + 14,
    lines: [p.id ?? '', [groupLabel(p.group), p.facet].filter(Boolean).join(' · '), format(p.value)],
  })

  return (
    <div ref={wrap} className="chart-wrap">
      {available > 0 && (
        <svg width={width} height={height} role="img" aria-label={`${yLabel} by diagnosis group${faceted ? ` and ${facetLabel ?? 'facet'}` : ''}`}>
          {ticks.map((t) => (
            <g key={t}>
              <line x1={m.l} x2={width - m.r} y1={y(t)} y2={y(t)} className="chart-grid" />
              <text x={m.l - 8} y={y(t) + 4} textAnchor="end" className="chart-tick">{tickText(t)}</text>
            </g>
          ))}
          <text x={14} y={(height - m.b + m.t) / 2} transform={`rotate(-90 14 ${(height - m.b + m.t) / 2})`} textAnchor="middle" className="chart-axis-title">
            {yLabel}{log ? ', log scale' : ''}
          </text>
          {facets.map((facet, fi) => {
            const x0 = m.l + slot * fi * groups.length
            return (
              <g key={facet}>
                {fi > 0 && <line x1={x0} x2={x0} y1={m.t} y2={height - m.b + 30} className="chart-divider" />}
                {faceted && <text x={x0 + (slot * groups.length) / 2} y={height - 10} textAnchor="middle" className="chart-facet">{facet || '—'}</text>}
                {groups.map((group, gi) => {
                  const cx = x0 + slot * gi + slot / 2
                  const vals = points.filter((p) => p.group === group && (p.facet ?? '') === facet)
                  const sorted = vals.map((p) => tx(p.value)).sort((a, b) => a - b)
                  const [q1, q2, q3] = [0.25, 0.5, 0.75].map((q) => quantile(sorted, q))
                  const reach = 1.5 * (q3 - q1)
                  const low = sorted.find((v) => v >= q1 - reach) ?? q1
                  const high = [...sorted].reverse().find((v) => v <= q3 + reach) ?? q3
                  const spread = Math.min(slot * 0.75, Math.max(box * 1.3, Math.sqrt(vals.length) * 8))  // more dots, wider cloud
                  return (
                    <g key={group}>
                      {sorted.length > 2 && (
                        <g className="chart-box">
                          <line x1={cx} x2={cx} y1={y(high)} y2={y(q3)} />
                          <line x1={cx} x2={cx} y1={y(q1)} y2={y(low)} />
                          <rect x={cx - box / 2} y={y(q3)} width={box} height={Math.max(1, y(q1) - y(q3))} rx={4} />
                          <line x1={cx - box / 2} x2={cx + box / 2} y1={y(q2)} y2={y(q2)} className="chart-median" />
                        </g>
                      )}
                      {vals.map((p, i) => {
                        const jitter = (((i * 9301 + 49297) % 233280) / 233280 - 0.5) * spread
                        return (
                          <g key={i} onMouseMove={(e) => show(e, p)} onMouseLeave={() => setTip(null)}>
                            <circle cx={cx + jitter} cy={y(tx(p.value))} r={4.5} fill={groupColor(group)} className="chart-dot" />
                            <circle cx={cx + jitter} cy={y(tx(p.value))} r={9} fill="transparent" />
                          </g>
                        )
                      })}
                      <text x={cx} y={height - m.b + 18} textAnchor="middle" className="chart-group">{groupLabel(group)}</text>
                      <text x={cx} y={height - m.b + 32} textAnchor="middle" className="chart-tick">n = {vals.length}</text>
                    </g>
                  )
                })}
              </g>
            )
          })}
        </svg>
      )}
      {tip && (
        <div className="chart-tooltip" style={{ left: tip.x, top: tip.y }}>
          {tip.lines.filter(Boolean).map((line, i) => <div key={i} className={i === 2 ? 'chart-tip-value' : 'muted'}>{line}</div>)}
        </div>
      )}
    </div>
  )
}

/** The colour key for diagnosis groups, in the project's order. */
export function GroupLegend({ groups }: { groups: string[] }) {
  const { groupColor, groupLabel } = useProject()
  return (
    <div className="chart-legend" aria-label="Diagnosis groups">
      {groups.map((g) => <span key={g}><span className="swatch round" style={{ background: groupColor(g) }} />{groupLabel(g)}</span>)}
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
