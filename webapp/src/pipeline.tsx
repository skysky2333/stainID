import { useMemo, useState } from 'react'
import type { Job, Step } from './api'
import { api } from './api'
import { Progress, StatusBadge } from './components'

const W = 118
const H = 50
const COL = 134
const ROW = 62

function wrap(text: string, width = 17): string[] {
  const lines: string[] = []
  for (const word of text.split(' ')) {
    const last = lines[lines.length - 1]
    if (last && (last + ' ' + word).length <= width) lines[lines.length - 1] = last + ' ' + word
    else lines.push(word)
  }
  return lines.slice(0, 2)
}

/** The steps as a dependency map: columns are the order things can run in; arrows point from a step to the steps that need it. */
export function PipelineMap({ steps, state, onOpen }: { steps: Step[]; state: (s: Step) => string; onOpen: (id: string) => void }) {
  const [hover, setHover] = useState<string | null>(null)
  const byId = useMemo(() => Object.fromEntries(steps.map((s) => [s.id, s])), [steps])
  const deps = (s: Step) => s.after.filter((a) => byId[a])
  const level = useMemo(() => {
    const memo: Record<string, number> = {}
    const depth = (id: string): number => (memo[id] ??= Math.max(-1, ...deps(byId[id]).map(depth)) + 1)
    steps.forEach((s) => depth(s.id))
    return memo
  }, [steps, byId])
  const position = useMemo(() => {
    const rows: Record<number, number> = {}
    return Object.fromEntries(steps.map((s) => { const col = level[s.id]; const row = (rows[col] = (rows[col] ?? -1) + 1); return [s.id, { x: col * COL + 4, y: row * ROW + 4 }] }))
  }, [steps, level])
  const related = useMemo(() => {
    if (!hover) return null
    const up = new Set<string>(); const down = new Set<string>()
    const climb = (id: string) => deps(byId[id]).forEach((d) => { up.add(d); climb(d) })
    const fall = (id: string) => steps.filter((s) => deps(s).includes(id)).forEach((s) => { down.add(s.id); fall(s.id) })
    climb(hover); fall(hover)
    return { up, down }
  }, [hover, steps, byId])
  const width = Math.max(...steps.map((s) => position[s.id].x)) + W + 8
  const height = Math.max(...steps.map((s) => position[s.id].y)) + H + 8

  return (
    <div className="pipeline-map">
      <svg viewBox={`0 0 ${width} ${height}`} width="100%" style={{ maxWidth: width }} role="img" aria-label="Order of the workflow steps">
        {steps.flatMap((s) => deps(s).map((d) => {
          const a = position[d]; const b = position[s.id]
          const lit = related && ((hover === s.id && related.up.has(d)) || (hover === d && related.down.has(s.id)) || (related.up.has(s.id) && related.up.has(d)) || (related.down.has(s.id) && related.down.has(d)))
          return <path key={`${d}-${s.id}`} className={`map-edge${lit ? ' lit' : ''}`}
            d={`M${a.x + W},${a.y + H / 2} C${a.x + W + 22},${a.y + H / 2} ${b.x - 22},${b.y + H / 2} ${b.x},${b.y + H / 2}`} markerEnd="url(#arrow)" />
        }))}
        <defs><marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L8,4 L0,8 z" className="map-arrow" /></marker></defs>
        {steps.map((s) => {
          const p = position[s.id]
          const dim = related && hover !== s.id && !related.up.has(s.id) && !related.down.has(s.id)
          return (
            <g key={s.id} transform={`translate(${p.x},${p.y})`} className={`map-node node-${state(s)}${dim ? ' dim' : ''}`} tabIndex={0}
              onMouseEnter={() => setHover(s.id)} onMouseLeave={() => setHover(null)} onFocus={() => setHover(s.id)} onBlur={() => setHover(null)}
              onClick={() => onOpen(s.id)} onKeyDown={(e) => e.key === 'Enter' && onOpen(s.id)}>
              <title>{`${s.title}: ${s.summary}`}</title>
              <rect width={W} height={H} rx={9} />
              {wrap(s.title).map((line, i, all) => (
                <text key={i} x={W / 2} y={H / 2 + (i - (all.length - 1) / 2) * 13 + 4} textAnchor="middle">{line}</text>
              ))}
            </g>
          )
        })}
      </svg>
      <div className="map-legend small muted">
        <span><i className="node-done" /> done</span><span><i className="node-partial" /> partly done</span><span><i className="node-running" /> running</span>
        <span><i className="node-queued" /> waiting</span><span><i className="node-failed" /> failed</span><span><i className="node-outdated" /> older model</span><span><i className="node-todo" /> not started</span>
        <span>Arrows: “needs the result of”. Hover a step to see what it needs and what needs it; click to open it.</span>
      </div>
    </div>
  )
}

function duration(seconds: number): string {
  if (seconds < 90) return `${Math.round(seconds)} s`
  if (seconds < 5400) return `${Math.round(seconds / 60)} min`
  return `${(seconds / 3600).toFixed(1)} h`
}

/** Running and waiting jobs: progress, elapsed time, estimated time left, and why a job is still waiting. */
export function NowRunning({ jobs, onChanged }: { jobs: Job[]; onChanged: () => void }) {
  const active = jobs.filter((j) => j.status === 'running' || j.status === 'queued').sort((a, b) => (a.status === b.status ? a.created - b.created : a.status === 'running' ? -1 : 1))
  const now = Date.now() / 1000
  return (
    <div className="card now-running">
      <h2>Now running</h2>
      {active.length ? (
        <table>
          <tbody>{active.map((job) => {
            const elapsed = job.started ? now - job.started : 0
            const { done, total } = job.progress
            const left = job.status === 'running' && done > 0 && total > done ? (elapsed / done) * (total - done) : null
            return (
              <tr key={job.id}>
                <td style={{ width: 220 }}><b>{job.title}</b></td>
                <td style={{ width: 150 }}><StatusBadge status={job.status} /></td>
                <td>
                  {job.status === 'running' ? (
                    <>
                      <Progress done={done} total={total} />
                      <div className="muted small">{total ? `${done} of ${total}` : 'starting…'} · running {duration(elapsed)}{left !== null ? ` · about ${duration(left)} left` : ''}</div>
                    </>
                  ) : <span className="small secondary">{job.waiting || 'Starts in a moment'}</span>}
                </td>
                <td style={{ width: 70 }}><button className="btn small" onClick={() => api.del(`/api/jobs/${job.id}`).then(onChanged)}>Stop</button></td>
              </tr>
            )
          })}</tbody>
        </table>
      ) : <p className="muted small" style={{ margin: 0 }}>Nothing is running. Press Run on a step below; it starts here.</p>}
    </div>
  )
}
