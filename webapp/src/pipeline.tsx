import { useMemo, useState } from 'react'
import type { Job, Step } from './api'
import { api } from './api'
import { CommandLineTag, ErrorNote, Progress, StatusBadge, withError } from './components'

/* ---- shared pieces of the Home and Run (workflow) pages ---- */

/** The three stages of the analysis, as the server names them, and the words the app shows for them. */
export const MAIN_STAGES = ['1 · Set up', '2 · Detect', '3 · Results']
export const STAGE: Record<string, { name: string; about: string }> = {
  '1 · Set up': { name: 'Set up', about: 'Slides, cores, TMA maps, tiles and stain thresholds' },
  '2 · Detect': { name: 'Run', about: 'Nuclei, neurons, plaques and tau found in every tile' },
  '3 · Results': { name: 'Results', about: 'One table row per donor and brain region' },
  '2 · Detect (v2)': { name: 'New detection (v2)', about: 'Candidates, AI calls, outlines and tables, stage by stage (not used by the results yet)' },
}
/** The v2 detection stages: shown on the Run page under the main stages, each run on its own (not by “Run what’s left”). */
export const V2_STAGE = '2 · Detect (v2)'
/** Steps the results do not need (they never count as “left to run”). */
export const OPTIONAL = new Set(['masks'])

export const isActive = (status: string | undefined) => status === 'running' || status === 'queued'

/** One word for where a step stands: running, queued (waiting), failed, outdated (needs attention), done, partial, todo. */
export function stepState(step: Step): string {
  if (step.job && isActive(step.job.status)) return step.job.status
  if (step.job?.status === 'failed' && step.progress.state !== 'done') return 'failed'
  if (step.progress.outdated) return 'outdated'
  return step.progress.state
}

/** Where a group of steps (a stage) stands, in the same words. */
export function groupState(steps: Step[]): string {
  const states = steps.map(stepState)
  const needed = steps.filter((s) => !OPTIONAL.has(s.id)).map(stepState)
  for (const word of ['failed', 'running', 'queued', 'outdated']) if (states.includes(word)) return word
  if (needed.length && needed.every((s) => s === 'done')) return 'done'
  return states.some((s) => s === 'done' || s === 'partial') ? 'partial' : 'todo'
}

/** “26,821 of 44,844 tiles”, or '' for a step that counts nothing. */
export function progressText(step: Step): string {
  const { done, total, unit } = step.progress
  return total ? `${done.toLocaleString()} of ${total.toLocaleString()} ${unit}` : ''
}

export function duration(seconds: number): string {
  if (seconds < 90) return `${Math.max(1, Math.round(seconds))} s`
  if (seconds < 5400) return `${Math.round(seconds / 60)} min`
  if (seconds < 172800) return `${Number((seconds / 3600).toFixed(1))} h`
  return `${Number((seconds / 86400).toFixed(1))} days`
}

/** Time a running job still needs, from its own pace so far (null before it has done anything). */
function timeLeft(job: Job, now: number): number | null {
  const { done, total } = job.progress
  if (job.status !== 'running' || !job.started || done <= 0 || total <= done) return null
  return ((now - job.started) / done) * (total - done)
}

/** Running and waiting work grouped by step (one entry per step, however many parts run), oldest first. */
export function activeByStep(jobs: Job[]): { workflow: string; title: string; jobs: Job[] }[] {
  const groups = new Map<string, Job[]>()
  for (const job of jobs.filter((j) => isActive(j.status)).sort((a, b) => a.created - b.created)) {
    groups.set(job.workflow, [...(groups.get(job.workflow) ?? []), job])
  }
  return [...groups.entries()].map(([workflow, list]) => ({ workflow, title: list[0].title, jobs: list }))
}

/** Ask, then stop these jobs; the question says what happens (finished work is kept, running again continues). */
export async function stopJobs(title: string, jobs: Job[]): Promise<boolean> {
  const terminal = jobs.some((j) => j.source === 'command line')
  const what = jobs.length > 1 ? `all ${jobs.length} parts of “${title}”` : `“${title}”`
  const ok = window.confirm(`Stop ${what}?\n\nWhat is finished is kept; running the step again continues where it stopped.`
    + (terminal ? '\n\nIt was started from a terminal: that command stops too.' : ''))
  if (!ok) return false
  await Promise.all(jobs.map((j) => api.del(`/api/jobs/${j.id}`)))
  return true
}

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

/** How far the running parts of one step have got, in one line: “8,883 of 14,895 fields · 3 parts · running 49 min · about
 *  2.7 h left” (counts from the step's files when there is a step, else from the jobs; `counts` false leaves them out), or
 *  why it waits. */
export function runFacts(jobs: Job[], step?: Step, counts = true): { running: Job[]; done: number; total: number; text: string } {
  const now = Date.now() / 1000
  const running = jobs.filter((j) => j.status === 'running')
  const counted = step !== undefined && step.progress.total > 0
  const done = counted ? step.progress.done : running.reduce((n, j) => n + j.progress.done, 0)
  const total = counted ? step.progress.total : running.reduce((n, j) => n + j.progress.total, 0)
  if (!running.length) return { running, done, total, text: jobs[0]?.waiting || 'Starts in a moment' }
  const lefts = running.map((j) => timeLeft(j, now)).filter((t): t is number => t !== null)
  const started = Math.min(...running.map((j) => j.started ?? now))
  const text = [
    !counts ? '' : total ? `${done.toLocaleString()} of ${total.toLocaleString()}${counted ? ` ${step.progress.unit}` : ''}` : 'starting…',
    running.length > 1 ? `${running.length} parts` : '',
    `running for ${duration(now - started)}`,
    lefts.length ? `about ${duration(Math.max(...lefts))} left` : '',
  ].filter(Boolean).join(' · ')
  return { running, done, total, text }
}

/** What is running or waiting now, one row per step: overall progress (from the step's files when `steps` are given, else
 *  from the jobs), how many parts run, time left, why it waits; Stop asks first. Nothing when idle. */
export function NowRunning({ jobs, steps = [], onChanged }: { jobs: Job[]; steps?: Step[]; onChanged: () => void }) {
  const [open, setOpen] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const groups = activeByStep(jobs)
  if (!groups.length) return null
  const now = Date.now() / 1000
  const byId: Record<string, Step | undefined> = Object.fromEntries(steps.map((s) => [s.id, s]))
  const stop = (title: string, list: Job[]) => withError(async () => { if (await stopJobs(title, list)) onChanged() }, setError)
  return (
    <div className="card now-running" id="running">
      <h2>Now running</h2>
      <ul className="run-list">
        {groups.map(({ workflow, title, jobs: list }) => {
          const { running, done, total, text } = runFacts(list, byId[workflow])
          return (
            <li key={workflow}>
              <div className="run-line">
                <b>{title}</b>
                <StatusBadge status={running.length ? 'running' : 'queued'} />
                <span className="spacer" />
                {list.length > 1 && <button className="link-btn small" onClick={() => setOpen(open === workflow ? null : workflow)}>
                  {open === workflow ? 'Hide parts' : `${list.length} parts`}</button>}
                <button className="btn small" onClick={() => stop(title, list)}>Stop</button>
              </div>
              {running.length > 0 && <Progress done={done} total={total} />}
              <div className="small secondary">{text}</div>
              {open === workflow && (
                <div className="table-wrap run-parts">
                  <table className="compact">
                    <tbody>{list.map((job, i) => {
                      const left = timeLeft(job, now)
                      return (
                        <tr key={job.id}>
                          <td>Part {i + 1} <CommandLineTag job={job} /></td>
                          <td className="num">{job.progress.total ? `${job.progress.done.toLocaleString()} of ${job.progress.total.toLocaleString()}` : job.status === 'queued' ? 'waiting' : 'starting…'}</td>
                          <td className="muted">{left !== null ? `about ${duration(left)} left` : ''}</td>
                          <td className="num"><button className="link-btn small" onClick={() => stop(title, [job])}>Stop this part</button></td>
                        </tr>
                      )
                    })}</tbody>
                  </table>
                </div>
              )}
            </li>
          )
        })}
      </ul>
      <ErrorNote error={error} />
    </div>
  )
}
