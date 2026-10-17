import { useMemo, useState } from 'react'
import type { Job, Step } from './api'
import { api, queueText } from './api'
import { CommandLineTag, ErrorNote, Progress, StatusBadge, withError } from './components'

/* ---- shared pieces of the Home and Run (workflow) pages ---- */

/** The three stages of the analysis, as the server names them, and the words the app shows for them. */
export const MAIN_STAGES = ['1 · Set up', '2 · Detect', '3 · Results']
/** The first detection (v1): still runnable from the Run page (under Advanced), never by “Run what’s left”. */
export const LEGACY_STAGE = 'Legacy (v1)'
export const STAGE: Record<string, { name: string; about: string }> = {
  '1 · Set up': { name: 'Set up', about: 'Slides, cores, TMA maps, tiles and stain thresholds' },
  '2 · Detect': { name: 'Detect', about: 'Nuclei, then every candidate, detection’s call on it and its outline, core by core' },
  '3 · Results': { name: 'Results', about: 'One table row per donor and brain region' },
  [LEGACY_STAGE]: { name: 'Legacy detection (v1)', about: 'The first detection and its tables: still runnable, not run by “Run what’s left”' },
}
/** Steps the results do not need (they never count as “left to run”). */
export const OPTIONAL = new Set<string>()

export const isActive = (status: string | undefined) => status === 'running' || status === 'queued' || status === 'paused'

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

/* ---- the flow chart of the Run page: the data flowing from slide scans to results tables ---- */

/** What a step hands on to the next one: the labels of what it makes that the other reads. */
function handedOn(from: Step, to: Step): string[] {
  const reads = new Set([...to.needs, ...to.uses].map((r) => r.id))
  return from.produces.filter((r) => reads.has(r.id)).map((r) => r.label)
}

/** How long the step's last finished run took (all its parts), or ''. */
function took(step: Step): string {
  const job = step.job
  if (!job || job.status !== 'finished' || !job.started || !job.ended) return ''
  return duration(job.ended - job.started)
}

const STATE_WORD: Record<string, string> = {
  done: 'done', partial: 'part done', running: 'running', queued: 'waiting', failed: 'stopped with an error', outdated: 'out of date', todo: 'not started',
}

/** One step of the flow chart: where it stands, its counts, what it makes, how long it took. */
function FlowNode({ step, state, lit, dim, chosen, onOpen, onHover }: {
  step: Step; state: string; lit: boolean; dim: boolean; chosen: boolean; onOpen: () => void; onHover: (on: boolean) => void
}) {
  const parts = step.active ?? (step.job && isActive(step.job.status) ? [step.job] : [])
  const running = parts.length ? runFacts(parts, step, false) : null
  const { done, total, unit } = step.progress
  const outdated = step.progress.outdated
  const time = took(step)
  const makes = step.produces.map((r) => r.label).join(', ')
  return (
    <button type="button" className={`flow-node node-${state}${lit ? ' lit' : ''}${dim ? ' dim' : ''}${chosen ? ' chosen' : ''}`} onClick={onOpen}
      onMouseEnter={() => onHover(true)} onMouseLeave={() => onHover(false)} onFocus={() => onHover(true)} onBlur={() => onHover(false)}
      aria-pressed={chosen} title={`${step.title}: ${step.summary}`}>
      <span className="flow-node-head"><span className="flow-dot" aria-hidden="true" /><b>{step.title}</b></span>
      <span className="flow-state small">{STATE_WORD[state] ?? state}{time && state === 'done' ? ` · took ${time}` : ''}</span>
      {total > 0 && <>
        <span className="flow-bar" aria-hidden="true"><span style={{ width: `${Math.min(100, (100 * done) / total)}%` }} /></span>
        <span className="flow-count small">{done.toLocaleString()} of {total.toLocaleString()} {unit}</span>
      </>}
      {outdated && <span className="flow-warn small">{step.progress.note || 'some out of date'}</span>}
      {running && <span className="flow-running small">{running.text}</span>}
      {makes && <span className="flow-makes small">makes <b>{makes}</b></span>}
    </button>
  )
}

/** An arrow between two steps, labelled with the data passed on. */
function FlowArrow({ labels, lit, down = false }: { labels: string[]; lit: boolean; down?: boolean }) {
  return (
    <span className={`flow-arrow${lit ? ' lit' : ''}${down ? ' down' : ''}`} aria-hidden="true">
      {labels.length > 0 && <span className="flow-arrow-label">{labels.slice(0, 2).join(', ')}{labels.length > 2 ? ' …' : ''}</span>}
      <span className="flow-arrow-line" />
    </span>
  )
}

/** The steps as a flow chart, stage by stage: each step's state, counts, what it makes and how long it took, with arrows
 *  labelled with the data each passes on. Hover a step to light up what it needs and what needs it; click one to open its
 *  details (`onOpen`). */
export function PipelineMap({ steps, state, stages, chosen, onOpen }: {
  steps: Step[]; state: (s: Step) => string; stages: string[]; chosen?: string | null; onOpen: (id: string) => void
}) {
  const [hover, setHover] = useState<string | null>(null)
  const byId = useMemo(() => Object.fromEntries(steps.map((s) => [s.id, s])), [steps])
  const deps = (s: Step) => s.after.filter((a) => byId[a])
  const related = useMemo(() => {
    if (!hover || !byId[hover]) return null
    const up = new Set<string>(); const down = new Set<string>()
    const climb = (id: string) => deps(byId[id]).forEach((d) => { if (!up.has(d)) { up.add(d); climb(d) } })
    const fall = (id: string) => steps.filter((s) => deps(s).includes(id)).forEach((s) => { if (!down.has(s.id)) { down.add(s.id); fall(s.id) } })
    climb(hover); fall(hover)
    return { up, down }
  }, [hover, steps, byId])  // eslint-disable-line react-hooks/exhaustive-deps
  const lanes = stages.map((stage) => ({ stage, steps: steps.filter((s) => s.stage === stage) })).filter((l) => l.steps.length)
  const isLit = (id: string) => !!related && (id === hover || related.up.has(id) || related.down.has(id))
  const edgeLit = (a: Step, b: Step) => !!related && isLit(a.id) && isLit(b.id) && deps(b).includes(a.id)
  /** What one lane hands on to the next: everything made in it that a step of the next reads. */
  const between = (from: Step[], to: Step[]) => [...new Set(from.flatMap((a) => to.flatMap((b) => handedOn(a, b))))]

  return (
    <div className="flow" role="group" aria-label="How the data flows through the steps">
      {lanes.map((lane, li) => (
        <div key={lane.stage} className="flow-lane-wrap">
          {li > 0 && <FlowArrow down labels={between(lanes[li - 1].steps, lane.steps)}
            lit={!!related && lanes[li - 1].steps.some((a) => lane.steps.some((b) => edgeLit(a, b)))} />}
          <section className="flow-lane">
            <h3 className="flow-lane-name">{STAGE[lane.stage]?.name ?? lane.stage}</h3>
            <div className="flow-row">
              {lane.steps.map((s, i) => {
                const previous = lane.steps[i - 1]
                return (
                  <span key={s.id} className="flow-cell">
                    {previous && <FlowArrow labels={handedOn(previous, s)} lit={edgeLit(previous, s)} />}
                    <FlowNode step={s} state={state(s)} lit={isLit(s.id)} dim={!!related && !isLit(s.id)} chosen={chosen === s.id}
                      onOpen={() => onOpen(s.id)} onHover={(on) => setHover(on ? s.id : null)} />
                  </span>
                )
              })}
            </div>
          </section>
        </div>
      ))}
      <div className="map-legend small muted">
        <span><i className="node-done" /> done</span><span><i className="node-partial" /> part done</span><span><i className="node-running" /> running</span>
        <span><i className="node-queued" /> waiting</span><span><i className="node-failed" /> error</span><span><i className="node-outdated" /> out of date</span>
        <span><i className="node-todo" /> not started</span>
        <span>Arrows carry what one step makes to the next. Hover a step to see what it needs and what needs it; click it for its details.</span>
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
  if (!running.length) return { running, done, total, text: jobs[0] ? queueText(jobs[0]) : 'Starts in a moment' }
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
