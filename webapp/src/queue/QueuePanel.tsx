import { useState } from 'react'
import type { Job, QueueOverview } from '../api'
import { api, queueText } from '../api'
import { ErrorNote, Loaded, Progress, StatusBadge, withError } from '../components'
import { useFetch } from '../hooks'
import { duration } from '../pipeline'

/** The one job queue of this computer (every project's steps, downloads, public datasets), highest priority first:
 * each job's place, progress and time left, with Pause / Resume / Cancel / Move up, and one switch for all background work. */
export function QueuePanel() {
  const queue = useFetch<QueueOverview>('/api/queue', 3000)
  const [error, setError] = useState<string | null>(null)
  const act = (job: Job, action: string, body: Record<string, unknown> = {}) =>
    withError(async () => { await api.post(`/api/jobs/${job.id}/${action}`, body); queue.reload() }, setError)
  const all = (action: 'pause-all' | 'resume-all') => withError(async () => { await api.post(`/api/queue/${action}`, {}); queue.reload() }, setError)
  return (
    <section className="card queue-card" aria-label="Job queue">
      <Loaded state={queue} label="Reading the job queue…">
        {(data) => {
          const live = data.classes.filter((c) => c.jobs.length)
          const { budget, used } = data
          return (
            <>
              <div className="row">
                <h2 className="queue-title">Queue</h2>
                <span className="muted small">
                  {used.cpus} of {budget.cpus} processors · {used.gpus} of {budget.gpus} GPUs · {data.memory
                    ? `RAM: ${Math.round(data.memory.used_gb)} of ${Math.round(data.memory.total_gb)} GB used · ${Math.round(data.memory.free_for_jobs_gb)} free for jobs`
                    : `${Math.round(used.ram_gb)} of ${Math.round(budget.ram_gb)} GB in use`}
                </span>
                <span className="spacer" />
                {data.paused_all
                  ? <button className="btn small primary" onClick={() => all('resume-all')}>Resume all background work</button>
                  : <button className="btn small" onClick={() => all('pause-all')} title="Every running job stops at its next unit boundary and nothing new starts until you resume">
                      Pause all background work</button>}
              </div>
              {!data.led && <div className="warning-note small">Nothing runs the queue right now: it starts again with the app or with <code>stainid jobs serve</code>.</div>}
              {(data.alerts ?? []).map((a) => <div key={a.id} className="warning-note small">Queue watchdog: {a.text}</div>)}
              {data.paused_all && <div className="muted small">All background work is paused. Running jobs stop at their next unit boundary; their finished units are kept.</div>}
              {live.length === 0 && <div className="muted small">Nothing is running or waiting.</div>}
              {live.map((group) => (
                <div key={group.id} className="queue-class">
                  <div className="queue-class-title small"><b>{group.rank}. {group.title}</b> <span className="muted">· {group.jobs.length}</span></div>
                  <ul className="queue-list">
                    {group.jobs.map((job) => <QueueRow key={job.id} job={job} onAct={act} />)}
                  </ul>
                </div>
              ))}
              <ErrorNote error={error} />
            </>
          )
        }}
      </Loaded>
    </section>
  )
}

function QueueRow({ job, onAct }: { job: Job; onAct: (job: Job, action: string, body?: Record<string, unknown>) => void }) {
  const { done, total } = job.progress
  const running = job.status === 'running'
  const waiting = job.status === 'queued' || job.status === 'paused'
  const chunks = job.chunks
  const chunkText = chunks ? `${chunks.finished} of ${chunks.total} chunks done${chunks.running ? ` · ${chunks.running} running` : ''}` : ''
  const facts = running
    ? [job.stopping ? (job.stopping === 'cancel' ? 'Stopping…' : 'Pausing at the next unit…') : '', chunkText,
       total ? `${done.toLocaleString()} of ${total.toLocaleString()} done` : 'Starting…',
       job.eta_s ? `about ${duration(job.eta_s)} left` : ''].filter(Boolean).join(' · ')
    : [job.queue?.position ? `#${job.queue.position} in its class` : '', chunkText, queueText(job)].filter(Boolean).join(' · ')
  const sizeText = job.size ? `${job.size}${job.estimate_s ? ` · ~${duration(job.estimate_s)}` : ''}` : ''
  const sizeTitle = job.needs ? `Needs ${job.needs.cpus} processors${job.needs.gpus ? `, ${job.needs.gpus} GPU` : ''}, ${job.needs.ram_gb} GB`
    + `${job.io === 'heavy' ? ', heavy disk use' : ''}${job.needs.network ? ', network only' : ''}` : ''

  return (
    <li className="queue-row">
      <div className="queue-main">
        <div className="row small">
          <StatusBadge status={job.status} />
          <b className="ellipsis">{job.title}</b>
          {job.project_name && <span className="muted ellipsis">{job.project_name}</span>}
          {sizeText && <span className={`badge size-${job.size}`} title={sizeTitle}>{sizeText}</span>}
        </div>
        {running && total > 0 && <Progress done={done} total={total} />}
        <div className="muted small ellipsis" title={facts}>{facts}</div>
        {running && job.running_chunks && job.running_chunks.length > 0 && (
          <div className="muted small ellipsis">Now: {job.running_chunks.map((t) => t.split(' · ').pop()).join(', ')}</div>)}
      </div>
      <div className="queue-actions small">
        {running && !job.stopping && <button className="link-btn" onClick={() => onAct(job, 'pause')}>Pause</button>}
        {(job.status === 'paused' || (running && job.stopping === 'pause')) && <button className="link-btn" onClick={() => onAct(job, 'resume')}>Resume</button>}
        {waiting && job.status === 'queued' && <button className="link-btn" onClick={() => onAct(job, 'pause')}>Pause</button>}
        {waiting && (job.queue?.position ?? 0) > 1 && <button className="link-btn" onClick={() => onAct(job, 'move', { where: 'up' })}>Move up</button>}
        {job.stopping !== 'cancel' && <button className="link-btn" onClick={() => onAct(job, 'cancel')}>Cancel</button>}
      </div>
    </li>
  )
}
