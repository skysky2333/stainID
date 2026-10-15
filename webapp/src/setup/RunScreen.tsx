import { useState } from 'react'
import { Link } from 'react-router-dom'
import type { Issue, Job, SetupOverview } from '../api'
import { api } from '../api'
import { ErrorNote, Loading, withError } from '../components'
import { useFetch } from '../hooks'
import { NowRunning } from '../pipeline'
import { Advanced, plural, ScreenFoot } from './parts'
import type { ScreenProps } from './Setup'

const DEVICE: Record<string, string> = { cuda: 'the NVIDIA GPU of this computer', mps: 'the Apple GPU of this computer', cpu: 'the processor of this computer (no GPU found, so the slow step is slow)' }

export function minutes(seconds: number): string {
  if (seconds < 90) return '1 min'
  if (seconds < 5400) return `${Math.round(seconds / 60)} min`
  return `${(seconds / 3600).toFixed(1)} h`
}

export default function RunScreen({ back, go, overview, onChanged }: ScreenProps) {
  const jobs = useFetch<Job[]>('/api/jobs', 3000)
  const issues = useFetch<Issue[]>('/api/setup/checks')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [started, setStarted] = useState('')
  const problems = (issues.data ?? []).filter((i) => i.level === 'problem')
  const active = (jobs.data ?? []).filter((j) => j.status === 'running' || j.status === 'queued')
  const run = () => withError(async () => {
    setBusy(true)
    try {
      const result = await api.post<{ queued: Job[] }>('/api/setup/run', {})
      setStarted(`Started ${plural(result.queued.length, 'step')}. They run one after the other; progress is below.`)
      jobs.reload(); onChanged()
    } finally { setBusy(false) }
  }, setError)

  if (!overview || !jobs.data) return <div className="screen"><Loading label="Looking at what is left to run…" /><ScreenFoot back={back} /></div>
  return (
    <div className="screen">
      {problems.length > 0 && (
        <div className="warning-note">{problems.length === 1 ? 'One thing' : `${problems.length} things`} on the Checks step would stop the analysis: {problems.map((p) => p.title).join('; ')}.{' '}
          <button className="link-btn" onClick={() => go('checks')}>Fix it first</button></div>
      )}
      <RunCard overview={overview} running={active.length > 0} busy={busy} onRun={run} />
      {started && <p className="small status-finished" role="status">{started}</p>}
      <ErrorNote error={error} />
      {active.length > 0 && <div className="run-progress"><NowRunning jobs={jobs.data} onChanged={jobs.reload} /></div>}
      <Advanced hint="time per step, every step on its own">
        {overview.estimate.steps.length > 0 && (
          <table className="compact small">
            <thead><tr><th>Step</th><th className="num">Rough time</th></tr></thead>
            <tbody>{overview.estimate.steps.map((s) => <tr key={s.id}><td>{stepTitle(overview, s.id)}</td><td className="num">{minutes(s.seconds)}</td></tr>)}</tbody>
          </table>
        )}
        <p className="small secondary">Every step can also be run on its own, with its options, from the <Link to="/workflow">list of all steps</Link>. The optional “Outline objects” step is not part of Run everything.</p>
      </Advanced>
      <ScreenFoot back={back} />
    </div>
  )
}

const stepTitle = (overview: SetupOverview, id: string) => overview.remaining.find((s) => s.id === id)?.title ?? id

function RunCard({ overview, running, busy, onRun }: { overview: SetupOverview; running: boolean; busy: boolean; onRun: () => void }) {
  const { estimate, remaining } = overview
  if (remaining.length) {
    return (
      <div className="card next-card">
        <h2>Run everything</h2>
        <p className="secondary">{plural(remaining.length, 'step is', 'steps are')} left: {remaining.map((s) => s.title).join(', ')}.</p>
        {estimate.seconds > 0 && (
          <p className="secondary">This takes <b>{estimate.text}</b> on {DEVICE[estimate.device] ?? estimate.device}
            {estimate.slowest ? `; most of it is “${stepTitle(overview, estimate.slowest)}”` : ''} ({plural(estimate.slides, 'slide')}, about {estimate.cores.toLocaleString()} cores
            and {estimate.tiles.toLocaleString()} fields). A rough guess.</p>
        )}
        <div className="row">
          <button className="btn primary" disabled={busy} onClick={onRun}>{busy ? 'Starting…' : 'Run everything'}</button>
          <span className="small muted">The steps start in the right order by themselves. You can close the browser; they keep running.</span>
        </div>
      </div>
    )
  }
  if (running) {
    return (
      <div className="card next-card">
        <h2>Running</h2>
        <p className="secondary">Every step left is running or waiting its turn. Progress is below. You can close the browser; the steps keep running.</p>
      </div>
    )
  }
  return (
    <div className="card next-card done">
      <h2>Everything has run</h2>
      <p className="secondary">The results are on the <Link to="/results">Results</Link> page; look at the detections core by core in <Link to="/cohort">Cohort &amp; cores</Link>.</p>
    </div>
  )
}
