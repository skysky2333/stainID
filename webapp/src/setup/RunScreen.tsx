import { useState } from 'react'
import { Link } from 'react-router-dom'
import type { Issue, Job, Step } from '../api'
import { api } from '../api'
import { ErrorNote, withError } from '../components'
import { useFetch } from '../hooks'
import { NowRunning } from '../pipeline'
import { MAIN_STAGES } from '../pages/Workflow'
import { Advanced, ScreenFoot } from './parts'
import type { ScreenProps } from './Setup'

const DEVICE: Record<string, string> = { cuda: 'the NVIDIA GPU of this computer', mps: 'the Apple GPU of this computer', cpu: 'the processor of this computer (no GPU found, so the slow step is slow)' }

export function minutes(seconds: number): string {
  if (seconds < 90) return '1 min'
  if (seconds < 5400) return `${Math.round(seconds / 60)} min`
  return `${(seconds / 3600).toFixed(1)} h`
}

export default function RunScreen({ back, go, overview, onChanged }: ScreenProps) {
  const jobs = useFetch<Job[]>('/api/jobs', 3000)
  const steps = useFetch<Step[]>('/api/steps', 5000)
  const issues = useFetch<Issue[]>('/api/setup/checks')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const problems = (issues.data ?? []).filter((i) => i.level === 'problem')
  const estimate = overview?.estimate
  const remaining = overview?.remaining ?? []
  const titles = Object.fromEntries((steps.data ?? []).map((s) => [s.id, s.title]))
  const active = (jobs.data ?? []).filter((j) => j.status === 'running' || j.status === 'queued')
  const run = () => withError(async () => {
    setBusy(true)
    try { await api.post('/api/setup/run', {}); jobs.reload(); steps.reload(); onChanged() } finally { setBusy(false) }
  }, setError)
  const main = (steps.data ?? []).filter((s) => MAIN_STAGES.includes(s.stage))

  return (
    <div className="screen">
      {problems.length > 0 && (
        <div className="warning-note">{problems.length === 1 ? 'One thing' : `${problems.length} things`} on the Checks screen would stop the analysis: {problems.map((p) => p.title).join('; ')}.{' '}
          <button className="link-btn" onClick={() => go('checks')}>Fix it first</button></div>
      )}
      <div className="card next-card">
        {remaining.length ? (
          <>
            <h2 style={{ margin: 0 }}>Run everything</h2>
            <p className="secondary">{remaining.length} steps are left: {remaining.map((s) => s.title).join(', ')}.</p>
            {estimate && estimate.seconds > 0 && (
              <p className="secondary">This takes <b>{estimate.text}</b> on {DEVICE[estimate.device] ?? estimate.device}
                {estimate.slowest ? `; most of it is “${titles[estimate.slowest] ?? estimate.slowest}”` : ''} ({estimate.slides} slides, about {estimate.cores.toLocaleString()} cores and {estimate.tiles.toLocaleString()} tiles). A rough guess.</p>
            )}
            <div className="row">
              <button className="btn primary" disabled={busy} onClick={run}>{busy ? 'Starting…' : 'Run everything'}</button>
              <span className="small muted">Steps start in the right order by themselves. You can close the browser; they keep running.</span>
            </div>
          </>
        ) : active.length ? (
          <><h2 style={{ margin: 0 }}>Running</h2><p className="secondary">Every step is running or waiting its turn. Progress is below and on the <Link to="/workflow">Workflow</Link> page.</p></>
        ) : (
          <><h2 style={{ margin: 0 }}>Everything has run</h2><p className="secondary">The results are on the <Link to="/results">Results</Link> page; look at the detections in <Link to="/cohort">Cohort &amp; cores</Link>.</p></>
        )}
        <ErrorNote error={error} />
      </div>
      {active.length > 0 && <NowRunning jobs={jobs.data ?? []} steps={main} onChanged={() => { jobs.reload(); steps.reload() }} />}
      <Advanced hint="time per step, the Workflow page">
        {estimate && estimate.steps.length > 0 && (
          <table className="compact small">
            <thead><tr><th>Step</th><th className="num">Rough time</th></tr></thead>
            <tbody>{estimate.steps.map((s) => <tr key={s.id}><td>{titles[s.id] ?? s.id}</td><td className="num">{minutes(s.seconds)}</td></tr>)}</tbody>
          </table>
        )}
        <p className="small secondary">Every step can also be run on its own, with its options, from the <Link to="/workflow">Workflow</Link> page. The optional “Outline objects” step is not part of Run everything.</p>
      </Advanced>
      <ScreenFoot back={back} />
    </div>
  )
}
