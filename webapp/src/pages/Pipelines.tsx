import { useState } from 'react'
import type { Job } from '../api'
import { api } from '../api'
import { ErrorNote, PageHead, Progress, StatusBadge } from '../components'
import { useFetch } from '../hooks'

type Option = { key: string; label: string; kind: 'stains' | 'stain' | 'select' | 'number' | 'text'; choices?: string[]; value: unknown }
interface Step { workflow: string; title: string; description: string; options: Option[] }

const device = (value: string): Option => ({ key: 'device', label: 'Device', kind: 'select', choices: ['cpu', 'mps'], value })
const shards: Option[] = [
  { key: 'shard_index', label: 'Shard', kind: 'number', value: 0 },
  { key: 'shard_count', label: 'of', kind: 'number', value: 1 },
]
const STEPS: Step[] = [
  { workflow: 'select', title: '0 · Field selection', description: 'Spatially balanced native-resolution fields per stain-core (tissue and focus only); the first N form the analysis manifest.',
    options: [{ key: 'fields_per_core', label: 'Fields per core', kind: 'number', value: 8 }, { key: 'primary', label: 'Primary', kind: 'number', value: 4 }] },
  { workflow: 'calibrate', title: '1 · Slide calibration', description: 'Per-slide DAB thresholds from pooled tissue pixels.',
    options: [{ key: 'output', label: 'Output file', kind: 'text', value: 'data/analysis/slide_dab_calibration_recomputed.csv' }] },
  { workflow: 'nuclei', title: '2 · Nuclei', description: 'Cellpose-SAM nuclei on the hematoxylin counterstain (needed before AT8 and for 6E10 niche features).',
    options: [{ key: 'stain', label: 'Stains', kind: 'stains', choices: ['AT8', '6E10', 'NeuN'], value: ['AT8'] }, device('mps'), { key: 'batch_size', label: 'Batch', kind: 'number', value: 8 }, ...shards] },
  { workflow: 'neun', title: '3a · NeuN neurons', description: 'Stain-contour + Cellpose-SAM candidates, context random forest (P ≥ 0.5).',
    options: [device('cpu'), { key: 'threads', label: 'Threads', kind: 'number', value: 8 }, ...shards] },
  { workflow: 'fields', title: '3b · 6E10 plaques / AT8 tau', description: 'Plaque segmentation + identity ensemble + morphotype; tau+ neurons + thread network.',
    options: [{ key: 'stain', label: 'Stains', kind: 'stains', choices: ['6E10', 'AT8'], value: ['6E10', 'AT8'] }, ...shards] },
  { workflow: 'masks', title: '4 · Object outlines', description: 'SAM ViT-B outlines prompted at every accepted object (shape measurements).',
    options: [{ key: 'stain', label: 'Stain', kind: 'stain', choices: ['NeuN', '6E10', 'AT8'], value: 'NeuN' }, device('cpu'), ...shards] },
  { workflow: 'neun_merge', title: '5a · Merge NeuN tables', description: 'Merge finished NeuN fields into cohort tables.', options: [] },
  { workflow: 'aggregate_fields', title: '5b · Aggregate field features', description: 'Core / donor-region plaque and tau features.',
    options: [{ key: 'level', label: 'Level', kind: 'select', choices: ['sample_region_id', 'core_id'], value: 'sample_region_id' }] },
  { workflow: 'aggregate_masks', title: '5c · Aggregate mask features', description: 'Core / donor-region shape features from outlines.',
    options: [{ key: 'level', label: 'Level', kind: 'select', choices: ['sample_region_id', 'core_id'], value: 'sample_region_id' }] },
]

export default function Pipelines() {
  const jobs = useFetch<Job[]>('/api/jobs', 3000)
  const [openLog, setOpenLog] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const submit = async (workflow: string, options: Record<string, unknown>) => {
    setError(null)
    try {
      await api.post('/api/jobs', { workflow, options })
      jobs.reload()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  return (
    <>
      <PageHead title="Pipelines & jobs" subtitle="Every step is resumable: finished cores and fields are skipped. Heavy jobs (GPU/CPU-bound) run at most two at a time; the rest wait in the queue." />
      <ErrorNote error={error} />
      <div className="grid grid-2" style={{ marginBottom: 20 }}>
        {STEPS.map((step) => <StepCard key={step.workflow} step={step} onRun={submit} />)}
      </div>
      <div className="card">
        <h2>Jobs</h2>
        {jobs.data && jobs.data.length ? (
          <table>
            <thead><tr><th>Job</th><th>Command</th><th>Status</th><th style={{ width: 220 }}>Progress</th><th>Started</th><th /></tr></thead>
            <tbody>
              {jobs.data.map((job) => (
                <tr key={job.id}>
                  <td>{job.title}<div className="muted small">{job.id}</div></td>
                  <td><code>stainid {job.argv.join(' ')}</code></td>
                  <td><StatusBadge status={job.status} /></td>
                  <td><Progress done={job.progress.done} total={job.progress.total} /><div className="muted small" style={{ maxWidth: 260, overflow: 'hidden', textOverflow: 'ellipsis' }}>{job.progress.last_line}</div></td>
                  <td className="muted small">{job.started ? new Date(job.started * 1000).toLocaleTimeString() : '—'}</td>
                  <td className="row">
                    <button className="btn small" onClick={() => setOpenLog(openLog === job.id ? null : job.id)}>Log</button>
                    {(job.status === 'running' || job.status === 'queued') && <button className="btn small" onClick={() => api.del(`/api/jobs/${job.id}`).then(jobs.reload)}>Cancel</button>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <p className="muted">No jobs yet — start one above.</p>}
        {openLog && <JobLog id={openLog} />}
      </div>
    </>
  )
}

function StepCard({ step, onRun }: { step: Step; onRun: (workflow: string, options: Record<string, unknown>) => void }) {
  const [values, setValues] = useState<Record<string, unknown>>(Object.fromEntries(step.options.map((o) => [o.key, o.value])))
  const set = (key: string, value: unknown) => setValues({ ...values, [key]: value })
  return (
    <div className="card">
      <div className="row"><h2 style={{ margin: 0 }}>{step.title}</h2><span className="spacer" /><button className="btn primary" onClick={() => onRun(step.workflow, values)}>Run</button></div>
      <p className="secondary small">{step.description}</p>
      <div className="row">
        {step.options.map((o) => (
          <label key={o.key} className="field">
            {o.label}
            {o.kind === 'stains' ? (
              <span className="row" style={{ gap: 8 }}>
                {o.choices!.map((c) => {
                  const current = (values[o.key] as string[]) ?? []
                  return <label key={c} className="layer-toggle"><input type="checkbox" checked={current.includes(c)}
                    onChange={() => set(o.key, current.includes(c) ? current.filter((x) => x !== c) : [...current, c])} />{c}</label>
                })}
              </span>
            ) : o.kind === 'select' || o.kind === 'stain' ? (
              <select value={String(values[o.key])} onChange={(e) => set(o.key, e.target.value)}>{o.choices!.map((c) => <option key={c}>{c}</option>)}</select>
            ) : (
              <input type={o.kind === 'number' ? 'number' : 'text'} value={String(values[o.key])} style={{ width: o.kind === 'number' ? 70 : 320 }}
                onChange={(e) => set(o.key, o.kind === 'number' ? Number(e.target.value) : e.target.value)} />
            )}
          </label>
        ))}
      </div>
    </div>
  )
}

function JobLog({ id }: { id: string }) {
  const log = useFetch<string>(`/api/jobs/${id}/log`, 3000)
  return <div className="log" style={{ marginTop: 12 }}>{log.data || '(empty)'}</div>
}
