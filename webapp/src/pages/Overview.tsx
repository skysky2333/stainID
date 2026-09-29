import { Link } from 'react-router-dom'
import type { Job, Summary } from '../api'
import { ErrorNote, GROUP_COLOR, GROUP_LABEL, GROUP_ORDER, PageHead, Progress, Stat, StatusBadge } from '../components'
import { useFetch } from '../hooks'

const STAGES: [keyof Summary['status'][string], string][] = [['detection', 'Detection'], ['nuclei', 'Nuclei'], ['masks', 'Object outlines']]

export default function Overview() {
  const { data, error } = useFetch<Summary>('/api/summary', 15000)
  const jobs = useFetch<Job[]>('/api/jobs', 5000)

  return (
    <>
      <PageHead title="Overview" subtitle="Cohort, pipeline progress and recent jobs." />
      <ErrorNote error={error} />
      {data && (
        <div className="grid" style={{ gap: 20 }}>
          <div className="grid grid-4">
            <Stat value={data.donors} label="donors" />
            <Stat value={data.donor_regions} label="donor-regions" />
            <Stat value={data.cores} label={`cores on ${data.tmas.length} TMAs`} />
            <Stat value={Object.values(data.fields).reduce((a, b) => a + b, 0).toLocaleString()} label="analysis fields (all stains)" />
          </div>
          <div className="grid grid-2">
            <div className="card">
              <h2>Donors by group</h2>
              <div style={{ display: 'grid', gap: 8 }}>
                {GROUP_ORDER.filter((g) => data.groups[g]).map((g) => (
                  <div key={g} className="row">
                    <span className="swatch" style={{ background: GROUP_COLOR[g] }} />
                    <span style={{ width: 90 }}>{GROUP_LABEL[g]}</span>
                    <div style={{ flex: 1, height: 12, background: 'var(--surface-0)', borderRadius: 4 }}>
                      <div style={{ width: `${(100 * data.groups[g]) / data.donors}%`, height: '100%', background: GROUP_COLOR[g], borderRadius: 4 }} />
                    </div>
                    <span style={{ width: 30, textAlign: 'right' }}>{data.groups[g]}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="card">
              <div className="row"><h2>Pipeline progress</h2><span className="spacer" /><Link to="/pipelines">Run pipelines →</Link></div>
              <table>
                <thead><tr><th>Stain</th>{STAGES.map(([, label]) => <th key={label}>{label}</th>)}</tr></thead>
                <tbody>
                  {Object.entries(data.status).map(([stain, stages]) => (
                    <tr key={stain}>
                      <td><b>{stain}</b></td>
                      {STAGES.map(([key]) => (
                        <td key={key} style={{ minWidth: 150 }}>
                          <Progress done={stages[key].done} total={stages[key].total} />
                          <span className="muted small">{stages[key].done}/{stages[key].total} {stages[key].unit}</span>
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <div className="card">
            <div className="row"><h2>Recent jobs</h2><span className="spacer" /><Link to="/pipelines">All jobs →</Link></div>
            {jobs.data && jobs.data.length ? (
              <table>
                <tbody>
                  {jobs.data.slice(0, 6).map((job) => (
                    <tr key={job.id}>
                      <td>{job.title}</td>
                      <td><StatusBadge status={job.status} /></td>
                      <td style={{ width: 200 }}><Progress done={job.progress.done} total={job.progress.total} /></td>
                      <td className="muted small">{new Date(job.created * 1000).toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : <p className="muted">No jobs yet.</p>}
          </div>
        </div>
      )}
    </>
  )
}
