import type { ProjectInfo } from '../api'
import { ErrorNote, PageHead } from '../components'
import { useFetch } from '../hooks'

export default function Project() {
  const { data, error } = useFetch<ProjectInfo>('/api/project')
  return (
    <>
      <PageHead title="Project" subtitle={data ? <>Configuration from <code>{data.root}/stainid.yaml</code> (defaults where absent). Edit the file and restart <code>stainid serve</code> to change paths.</> : null} />
      <ErrorNote error={error} />
      {data && (
        <div className="grid">
          <div className="card">
            <table><tbody>
              <tr><td className="secondary">Name</td><td>{data.name}</td></tr>
              <tr><td className="secondary">Root</td><td><code>{data.root}</code></td></tr>
              <tr><td className="secondary">Pixel size</td><td>{data.pixel_size_um} µm</td></tr>
              <tr><td className="secondary">Stains</td><td>{Object.entries(data.stains).map(([s, p]) => `${s} → ${p}`).join(' · ')}</td></tr>
            </tbody></table>
          </div>
          {Object.entries(data.paths).map(([section, entries]) => (
            <div key={section} className="card">
              <h2 style={{ textTransform: 'capitalize' }}>{section}</h2>
              <table><tbody>
                {Object.entries(entries).map(([key, v]) => (
                  <tr key={key}><td style={{ width: 180 }}>{key}</td><td><span className={`badge ${v.exists ? 'status-finished' : 'status-failed'}`}>{v.exists ? 'found' : 'missing'}</span></td><td><code>{v.path}</code></td></tr>
                ))}
              </tbody></table>
            </div>
          ))}
        </div>
      )}
    </>
  )
}
