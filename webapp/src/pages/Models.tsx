import type { ModelInfo } from '../api'
import { ErrorNote, PageHead } from '../components'
import { useFetch } from '../hooks'

const LABELS: Record<string, string> = {
  n_features: 'features', training_labels: 'training labels', training_n: 'training labels', training_positive_n: 'positive labels',
  probability_threshold: 'accept at P ≥', context_threshold: 'accept at P ≥', identity_threshold: 'accept at P ≥',
  morphotype_threshold: 'compact if inner DAB ≥ × threshold', morphotype_minimum_diameter_um: 'morphotype min diameter (µm)',
  nms_radius_um: 'duplicate radius (µm)', morphotype_training_n: 'morphotype labels',
}

export default function Models() {
  const { data, error } = useFetch<ModelInfo[]>('/api/models')
  return (
    <>
      <PageHead title="Models" subtitle="Frozen models used by the pipelines (paths are set in stainid.yaml). Random forests are trained on the project's reference labels; deep models are used as published." />
      <ErrorNote error={error} />
      <div className="grid grid-2">
        {(data ?? []).map((m) => (
          <div key={m.key} className="card">
            <div className="row"><h2 style={{ margin: 0 }}>{m.key}</h2><span className="spacer" />
              <span className={`badge ${m.exists ? 'status-finished' : 'status-failed'}`}>{m.exists ? 'available' : 'missing'}</span></div>
            <p className="secondary">{m.role}</p>
            <p className="small muted" style={{ overflowWrap: 'anywhere' }}><code>{m.path}</code>{m.size_mb !== undefined && <> · {m.size_mb} MB · {m.modified}</>}</p>
            {m.bundle && (
              <table><tbody>
                {Object.entries(m.bundle).filter(([k]) => k in LABELS).map(([k, v]) => (
                  <tr key={k}><td className="secondary">{LABELS[k]}</td><td className="num">{typeof v === 'number' ? +v.toPrecision(4) : String(v)}</td></tr>
                ))}
              </tbody></table>
            )}
          </div>
        ))}
      </div>
    </>
  )
}
