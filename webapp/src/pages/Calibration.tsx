import { useNavigate } from 'react-router-dom'
import type { Row } from '../api'
import { api } from '../api'
import { Bars } from '../charts'
import { ErrorNote, PageHead, fmt } from '../components'
import { useFetch } from '../hooks'

const STAIN_COLOR: Record<string, string> = { NeuN: 'var(--series-1)', '6E10': 'var(--series-2)', AT8: 'var(--series-3)' }

export default function Calibration() {
  const { data, error } = useFetch<Row[]>('/api/calibration')
  const navigate = useNavigate()
  const stains = Array.from(new Set((data ?? []).map((r) => String(r.stain))))

  const recompute = async () => {
    await api.post('/api/jobs', { workflow: 'calibrate', options: { output: 'data/analysis/slide_dab_calibration_recomputed.csv' } })
    navigate('/pipelines')
  }

  return (
    <>
      <PageHead title="Slide calibration" subtitle="One DAB threshold per slide (TMA × stain) = max(Otsu on pooled tissue DAB, median + 6 MAD, 0.01 OD), computed blind to diagnosis. Every rule downstream is relative to it.">
        <button className="btn" onClick={recompute} title="Writes a separate file; the frozen calibration is not overwritten">Recompute (new file)</button>
      </PageHead>
      <ErrorNote error={error} />
      <div className="grid grid-3" style={{ marginBottom: 16 }}>
        {stains.map((stain) => (
          <div key={stain} className="card">
            <h2>{stain}</h2>
            <Bars unit=" OD" items={(data ?? []).filter((r) => r.stain === stain).map((r) => ({ label: `LIP-${r.tma}`, value: Number(r.threshold_dab_od), color: STAIN_COLOR[stain] }))} />
          </div>
        ))}
      </div>
      <div className="table-wrap">
        <table>
          <thead><tr><th>TMA</th><th>Stain</th><th className="num">Threshold (OD)</th><th className="num">Sampled pixels</th><th className="num">p50</th><th className="num">p90</th><th className="num">p99</th></tr></thead>
          <tbody>
            {(data ?? []).map((r) => (
              <tr key={`${r.tma}-${r.stain}`}>
                <td>LIP-{r.tma}</td><td>{r.stain}</td><td className="num"><b>{fmt(r.threshold_dab_od, 4)}</b></td><td className="num">{fmt(r.sampled_tissue_pixels)}</td>
                <td className="num">{fmt(r.dab_p50, 3)}</td><td className="num">{fmt(r.dab_p90, 3)}</td><td className="num">{fmt(r.dab_p99, 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}
