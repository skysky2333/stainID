import { useNavigate } from 'react-router-dom'
import type { Row } from '../api'
import { api } from '../api'
import { Bars } from '../charts'
import { ErrorNote, HelpBox, PageHead, Term, fmt } from '../components'
import { useFetch } from '../hooks'
import { useProject } from '../project'

const STAIN_COLOR: Record<string, string> = { NeuN: 'var(--series-1)', '6E10': 'var(--series-2)', AT8: 'var(--series-3)' }

export default function Calibration() {
  const { data, error } = useFetch<Row[]>('/api/calibration')
  const { tmaName } = useProject()
  const navigate = useNavigate()
  const stains = Array.from(new Set((data ?? []).map((r) => String(r.stain))))

  const recompute = async () => {
    await api.post('/api/jobs', { workflow: 'calibrate', options: { output: 'data/analysis/slide_dab_calibration_recomputed.csv' } })
    navigate('/workflow#calibrate')
  }

  return (
    <>
      <PageHead title="Stain thresholds" subtitle="The brown-stain (DAB) cut-off used for each slide.">
        <button className="btn" onClick={recompute} title="Writes a separate file; the frozen calibration is not overwritten">Recompute (new file)</button>
      </PageHead>
      <HelpBox id="calibration">
        Slides are never stained exactly alike. So instead of one fixed cut-off, stainID sets a <Term>stain threshold</Term> for each slide from that slide’s own tissue,
        without knowing any diagnosis: the highest of (a) the Otsu split of all tissue pixels, (b) the median + 6 × MAD, and (c) 0.01 <Term>OD</Term>.
        Every later rule (what counts as stained, a compact plaque core, a tau+ neuron) is measured relative to it. Unusually high or low bars point to a slide worth checking.
      </HelpBox>
      <ErrorNote error={error} />
      <div className="grid grid-3" style={{ marginBottom: 16 }}>
        {stains.map((stain) => (
          <div key={stain} className="card">
            <h2>{stain}</h2>
            <Bars unit=" OD" items={(data ?? []).filter((r) => r.stain === stain).map((r) => ({ label: tmaName(String(r.tma)), value: Number(r.threshold_dab_od), color: STAIN_COLOR[stain] }))} />
          </div>
        ))}
      </div>
      <div className="table-wrap">
        <table>
          <thead><tr><th>TMA</th><th>Stain</th><th className="num">Threshold (OD)</th><th className="num">Sampled pixels</th><th className="num">p50</th><th className="num">p90</th><th className="num">p99</th></tr></thead>
          <tbody>
            {(data ?? []).map((r) => (
              <tr key={`${r.tma}-${r.stain}`}>
                <td>{tmaName(String(r.tma))}</td><td>{r.stain}</td><td className="num"><b>{fmt(r.threshold_dab_od, 4)}</b></td><td className="num">{fmt(r.sampled_tissue_pixels)}</td>
                <td className="num">{fmt(r.dab_p50, 3)}</td><td className="num">{fmt(r.dab_p90, 3)}</td><td className="num">{fmt(r.dab_p99, 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}
