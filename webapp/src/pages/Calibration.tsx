import { Link } from 'react-router-dom'
import type { Row } from '../api'
import { enc } from '../api'
import { Empty, HelpBox, Loaded, PageHead, Term, fmt } from '../components'
import { useFetch } from '../hooks'
import { useProject } from '../project'
import { Advanced } from '../setup/parts'

const STAIN_COLOR: Record<string, string> = { NeuN: 'var(--series-1)', '6E10': 'var(--series-2)', AT8: 'var(--series-3)' }
const UNUSUAL = 2  // a slide whose threshold is under half or over twice the stain's median is worth a look

const median = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

export default function Calibration() {
  const calibration = useFetch<Row[]>('/api/calibration')
  return (
    <>
      <PageHead title="Stain thresholds" subtitle="How dark the brown (DAB) stain has to be to count as stained, set for each slide from its own tissue." />
      <HelpBox id="calibration">
        Slides are never stained exactly alike, so one fixed cut-off would make darker slides look more diseased. stainID therefore sets a
        {' '}<Term>stain threshold</Term> for every slide from that slide’s own tissue, without knowing any diagnosis, and measures everything
        (stained area, plaques, tau) relative to it. A slide marked <b>check</b> has a threshold under half or over twice that of the other
        slides with the same stain: open its cores and see whether the staining or the scan looks unusual.
      </HelpBox>
      <Loaded state={calibration} label="Loading the stain thresholds…">
        {(rows) => rows.length ? <Thresholds rows={rows} /> : (
          <Empty>No stain thresholds yet. They are set by the <b>Calibrate stain thresholds</b> step, after the cores are cut out.
            <div style={{ marginTop: 10 }}><Link className="btn primary" to="/workflow#calibrate">Go to Run</Link></div></Empty>
        )}
      </Loaded>
    </>
  )
}

function Thresholds({ rows }: { rows: Row[] }) {
  const { tmaName } = useProject()
  const stains = Array.from(new Set(rows.map((r) => String(r.stain)))).sort()
  const tmas = Array.from(new Set(rows.map((r) => String(r.tma)))).sort((a, b) => a.length - b.length || a.localeCompare(b))
  const value = new Map(rows.map((r) => [`${r.tma}|${r.stain}`, Number(r.threshold_dab_od)]))
  const stats = Object.fromEntries(stains.map((s) => {
    const values = tmas.map((t) => value.get(`${t}|${s}`)).filter((v): v is number => v !== undefined && Number.isFinite(v))
    return [s, { median: median(values), max: Math.max(...values) }]
  }))
  const unusual = (tma: string, stain: string) => {
    const v = value.get(`${tma}|${stain}`)
    const m = stats[stain].median
    return v !== undefined && m > 0 && (v < m / UNUSUAL || v > m * UNUSUAL)
  }
  const flagged = tmas.flatMap((t) => stains.filter((s) => unusual(t, s)).map((s) => `${tmaName(t)} ${s}`))

  return (
    <>
      <div className="card">
        <p className="secondary" style={{ marginTop: 0 }}>
          {rows.length} slides ({tmas.length} TMAs × {stains.length} stains).{' '}
          {flagged.length ? <><b className="warning-text">{flagged.length} worth a look:</b> {flagged.join(', ')}.</> : 'None stands out from the others.'}
          {' '}Click a value to see that slide’s cores.
        </p>
        <div className="table-wrap threshold-table">
          <table>
            <thead>
              <tr><th>TMA</th>{stains.map((s) => <th key={s}>{s} <span className="muted small">(median {fmt(stats[s].median, 2)})</span></th>)}</tr>
            </thead>
            <tbody>
              {tmas.map((t) => (
                <tr key={t}>
                  <td>{tmaName(t)}</td>
                  {stains.map((s) => {
                    const v = value.get(`${t}|${s}`)
                    if (v === undefined) return <td key={s} className="muted">—</td>
                    const odd = unusual(t, s)
                    return (
                      <td key={s}>
                        <Link className="threshold-cell" to={`/cohort?tma=${enc(t)}&stain=${enc(s)}`} title={`${tmaName(t)} ${s}: threshold ${fmt(v, 3)} OD${odd ? ' — far from the other slides' : ''}. Open its cores.`}>
                          <span className="threshold-bar"><span style={{ width: `${(100 * v) / (stats[s].max || 1)}%`, background: STAIN_COLOR[s] ?? 'var(--accent)' }} /></span>
                          <span className="num">{fmt(v, 2)}</span>
                          {odd && <span className="badge check-badge">check</span>}
                        </Link>
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted small" style={{ marginBottom: 0 }}>Values are in <Term>OD</Term> (optical density of the brown stain); higher means a pixel must be darker to count as stained.</p>
      </div>
      <Advanced title="How the threshold is set, and all numbers">
        <p className="secondary" style={{ marginTop: 0 }}>
          For each slide, stainID samples tissue pixels from its cores and takes the highest of (a) the Otsu split of their DAB values,
          (b) the median + 6 × MAD and (c) 0.01 OD. The percentiles show how the DAB values of the tissue are spread (every tile of every core of the slide).
        </p>
        <div className="table-wrap">
          <table>
            <thead><tr><th>TMA</th><th>Stain</th><th className="num">Threshold (OD)</th><th className="num">Tissue pixels used</th>
              {['p50', 'p75', 'p90', 'p95', 'p99', 'p999'].map((p) => <th key={p} className="num">{p === 'p999' ? 'p99.9' : p}</th>)}</tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={`${r.tma}-${r.stain}`}>
                  <td>{tmaName(String(r.tma))}</td><td>{r.stain}</td><td className="num"><b>{fmt(r.threshold_dab_od, 4)}</b></td><td className="num">{fmt(r.sampled_tissue_pixels)}</td>
                  {['dab_p50', 'dab_p75', 'dab_p90', 'dab_p95', 'dab_p99', 'dab_p999'].map((k) => <td key={k} className="num">{fmt(r[k], 3)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Advanced>
    </>
  )
}
