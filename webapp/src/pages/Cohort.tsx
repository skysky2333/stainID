import { useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import type { CoreRow, Row, Summary } from '../api'
import { coreThumb } from '../api'
import { ErrorNote, GROUP_COLOR, GROUP_LABEL, GroupBadge, PageHead } from '../components'
import { useFetch, useStored } from '../hooks'

export default function Cohort() {
  const summary = useFetch<Summary>('/api/summary')
  const [tma, setTma] = useStored('stainid.cohort.tma', '1')
  const [stain, setStain] = useStored('stainid.cohort.stain', 'NeuN')
  const [view, setView] = useStored<'grid' | 'table'>('stainid.cohort.view', 'grid')
  const cores = useFetch<CoreRow[]>(`/api/cores?tma=${tma}`)
  const layout = useFetch<Row[]>(`/api/tmas/${tma}/layout`)
  const navigate = useNavigate()
  const stains = summary.data ? Object.keys(summary.data.fields) : ['NeuN', '6E10', 'AT8']

  const grid = useMemo(() => {
    const cells = (layout.data ?? []).map((r) => String(r.core_label))
    const cols = Array.from(new Set(cells.map((c) => c.split('-')[0]))).sort()
    const rows = Array.from(new Set(cells.map((c) => Number(c.split('-')[1])))).sort((a, b) => a - b)
    return { cols, rows }
  }, [layout.data])
  const byLabel = useMemo(() => new Map((cores.data ?? []).map((c) => [c.core_label, c])), [cores.data])
  const layoutByLabel = useMemo(() => new Map((layout.data ?? []).map((r) => [String(r.core_label), r])), [layout.data])

  return (
    <>
      <PageHead title="Cohort & cores" subtitle="Tissue microarray layout. Border colour = group; click a core to open the field viewer.">
        <label className="field">TMA
          <select value={tma} onChange={(e) => setTma(e.target.value)}>{(summary.data?.tmas ?? [tma]).map((t) => <option key={t} value={t}>LIP-{t}</option>)}</select>
        </label>
        <label className="field">Stain
          <select value={stain} onChange={(e) => setStain(e.target.value)}>{stains.map((s) => <option key={s}>{s}</option>)}</select>
        </label>
        <label className="field">View
          <select value={view} onChange={(e) => setView(e.target.value as 'grid' | 'table')}><option value="grid">Grid</option><option value="table">Table</option></select>
        </label>
      </PageHead>
      <ErrorNote error={cores.error ?? layout.error} />
      <div className="row small" style={{ marginBottom: 12 }}>
        {Object.entries(GROUP_LABEL).map(([g, label]) => <span key={g} className="row" style={{ gap: 5 }}><span className="swatch" style={{ background: GROUP_COLOR[g] }} />{label}</span>)}
        <span className="row" style={{ gap: 5 }}><span className="swatch" style={{ border: '1px dashed var(--text-muted)' }} />control / empty position</span>
      </div>
      {view === 'grid' ? (
        <div className="card">
          <div className="core-grid" style={{ gridTemplateColumns: `28px repeat(${grid.cols.length}, minmax(70px, 1fr))` }}>
            <div />
            {grid.cols.map((c) => <div key={c} className="muted small" style={{ textAlign: 'center' }}>{c}</div>)}
            {grid.rows.map((r) => (
              <FragmentRow key={r} row={r} cols={grid.cols} stain={stain} byLabel={byLabel} layoutByLabel={layoutByLabel} onOpen={(id) => navigate(`/cores/${id}?stain=${stain}`)} />
            ))}
          </div>
        </div>
      ) : (
        <div className="table-wrap">
          <table>
            <thead><tr><th>Core</th><th>Donor</th><th>Group</th><th>Region</th><th>Replicate</th><th>Stains</th></tr></thead>
            <tbody>
              {(cores.data ?? []).map((c) => (
                <tr key={c.core_id} className="clickable" onClick={() => navigate(`/cores/${c.core_id}?stain=${stain}`)}>
                  <td>{c.core_id}</td><td>{c.donor_id}</td><td><GroupBadge group={c.disease_group} /></td><td>{c.region}</td>
                  <td>{c.technical_replicate}</td><td>{c.stains.join(', ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  )
}

function FragmentRow({ row, cols, stain, byLabel, layoutByLabel, onOpen }: {
  row: number; cols: string[]; stain: string; byLabel: Map<string, CoreRow>; layoutByLabel: Map<string, Row>; onOpen: (id: string) => void
}) {
  return (
    <>
      <div className="muted small" style={{ alignSelf: 'center', textAlign: 'center' }}>{row}</div>
      {cols.map((col) => {
        const label = `${col}-${row}`
        const core = byLabel.get(label)
        const info = layoutByLabel.get(label)
        if (!core || !core.stains.includes(stain)) {
          return <div key={label} className="core-cell empty" title={info ? `${label} ${info.tissue_control ?? ''}` : label}><span className="tag">{label}</span></div>
        }
        return (
          <div key={label} className="core-cell" style={{ borderColor: core.disease_group ? GROUP_COLOR[core.disease_group] : 'var(--border)' }}
            title={`${core.core_id} · donor ${core.donor_id} · ${core.region}`} onClick={() => onOpen(core.core_id)}>
            <img loading="lazy" src={coreThumb(core.core_id, stain, 256)} alt={core.core_id} />
            <span className="tag">{label} · {core.region === 'frontal' ? 'F' : core.region === 'occipital' ? 'O' : ''}</span>
          </div>
        )
      })}
    </>
  )
}
