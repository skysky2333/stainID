import { useEffect, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { coreThumb, enc } from '../api'
import { Empty, GroupBadge, HelpBox, Loaded, Loading, PageHead, Progress, StatusBadge } from '../components'
import { useFetch, useStored } from '../hooks'
import { useProject } from '../project'

interface CohortInfo { tmas: { tma: string; cores: number; stains: string[] }[]; stains: string[] }
type Detection = 'done' | 'partial' | 'not_run'
interface GridCell {
  core_label: string; core_id: string | null; tissue_control: string | null; donor_id?: string | null; region?: string | null
  disease_group?: string | null; technical_replicate?: string | null; thumb?: boolean; placeholder?: string | null; detection?: Detection
}
interface TmaGrid { tma: string; stain: string; columns: string[]; rows: number[]; cells: GridCell[]; thumbs_waiting: number; detection: { done: number; total: number } }

const DETECTION_TEXT: Record<Detection, string> = { done: 'detected', partial: 'detection part-way', not_run: 'not detected yet' }
const DETECTION_STATUS: Record<Detection, string> = { done: 'done', partial: 'partial', not_run: 'todo' }

export default function Cohort() {
  const cohort = useFetch<CohortInfo>('/api/cohort')
  return (
    <>
      <PageHead title="Cores" subtitle="Every core of each TMA. Open one to look at it closely, with what the AI found drawn on top." />
      <HelpBox id="cohort">
        Pick a TMA and a stain to see all its cores as they sit on the slide; the coloured frame is the diagnostic group. Click a core to open it:
        you can zoom in down to single cells, and switch on what the AI detected (neurons, plaques, tau tangles), the stained area and the
        tile grid. A <DetectionMark detection="not_run" /> marks a core the AI has not looked at yet, <DetectionMark detection="partial" /> one it is part-way through.
      </HelpBox>
      <Loaded state={cohort} label="Loading the TMAs…">
        {(data) => data.tmas.length ? <CohortGrid info={data} /> : (
          <Empty>There are no core pictures yet. They are made when the cores are cut out of the slides.<div style={{ marginTop: 10 }}><Link className="btn primary" to="/workflow">Go to Run</Link></div></Empty>
        )}
      </Loaded>
    </>
  )
}

function CohortGrid({ info }: { info: CohortInfo }) {
  const { tmaName } = useProject()
  const [params, setParams] = useSearchParams()
  const [storedTma, storeTma] = useStored('stainid.cohort.tma', info.tmas[0].tma)
  const [storedStain, storeStain] = useStored('stainid.cohort.stain', info.stains[0])
  const [view, setView] = useStored<'grid' | 'table'>('stainid.cohort.view', 'grid')
  const tmas = info.tmas.map((t) => t.tma)
  const wantedTma = params.get('tma') ?? storedTma
  const tma = tmas.includes(wantedTma) ? wantedTma : tmas[0]
  const stains = info.tmas.find((t) => t.tma === tma)?.stains ?? info.stains
  const wantedStain = params.get('stain') ?? storedStain
  const stain = stains.includes(wantedStain) ? wantedStain : stains[0]
  const choose = (next: { tma?: string; stain?: string }) => {
    const t = next.tma ?? tma
    const s = next.stain ?? stain
    storeTma(t)
    storeStain(s)
    setParams({ tma: t, stain: s }, { replace: true })
  }
  const at = tmas.indexOf(tma)

  return (
    <>
      <div className="cohort-bar">
        <div className="row">
          <button className="btn" disabled={at <= 0} onClick={() => choose({ tma: tmas[at - 1] })} aria-label="Previous TMA" title="Previous TMA">‹</button>
          <select value={tma} onChange={(e) => choose({ tma: e.target.value })} aria-label="TMA">
            {info.tmas.map((t) => <option key={t.tma} value={t.tma}>{tmaName(t.tma)} · {t.cores} cores</option>)}
          </select>
          <button className="btn" disabled={at >= tmas.length - 1} onClick={() => choose({ tma: tmas[at + 1] })} aria-label="Next TMA" title="Next TMA">›</button>
        </div>
        <div className="segmented" role="tablist" aria-label="Stain">
          {stains.map((s) => <button key={s} role="tab" aria-selected={s === stain} className={s === stain ? 'active' : ''} onClick={() => choose({ stain: s })}>{s}</button>)}
        </div>
        <span className="spacer" />
        <div className="segmented" aria-label="Layout">
          <button className={view === 'grid' ? 'active' : ''} onClick={() => setView('grid')}>Pictures</button>
          <button className={view === 'table' ? 'active' : ''} onClick={() => setView('table')}>List</button>
        </div>
      </div>
      <TmaCores key={`${tma}-${stain}`} tma={tma} stain={stain} view={view} />
    </>
  )
}

function TmaCores({ tma, stain, view }: { tma: string; stain: string; view: 'grid' | 'table' }) {
  const { tmaName } = useProject()
  const navigate = useNavigate()
  const [poll, setPoll] = useState(false)
  const grid = useFetch<TmaGrid>(`/api/tmas/${enc(tma)}/grid?stain=${enc(stain)}`, poll ? 3000 : undefined)
  // while pictures are being made, ask again every few seconds (asking also keeps them queued)
  useEffect(() => setPoll((grid.data?.thumbs_waiting ?? 0) > 0), [grid.data])
  const open = (cell: GridCell) => cell.core_id && navigate(`/cores/${enc(cell.core_id)}?stain=${enc(stain)}`)

  return (
    <Loaded state={grid} label={`Loading ${tmaName(tma)}…`}>
      {(g) => (
        <>
          <GridStatus grid={g} />
          {view === 'grid' ? <CoreGrid grid={g} onOpen={open} /> : <CoreTable grid={g} onOpen={open} />}
        </>
      )}
    </Loaded>
  )
}

function GridStatus({ grid }: { grid: TmaGrid }) {
  const { groups, groupColor, groupLabel, tmaName } = useProject()
  const { done, total } = grid.detection
  return (
    <div className="card cohort-status">
      <div className="row">
        <span><b>AI detection</b> on {tmaName(grid.tma)} {grid.stain}:</span>
        {done === total ? <StatusBadge status="done" /> : done === 0 ? <span className="badge status-todo">Not detected yet</span>
          : <span className="row" style={{ gap: 8 }}><span>{done} of {total} cores</span><span style={{ width: 120 }}><Progress done={done} total={total} /></span></span>}
        <span className="spacer" />
        {grid.thumbs_waiting > 0 && <Loading inline label={`Making pictures of ${grid.thumbs_waiting} core${grid.thumbs_waiting === 1 ? '' : 's'} (first visit only)…`} />}
      </div>
      <div className="row small legend">
        {groups.map((g) => <span key={g} className="row"><span className="swatch" style={{ background: groupColor(g) }} />{groupLabel(g)}</span>)}
        <span className="row"><span className="swatch outline dashed" />control tissue or empty</span>
        <span className="row"><DetectionMark detection="not_run" /> not detected yet</span>
        <span className="row"><DetectionMark detection="partial" /> detection part-way</span>
      </div>
    </div>
  )
}

function CoreGrid({ grid, onOpen }: { grid: TmaGrid; onOpen: (cell: GridCell) => void }) {
  const { groupColor, groupLabel } = useProject()
  const byLabel = new Map(grid.cells.map((c) => [c.core_label, c]))
  return (
    <div className="card">
      <div className="core-grid" style={{ gridTemplateColumns: `20px repeat(${grid.columns.length}, minmax(0, 200px))` }}>
        <div />
        {grid.columns.map((c) => <div key={c} className="muted small core-axis">{c}</div>)}
        {grid.rows.map((row) => [
          <div key={`r${row}`} className="muted small core-axis" style={{ alignSelf: 'center' }}>{row}</div>,
          ...grid.columns.map((col) => {
            const label = `${col}-${row}`
            const cell = byLabel.get(label)
            if (!cell?.core_id) {
              return <div key={label} className="core-cell empty" title={cell?.tissue_control ? `${label}: control tissue (${cell.tissue_control})` : `${label}: no core`}>
                <span className="tag">{label}{cell?.tissue_control ? ` · ${cell.tissue_control}` : ''}</span></div>
            }
            const detection = cell.detection ?? 'not_run'
            const about = [label, cell.donor_id && `donor ${cell.donor_id}`, cell.region, cell.disease_group && groupLabel(cell.disease_group), DETECTION_TEXT[detection]].filter(Boolean).join(' · ')
            return (
              <button key={label} className="core-cell" style={{ borderColor: cell.disease_group ? groupColor(cell.disease_group) : 'var(--border)' }} title={about} aria-label={about} onClick={() => onOpen(cell)}>
                <CorePicture src={cell.thumb ? coreThumb(cell.core_id, grid.stain, 256) : null} placeholder={cell.placeholder} />
                {detection !== 'done' && <DetectionMark detection={detection} corner />}
                <span className="tag">{label}{cell.region ? ` · ${cell.region}` : ''}</span>
              </button>
            )
          }),
        ])}
      </div>
    </div>
  )
}

/** A core's thumbnail, faded in over its blurred stand-in (cut from the slide picture) once it has loaded. Both fill the
 *  same square, so nothing moves. The stand-in is kept: the grid stops sending it once the thumbnail is made. */
function CorePicture({ src, placeholder }: { src: string | null; placeholder?: string | null }) {
  const [standIn] = useState(placeholder)
  const [loaded, setLoaded] = useState(false)
  if (!src && !standIn) return <span className="core-cell-wait"><span className="spinner" aria-hidden="true" /></span>
  return (
    <>
      {standIn && <img className="core-cell-standin" src={standIn} alt="" />}
      {src && <img className={`core-cell-pic${loaded ? ' loaded' : ''}`} loading="lazy" src={src} alt="" onLoad={() => setLoaded(true)} />}
    </>
  )
}

function CoreTable({ grid, onOpen }: { grid: TmaGrid; onOpen: (cell: GridCell) => void }) {
  const cores = grid.cells.filter((c) => c.core_id)
  return (
    <div className="table-wrap">
      <table>
        <thead><tr><th>Position</th><th>Donor</th><th>Group</th><th>Region</th><th className="num">Replicate</th><th>AI detection</th></tr></thead>
        <tbody>
          {cores.map((c) => (
            <tr key={c.core_label} className="clickable" onClick={() => onOpen(c)}>
              <td>{c.core_label}</td><td>{c.donor_id ?? '—'}</td><td><GroupBadge group={c.disease_group} /></td><td>{c.region ?? '—'}</td>
              <td className="num">{c.technical_replicate ?? '—'}</td>
              <td><StatusBadge status={DETECTION_STATUS[c.detection ?? 'not_run']} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** The small ring on a core the AI has not (fully) looked at: hollow = not yet, half filled = part-way. */
function DetectionMark({ detection, corner = false }: { detection: Detection; corner?: boolean }) {
  return <span className={`det-mark det-${detection}${corner ? ' corner' : ''}`} aria-hidden="true" />
}
