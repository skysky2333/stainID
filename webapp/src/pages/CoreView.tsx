import { useState } from 'react'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import type { CoreDetail, FieldObject } from '../api'
import { coreThumb, enc } from '../api'
import { ErrorNote, GroupBadge, PageHead, fmt } from '../components'
import { useFetch, useStored } from '../hooks'
import { CLASS_STYLE, RASTER_LAYERS, ZoomViewer } from '../viewer'

const SIZES: [number, string][] = [[800, 'small (220 µm)'], [1600, 'medium (440 µm)'], [2400, 'large (660 µm)']]
interface Region { x0: number; y0: number; size: number; objects: FieldObject[] }

export default function CoreView() {
  const { coreId = '' } = useParams()
  const [params, setParams] = useSearchParams()
  const core = useFetch<CoreDetail>(`/api/cores/${coreId}`)
  const stains = (core.data?.images ?? []).map((i) => i.stain).sort()
  const stain = params.get('stain') && stains.includes(params.get('stain')!) ? params.get('stain')! : stains[0]
  const image = core.data?.images.find((i) => i.stain === stain)
  const [size, setSize] = useStored('stainid.core.region', 1600)
  const point = params.get('x') && params.get('y') ? { x: Number(params.get('x')), y: Number(params.get('y')) } : null
  const go = (x: number, y: number) => setParams({ stain: stain ?? '', x: String(Math.round(x)), y: String(Math.round(y)) })

  return (
    <>
      <PageHead title={coreId} subtitle={core.data ? <span className="row"><GroupBadge group={core.data.disease_group} />
        donor {core.data.donor_id ?? '—'} · {core.data.region ?? '—'} · replicate {core.data.technical_replicate ?? '—'} · <Link to="/cohort">back to cohort</Link></span> : null}>
        <div className="tabs" style={{ marginBottom: 0 }}>
          {stains.map((s) => <button key={s} className={`tab${s === stain ? ' active' : ''}`}
            onClick={() => setParams({ stain: s, ...(point ? { x: String(point.x), y: String(point.y) } : {}) })}>{s}</button>)}
        </div>
      </PageHead>
      <ErrorNote error={core.error} />
      {core.data && stain && image && (
        <div className="grid" style={{ gridTemplateColumns: 'minmax(280px, 380px) 1fr', alignItems: 'start' }}>
          <div className="card">
            <h3>Whole core · {stain}</h3>
            <div style={{ position: 'relative', cursor: 'crosshair' }} onClick={(e) => {
              const rect = e.currentTarget.getBoundingClientRect()
              go(((e.clientX - rect.left) / rect.width) * image.native_width_px, ((e.clientY - rect.top) / rect.height) * image.native_height_px)
            }}>
              <img src={coreThumb(coreId, stain, 1200)} style={{ width: '100%', display: 'block', borderRadius: 8 }} alt="core" draggable={false} />
              {point && (
                <svg viewBox={`0 0 ${image.native_width_px} ${image.native_height_px}`} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
                  <rect x={point.x - size / 2} y={point.y - size / 2} width={size} height={size} fill="rgb(255 210 30 / 18%)" stroke="#ffd21f" strokeWidth={70} />
                </svg>
              )}
            </div>
            <p className="muted small">Click anywhere on the core to see that spot at full resolution.</p>
            <label className="field">Preview size<select value={size} onChange={(e) => setSize(Number(e.target.value))}>
              {SIZES.map(([v, t]) => <option key={v} value={v}>{t}</option>)}</select></label>
          </div>
          {point ? <RegionViewer key={`${stain}-${point.x}-${point.y}-${size}`} coreId={coreId} stain={stain} x={point.x} y={point.y} size={size} onMove={go} />
            : <div className="card"><p className="muted">Click a spot on the core on the left to open it here.</p></div>}
        </div>
      )}
    </>
  )
}

function RegionViewer({ coreId, stain, x, y, size, onMove }: { coreId: string; stain: string; x: number; y: number; size: number; onMove: (x: number, y: number) => void }) {
  const query = `x=${x}&y=${y}&size=${size}`
  const region = useFetch<Region>(`/api/cores/${enc(coreId)}/${enc(stain)}/region?${query}`)
  const [layers, setLayers] = useStored<Record<string, boolean>>('stainid.core.layers', { objects: true, rejected: false, dab: false })
  const [selected, setSelected] = useState<number | null>(null)
  const [focus, setFocus] = useState<{ x: number; y: number; key: number } | null>(null)
  const found = (region.data?.objects ?? []).map((o, i) => ({ ...o, index: i }))
  const visible = found.filter((o) => layers.rejected || o.model_class !== 'rejected')
  const classes = Array.from(new Set(found.map((o) => o.model_class)))
  const toggle = (key: string) => setLayers({ ...layers, [key]: !layers[key] })
  const step = (dx: number, dy: number) => region.data && onMove(region.data.x0 + region.data.size / 2 + dx * region.data.size / 2, region.data.y0 + region.data.size / 2 + dy * region.data.size / 2)

  return (
    <div className="grid" style={{ gridTemplateColumns: 'minmax(0, 1fr) 280px', alignItems: 'start' }}>
      <div>
        <ZoomViewer src={`/api/images/cores/${enc(coreId)}/${enc(stain)}/region.jpg?${query}`} focus={focus}
          layers={layers.dab ? [`/api/images/cores/${enc(coreId)}/${enc(stain)}/region_dab.png?${query}`] : []}
          title={<span className="row"><button className="btn small" onClick={() => step(-1, 0)}>←</button><button className="btn small" onClick={() => step(0, -1)}>↑</button>
            <button className="btn small" onClick={() => step(0, 1)}>↓</button><button className="btn small" onClick={() => step(1, 0)}>→</button>
            <span className="muted small">{region.data ? `x ${region.data.x0}–${region.data.x0 + region.data.size}, y ${region.data.y0}–${region.data.y0 + region.data.size} px` : 'loading…'}</span></span>}>
          {(zoom) => layers.objects && visible.map((o) => (
            <circle key={o.index} cx={o.x} cy={o.y} r={Math.max(8, o.radius_px) + 4} fill="none" stroke={CLASS_STYLE[o.model_class]?.color ?? '#fff'}
              strokeWidth={(o.index === selected ? 6 : 3) / Math.sqrt(zoom)} style={{ cursor: 'pointer' }} onClick={() => setSelected(o.index)}>
              <title>{`${CLASS_STYLE[o.model_class]?.label ?? o.model_class} · p=${fmt(o.probability, 2)} · ${fmt(o.area_um2, 3)} µm²`}</title>
            </circle>
          ))}
        </ZoomViewer>
        <p className="muted small">Scroll to zoom, drag to pan; the arrows move to the neighbouring spot. The first preview of a core takes a few seconds while it is prepared.</p>
        <ErrorNote error={region.error} />
      </div>
      <div className="grid">
        <div className="card">
          <h3>Show</h3>
          <label className="layer-toggle"><input type="checkbox" checked={!!layers.objects} onChange={() => toggle('objects')} />AI detections ({visible.length})</label>
          {classes.map((c) => (
            <div key={c} className="layer-toggle" style={{ paddingLeft: 22 }}>
              <span className="swatch" style={{ background: CLASS_STYLE[c]?.color ?? '#999' }} />{CLASS_STYLE[c]?.label ?? c}<span className="spacer" /><span className="muted">{found.filter((o) => o.model_class === c).length}</span>
            </div>
          ))}
          {classes.includes('rejected') && <label className="layer-toggle" style={{ paddingLeft: 22 }}><input type="checkbox" checked={!!layers.rejected} onChange={() => toggle('rejected')} />show rejected</label>}
          <label className="layer-toggle"><input type="checkbox" checked={!!layers.dab} onChange={() => toggle('dab')} /><span className="swatch" style={{ background: RASTER_LAYERS.dab.color }} />{RASTER_LAYERS.dab.label}</label>
          {region.data && !found.length && <p className="muted small">No detections here yet: detection has only run on the analysed fields so far.</p>}
        </div>
        {visible.length > 0 && (
          <div className="card">
            <h3>Objects ({visible.length})</h3>
            <div className="table-wrap" style={{ maxHeight: 360 }}>
              <table>
                <thead><tr><th>class</th><th className="num">p</th><th className="num">µm²</th></tr></thead>
                <tbody>
                  {[...visible].sort((a, b) => (b.probability ?? 0) - (a.probability ?? 0)).map((o) => (
                    <tr key={o.index} className={`clickable${o.index === selected ? ' selected' : ''}`} onClick={() => { setSelected(o.index); setFocus((f) => ({ x: o.x, y: o.y, key: (f?.key ?? 0) + 1 })) }}>
                      <td><span className="swatch" style={{ background: CLASS_STYLE[o.model_class]?.color }} /> {CLASS_STYLE[o.model_class]?.short ?? o.model_class}</td>
                      <td className="num">{fmt(o.probability, 2)}</td><td className="num">{fmt(o.area_um2, 3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
