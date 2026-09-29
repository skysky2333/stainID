import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import type { CoreDetail, FieldObject, FieldRow, Outline } from '../api'
import { coreThumb, fieldImage, fieldLayer } from '../api'
import { ErrorNote, GroupBadge, PageHead, fmt } from '../components'
import { useFetch, useStored } from '../hooks'

const CLASS_STYLE: Record<string, { color: string; label: string; short?: string }> = {
  neuron: { color: '#00d5ff', label: 'NeuN+ neuron', short: 'neuron' },
  rejected: { color: '#ff3b3b', label: 'rejected candidate', short: 'rejected' },
  compact: { color: '#ff00c8', label: 'compact plaque', short: 'compact' },
  diffuse: { color: '#00dcff', label: 'diffuse plaque', short: 'diffuse' },
  small_plaque: { color: '#c8c8c8', label: 'small deposit (<15 µm)', short: 'small' },
  tau_neuron_ring: { color: '#00e050', label: 'tau+ neuron (nucleus-ring)', short: 'ring' },
  tau_neuron_dense: { color: '#ffd000', label: 'tau+ neuron (dense body)', short: 'dense body' },
}
const RASTER_LAYERS: Record<string, { label: string; stains: string[]; color: string }> = {
  dab: { label: 'DAB above slide threshold', stains: ['NeuN', '6E10', 'AT8'], color: '#ff2828' },
  exclusions: { label: 'excluded (non-tissue, artifacts, folds)', stains: ['NeuN', '6E10', 'AT8'], color: '#5a6eff' },
  threads: { label: 'neuropil threads (computed on demand)', stains: ['AT8'], color: '#286eff' },
}
const CONTEXT = 128

export default function CoreView() {
  const { coreId = '' } = useParams()
  const [params, setParams] = useSearchParams()
  const core = useFetch<CoreDetail>(`/api/cores/${coreId}`)
  const stains = useMemo(() => Array.from(new Set((core.data?.fields ?? []).map((f) => f.stain))).sort(), [core.data])
  const stain = params.get('stain') && stains.includes(params.get('stain')!) ? params.get('stain')! : stains[0]
  const fields = (core.data?.fields ?? []).filter((f) => f.stain === stain).sort((a, b) => a.selection_order - b.selection_order)
  const tileId = params.get('field') && fields.some((f) => f.tile_id === params.get('field')) ? params.get('field')! : fields[0]?.tile_id
  const image = core.data?.images.find((i) => i.stain === stain)

  const select = (next: Record<string, string>) => setParams({ stain: stain ?? '', ...(tileId ? { field: tileId } : {}), ...next })

  return (
    <>
      <PageHead title={coreId} subtitle={core.data ? <span className="row"><GroupBadge group={core.data.disease_group} />
        donor {core.data.donor_id ?? '—'} · {core.data.region ?? '—'} · replicate {core.data.technical_replicate ?? '—'} · <Link to="/cohort">back to cohort</Link></span> : null}>
        <div className="tabs" style={{ marginBottom: 0 }}>
          {stains.map((s) => <button key={s} className={`tab${s === stain ? ' active' : ''}`} onClick={() => setParams({ stain: s })}>{s}</button>)}
        </div>
      </PageHead>
      <ErrorNote error={core.error} />
      {core.data && stain && (
        <div className="grid" style={{ gridTemplateColumns: 'minmax(260px, 320px) 1fr', alignItems: 'start' }}>
          <div className="grid">
            <div className="card">
              <h3>Core · {stain}</h3>
              <div style={{ position: 'relative' }}>
                <img src={coreThumb(coreId, stain, 640)} style={{ width: '100%', display: 'block', borderRadius: 8 }} alt="core" />
                {image && (
                  <svg viewBox={`0 0 ${image.native_width_px} ${image.native_height_px}`} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}>
                    {fields.map((f) => (
                      <g key={f.tile_id} style={{ cursor: 'pointer' }} onClick={() => select({ field: f.tile_id })}>
                        <rect x={f.x_px} y={f.y_px} width={f.width_px} height={f.height_px} fill={f.tile_id === tileId ? 'rgb(255 210 30 / 25%)' : 'transparent'}
                          stroke="#ffd21f" strokeWidth={f.tile_id === tileId ? 120 : 60} />
                        <text x={f.x_px + 200} y={f.y_px + 900} fontSize={800} fill="#ffd21f" fontWeight={700}>{f.selection_order}</text>
                      </g>
                    ))}
                  </svg>
                )}
              </div>
              <p className="muted small">Yellow squares = analysed 560 µm fields. Click one to open it.</p>
            </div>
          </div>
          {tileId && <FieldViewer key={tileId} field={fields.find((f) => f.tile_id === tileId)!} stain={stain} />}
        </div>
      )}
    </>
  )
}

function FieldViewer({ field, stain }: { field: FieldRow; stain: string }) {
  const tileId = field.tile_id
  const inner = { x: field.x_px - Math.max(0, field.x_px - CONTEXT), y: field.y_px - Math.max(0, field.y_px - CONTEXT) }
  const objects = useFetch<{ objects: FieldObject[] }>(`/api/fields/${tileId}/objects`)
  const outlines = useFetch<Outline[]>(`/api/fields/${tileId}/outlines`)
  const [layers, setLayers] = useStored<Record<string, boolean>>('stainid.viewer.layers', { objects: true, outlines: true, rejected: false, dab: false, exclusions: false, threads: false, analysed: true })
  const [zoom, setZoom] = useState(1)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const [selected, setSelected] = useState<number | null>(null)
  const [size, setSize] = useState({ w: 2304, h: 2304 })
  const drag = useRef<{ x: number; y: number; ox: number; oy: number } | null>(null)
  const box = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const img = new Image()
    img.onload = () => setSize({ w: img.naturalWidth, h: img.naturalHeight })
    img.src = fieldImage(tileId)
  }, [tileId])

  const visible = (objects.data?.objects ?? []).map((o, i) => ({ ...o, index: i })).filter((o) => layers.rejected || o.model_class !== 'rejected')
  const classes = Array.from(new Set((objects.data?.objects ?? []).map((o) => o.model_class)))
  const counts = classes.map((c) => [c, (objects.data?.objects ?? []).filter((o) => o.model_class === c).length] as const)
  const toggle = (key: string) => setLayers({ ...layers, [key]: !layers[key] })

  const focus = (o: FieldObject & { index: number }) => {
    setSelected(o.index)
    const width = box.current?.clientWidth ?? 600
    const scale = width / size.w
    setZoom(4)
    setOffset({ x: width / 2 - o.x * scale * 4, y: width / 2 - o.y * scale * 4 })
  }

  return (
    <div className="grid" style={{ gridTemplateColumns: 'minmax(0, 1fr) 320px', alignItems: 'start' }}>
      <div className="card">
        <div className="row" style={{ marginBottom: 10 }}>
          <h3 style={{ margin: 0 }}>{tileId}</h3>
          <span className="spacer" />
          <button className="btn small" onClick={() => setZoom(Math.max(1, zoom / 1.5))}>−</button>
          <span className="small muted">{zoom.toFixed(1)}×</span>
          <button className="btn small" onClick={() => setZoom(Math.min(12, zoom * 1.5))}>+</button>
          <button className="btn small" onClick={() => { setZoom(1); setOffset({ x: 0, y: 0 }) }}>Reset</button>
        </div>
        <div ref={box} className="viewer" style={{ aspectRatio: `${size.w} / ${size.h}`, cursor: zoom > 1 ? 'grab' : 'default' }}
          onWheel={(e) => { const next = Math.min(12, Math.max(1, zoom * (e.deltaY < 0 ? 1.15 : 1 / 1.15))); setZoom(next); if (next === 1) setOffset({ x: 0, y: 0 }) }}
          onMouseDown={(e) => { drag.current = { x: e.clientX, y: e.clientY, ox: offset.x, oy: offset.y } }}
          onMouseMove={(e) => { if (drag.current && zoom > 1) setOffset({ x: drag.current.ox + e.clientX - drag.current.x, y: drag.current.oy + e.clientY - drag.current.y }) }}
          onMouseUp={() => { drag.current = null }} onMouseLeave={() => { drag.current = null }}>
          <div style={{ position: 'absolute', inset: 0, transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})`, transformOrigin: '0 0' }}>
            <img src={fieldImage(tileId)} alt={tileId} draggable={false} />
            {Object.entries(RASTER_LAYERS).filter(([key, def]) => layers[key] && def.stains.includes(stain)).map(([key]) => (
              <img key={key} src={fieldLayer(tileId, key)} alt={key} draggable={false} style={{ imageRendering: 'pixelated' }} />
            ))}
            <svg viewBox={`0 0 ${size.w} ${size.h}`}>
              {layers.analysed && <rect x={inner.x} y={inner.y} width={field.width_px} height={field.height_px} fill="none" stroke="#ffd21f" strokeWidth={4} strokeDasharray="24 16" />}
              {layers.outlines && (outlines.data ?? []).map((o) => (
                <polygon key={o.label} points={o.points.map((p) => p.join(',')).join(' ')} fill="none" stroke="#ffffff" strokeOpacity={0.85} strokeWidth={2.5 / zoom} />
              ))}
              {layers.objects && visible.map((o) => (
                <circle key={o.index} cx={o.x} cy={o.y} r={Math.max(8, o.radius_px) + 4} fill="none" stroke={CLASS_STYLE[o.model_class]?.color ?? '#fff'}
                  strokeWidth={(o.index === selected ? 6 : 3) / Math.sqrt(zoom)} style={{ cursor: 'pointer' }} onClick={() => setSelected(o.index)}>
                  <title>{`${CLASS_STYLE[o.model_class]?.label ?? o.model_class} · p=${fmt(o.probability, 2)} · ${fmt(o.area_um2, 3)} µm²`}</title>
                </circle>
              ))}
            </svg>
          </div>
        </div>
        <p className="muted small">Scroll to zoom, drag to pan. The dashed square is the analysed field; the margin is context only.</p>
      </div>
      <div className="grid">
        <div className="card">
          <h3>Layers</h3>
          <label className="layer-toggle"><input type="checkbox" checked={!!layers.objects} onChange={() => toggle('objects')} />Detections</label>
          {counts.map(([c, n]) => (
            <div key={c} className="layer-toggle" style={{ paddingLeft: 22 }}>
              <span className="swatch" style={{ background: CLASS_STYLE[c]?.color ?? '#999' }} />{CLASS_STYLE[c]?.label ?? c}<span className="spacer" /><span className="muted">{n}</span>
            </div>
          ))}
          {stain === 'NeuN' && <label className="layer-toggle" style={{ paddingLeft: 22 }}><input type="checkbox" checked={!!layers.rejected} onChange={() => toggle('rejected')} />show rejected</label>}
          <label className="layer-toggle"><input type="checkbox" checked={!!layers.outlines} onChange={() => toggle('outlines')} /><span className="swatch" style={{ border: '2px solid #fff', background: '#555' }} />SAM outlines ({outlines.data?.length ?? 0})</label>
          {Object.entries(RASTER_LAYERS).filter(([, d]) => d.stains.includes(stain)).map(([key, d]) => (
            <label key={key} className="layer-toggle"><input type="checkbox" checked={!!layers[key]} onChange={() => toggle(key)} /><span className="swatch" style={{ background: d.color }} />{d.label}</label>
          ))}
          <label className="layer-toggle"><input type="checkbox" checked={!!layers.analysed} onChange={() => toggle('analysed')} />analysed-field boundary</label>
        </div>
        <div className="card">
          <h3>Objects ({visible.length})</h3>
          <div className="table-wrap" style={{ maxHeight: 360 }}>
            <table>
              <thead><tr><th>class</th><th className="num">p</th><th className="num">µm²</th></tr></thead>
              <tbody>
                {visible.sort((a, b) => (b.probability ?? 0) - (a.probability ?? 0)).map((o) => (
                  <tr key={o.index} className={`clickable${o.index === selected ? ' selected' : ''}`} onClick={() => focus(o)}>
                    <td><span className="swatch" style={{ background: CLASS_STYLE[o.model_class]?.color }} /> {CLASS_STYLE[o.model_class]?.short ?? o.model_class}</td>
                    <td className="num">{fmt(o.probability, 2)}</td><td className="num">{fmt(o.area_um2, 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}
