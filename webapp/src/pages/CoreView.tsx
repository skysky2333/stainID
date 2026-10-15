import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import type { CoreDetail } from '../api'
import { apiUrl, coreThumb, enc } from '../api'
import { ErrorNote, GroupBadge, Loaded, Loading, PageHead, Progress, fmt } from '../components'
import type { Line, Mark, Pyramid, ViewTarget } from '../deepzoom'
import { DeepZoom } from '../deepzoom'
import { useFetch, useStored } from '../hooks'
import { useProject } from '../project'
import { Advanced } from '../setup/parts'
import { CLASS_STYLE, RASTER_LAYERS } from '../viewer'

interface CoreInfo extends CoreDetail {
  images: (CoreDetail['images'][number] & { pixel_width_um?: number })[]
  neighbours: { previous: string | null; next: string | null }
}
type PictureState = ({ ready: true } & Pyramid) | { ready: false; state: 'waiting' | 'preparing' | 'failed'; ahead?: number; error?: string; thumb_size: number | null }
/** What detection measured on an AT8 core's neuropil threads (all tiles). */
interface ThreadTotals { length_mm: number; skeleton_px: number }
interface CoreObjects {
  status: 'done' | 'partial' | 'not_run'; tiles_done: number; tiles_total: number; columns: string[]; rows: (string | number | null)[][]
  tiles: number[][]; outlines: boolean; threads: ThreadTotals | null
}
/** Segment Anything outlines: per detection (its exact x, y) the outlined area and polygons, in core-image pixels. */
interface CoreOutlines { status: 'done' | 'not_made'; columns: string[]; rows: [number, number, number | null, number[][]][] }
/** AT8 neuropil threads traced on the server (the first time, a few minutes; tiles done so far meanwhile): per piece its
 *  mean width (µm) and polyline, in core-image pixels. */
interface CoreThreads {
  status: 'done' | 'waiting' | 'tracing' | 'failed' | 'not_run' | 'not_started'; tiles_done: number; tiles_total: number
  pieces: number; length_mm: number; measured: ThreadTotals | null; rows: [number, number[]][]; error?: string; ahead?: number
}
interface Outline { area: number | null; rings: number[][] }
interface Found { index: number; x: number; y: number; r: number; cls: string; p: number | null; area: number | null; outline?: Outline }

// exact outlines are drawn instead of circles wherever the Outline objects step has made them
const LAYER_DEFAULTS = { objects: true, rejected: false, outlines: true, threads: false, dab: false, tiles: false }
type Layers = typeof LAYER_DEFAULTS
const TILE_COLOR = 'rgb(30 30 30 / 55%)'
const THREAD_COLOR = RASTER_LAYERS.threads.color
const TRACING = ['waiting', 'tracing']
/** A thread's line on screen (px), by its measured width: thin, medium (1–2 µm) or thick (2 µm or more). */
const threadLine = (widthUm: number) => (widthUm < 1 ? 1.2 : widthUm < 2 ? 1.9 : 2.8)

export default function CoreView() {
  const { coreId = '' } = useParams()
  const [params, setParams] = useSearchParams()
  const navigate = useNavigate()
  const { tmaName } = useProject()
  const core = useFetch<CoreInfo>(`/api/cores/${enc(coreId)}`)
  const info = core.data?.core_id === coreId ? core.data : null
  const stains = (info?.images ?? []).map((i) => i.stain)
  const asked = params.get('stain')
  const stain = asked && stains.includes(asked) ? asked : stains[0]
  const back = info ? `/cohort?tma=${enc(info.tma)}${stain ? `&stain=${enc(stain)}` : ''}` : '/cohort'
  const open = (id: string | null) => id && navigate(`/cores/${enc(id)}${stain ? `?stain=${enc(stain)}` : ''}`)

  return (
    <>
      <Link to={back} className="back-link">← All cores{info ? ` of ${tmaName(info.tma)}` : ''}</Link>
      <PageHead title={info ? `${tmaName(info.tma)} · ${info.core_label}` : coreId}
        subtitle={info ? <span className="row core-facts"><GroupBadge group={info.disease_group} />
          {info.donor_id ? <span>donor {info.donor_id}</span> : <span>control tissue</span>}
          {info.region && <span>{info.region}</span>}{info.technical_replicate && <span>replicate {info.technical_replicate}</span>}</span> : 'Loading the core…'}>
        {info && <span className="row">
          <button className="btn" disabled={!info.neighbours.previous} onClick={() => open(info.neighbours.previous)} title="The core before this one on the TMA">← Previous core</button>
          <button className="btn" disabled={!info.neighbours.next} onClick={() => open(info.neighbours.next)} title="The core after this one on the TMA">Next core →</button>
        </span>}
      </PageHead>
      <Loaded state={core} label="Loading the core…">
        {() => !info ? <Loading label="Loading the core…" /> : !stain ? (
          <div className="empty-state">There is no picture of this core yet. Pictures of the cores are made by the <b>Cut out cores</b> step; see <Link to="/workflow">Run</Link>.</div>
        ) : (
          <>
            <div className="tabs" role="tablist" aria-label="Stain">
              {stains.map((s) => <button key={s} role="tab" aria-selected={s === stain} className={`tab${s === stain ? ' active' : ''}`}
                onClick={() => setParams({ stain: s }, { replace: true })}>{s}</button>)}
            </div>
            <CoreImage key={`${coreId}-${stain}`} core={info} stain={stain} start={params.get('x') && params.get('y') ? { x: Number(params.get('x')), y: Number(params.get('y')) } : null} />
          </>
        )}
      </Loaded>
    </>
  )
}

/** One stain of one core: the zoomable picture with the detections on top, and what is shown beside it. */
function CoreImage({ core, stain, start }: { core: CoreInfo; stain: string; start: { x: number; y: number } | null }) {
  const base = `/api/cores/${enc(core.core_id)}/${enc(stain)}`
  const [poll, setPoll] = useState(true)
  const picture = useFetch<PictureState>(`${base}/picture`, poll ? 1500 : undefined)
  const found = useFetch<CoreObjects>(`${base}/objects`)
  const [stored, setLayers] = useStored<Layers>('stainid.core.layers.v3', LAYER_DEFAULTS)
  const layers = { ...LAYER_DEFAULTS, ...stored }
  const outlinesMade = !!found.data?.outlines
  const [wantOutlines, setWantOutlines] = useState(false)  // fetched once asked for, kept when the layer is turned off
  useEffect(() => { if (layers.outlines) setWantOutlines(true) }, [layers.outlines])
  const shapes = useFetch<CoreOutlines>(wantOutlines && outlinesMade ? `${base}/outlines` : null)
  const threadsMeasured = found.data?.threads ?? null
  const [threadPoll, setThreadPoll] = useState(true)
  const traced = useFetch<CoreThreads>(layers.threads && threadsMeasured ? `${base}/threads` : null, threadPoll ? 3000 : undefined)
  const tracing = TRACING.includes(traced.data?.status ?? 'waiting')
  useEffect(() => { setThreadPoll(tracing) }, [tracing])
  const [selected, setSelected] = useState<number | null>(null)
  const [target, setTarget] = useState<ViewTarget | null>(start ? { ...start, scale: 0.5, key: 0 } : null)
  const image = core.images.find((i) => i.stain === stain)
  const ready = picture.data?.ready ? picture.data : null
  const settled = !!ready || (picture.data && !picture.data.ready && picture.data.state === 'failed')
  useEffect(() => { if (settled) setPoll(false) }, [settled])

  const objects = useMemo<Found[]>(() => {
    const data = found.data
    if (!data) return []
    const col = Object.fromEntries(data.columns.map((c, i) => [c, i]))
    const outlines = new Map((shapes.data?.rows ?? []).map(([x, y, area, rings]) => [`${x},${y}`, { area, rings }]))
    return data.rows.map((r, index) => ({ index, x: Number(r[col.x]), y: Number(r[col.y]), r: Number(r[col.radius_px] ?? 6), cls: String(r[col.model_class]),
      p: r[col.probability] === null ? null : Number(r[col.probability]), area: r[col.area_um2] === null ? null : Number(r[col.area_um2]),
      outline: outlines.get(`${r[col.x]},${r[col.y]}`) }))
  }, [found.data, shapes.data])
  const counts = useMemo(() => objects.reduce<Record<string, number>>((acc, o) => ({ ...acc, [o.cls]: (acc[o.cls] ?? 0) + 1 }), {}), [objects])
  const classes = Object.keys(counts).filter((c) => c !== 'rejected').sort()
  const showOutlines = layers.outlines && outlinesMade && !!shapes.data
  const drawn = useMemo(() => objects.filter((o) => (layers.rejected || o.cls !== 'rejected') && (layers.objects || (showOutlines && o.outline))),
    [objects, layers.objects, layers.rejected, showOutlines])
  // an object with an exact outline shown is drawn by its outline instead of the circle
  const marks = useMemo<Mark[]>(() => drawn.map((o) => {
    const outlined = showOutlines && !!o.outline
    return { x: o.x, y: o.y, r: o.r, color: CLASS_STYLE[o.cls]?.color ?? '#fff', rings: outlined ? o.outline!.rings : undefined, circle: !outlined }
  }), [drawn, showOutlines])
  const lines = useMemo<Line[]>(() => (layers.threads && traced.data ? traced.data.rows.map(([width, points]) => ({ points, width: threadLine(width) })) : []),
    [layers.threads, traced.data])
  const boxes = useMemo<Mark[]>(() => (layers.tiles ? (found.data?.tiles ?? []).map(([x, y, w, h]) => ({ x, y, w, h, color: TILE_COLOR, dashed: true })) : []),
    [layers.tiles, found.data])
  const toggle = (key: keyof Layers) => setLayers({ ...layers, [key]: !layers[key] })
  const pick = drawn.find((o) => o.index === selected) ?? null
  const describe = (i: number) => {
    const o = drawn[i]
    return [CLASS_STYLE[o.cls]?.label ?? o.cls, o.p === null ? null : `${Math.round(o.p * 100)}% sure`, o.area === null ? null : `${area(o.area)} µm²`,
      showOutlines && o.outline?.area != null ? `outlined ${area(o.outline.area)} µm²` : null].filter(Boolean).join(' · ')
  }
  const go = (x: number, y: number, scale?: number) => setTarget((t) => ({ x, y, scale, key: (t?.key ?? 0) + 1 }))
  const tileUrl = ready ? (level: number, col: number, row: number, layer?: string) =>
    apiUrl(`/api/images/cores/${enc(core.core_id)}/${enc(stain)}/tiles/${level}/${col}_${row}${layer ? `_${layer}.png` : '.jpg'}?v=${ready.version}`) : null

  return (
    <div className="core-layout">
      <div className="card core-picture">
        {ready && tileUrl ? (
          <DeepZoom pyramid={ready} tileUrl={tileUrl} layers={layers.dab ? ['dab'] : []} marks={marks} lines={lines} lineColor={THREAD_COLOR}
            boxes={boxes} pixelUm={image?.pixel_width_um}
            selected={pick ? drawn.indexOf(pick) : null} describe={describe} target={target}
            onPick={(i) => setSelected(i === null ? null : drawn[i].index)}
            toolbar={<span className="small muted">Scroll to zoom · drag to move · click an object for details</span>} />
        ) : (
          <Preparing core={core} stain={stain} state={picture.data} error={picture.error} />
        )}
      </div>
      <div className="core-side">
        <div className="card">
          <h3>On the picture</h3>
          <Detections status={found.data} error={found.error} stain={stain} tma={core.tma} tissue={image?.tissue_status} />
          {found.data && found.data.status !== 'not_run' && (
            <>
              <label className="layer-toggle"><input type="checkbox" checked={layers.objects} onChange={() => toggle('objects')} />
                <b>What the AI found</b><span className="spacer" /><span className="muted">{(objects.length - (counts.rejected ?? 0)).toLocaleString()}</span></label>
              {classes.map((c) => (
                <div key={c} className="layer-toggle indent">
                  <span className="swatch" style={{ background: CLASS_STYLE[c]?.color ?? '#999' }} />{CLASS_STYLE[c]?.label ?? c}<span className="spacer" /><span className="muted">{counts[c].toLocaleString()}</span>
                </div>
              ))}
              {!!counts.rejected && (
                <label className="layer-toggle indent" title="Candidates the AI looked at and decided were not real objects">
                  <input type="checkbox" checked={layers.rejected} disabled={!layers.objects} onChange={() => toggle('rejected')} />
                  <span className="swatch" style={{ background: CLASS_STYLE.rejected.color }} />also show rejected candidates<span className="spacer" /><span className="muted">{counts.rejected.toLocaleString()}</span>
                </label>
              )}
              <OutlinesToggle made={outlinesMade} on={layers.outlines} onToggle={() => toggle('outlines')} state={shapes}
                count={objects.filter((o) => o.outline).length} />
              {stain === 'AT8' && <ThreadsToggle measured={threadsMeasured} on={layers.threads} onToggle={() => toggle('threads')} state={traced} />}
            </>
          )}
          <label className="layer-toggle"><input type="checkbox" checked={layers.dab} onChange={() => toggle('dab')} />
            <span className="swatch" style={{ background: RASTER_LAYERS.dab.color }} />Stained area (brown above this slide’s threshold)</label>
          {(found.data?.tiles.length ?? 0) > 0 && <label className="layer-toggle"><input type="checkbox" checked={layers.tiles} onChange={() => toggle('tiles')} />
            <span className="swatch outline dashed" />Tile grid ({found.data!.tiles.length} tiles analysed)</label>}
        </div>
        {pick && (
          <div className="card">
            <h3>Selected object</h3>
            <p className="row" style={{ margin: '0 0 6px' }}><span className="swatch" style={{ background: CLASS_STYLE[pick.cls]?.color }} /><b>{CLASS_STYLE[pick.cls]?.label ?? pick.cls}</b></p>
            <dl className="facts">
              {pick.p !== null && <><dt>AI confidence</dt><dd>{Math.round(pick.p * 100)}%</dd></>}
              {pick.area !== null && <><dt>Area</dt><dd>{area(pick.area)} µm²</dd></>}
              {showOutlines && pick.outline?.area != null && <><dt title="The area inside the Segment Anything outline">Outlined area</dt><dd>{area(pick.outline.area)} µm²</dd></>}
              <dt>Position</dt><dd>{Math.round(pick.x).toLocaleString()}, {Math.round(pick.y).toLocaleString()} px</dd>
            </dl>
            <div className="row"><button className="btn small" onClick={() => go(pick.x, pick.y, 1)}>Zoom to it</button>
              <button className="btn small" onClick={() => setSelected(null)}>Clear</button></div>
          </div>
        )}
        <Advanced title="Details">
          <dl className="facts">
            <dt>Core</dt><dd><code>{core.core_id}</code></dd>
            {image && <><dt>Image size</dt><dd>{image.native_width_px.toLocaleString()} × {image.native_height_px.toLocaleString()} px</dd></>}
            {image?.pixel_width_um && <><dt>Pixel size</dt><dd>{fmt(image.pixel_width_um, 3)} µm</dd></>}
            {image?.tissue_fraction !== undefined && <><dt>Tissue</dt><dd>{Math.round(image.tissue_fraction * 100)}% of the core{image.tissue_status ? ` (${image.tissue_status})` : ''}</dd></>}
            {core.sample_region_id && <><dt>Sample</dt><dd><code>{core.sample_region_id}</code></dd></>}
            {found.data && <><dt>Detection</dt><dd>{found.data.tiles_done} of {found.data.tiles_total} tiles</dd></>}
          </dl>
          {drawn.length > 0 && <ObjectList objects={drawn} selected={selected} onPick={(o) => { setSelected(o.index); go(o.x, o.y, 1) }} />}
        </Advanced>
      </div>
    </div>
  )
}

/** While the server prepares the zoomable picture (first view of a core only): the small picture, dimmed, and why. */
function Preparing({ core, stain, state, error }: { core: CoreInfo; stain: string; state: PictureState | null; error: string | null }) {
  const failed = error ?? (state && !state.ready && state.state === 'failed' ? state.error ?? 'unknown error' : null)
  const ahead = state && !state.ready ? state.ahead ?? 0 : 0
  return (
    <div className="dz-preparing" style={{ height: 'min(72vh, 760px)' }}>
      {state && !state.ready && state.thumb_size && <img src={coreThumb(core.core_id, stain, state.thumb_size)} alt="" />}
      <div className="dz-preparing-note">
        {failed ? <><b>This picture could not be prepared.</b><ErrorNote error={failed} /></>
          : !state ? <Loading label="Loading the picture…" />
          : <Loading label={<span><b>Preparing this core for viewing…</b><br />
              <span className="small">{ahead ? `${ahead} other picture${ahead === 1 ? '' : 's'} first, then this one. ` : ''}The full scan is read once (about 10 s); after that it opens instantly.</span></span>} />}
      </div>
    </div>
  )
}

/** The "Exact outlines" layer: on wherever the outlines are made (drawn instead of circles); where they are not, why not. */
function OutlinesToggle({ made, on, onToggle, state, count }: {
  made: boolean; on: boolean; onToggle: () => void; state: { data: CoreOutlines | null; error: string | null; reload: () => void }; count: number
}) {
  const ready = made && state.data?.status !== 'not_made'
  return (
    <>
      <label className="layer-toggle" title="The exact shape of each object, drawn by Segment Anything (the optional Outline objects step)">
        <input type="checkbox" checked={on && ready} disabled={!ready} onChange={onToggle} />
        <span className="swatch outline" />Exact outlines (Segment Anything)<span className="spacer" />
        {on && ready && state.data && <span className="muted">{count.toLocaleString()}</span>}
      </label>
      {!ready && <p className="layer-note small">Not made for this core yet, so each object is marked with a circle. Exact outlines come from
        the optional <Link to="/workflow#masks">Outline objects</Link> step, which has not reached this core.</p>}
      {on && ready && !state.data && !state.error && <div className="layer-toggle indent"><Loading inline label="Loading the outlines…" /></div>}
      {on && ready && state.error && <div className="layer-toggle indent wrap"><ErrorNote error={state.error} />
        <button className="btn small" onClick={state.reload}>Try again</button></div>}
    </>
  )
}

/** The AT8 "Neuropil threads" layer: what detection measured, and (when turned on) the threads traced on the picture. */
function ThreadsToggle({ measured, on, onToggle, state }: {
  measured: ThreadTotals | null; on: boolean; onToggle: () => void; state: { data: CoreThreads | null; error: string | null; reload: () => void }
}) {
  const data = state.data
  const tracing = !!data && TRACING.includes(data.status)
  return (
    <>
      <label className="layer-toggle" title="Thin tau-positive threads in the neuropil, as the AI traced them; line thickness shows each thread’s width">
        <input type="checkbox" checked={on && !!measured} disabled={!measured} onChange={onToggle} />
        <span className="swatch" style={{ background: THREAD_COLOR }} />Neuropil threads<span className="spacer" />
        {measured && <span className="muted" title="Total length of the threads the AI measured in this core">{lengthMm(measured.length_mm)}</span>}
      </label>
      {!measured && <p className="layer-note small">Threads are measured when the AI detects tau on this core; it has not finished this core yet.</p>}
      {on && measured && !data && !state.error && <div className="layer-toggle indent"><Loading inline label="Loading the threads…" /></div>}
      {on && tracing && (
        <div className="layer-note small">
          {data!.status === 'waiting' ? 'Waiting for another core’s threads to finish, then tracing this one… '
            : `Tracing the threads on this core (first time only, a few minutes)… ${data!.tiles_done} of ${data!.tiles_total} tiles.`}
          <Progress done={data!.tiles_done} total={data!.tiles_total} />
        </div>
      )}
      {on && data?.status === 'done' && <p className="layer-note small">{data.pieces.toLocaleString()} thread pieces drawn; thicker lines are thicker threads (1 µm, 2 µm and up).</p>}
      {on && (state.error || data?.status === 'failed') && <div className="layer-toggle indent wrap"><ErrorNote error={state.error ?? data?.error ?? 'unknown error'} />
        <button className="btn small" onClick={state.reload}>Try again</button></div>}
    </>
  )
}

/** A length given in mm: in µm below 1 mm. */
function lengthMm(mm: number) {
  return mm >= 1 ? `${fmt(mm, 3)} mm` : `${Math.round(mm * 1000).toLocaleString()} µm`
}

/** An area in µm², with thousands separators. */
function area(um2: number) {
  return um2 >= 100 ? Math.round(um2).toLocaleString() : fmt(um2, 3)
}

function Detections({ status, error, stain, tma, tissue }: { status: CoreObjects | null; error: string | null; stain: string; tma: string; tissue?: string | null }) {
  const { tmaName } = useProject()
  if (error) return <ErrorNote error={error} />
  if (!status) return <Loading inline label="Loading the detections…" />
  if (status.status === 'done') return null
  if (status.status === 'partial') {
    return (
      <div className="note-box">
        <b>Detection is part-way through this core.</b> Showing what it has found so far ({status.tiles_done} of {status.tiles_total} tiles).
        <Progress done={status.tiles_done} total={status.tiles_total} />
      </div>
    )
  }
  if (!status.tiles_total && (tissue === 'empty' || tissue === 'sparse')) {
    return (
      <div className="note-box">
        <b>Skipped: the tissue check found {tissue === 'empty' ? 'no' : 'too little'} tissue on this {stain} picture.</b> The AI only looks
        at pictures with tissue, so nothing was detected here. If you can see tissue, the check was wrong: ask whoever runs the analysis
        to look at this core again.
      </div>
    )
  }
  return (
    <div className="note-box">
      <b>Not detected yet.</b> The AI has not looked at the {stain} slide of {tmaName(tma)} yet{status.tiles_total ? '' : ' (no tiles were chosen for it)'}, so only the picture is shown.
      {' '}<Link to="/workflow">See progress on the Run page.</Link>
    </div>
  )
}

function ObjectList({ objects, selected, onPick }: { objects: Found[]; selected: number | null; onPick: (o: Found) => void }) {
  const [page, setPage] = useState(0)
  const size = 100
  const sorted = useMemo(() => [...objects].sort((a, b) => (b.p ?? 0) - (a.p ?? 0)), [objects])
  const pages = Math.ceil(sorted.length / size)
  return (
    <>
      <div className="row small" style={{ margin: '10px 0 6px' }}><b>All objects ({objects.length.toLocaleString()})</b><span className="spacer" />
        {pages > 1 && <><button className="btn small" disabled={page === 0} onClick={() => setPage(page - 1)}>‹</button>
          <span className="muted">{page + 1} / {pages}</span><button className="btn small" disabled={page >= pages - 1} onClick={() => setPage(page + 1)}>›</button></>}</div>
      <div className="table-wrap" style={{ maxHeight: 320 }}>
        <table className="compact">
          <thead><tr><th>Class</th><th className="num">Sure</th><th className="num">µm²</th></tr></thead>
          <tbody>
            {sorted.slice(page * size, (page + 1) * size).map((o) => (
              <tr key={o.index} className={`clickable${o.index === selected ? ' selected' : ''}`} onClick={() => onPick(o)}>
                <td><span className="swatch" style={{ background: CLASS_STYLE[o.cls]?.color }} /> {CLASS_STYLE[o.cls]?.short ?? o.cls}</td>
                <td className="num">{o.p === null ? '—' : `${Math.round(o.p * 100)}%`}</td><td className="num">{o.area === null ? '—' : area(o.area)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}
