/* "Mark bad areas" (sessions of kind "areas", routes/areas.py): one core at a time with all its stain images. The
   picture shows each type's area as it stands (GET …/areas: detection's with everyone's saved edits, hatched, the
   chosen type strong; detection's own areas, what the analysis uses, never change here) and detection's objects
   faintly, for context. The brush is always on and is the field tool's (`BrushMasks`, labels/brush.ts): 1–4 choose
   the type, whose current area (detection's with everyone's saved edits) is drawn as one merged region; a drag adds to
   it, Shift-drag takes out of it (detection's part too); Tab / Shift+Tab (or + / −) the brush size, the wheel zooms, the
   middle button pans, Z takes back a stroke, Space hides everything drawn over the tissue, [ / ] the other stain.
   Enter says "this stain is done" and goes to the next stain; after the last one the core is
   saved (every stain looked at, an edit for each stain painted on) and the next core opens. People's areas teach and
   test the bad-area detection; the analysis keeps detection's own. The page is blinded: the core and its stains, never
   a group or a diagnosis. */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import type { AreaCore, AreaImage, AreaNext, FieldAreas, PictureState, RoundProgress, SessionDetail } from '../api'
import { api, apiUrl, enc } from '../api'
import { ErrorNote, Loading, Progress, withError } from '../components'
import { DeepZoom } from '../deepzoom'
import type { Mark, PaintPoint, Pyramid } from '../deepzoom'
import { useFetch, useStored } from '../hooks'
import { AREA_SCALE, BRUSH_UM, BrushMasks, CLEARED, EXCLUDE_TYPES, brushStep, colourOfType } from './brush'
import { useFitScreen, useLeaveGuard } from './rounds'

export const AREAS_TITLE = 'Mark bad areas'
export const AREAS_TEXT = 'One core at a time with all its stains: paint where the tissue is folded, torn, dirty, missing or out of focus, or press Enter when a stain looks fine.'
export const PRACTICE_AREAS = 'practice-areas'
/** What someone does on the page, as the tutorial follows it. */
export type AreaEvent = 'type' | 'paint' | 'clear' | 'undo' | 'size' | 'stain' | 'fine' | 'overlays' | 'saved'
const HOLD_MS = 10000  // how often the page renews its hold on the core (and hears who else is on it)
const OBJECT_COLOUR = 'rgb(255 255 255 / 55%)'

const typing = (e: KeyboardEvent) => {
  const t = e.target as HTMLElement | null
  return !!t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)
}

/** The core-by-core progress: cores done by anyone, in progress, and the people. */
export function AreaProgressLine({ progress, me }: { progress: RoundProgress | null; me?: string | null }) {
  if (!progress) return null
  const open = progress.open_now ?? 0
  const others = progress.people.slice(0, 4)
  return (
    <div className="round-progress">
      <span className="small secondary"><b>{progress.decided.toLocaleString()}</b> of {progress.total.toLocaleString()} core{progress.total === 1 ? '' : 's'} checked
        {open > 0 && <> · {open.toLocaleString()} in progress</>}
        {others.length > 0 && <span className="muted"> · {others.map((p) => `${p.person === me ? 'you' : p.person} ${p.decided.toLocaleString()}`).join(' · ')}</span>}</span>
      <Progress done={progress.decided} total={progress.total} />
    </div>
  )
}

/** The session page: the next core for this person, then the one after, until none is left. */
export function MarkAreas({ detail, onEvent, coach, onFinished }: {
  detail: SessionDetail; onEvent?: (e: AreaEvent) => void; coach?: ReactNode; onFinished?: () => void
}) {
  useFitScreen()
  const meta = detail.meta as SessionDetail['meta'] & { stains_of?: Record<string, string[]>; progress?: RoundProgress }
  const name = meta.name
  const practice = name === PRACTICE_AREAS
  const [next, setNext] = useState<AreaNext | null>(null)
  const [progress, setProgress] = useState<RoundProgress | null>(meta.progress ?? null)
  const [error, setError] = useState<string | null>(null)
  const [practised, setPractised] = useState(0)
  const ask = useCallback(async (skip = '') => {
    const found = await withError(() => api.get<AreaNext>(`/api/annotate/${name}/areas/next${skip ? `?skip=${enc(skip)}` : ''}`), setError)
    if (found) { setNext(found); setProgress(found.progress) }
  }, [name])
  useEffect(() => { void ask() }, [ask])

  const top = (
    <div className="annotate-top">
      <div className="annotate-heading">
        <span className="row small"><Link to="/annotate">← All sessions</Link>
          {!practice && <Link to={`/annotate/tutorial?kind=areas&next=${enc(`/annotate/${name}`)}`} title="The short tutorial of this page again">Tutorial</Link>}</span>
        <h1>{meta.title}</h1>
        <div className="muted small">{AREAS_TITLE}{next?.core_id ? ` · core ${next.core_id}` : ''}</div>
      </div>
      <div className="annotate-progress">{!practice && <AreaProgressLine progress={progress} me={detail.annotator} />}</div>
    </div>
  )
  if (!next) return <div className="annotate-page round-page fit-page">{top}<ErrorNote error={error} />{!error && <Loading label="Finding a core for you…" />}</div>
  if (!next.core_id) {
    return (
      <div className="annotate-page round-page fit-page">{top}
        <div className="card empty-state">
          <p style={{ marginTop: 0 }}><b>{next.taken ? 'The other cores are open with someone else right now.' : 'Every core of this session has been checked — thank you!'}</b></p>
          <div className="row">{next.taken > 0 && <button className="btn" onClick={() => void ask()}>Look again</button>}<Link className="btn primary" to="/annotate">All sessions</Link></div>
        </div>
      </div>
    )
  }
  return (
    <div className="annotate-page round-page fit-page areas-page">
      {top}
      {meta.instructions && <div className="instructions">{meta.instructions}</div>}
      <ErrorNote error={error} />
      {practice && practised > 0 && <div className="found found-good small" role="status">Saved (this practice keeps nothing). In a real session the next core opens now.</div>}
      <CoreTask key={`${next.core_id}-${practised}`} name={name} core={next.core_id} after={next.after} afterStains={next.after ? meta.stains_of?.[next.after] ?? [] : []}
        practice={practice} onEvent={onEvent} coach={coach}
        onSkip={() => void ask(next.core_id!)}
        onSaved={(p) => {
          onEvent?.('saved')
          if (p) setProgress(p)
          if (practice) { setPractised((n) => n + 1); onFinished?.(); return }
          void ask()
        }} />
    </div>
  )
}

/** One core: the stain tabs (with a tick for each stain looked at), the picture of the chosen stain and the brush. */
function CoreTask({ name, core, after, afterStains, practice, onEvent, coach, onSkip, onSaved }: {
  name: string; core: string; after: string | null; afterStains: string[]; practice: boolean
  onEvent?: (e: AreaEvent) => void; coach?: ReactNode; onSkip: () => void; onSaved: (progress: RoundProgress | null) => void
}) {
  const base = `/api/annotate/${name}/cores/${enc(core)}`
  const task = useFetch<AreaCore>(base)
  const [at, setAt] = useState(0)
  const [looked, setLooked] = useState<Set<string>>(() => new Set())
  const [kind, setKind] = useState('1')
  const [brushUm, setBrushUm] = useStored<number>('stainid.round2.brush', BRUSH_UM.start)  // the field tool's brush size too
  const [hidden, setHidden] = useState(false)
  const [objectsOn, setObjectsOn] = useStored('stainid.areas.objects', true)
  const [tick, setTick] = useState(0)  // the strokes changed (undo, a stroke ended)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [openBy, setOpenBy] = useState<string[]>([])
  const masks = useRef(new Map<string, BrushMasks>())
  const images = useMemo(() => task.data?.images ?? [], [task.data])
  const image: AreaImage | null = images[at] ?? null
  const tell = useCallback((e: AreaEvent) => onEvent?.(e), [onEvent])
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const painted = useMemo(() => images.filter((i) => masks.current.get(i.stain)?.dirty).map((i) => i.stain), [images, tick])
  const mayLeave = useLeaveGuard(saving ? 0 : painted.length)

  useEffect(() => { if (task.data) setOpenBy(task.data.open_by) }, [task.data])
  useEffect(() => {  // keep the core this person's while the page is open, and hear who else is on it
    if (practice) return
    const timer = setInterval(() => {
      if (document.hidden) return
      api.post<{ open_by: string[] }>(`${base}/hold`, {}).then((r) => setOpenBy(r.open_by)).catch(() => undefined)
    }, HOLD_MS)
    return () => clearInterval(timer)
  }, [base, practice])

  // ahead of time: the next stain's picture (made and its coarse tiles fetched) and the next core's pictures (made)
  useEffect(() => {
    const ahead = images[at + 1]
    if (!ahead) return
    const where = `${base}/${enc(ahead.stain)}`
    api.get<PictureState>(`${where}/picture`).then((p) => { if (p.ready) prefetchTiles(where, p) }).catch(() => undefined)
  }, [images, at, base])
  useEffect(() => {
    if (!after || !task.data) return
    for (const stain of afterStains) api.get(`/api/annotate/${name}/cores/${enc(after)}/${enc(stain)}/picture`).catch(() => undefined)
  }, [after, afterStains, name, task.data])

  const go = (i: number) => { if (images.length) { setAt((i + images.length) % images.length); tell('stain') } }
  const save = async () => {
    if (saving || !images.length) return
    const edits: Record<string, { pngs: Record<string, string>; scale: number }> = {}
    for (const i of images) {
      const pngs = masks.current.get(i.stain)?.pngs()
      if (pngs) edits[i.stain] = { pngs, scale: AREA_SCALE }
    }
    setSaving(true)
    const done = await withError(() => api.post<{ progress: RoundProgress }>(`${base}/save`, { edits, checked: images.map((i) => i.stain) }), setError)
    setSaving(false)
    if (done) onSaved(done.progress ?? null)
  }
  /** Enter: this stain is done; the next one not looked at yet, or (all looked at) save the core. */
  const fine = () => {
    if (!image) return
    const now = new Set(looked).add(image.stain)
    setLooked(now)
    tell('fine')
    const left = images.findIndex((i, k) => k > at && !now.has(i.stain))
    const first = images.findIndex((i) => !now.has(i.stain))
    if (left >= 0) setAt(left)
    else if (first >= 0) setAt(first)
    else void save()
  }
  const undo = () => {
    if (!image || !masks.current.get(image.stain)?.undo()) return
    setTick((t) => t + 1)
    tell('undo')
  }
  const size = (bigger: boolean) => { setBrushUm(brushStep(brushUm, bigger)); tell('size') }
  const latest = useRef({ fine, undo, go, at, hidden, size })
  latest.current = { fine, undo, go, at, hidden, size }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (typing(e) || e.altKey) return
      const l = latest.current
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); l.undo(); return }
      if (e.ctrlKey || e.metaKey) return
      const type = EXCLUDE_TYPES.find((t) => t.key === e.key)
      if (type) { setKind(type.key); tell('type'); return }
      if (e.key === 'Tab') { e.preventDefault(); l.size(e.shiftKey) }  // the brush size, as in the field tool
      else if (e.key === '+' || e.key === '=' || e.key === '-' || e.key === '−') { e.preventDefault(); e.stopPropagation(); l.size(e.key !== '-' && e.key !== '−') }
      else if (e.key === '[') { e.preventDefault(); l.go(l.at - 1) }
      else if (e.key === ']') { e.preventDefault(); l.go(l.at + 1) }
      else if (e.key.toLowerCase() === 'z') l.undo()
      else if (e.key === 'Enter') { e.preventDefault(); l.fine() }
      else if (e.key === ' ') { e.preventDefault(); setHidden(!l.hidden); tell('overlays') }
    }
    window.addEventListener('keydown', onKey, true)  // first, so + / − size the brush rather than zoom the picture
    return () => window.removeEventListener('keydown', onKey, true)
  }, [tell])

  if (!task.data) return <><ErrorNote error={task.error} />{!task.error && <Loading label="Opening the core…" />}</>
  const allLooked = images.every((i) => looked.has(i.stain))
  const notYet = images.filter((i) => !looked.has(i.stain)).map((i) => i.stain)
  const second = task.data.second_check
  return (
    <>
      {(second || openBy.length > 0) && <div className="found found-warn small field-others" role="status">
        {second && <>This core is a <b>second check</b>: someone else checks it too, so we learn how well people agree. Mark what you see yourself. </>}
        {openBy.length > 0 && <>{openBy.join(', ')} {openBy.length === 1 ? 'has' : 'have'} this core open too. {!practice && <button className="link-btn small"
          onClick={() => { if (mayLeave()) onSkip() }}>Take another core</button>}</>}
      </div>}
      <ErrorNote error={error} />
      <div className="annotate-workspace">
        <div className="annotate-image" onContextMenu={(e) => e.preventDefault()}>
          <div className="area-tabs" role="tablist" aria-label="Stains of this core">
            {images.map((i, k) => (
              <button key={i.stain} role="tab" aria-selected={k === at} onMouseDown={(e) => e.preventDefault()} className={`area-tab${k === at ? ' chosen' : ''}${looked.has(i.stain) ? ' is-looked' : ''}`}
                onClick={() => go(k)} title={looked.has(i.stain) ? `${i.stain}: looked at` : `${i.stain}: not looked at yet`}>
                <img src={apiUrl(`${base}/${enc(i.stain)}/thumb.jpg?size=256`)} alt="" draggable={false} />
                <span>{looked.has(i.stain) ? <span className="area-tick" aria-label="looked at">✓</span> : null}{i.stain}
                  {masks.current.get(i.stain)?.dirty ? <span className="area-dot" title="painted, not saved yet" /> : null}</span>
              </button>
            ))}
            <span className="small muted area-tabs-keys"><span className="kbd">[</span> <span className="kbd">]</span> switch stains</span>
          </div>
          {image && <StainView key={image.stain} base={`${base}/${enc(image.stain)}`} image={image} masks={masks.current} kind={kind}
            brushRadius={brushUm / (task.data.pixel_size_um || 0.27) / 2} hidden={hidden} objectsOn={objectsOn} tick={tick}
            onStroke={(erase) => { setTick((t) => t + 1); tell(erase ? 'clear' : 'paint') }} />}
        </div>
        <aside className="annotate-side round-side" aria-label="The brush, the keys and this core">
          {coach}
          <div className="card side-block">
            <h3>Brush</h3>
            <div className="area-types">
              {EXCLUDE_TYPES.map((t) => (
                <button key={t.key} className={`area-type${kind === t.key ? ' chosen' : ''}`} onMouseDown={(e) => e.preventDefault()} onClick={() => { setKind(t.key); tell('type') }} aria-pressed={kind === t.key}>
                  <span className="kbd">{t.key}</span><span className="swatch" style={{ background: t.colour }} />{t.name}</button>
              ))}
              <div className="area-type is-clear"><span className="kbd">Shift</span><span className="swatch" style={{ background: CLEARED.colour }} />drag: take it out of the
                {' '}{EXCLUDE_TYPES.find((t) => t.key === kind)?.name} area, detection’s part too</div>
            </div>
            <p className="small secondary area-size" style={{ margin: '6px 0 0' }}>Brush <b>{brushUm} µm</b> · <span className="kbd">Tab</span> smaller,
              {' '}<span className="kbd">Shift</span>+<span className="kbd">Tab</span> bigger</p>
            <p className="small muted" style={{ margin: '4px 0 0' }}>The hatched region is the type’s area as it stands: detection’s with everyone’s edits. A drag adds to
              it (one region, however many strokes; another type’s pixels become this type), Shift-drag takes out of it.</p>
          </div>
          <div className="card side-block">
            <h3>This core</h3>
            <p className="small secondary" style={{ marginTop: 0 }}>Look at each stain. Paint what should not be counted; press <span className="kbd">Enter</span> when a stain is done.
              {' '}{allLooked ? 'Every stain is looked at.' : `Still to look at: ${notYet.join(', ')}.`}</p>
            <button className="btn primary" disabled={saving || !allLooked} onClick={() => void save()}
              title={allLooked ? 'Save this core and go on to the next one' : `Look at every stain first (Enter on each): ${notYet.join(', ')}`}>
              {saving ? 'Saving…' : 'Save and continue'}</button>
            {painted.length > 0 && <p className="small muted" style={{ marginBottom: 0 }}>Painted, not saved yet: {painted.join(', ')}.</p>}
          </div>
          <div className="card side-block key-help small">
            <h3>Keys</h3>
            <div><span className="kbd">1</span>–<span className="kbd">4</span> type · drag adds to it · <span className="kbd">Shift</span>-drag takes out of it</div>
            <div><span className="kbd">Tab</span> / <span className="kbd">Shift</span>+<span className="kbd">Tab</span> (or <span className="kbd">+</span> <span className="kbd">−</span>) brush size · wheel: zoom · middle button: move</div>
            <div><span className="kbd">Z</span> undo a stroke · <span className="kbd">Space</span> hide the overlays</div>
            <div><span className="kbd">Enter</span> this stain is done · <span className="kbd">[</span> <span className="kbd">]</span> other stain</div>
            <label className="layer-toggle" style={{ marginTop: 6 }}><input type="checkbox" checked={objectsOn} onChange={(e) => setObjectsOn(e.target.checked)} />Show detection’s objects (faint)</label>
            <p className="muted" style={{ margin: '6px 0 0' }}>The areas you mark teach and test the bad-area detection; the analysis uses detection’s own areas (hatched).</p>
          </div>
        </aside>
      </div>
    </>
  )
}

/** A stain's coarse tiles (the level the whole core is first shown at, and the one under it), fetched ahead. */
function prefetchTiles(where: string, p: Pyramid) {
  const dpr = window.devicePixelRatio || 1
  const fitLevel = Math.max(0, Math.min(p.levels.length - 1, Math.floor(Math.log2(Math.max(1, p.width / (900 * dpr))))))
  for (const level of [fitLevel, Math.min(p.levels.length - 1, fitLevel + 1)]) {
    const info = p.levels[level]
    if (info.cols * info.rows > 64) continue
    for (let row = 0; row < info.rows; row++) for (let col = 0; col < info.cols; col++) {
      const img = new Image()
      img.src = apiUrl(`${where}/tiles/${level}/${col}_${row}.jpg?v=${p.version}`)
    }
  }
}

/** One stain image of the core: each type's area now (detection's with everyone's saved edits and the strokes since
 *  Save; the chosen type strong), detection's objects faintly, and the brush (`brushRadius` in core px). */
function StainView({ base, image, masks, kind, brushRadius, hidden, objectsOn, tick, onStroke }: {
  base: string; image: AreaImage; masks: Map<string, BrushMasks>; kind: string; brushRadius: number; hidden: boolean; objectsOn: boolean; tick: number
  onStroke: (erase: boolean) => void
}) {
  const [poll, setPoll] = useState(!image.picture?.ready)
  const picture = useFetch<PictureState>(image.picture?.ready ? null : `${base}/picture`, poll ? 1500 : undefined)
  const state = image.picture?.ready ? image.picture : picture.data
  const pyramid = state?.ready ? state : null
  useEffect(() => { if (pyramid || (state && !state.ready && state.state === 'failed')) setPoll(false) }, [pyramid, state])
  const current = useFetch<FieldAreas>(`${base}/areas`)
  const objects = useFetch<{ columns: string[]; rows: [number, number, boolean, number[][] | null][] }>(objectsOn ? `${base}/objects` : null)
  const marks = useMemo<Mark[]>(() => (objectsOn && !hidden ? (objects.data?.rows ?? []).map(([x, y, accepted, rings]) => ({
    x, y, r: 6, color: OBJECT_COLOUR, dashed: !accepted, rings: rings ?? undefined, circle: rings?.length ? false : undefined,
  })) : []), [objects.data, objectsOn, hidden])
  const ex = image.excluded

  const masksFor = (): BrushMasks | null => {
    if (!pyramid) return null
    let m = masks.get(image.stain)
    if (!m) { m = new BrushMasks(pyramid.width, pyramid.height, AREA_SCALE); masks.set(image.stain, m) }
    return m
  }
  const canvas = useRef<HTMLCanvasElement | null>(null)
  const cursor = useRef<SVGCircleElement | null>(null)
  const frame = useRef(0)
  const latestKind = useRef(kind)
  latestKind.current = kind
  const redraw = () => {
    if (frame.current) return
    frame.current = requestAnimationFrame(() => { frame.current = 0; const m = masks.get(image.stain); if (m && canvas.current) m.draw(canvas.current, latestKind.current) })
  }
  useEffect(() => () => cancelAnimationFrame(frame.current), [])
  useEffect(redraw, [tick, pyramid, kind])  // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {  // each type's current area from the server, under the strokes
    const m = masksFor()
    if (!m || !current.data) return
    let live = true
    void m.setBase(current.data).then(() => { if (live) redraw() })
    return () => { live = false }
  }, [current.data, pyramid])  // eslint-disable-line react-hooks/exhaustive-deps
  const setCanvas = useCallback((c: HTMLCanvasElement | null) => { canvas.current = c; if (c) { const m = masks.get(image.stain); if (m) m.draw(c, latestKind.current) } }, [masks, image.stain])
  const clearing = useRef(false)
  const last = useRef<PaintPoint | null>(null)
  const showCursor = (p: PaintPoint | null) => {
    const c = cursor.current
    last.current = p
    if (!c) return
    if (!p) { c.style.display = 'none'; return }
    c.style.display = ''
    c.setAttribute('cx', String(p.x))
    c.setAttribute('cy', String(p.y))
    c.setAttribute('r', String(brushRadius))
    c.setAttribute('stroke-width', String(1.6 / p.s))
    c.setAttribute('stroke', p.shift || clearing.current ? CLEARED.colour : colourOfType(kind))
  }
  useEffect(() => { if (last.current) showCursor(last.current) })  // the brush size or type changed
  const latestBrush = useRef(brushRadius)
  latestBrush.current = brushRadius

  if (!pyramid) {
    const failed = state && !state.ready && state.state === 'failed' ? state.error ?? 'unknown error' : null
    const ahead = state && !state.ready ? state.ahead ?? 0 : 0
    return (
      <div className="dz-preparing" style={{ height: 'calc(100vh - 260px)' }}>
        {state && !state.ready && state.thumb_size && <img src={apiUrl(`${base}/thumb.jpg?size=${state.thumb_size}`)} alt="" />}
        <div className="dz-preparing-note">{failed ? <><b>This picture could not be prepared.</b><ErrorNote error={failed} /></>
          : <Loading label={<span><b>Preparing this picture…</b><br /><span className="small">{ahead ? `${ahead} other picture${ahead === 1 ? '' : 's'} first. ` : ''}It takes about 10 s once.</span></span>} />}</div>
      </div>
    )
  }
  const types = (ex.kinds ?? []).filter((k) => (k.fraction ?? 0) > 0)
  return (
    <DeepZoom pyramid={pyramid} height="calc(100vh - 260px)" panLock cursor="none"
      tileUrl={(level, col, row) => apiUrl(`${base}/tiles/${level}/${col}_${row}.jpg?v=${pyramid.version}`)}
      marks={marks}
      onTileError={() => { setPoll(true); picture.reload() }}
      toolbar={<span className="row small">
        {hidden ? <span className="badge">Overlays hidden · <span className="kbd">Space</span></span>
          : ex.status !== 'done' ? <span className="muted">Detection has not marked bad areas on this image yet.</span>
            : <span className="muted" title="What the analysis leaves out: detection's automatic areas. They do not change from what you mark here.">
              Excluded from analysis (detection){types.length ? ': ' + types.map((k) => `${k.label} ${Math.round(100 * (k.fraction ?? 0))}%`).join(', ') : ': nothing'}</span>}
        {!hidden && <span className="muted"> · hatched: each type’s area as it stands (detection’s with people’s edits)</span>}
      </span>}
      onPress={(p) => {
        const m = masksFor()
        if (!m) return
        clearing.current = p.shift
        m.begin(p.x, p.y, latestBrush.current, kind, p.shift)
        redraw()
      }}
      onMove={(p) => {
        showCursor(p)
        const m = masks.get(image.stain)
        if (p && m?.painting) { m.extend(p.x, p.y); redraw() }
      }}
      onRelease={() => {
        const m = masks.get(image.stain)
        if (m?.end()) { redraw(); onStroke(clearing.current) }  // one step for Z
        clearing.current = false
      }}
      over={(unit) => (
        <>
          <canvas ref={setCanvas} className="area-pending" style={{ width: pyramid.width, height: pyramid.height, display: hidden ? 'none' : undefined }} />
          <svg className="area-svg" width={pyramid.width} height={pyramid.height} viewBox={`0 0 ${pyramid.width} ${pyramid.height}`} data-unit={unit}>
            <circle ref={cursor} r={0} fill="none" style={{ display: 'none' }} />
          </svg>
        </>
      )} />
  )
}
