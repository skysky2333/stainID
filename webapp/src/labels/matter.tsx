/* "Grey / white matter" (sessions of kind "matter", routes/matter.py, labels/matter.py): people paint the white matter
   of each core (to leave out), first on its NeuN image (detection's neurons drawn as a guide: grey matter is where they
   are), then on its other stain images, which start from the NeuN area carried over through the registration. There
   it is adjusted, or accepted as carried with one click.

   The session page is the picker: the cores by TMA, each with where it stands (to do, in progress, done, finished by
   you), and "Next core". A core page holds the core for this person while it is open (anyone else is told who is on
   it); a core they finished opens to look at, with "Redo this core". The brush (mattermask.ts) paints white matter, E
   or Shift-drag erases, Tab / Shift+Tab a smaller / bigger brush, Z undoes a stroke, Space hides the drawing, [ / ] the
   other stain, Enter saves the stain (on a carried stain with nothing changed: accepts it as carried). The page is
   blinded: the core and its stains, never a group or a diagnosis. People's areas train and evaluate a grey / white
   matter model; the analysis never reads them. */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import type { MatterArea, MatterCore, MatterCoreRow, PictureState, RoundProgress, SessionDetail } from '../api'
import { api, apiUrl, enc } from '../api'
import { ErrorNote, Loading, withError } from '../components'
import { DeepZoom } from '../deepzoom'
import type { Mark, PaintPoint } from '../deepzoom'
import { useFetch, useStored } from '../hooks'
import { PALETTE } from '../palette'
import { BRUSH_KEY, BRUSH_PX, CLEARED, brushStep } from './brush'
import { AreaProgressLine } from './markareas'
import { MATTER_COLOUR, MATTER_SCALE, MatterMask } from './mattermask'
import { BackToSets, LabellingBar, useFitScreen, useLeaveGuard } from './rounds'

export const MATTER_TITLE = 'Grey / white matter'
export const MATTER_TEXT = 'One core at a time: paint the white matter on the NeuN image, then check it on the other stains, where it is carried over for you.'
export const PRACTICE_MATTER = 'practice-matter'  // the tutorial's practice (labels/tutorial.tsx): nothing is kept
/** What someone does on a core page, as the tutorial follows it. */
export type MatterEvent = 'guide' | 'paint' | 'erase' | 'undo' | 'stain' | 'saved-source' | 'accepted' | 'saved-other' | 'finished'
const HOLD_MS = 10000
const NEURON_COLOUR = PALETTE.classes[1]
const STATE_TEXT: Record<MatterCoreRow['state'] | 'mine', string> = { not_started: 'To do', in_progress: 'In progress', done: 'Done', mine: 'Finished by you' }
const STATE_BADGE: Record<MatterCoreRow['state'] | 'mine', string> = { not_started: 'status-todo', in_progress: 'status-running', done: 'status-done', mine: 'status-done' }
const REGISTRATION_TEXT: Record<string, string> = { pass: 'lined up well', review: 'lined up roughly: check the edges', fail: 'could not be lined up well: check closely', same: 'the same image' }

const typing = (e: KeyboardEvent) => {
  const t = e.target as HTMLElement | null
  return !!t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)
}
type Meta = SessionDetail['meta'] & { progress?: RoundProgress; core_rows?: MatterCoreRow[] }

function Heading({ meta, core, progress, me }: { meta: Meta; core?: string; progress: RoundProgress | null; me?: string | null }) {
  return (
    <LabellingBar title={meta.title} about={`${MATTER_TITLE}${core ? ` · core ${core}` : ''}`}
      links={<><BackToSets />{core && meta.name !== PRACTICE_MATTER && <Link to={`/annotate/${meta.name}`}>All cores of this session</Link>}
        {meta.name !== PRACTICE_MATTER && <Link to={`/tutorial?kind=matter&next=${enc(`/annotate/${meta.name}`)}`} title="The short tutorial of this page again">Tutorial</Link>}</>}
      progress={<AreaProgressLine progress={progress} me={me} verb="done" />} />
  )
}

/** The session page: the picker (the cores by TMA, each with where it stands) and the next core. */
export function MatterSession({ detail }: { detail: SessionDetail }) {
  const meta = detail.meta as Meta
  const navigate = useNavigate()
  const [error, setError] = useState<string | null>(null)
  const [none, setNone] = useState<string | null>(null)
  const rows = useMemo(() => meta.core_rows ?? [], [meta.core_rows])
  const me = detail.annotator
  const byTma = useMemo(() => {
    const out = new Map<string, MatterCoreRow[]>()
    for (const r of rows) out.set(r.tma, [...(out.get(r.tma) ?? []), r])
    return [...out.entries()].sort(([a], [b]) => a.localeCompare(b, undefined, { numeric: true }))
  }, [rows])
  const next = async () => {
    const found = await withError(() => api.get<{ core_id: string | null; taken: number }>(`/api/annotate/${meta.name}/matter/next`), setError)
    if (!found) return
    if (found.core_id) navigate(`/annotate/${meta.name}/${enc(found.core_id)}`)
    else setNone(found.taken ? 'The cores left are open with someone else right now.' : 'Every core of this session is done — thank you!')
  }
  const look = (r: MatterCoreRow): MatterCoreRow['state'] | 'mine' => (r.done_by_me && r.state !== 'done' ? 'mine' : r.state)
  const mine = rows.filter((r) => r.done_by_me).length
  return (
    <div className="annotate-page matter-home">
      <Heading meta={meta} progress={meta.progress ?? null} me={me} />
      {meta.instructions && <div className="instructions">{meta.instructions}</div>}
      <ErrorNote error={error} />
      <div className="card matter-start">
        <p style={{ marginTop: 0 }}>{MATTER_TEXT} Detection’s neurons are drawn on NeuN as a guide: the grey matter is where they are.</p>
        <div className="row"><button className="btn primary" onClick={() => void next()}>{mine ? 'Next core' : 'Start'}</button>
          <span className="small muted">{mine ? `You finished ${mine} core${mine === 1 ? '' : 's'}.` : ''} Or open any core below.</span></div>
        {none && <p className="small found found-good" role="status" style={{ marginBottom: 0 }}>{none}</p>}
      </div>
      {byTma.map(([tma, cores]) => (
        <div key={tma} className="card matter-tma">
          <h3>{tma ? `TMA ${tma}` : 'Cores'} <span className="muted small">· {cores.filter((r) => r.state === 'done').length} of {cores.length} done</span></h3>
          <div className="matter-cores">
            {cores.map((r) => {
              const state = look(r)
              const others = r.open_by.filter((p) => p !== me)
              return (
                <Link key={r.core_id} className={`matter-core${others.length ? ' is-taken' : ''}`} to={`/annotate/${meta.name}/${enc(r.core_id)}`}
                  title={`${r.core_id} · ${r.stains.join(', ')}${r.done_by.length ? ` · done by ${r.done_by.map((p) => (p === me ? 'you' : p)).join(', ')}` : ''}${others.length ? ` · ${others.join(', ')} working on it now` : ''}`}>
                  <img src={apiUrl(`/api/annotate/${meta.name}/cores/${enc(r.core_id)}/${enc(r.stains[0] ?? 'NeuN')}/thumb.jpg?size=256`)} alt="" loading="lazy" />
                  <span className="matter-core-name">{r.core}</span>
                  <span className={`badge ${STATE_BADGE[state]}`}>{others.length ? `${others.join(', ')} on it` : STATE_TEXT[state]}</span>
                  {r.saved_by_me.length > 0 && state !== 'mine' && state !== 'done' && <span className="small muted">{r.saved_by_me.join(', ')} saved</span>}
                </Link>
              )
            })}
          </div>
        </div>
      ))}
    </div>
  )
}

/** One core of the session: its stain tabs (NeuN first), the picture of the chosen stain and the brush. */
export function MatterCorePage({ detail, core, onEvent, coach, onFinished }: {
  detail: SessionDetail; core: string; onEvent?: (e: MatterEvent) => void; coach?: ReactNode; onFinished?: () => void
}) {
  useFitScreen()
  const meta = detail.meta as Meta
  const name = meta.name
  const practice = name === PRACTICE_MATTER  // the tutorial: saves are echoed; what was saved is kept on this page only
  const [practised, setPractised] = useState<Record<string, string>>({})  // practice: each saved stain's area (its PNG)
  const tell = useCallback((e: MatterEvent) => onEvent?.(e), [onEvent])
  const navigate = useNavigate()
  const [params, setParams] = useSearchParams()
  const edit = params.get('edit') === '1'
  const base = `/api/annotate/${name}/matter/cores/${enc(core)}`
  const task = useFetch<MatterCore>(`${base}${edit ? '?edit=true' : ''}`)
  const data = task.data
  const readOnly = !!data?.read_only
  const [at, setAt] = useState(0)
  const [erase, setErase] = useState(false)
  const [brushSize, setBrushSize] = useStored<number>(BRUSH_KEY, BRUSH_PX.start)
  const [hidden, setHidden] = useState(false)
  const [neuronsOn, setNeuronsOn] = useStored('stainid.matter.neurons', true)
  const [tick, setTick] = useState(0)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [finished, setFinished] = useState(false)
  const [saves, setSaves] = useState(0)  // a saved stain's area is fetched again
  const [progress, setProgress] = useState<RoundProgress | null>(meta.progress ?? null)
  const masks = useRef(new Map<string, MatterMask>())
  const [areaOf, setAreaOf] = useState<Record<string, MatterArea | undefined>>({})
  const images = useMemo(() => (data?.images ?? []).map((i) => (practice ? { ...i, saved_by_me: i.stain in practised } : i)), [data, practice, practised])
  const image = images[at] ?? null
  const source = data?.source ?? 'NeuN'
  const sourceSaved = !!images.find((i) => i.stain === source)?.saved_by_me
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const painted = useMemo(() => images.filter((i) => masks.current.get(i.stain)?.dirty).map((i) => i.stain), [images, tick])
  const mayLeave = useLeaveGuard(saving || readOnly ? 0 : painted.length)
  useEffect(() => { if (data) setProgress(data.progress) }, [data])
  useEffect(() => {  // keep the core this person's while the page is open; closing it frees it for the others at once
    if (!data || readOnly || practice) return
    const timer = setInterval(() => { if (!document.hidden) api.post(`${base}/hold`, {}).catch(() => undefined) }, HOLD_MS)
    const leave = () => { void fetch(apiUrl(`${base}/leave`), { method: 'POST', keepalive: true, credentials: 'include' }).catch(() => undefined) }
    window.addEventListener('pagehide', leave)
    return () => { clearInterval(timer); window.removeEventListener('pagehide', leave); leave() }
  }, [data, readOnly, practice, base])

  const locked = (stain: string) => !readOnly && stain !== source && !sourceSaved
  const go = (i: number) => {
    if (!images.length) return
    const k = (i + images.length) % images.length
    if (locked(images[k].stain)) { setError(`Paint and save ${source} first: the other stains start from its white matter.`); return }
    setError(null)
    if (k !== at) tell('stain')
    setAt(k)
  }
  const area = image ? areaOf[image.stain] : undefined
  const mask = image ? masks.current.get(image.stain) : undefined
  const carried = area?.from === 'carried'
  const dirty = !!mask?.dirty

  const save = async (accepted: boolean) => {
    if (!image || saving || readOnly) return
    const m = masks.current.get(image.stain)
    if (!accepted && !m) return
    setSaving(true)
    const done = await withError(() => api.post<{ finished: boolean; progress: RoundProgress }>(`${base}/${enc(image.stain)}/save`,
      accepted ? { accepted: true } : { png: m!.png(), scale: m!.scale }), setError)
    setSaving(false)
    if (!done) return
    tell(image.stain === source ? 'saved-source' : accepted ? 'accepted' : 'saved-other')
    if (practice) {  // nothing is on the server: the page keeps what was saved, the other stains are carried from it
      const png = accepted ? area?.kinds?.white ?? '' : m!.png()
      const now = { ...practised, [image.stain]: png }
      m?.saved()
      setPractised(now)
      if (image.stain === source) for (const [stain, other] of masks.current) if (stain !== source && !other.dirty) masks.current.delete(stain)
      const next = images.findIndex((i) => !(i.stain in now))
      if (next < 0) { setFinished(true); tell('finished'); onFinished?.(); return }
      setAt(next)
      return
    }
    setProgress(done.progress)
    if (image.stain === source) for (const [stain, other] of masks.current) if (stain !== source && !other.dirty) masks.current.delete(stain)  // they start from the new carry
    const left = images.findIndex((i, k) => k !== at && !i.saved_by_me)
    masks.current.delete(image.stain)
    task.reload()
    setAreaOf((a) => ({ ...a, [image.stain]: undefined }))
    setSaves((n) => n + 1)
    if (done.finished) { setFinished(true); return }
    if (left >= 0) setAt(left)
  }
  const nextCore = async () => {
    const found = await withError(() => api.get<{ core_id: string | null }>(`/api/annotate/${name}/matter/next?skip=${enc(core)}`), setError)
    if (found?.core_id) navigate(`/annotate/${name}/${enc(found.core_id)}`)
    else if (found) navigate(`/annotate/${name}`)
  }
  const undo = () => { if (image && masks.current.get(image.stain)?.undo()) { setTick((t) => t + 1); tell('undo') } }
  const enter = () => { if (!image || readOnly || locked(image.stain)) return; void save(carried && !dirty) }
  const latest = useRef({ go, at, undo, enter, hidden, brushSize })
  latest.current = { go, at, undo, enter, hidden, brushSize }
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (typing(e) || e.altKey) return
      const l = latest.current
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); l.undo(); return }
      if (e.ctrlKey || e.metaKey) return
      const key = e.key.toLowerCase()
      if (key === 'b') setErase(false)
      else if (key === 'e') setErase(true)
      else if (e.key === 'Tab') { e.preventDefault(); setBrushSize(brushStep(l.brushSize, e.shiftKey)) }
      else if (e.key === '[') { e.preventDefault(); l.go(l.at - 1) }
      else if (e.key === ']') { e.preventDefault(); l.go(l.at + 1) }
      else if (key === 'z') l.undo()
      else if (e.key === 'Enter') { e.preventDefault(); l.enter() }
      else if (e.key === ' ') { e.preventDefault(); setHidden(!l.hidden) }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [setBrushSize])

  const taken = task.error && /working on this core/.test(task.error) ? task.error.replace(/^\d{3}: /, '') : null
  if (taken) {
    return (
      <div className="annotate-page"><Heading meta={meta} core={core} progress={progress} me={detail.annotator} />
        <div className="card" style={{ maxWidth: 560 }}>
          <h2 style={{ marginTop: 0 }}>{taken}</h2>
          <p className="secondary">Only one person works on a core at a time. It opens for you again once they close it.</p>
          <Link className="btn primary" to={`/annotate/${name}`}>Back to the session’s cores</Link>
        </div>
      </div>
    )
  }
  if (!data) return <div className="annotate-page round-page fit-page"><Heading meta={meta} core={core} progress={progress} me={detail.annotator} />
    <ErrorNote error={task.error} />{!task.error && <Loading label="Opening the core…" />}</div>

  const savedHere = image?.mine
  const status: ReactNode = !image ? null
    : area?.from === 'waiting' ? <>Paint and save <b>{source}</b> first: this stain starts from its white matter.</>
      : area?.from === 'carried' ? <>Carried from <b>{area.carried_from}</b>: the images {REGISTRATION_TEXT[area.registration?.status ?? ''] ?? 'are lined up'}
        {area.registration?.how === 'computed' ? ' (lined up here, not in the registration table)' : ''}. Adjust it, or accept it as carried.</>
        : area?.from === 'saved' ? <>Saved{area.saved?.person && area.saved.person !== detail.annotator ? ` by ${area.saved.person}` : ' by you'}
          {area.saved?.carried_from ? (area.saved.accepted ? `, accepted as carried from ${area.saved.carried_from}` : `, adjusted from ${area.saved.carried_from}`) : ''}.</>
          : image.stain === source ? <>Paint the white matter: the paler tissue with few neurons. Nothing painted means no white matter on this core.</> : null
  return (
    <div className="annotate-page round-page fit-page areas-page matter-page">
      <Heading meta={meta} core={data.core} progress={practice ? null : progress} me={detail.annotator} />
      {meta.instructions && <div className="instructions">{meta.instructions}</div>}
      {readOnly && <div className="found found-good small field-others" role="status">
        {data.done_by.includes(detail.annotator ?? '') ? 'You finished this core' : `Done by ${data.done_by.join(', ')}`}: it opens to look at.{' '}
        {data.done_by.includes(detail.annotator ?? '') && <button className="link-btn small" onClick={() => { masks.current.clear(); setAreaOf({}); setParams({ edit: '1' }) }}>Redo this core</button>}
      </div>}
      <ErrorNote error={error} />
      {finished && (practice ? <div className="found found-good small" role="status">Saved (this practice keeps nothing). In a real session the next core opens now.</div>
        : <div className="found found-good small" role="status">This core is done — thank you.{' '}
          <button className="btn small primary" onClick={() => void nextCore()}>Next core</button>{' '}<Link className="btn small" to={`/annotate/${name}`}>All cores</Link></div>)}
      <div className="annotate-workspace">
        <div className="annotate-image" onContextMenu={(e) => e.preventDefault()}>
          <div className="area-tabs" role="tablist" aria-label="Stains of this core">
            {images.map((i, k) => (
              <button key={i.stain} role="tab" aria-selected={k === at} onMouseDown={(e) => e.preventDefault()} disabled={locked(i.stain)}
                className={`area-tab${k === at ? ' chosen' : ''}${i.saved_by_me ? ' is-looked' : ''}`} onClick={() => go(k)}
                title={locked(i.stain) ? `${i.stain}: after ${source}` : i.saved_by_me ? `${i.stain}: saved` : `${i.stain}: not saved yet`}>
                <img src={apiUrl(`/api/annotate/${name}/cores/${enc(core)}/${enc(i.stain)}/thumb.jpg?size=256`)} alt="" draggable={false} />
                <span>{i.saved_by_me ? <span className="area-tick" aria-label="saved">✓</span> : null}{i.stain}{i.stain === source ? ' · first' : ''}
                  {masks.current.get(i.stain)?.dirty ? <span className="area-dot" title="painted, not saved yet" /> : null}</span>
              </button>
            ))}
            <span className="small muted area-tabs-keys"><span className="kbd">[</span> <span className="kbd">]</span> switch stains</span>
          </div>
          {image && <MatterView key={`${image.stain}:${readOnly}:${saves}`} name={name} core={core} stain={image.stain} picture={image.picture} masks={masks.current}
            readOnly={readOnly} erase={erase} brushRadius={brushSize / 2} hidden={hidden} neurons={neuronsOn && image.stain === source}
            carryFrom={practice && image.stain !== source ? practised[source] ?? '' : undefined}
            onArea={(a) => setAreaOf((was) => ({ ...was, [image.stain]: a }))} onStroke={(erased) => { setTick((t) => t + 1); tell(erased ? 'erase' : 'paint') }} />}
        </div>
        <aside className="annotate-side round-side" aria-label="The brush, this stain and the keys">
          {coach}
          {readOnly ? (image?.stain === source && <div className="card side-block"><label className="layer-toggle"><input type="checkbox" checked={neuronsOn} onChange={(e) => setNeuronsOn(e.target.checked)} />
            <span className="swatch" style={{ background: 'transparent', border: `2px solid ${NEURON_COLOUR}`, borderRadius: '50%' }} />Show detection’s neurons (a guide)</label></div>)
            : <div className="card side-block">
            <h3>Brush</h3>
            <div className="area-types">
              <button className={`area-type${!erase ? ' chosen' : ''}`} disabled={readOnly} onMouseDown={(e) => e.preventDefault()} onClick={() => setErase(false)} aria-pressed={!erase}>
                <span className="kbd">B</span><span className="swatch" style={{ background: MATTER_COLOUR }} />paint white matter</button>
              <button className={`area-type${erase ? ' chosen' : ''}`} disabled={readOnly} onMouseDown={(e) => e.preventDefault()} onClick={() => setErase(true)} aria-pressed={erase}>
                <span className="kbd">E</span><span className="swatch" style={{ background: CLEARED.colour }} />eraser (or <span className="kbd">Shift</span>-drag)</button>
            </div>
            <p className="small secondary area-size" style={{ margin: '6px 0 0' }}>Brush <b>{brushSize} px</b> · <span className="kbd">Tab</span> smaller,
              {' '}<span className="kbd">Shift</span>+<span className="kbd">Tab</span> bigger · zoom in for a finer one</p>
            {image?.stain === source && <label className="layer-toggle" style={{ marginTop: 6 }}><input type="checkbox" checked={neuronsOn} onChange={(e) => { setNeuronsOn(e.target.checked); tell('guide') }} />
              <span className="swatch" style={{ background: 'transparent', border: `2px solid ${NEURON_COLOUR}`, borderRadius: '50%' }} />Show detection’s neurons (a guide)</label>}
          </div>}
          <div className="card side-block">
            <h3>{image ? image.stain : 'This stain'}</h3>
            <p className="small secondary" style={{ marginTop: 0 }}>{status}</p>
            {!readOnly && image && !locked(image.stain) && area && area.from !== 'waiting' && <div className="row">
              {carried && <button className={`btn${dirty ? '' : ' primary'}`} disabled={saving} onClick={() => void save(true)}
                title="Save the white matter carried from NeuN as it is (Enter when nothing is changed)">Accept as carried</button>}
              {(!carried || dirty) && <button className="btn primary" disabled={saving} onClick={() => void save(false)}
                title="Save this stain's white matter as painted (Enter)">{saving ? 'Saving…' : savedHere && !dirty ? `Save ${image.stain} again` : `Save ${image.stain}`}</button>}
            </div>}
            {painted.length > 0 && !readOnly && <p className="small muted" style={{ marginBottom: 0 }}>Painted, not saved yet: {painted.join(', ')}.</p>}
            {!readOnly && <p className="small muted" style={{ marginBottom: 0 }}>{images.filter((i) => i.saved_by_me).length} of {images.length} stains saved.{' '}
              {!practice && <button className="link-btn small" onClick={() => { if (mayLeave()) void nextCore() }}>Take another core</button>}</p>}
          </div>
          <div className="card side-block key-help small">
            <h3>Keys</h3>
            <div>drag paints · <span className="kbd">E</span> eraser · <span className="kbd">B</span> brush · <span className="kbd">Shift</span>-drag erases</div>
            <div><span className="kbd">Tab</span> / <span className="kbd">Shift</span>+<span className="kbd">Tab</span> a smaller / bigger brush · <span className="kbd">+</span> <span className="kbd">−</span> zoom · arrow keys move · middle button: move</div>
            <div><span className="kbd">Z</span> undo a stroke · <span className="kbd">Space</span> hide the drawing</div>
            <div><span className="kbd">Enter</span> save this stain (accept as carried when nothing changed) · <span className="kbd">[</span> <span className="kbd">]</span> other stain</div>
            <p className="muted" style={{ margin: '6px 0 0' }}>What you mark teaches and tests a grey / white matter model; the analysis does not use it.</p>
          </div>
        </aside>
      </div>
    </div>
  )
}

/** One stain image: the picture, its white matter (the brush's start: saved, carried or none) and on NeuN detection's
 *  neurons as a guide. */
function MatterView({ name, core, stain, picture, masks, readOnly, erase, brushRadius, hidden, neurons, carryFrom, onArea, onStroke }: {
  name: string; core: string; stain: string; picture: PictureState | null; masks: Map<string, MatterMask>; readOnly: boolean; erase: boolean
  brushRadius: number; hidden: boolean; neurons: boolean; onArea: (a: MatterArea) => void; onStroke: (erased: boolean) => void
  /** the tutorial: the source stain's area as painted on this page (nothing of it is on the server), carried here */
  carryFrom?: string
}) {
  const where = `/api/annotate/${name}/cores/${enc(core)}/${enc(stain)}`
  const [poll, setPoll] = useState(!picture?.ready)
  const fetched = useFetch<PictureState>(picture?.ready ? null : `${where}/picture`, poll ? 1500 : undefined)
  const state = picture?.ready ? picture : fetched.data
  const pyramid = state?.ready ? state : null
  useEffect(() => { if (pyramid || (state && !state.ready && state.state === 'failed')) setPoll(false) }, [pyramid, state])
  const stainBase = `/api/annotate/${name}/matter/cores/${enc(core)}/${enc(stain)}`
  const asked = useFetch<MatterArea>(carryFrom === undefined ? `${stainBase}/area${readOnly ? '?look=true' : ''}` : null)
  const [carriedHere, setCarriedHere] = useState<{ data: MatterArea | null; error: string | null }>({ data: null, error: null })
  useEffect(() => {
    if (carryFrom === undefined) return
    let live = true
    api.post<MatterArea>(`${stainBase}/carry`, { png: carryFrom, scale: MATTER_SCALE })
      .then((data) => { if (live) setCarriedHere({ data, error: null }) }).catch((e: Error) => { if (live) setCarriedHere({ data: null, error: e.message }) })
    return () => { live = false }
  }, [carryFrom, stainBase])
  const area = carryFrom === undefined ? asked : carriedHere
  const objects = useFetch<{ rows: [number, number, boolean, number[][] | null][] }>(neurons ? `${where}/objects` : null)
  const marks = useMemo<Mark[]>(() => (neurons && !hidden ? (objects.data?.rows ?? []).filter(([, , accepted]) => accepted).map(([x, y, , rings]) => ({
    x, y, r: 6, color: NEURON_COLOUR, rings: rings ?? undefined, circle: rings?.length ? false : undefined,
  })) : []), [objects.data, neurons, hidden])
  const host = useRef<HTMLDivElement | null>(null)
  const cursor = useRef<SVGCircleElement | null>(null)
  const maskFor = useCallback((): MatterMask | null => {
    if (!pyramid) return null
    let m = masks.get(stain)
    if (!m) { m = new MatterMask(pyramid.width, pyramid.height); masks.set(stain, m) }
    return m
  }, [pyramid, masks, stain])
  const [started, setStarted] = useState(false)
  useEffect(() => {  // the area to start from (unless strokes are kept from before a stain switch)
    if (!area.data) return
    onArea(area.data)
    const m = maskFor()
    if (!m) return
    if (m.dirty) { m.attach(host.current); setStarted(true); return }
    let live = true
    void m.start(area.data.kinds?.white, area.data.width, area.data.height).then(() => { if (live) { m.attach(host.current); setStarted(true) } })
    return () => { live = false }
  }, [area.data, maskFor])  // eslint-disable-line react-hooks/exhaustive-deps
  const setHost = useCallback((h: HTMLDivElement | null) => {
    host.current = h
    masks.get(stain)?.attach(h)
  }, [masks, stain])
  const last = useRef<PaintPoint | null>(null)
  const erasing = useRef(false)
  const showCursor = (p: PaintPoint | null) => {
    const c = cursor.current
    last.current = p
    if (!c) return
    if (!p || readOnly) { c.style.display = 'none'; return }
    c.style.display = ''
    c.setAttribute('cx', String(p.x))
    c.setAttribute('cy', String(p.y))
    c.setAttribute('r', String(brushRadius / p.s))
    c.setAttribute('stroke-width', String(1.6 / p.s))
    c.setAttribute('stroke', p.shift || erase || erasing.current ? CLEARED.colour : MATTER_COLOUR)
  }
  useEffect(() => { if (last.current) showCursor(last.current) })

  if (!pyramid) {
    const failed = state && !state.ready && state.state === 'failed' ? state.error ?? 'unknown error' : null
    return (
      <div className="dz-preparing dz-preparing-fill">
        {state && !state.ready && state.thumb_size && <img src={apiUrl(`${where}/thumb.jpg?size=${state.thumb_size}`)} alt="" />}
        <div className="dz-preparing-note">{failed ? <><b>This picture could not be prepared.</b><ErrorNote error={failed} /></>
          : <Loading label={<span><b>Preparing this picture…</b><br /><span className="small">It takes about 10 s once.</span></span>} />}</div>
      </div>
    )
  }
  return (
    <DeepZoom pyramid={pyramid} height="fill" panLock={!readOnly} cursor={readOnly ? undefined : 'none'}
      tileUrl={(level, col, row) => apiUrl(`${where}/tiles/${level}/${col}_${row}.jpg?v=${pyramid.version}`)}
      marks={marks}
      onTileError={() => { setPoll(true); fetched.reload() }}
      toolbar={<span className="row small">
        {hidden ? <span className="badge">Drawing hidden · <span className="kbd">Space</span></span>
          : !started ? <Loading inline label="Loading the white matter…" />
            : <span className="muted"><span className="swatch" style={{ background: MATTER_COLOUR, display: 'inline-block', width: 10, height: 10, marginRight: 4 }} />hatched: white matter
              {neurons ? <> · <span style={{ color: NEURON_COLOUR }}>outlined</span>: detection’s neurons</> : null}</span>}
        <ErrorNote error={area.error} />
      </span>}
      onPress={readOnly ? undefined : (p) => {
        const m = maskFor()
        if (!m) return
        erasing.current = p.shift || erase
        m.begin(p.x, p.y, brushRadius / p.s, erasing.current)
      }}
      onMove={(p) => {
        showCursor(p)
        const m = masks.get(stain)
        if (p && m?.painting) m.extend(p.x, p.y)
      }}
      onRelease={readOnly ? undefined : () => {
        if (masks.get(stain)?.end()) onStroke(erasing.current)
        erasing.current = false
      }}
      over={(unit) => (
        <>
          <div ref={setHost} className="area-pending" style={{ width: pyramid.width, height: pyramid.height, display: hidden ? 'none' : undefined }} />
          <svg className="area-svg" width={pyramid.width} height={pyramid.height} viewBox={`0 0 ${pyramid.width} ${pyramid.height}`} data-unit={unit}>
            <circle ref={cursor} r={brushRadius * unit} strokeWidth={1.6 * unit} fill="none" style={{ display: 'none' }} />
          </svg>
        </>
      )} />
  )
}
