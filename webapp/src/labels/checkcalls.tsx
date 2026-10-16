/* Round 1, "Check the AI's calls": one candidate at a time, accepted or rejected, with the AI's class, type, score and
   outline. The keys are those of round 2, acting on the object on screen: Enter keeps the AI's call, 1–9 (and the type
   keys) choose the class, 0 unsure, Delete not an object, Tab / Shift+Tab a bigger or smaller outline (the three
   Segment Anything masks come with the object: no waiting), Space precise editing (right-click a spot to include,
   left-click one to leave out, drag the corners; Space again saves), Z undo. The chosen outline is saved with the answer.
   The next objects are on the page already with their pictures (the quick-check machinery: held for this person, saved
   in the background). Every few seconds the page asks what others decided meanwhile: those objects leave the page at
   once, and the progress counts everyone's work. */
import { useCallback, useEffect, useRef, useState } from 'react'
import type { MouseEvent as ReactMouseEvent } from 'react'
import { Link } from 'react-router-dom'
import type { QueuePictures, RoundDetail, RoundItem, RoundNext, RoundProgress, RoundSamOptions, SessionDetail } from '../api'
import { api, apiUrl, enc } from '../api'
import { Empty, ErrorNote, Loading, fmt } from '../components'
import type { ClassPick } from './classKeys'
import { classOf } from './classKeys'
import { ClassChooser } from './classes'
import { PicturesBeingMade, type Picture, useLasting, usePreload, useSaver } from './quickcheck'
import type { RoundEvent } from './rounds'
import { KeyLegend, MaskSwitch, ROUND_TITLE, RoundProgressLine, TIER_TEXT, callText, lookOf, lookSwatch, polygonPoints, stepMask, unsureClass, useCandidateKeys,
  useLive } from './rounds'

const ASK = 8  // candidates asked for at a time: the one on screen and the next ones, pictures loaded
const REFILL = 4
const GUARD_MS = 150  // a key pressed sooner after a new object appears does not answer it (a double press)
const pictureUrl = (session: string, o: RoundItem) => apiUrl(`/api/annotate/${session}/object.jpg?tile_id=${enc(o.tile_id)}&x=${o.x}&y=${o.y}`)
type Answer = { kind: 'keep' } | { kind: 'class'; pick: ClassPick }
/** Precise editing of the outline on screen: the polygon (crop pixels) and the spots clicked (x, y, 1 include / 0 leave out). */
interface Precise { points: number[][]; clicks: number[][]; busy: boolean }

function circle(x: number, y: number, r: number) {
  return Array.from({ length: 16 }, (_, i) => [x + r * Math.cos((i * Math.PI) / 8), y + r * Math.sin((i * Math.PI) / 8)])
}

/** `onEvent` hears what the person does (the tutorial follows it). */
export function CheckCalls({ detail, onEvent }: { detail: SessionDetail; onEvent?: (event: RoundEvent) => void }) {
  const tell = (event: RoundEvent) => onEvent?.(event)
  const meta = detail.meta as SessionDetail['meta'] & RoundDetail
  const { vocabulary, keys, subtypes: typesAsked } = meta
  const waitMs = Math.round(1000 * vocabulary.subtype_wait_s)
  const unsure = unsureClass(vocabulary)
  const [items, setItems] = useState<RoundItem[]>([])
  const [masks, setMasks] = useState<Record<string, number>>({})  // the mask chosen per candidate (else its default)
  const [edited, setEdited] = useState<Record<string, number[][]>>({})  // outlines edited precisely (crop pixels), saved
  const [precise, setPrecise] = useState<Precise | null>(null)
  const [dragging, setDragging] = useState<number | null>(null)
  const [history, setHistory] = useState<{ item: RoundItem; save: number; text: string; mask: boolean }[]>([])
  const [remaining, setRemaining] = useState(0)
  const [progress, setProgress] = useState<RoundProgress | null>(meta.progress)
  const [pictures, setPictures] = useState<QueuePictures | null>(null)
  const [asked, setAsked] = useState<'never' | 'asking' | 'more' | 'nothing'>('never')
  const [askError, setAskError] = useState<string | null>(null)
  const [editError, setEditError] = useState<string | null>(null)
  const [taken, setTaken] = useState<string | null>(null)  // "someone else just decided the one on screen"
  const answered = useRef(new Set<string>())
  const asking = useRef(false)
  const shownAt = useRef(0)
  const overlay = useRef<SVGSVGElement>(null)
  const saver = useSaver(meta.name)
  const pendingRef = useRef(0)
  pendingRef.current = saver.waiting
  const itemsRef = useRef(items)
  itemsRef.current = items

  const ask = useCallback(async () => {
    if (asking.current) return
    asking.current = true
    setAsked((a) => (a === 'never' ? 'asking' : a))
    try {
      const next = await api.get<RoundNext>(`/api/annotate/${meta.name}/next?count=${ASK}`)
      const known = new Set(itemsRef.current.map((i) => i.key))
      const fresh = next.objects.filter((o) => !known.has(o.key) && !answered.current.has(o.key))
      setItems((have) => [...have, ...fresh.filter((o) => !have.some((i) => i.key === o.key))])
      setRemaining(Math.max(0, next.remaining - pendingRef.current))
      setPictures(next.pictures)
      setProgress(next.progress)
      setAsked(fresh.length ? 'more' : 'nothing')
      setAskError(null)
    } catch (e) {
      setAskError((e as Error).message)
      setAsked((a) => (a === 'asking' ? 'never' : a))
    } finally {
      asking.current = false
    }
  }, [meta.name])

  useEffect(() => { void ask() }, [ask])
  useEffect(() => { if (items.length < REFILL && asked === 'more') void ask() }, [items.length, asked, ask])

  // what others decided meanwhile leaves this page (the one on screen too, with a word), and the progress follows
  useLive(meta.name, { keys: items.map((i) => i.key) }, (live) => {
    if (live.progress) setProgress(live.progress)
    const gone = new Set((live.gone ?? []).filter((k) => !answered.current.has(k)))
    if (!gone.size) return
    if (itemsRef.current[0] && gone.has(itemsRef.current[0].key)) { setTaken('Someone else just checked that object, so here is the next one.'); setPrecise(null) }
    setItems((have) => have.filter((i) => !gone.has(i.key)))
    setRemaining((r) => Math.max(0, r - gone.size))
    setAsked((a) => (a === 'nothing' ? 'more' : a))
  })
  useEffect(() => { if (!taken) return; const timer = setTimeout(() => setTaken(null), 4000); return () => clearTimeout(timer) }, [taken])

  const urls = items.map((o) => pictureUrl(meta.name, o))
  const [states, reload] = usePreload(urls)
  const item = items[0]
  const picture: Picture = item ? states[urls[0]] ?? 'loading' : 'loading'
  useEffect(() => { if (picture === 'ready') shownAt.current = performance.now() }, [item?.key, picture])
  const statesRef = useRef(states)
  statesRef.current = states
  const choice = useRef({ masks, edited })
  choice.current = { masks, edited }
  const maskOf = (o: RoundItem) => masks[o.key] ?? o.mask
  const origin = (o: RoundItem) => [Math.round(o.x) - Math.floor(o.crop_px / 2), Math.round(o.y) - Math.floor(o.crop_px / 2)]
  const outlineNow = (o: RoundItem) => edited[o.key] ?? o.masks[maskOf(o)] ?? circle(o.crop_px / 2, o.crop_px / 2, o.radius_px)

  /** Answers the object on screen and moves on at once (the answer and its outline are saved in the background). */
  const answer = useCallback((what: Answer) => {
    const current = itemsRef.current[0]
    if (!current || statesRef.current[pictureUrl(meta.name, current)] !== 'ready' || performance.now() - shownAt.current < GUARD_MS) return
    const withMask = !choice.current.edited[current.key] && current.masks.length > 1
    const mask = withMask ? (choice.current.masks[current.key] ?? current.mask) : null
    const record = what.kind === 'keep'
      ? { task: 'class', candidate: current.candidate_id, keep: true, mask }
      : { task: 'class', candidate: current.candidate_id, value: what.pick.identity, subtype: what.pick.subtype ?? '', mask }
    const save = saver.save(record)
    answered.current.add(current.key)
    itemsRef.current = itemsRef.current.slice(1)
    const text = what.kind === 'keep' ? `kept: ${callText(vocabulary, current.ai_value, current.ai_subtype, typesAsked)}`
      : callText(vocabulary, what.pick.identity, what.pick.subtype, typesAsked)
    setHistory((h) => [{ item: current, save, text, mask: withMask }, ...h].slice(0, 50))
    setItems((have) => have.filter((i) => i.key !== current.key))
    setRemaining((r) => Math.max(0, r - 1))
    setTaken(null)
    setAsked((a) => (a === 'nothing' ? 'more' : a))
  }, [meta.name, saver, vocabulary, typesAsked])

  const undo = useCallback(() => {
    if (precise) { setPrecise(null); return }  // inside precise editing: undo leaves it without saving
    const last = history[0]
    if (!last) return
    if (!saver.cancel(last.save)) saver.save({ task: 'class', candidate: last.item.candidate_id, deleted: true, mask: last.mask ? 0 : null })
    answered.current.delete(last.item.key)
    setHistory((h) => h.slice(1))
    setItems((have) => [last.item, ...have.filter((i) => i.key !== last.item.key)])
    setRemaining((r) => r + 1)
  }, [history, saver, precise])

  /** Segment Anything again with the spots clicked (the object's centre included). */
  const refine = useCallback(async (clicks: number[][]) => {
    const current = itemsRef.current[0]
    if (!current) return
    const [x0, y0] = origin(current)
    setPrecise((p) => (p ? { ...p, clicks, busy: true } : p))
    try {
      const points = [[Math.round(current.x), Math.round(current.y), 1], ...clicks.map(([x, y, on]) => [Math.round(x + x0), Math.round(y + y0), on])]
      const found = await api.post<RoundSamOptions>(`/api/annotate/${meta.name}/fields/${enc(current.tile_id)}/segment`, { points })
      const best = found.options[found.chosen]
      setPrecise((p) => (p && best ? { ...p, points: best.points.map(([x, y]) => [x - x0, y - y0]), busy: false } : p && { ...p, busy: false }))
      setEditError(best ? null : 'Segment Anything found no outline with these spots: drag the corners instead.')
    } catch (e) {
      setPrecise((p) => (p ? { ...p, busy: false } : p))
      setEditError((e as Error).message)
    }
  }, [meta.name])

  const togglePrecise = useCallback(() => {
    const current = itemsRef.current[0]
    if (!current) return
    if (!precise) { setEditError(null); setPrecise({ points: outlineNow(current).map((p) => [...p]), clicks: [], busy: false }); return }
    const [x0, y0] = origin(current)
    saver.save({ task: 'outline', candidate: current.candidate_id, value: 'edited', points: precise.points.map(([x, y]) => [Math.round(x + x0), Math.round(y + y0)]) })
    setEdited((e) => ({ ...e, [current.key]: precise.points }))
    setPrecise(null)
  }, [precise, saver])  // eslint-disable-line react-hooks/exhaustive-deps

  const changeMask = useCallback((bigger: boolean) => {
    const current = itemsRef.current[0]
    if (!current || precise) return
    setEdited((e) => { const next = { ...e }; delete next[current.key]; return next })
    setMasks((m) => ({ ...m, [current.key]: stepMask(current.masks, m[current.key] ?? current.mask, bigger) }))
  }, [precise])

  const classKeys = useCandidateKeys(keys, waitMs, {
    answering: picture === 'ready' && !precise,
    onClass: (pick) => { tell(pick.subtype ? 'subtype' : 'class'); answer({ kind: 'class', pick }) },
    onUnsure: () => { if (unsure) { tell('unsure'); answer({ kind: 'class', pick: { identity: unsure, subtype: null } }) } },
    onRemove: () => { if (meta.remove_class) { tell('remove'); answer({ kind: 'class', pick: { identity: meta.remove_class, subtype: null } }) } },
    onMask: (bigger) => { tell('mask'); changeMask(bigger) },
    onPrecise: () => { tell(precise ? 'precise-out' : 'precise-in'); togglePrecise() },
    onUndo: () => { tell('undo'); undo() },
    onEnter: () => { tell('keep'); answer({ kind: 'keep' }) },
    onEscape: () => setPrecise(null),
  })

  /** A spot on the picture in crop pixels. */
  const spot = (e: ReactMouseEvent) => {
    const box = overlay.current?.getBoundingClientRect()
    if (!box || !item) return null
    return [((e.clientX - box.left) / box.width) * item.crop_px, ((e.clientY - box.top) / box.height) * item.crop_px]
  }
  const preciseClick = (e: ReactMouseEvent, include: boolean) => {
    e.preventDefault()
    const at = spot(e)
    if (!precise || !at || precise.busy) return
    void refine([...precise.clicks, [at[0], at[1], include ? 1 : 0]])
  }
  useEffect(() => {
    if (dragging === null) return
    const move = (e: MouseEvent) => {
      const at = spot(e as unknown as ReactMouseEvent)
      if (at) setPrecise((p) => (p ? { ...p, points: p.points.map((q, i) => (i === dragging ? at : q)) } : p))
    }
    const up = () => setDragging(null)
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
    return () => { window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up) }
  }, [dragging])  // eslint-disable-line react-hooks/exhaustive-deps

  const saving = useLasting(saver.waiting > 0, 800)
  const done = history.length
  const look = item ? lookOf(vocabulary, meta.remove_class, item.ai_value) : null
  const shownMask = item ? maskOf(item) : 0
  const outline = item ? (precise ? precise.points : outlineNow(item)) : null
  const unit = item ? item.crop_px / 300 : 1
  return (
    <div className="annotate-page round-page">
      <div className="annotate-top">
        <div className="annotate-heading">
          <Link to="/annotate" className="small">← All sessions</Link>
          <h1>{meta.title}</h1>
          <div className="muted small">{meta.stain} · Round 1 · {ROUND_TITLE[1]}</div>
        </div>
        <div className="annotate-progress">
          <RoundProgressLine progress={progress} unit="objects" me={detail.annotator} />
          <span className="small muted">{done ? `${done.toLocaleString()} by you this visit` : 'Enter keeps the AI’s call'}
            {asked !== 'never' && asked !== 'asking' ? ` · ${remaining.toLocaleString()} left for you` : ''}
            {done > 0 && <span className={saver.error ? ' warning-text' : ''}> · {saver.error ? 'not saved yet' : saving ? 'saving…' : 'all saved ✓'}</span>}</span>
        </div>
      </div>
      {meta.instructions && <div className="instructions">{meta.instructions}</div>}
      {saver.error && (
        <div className="row quick-save-problem">
          <ErrorNote error={`Your last ${saver.waiting === 1 ? 'answer is' : `${saver.waiting} answers are`} not saved yet: ${saver.error.replace(/^\d{3}: /, '')}${saver.refused ? '' : ' — trying again…'}`} />
          {saver.refused && <button className="btn small" onClick={() => void saver.retry()}>Try again</button>}
        </div>
      )}
      {pictures && pictures.ready < pictures.total && <PicturesBeingMade pictures={pictures} />}
      {taken && <div className="found found-good small round-taken" role="status">{taken}</div>}
      {item && look ? (
        <div className="queue-layout">
          <div className={`card queue-image${precise ? ' is-precise' : ''}`} data-candidate={item.candidate_id} data-tile={item.tile_id}>
            <div className="queue-crop">
              <img key={item.key} src={urls[0]} alt="the object to check" />
              <svg ref={overlay} viewBox={`0 0 ${item.crop_px} ${item.crop_px}`} className={`round-outline${precise ? ' editing' : ''}`}
                onClick={precise ? (e) => preciseClick(e, false) : undefined} onContextMenu={precise ? (e) => preciseClick(e, true) : (e) => e.preventDefault()}>
                {outline && <polygon points={polygonPoints(outline)} fill={precise ? look.colour : 'none'} fillOpacity={0.15}
                  style={{ stroke: precise ? '#ffffff' : look.colour }} strokeWidth={(precise ? 1.6 : 1.4) * unit} strokeDasharray={look.dash && !precise ? `${4 * unit} ${3 * unit}` : undefined} />}
                {precise && precise.points.map((p, i) => (
                  <circle key={i} cx={p[0]} cy={p[1]} r={2.6 * unit} fill="#fff" stroke="#000" strokeWidth={0.6 * unit} style={{ cursor: 'move' }}
                    onMouseDown={(e) => { e.stopPropagation(); e.preventDefault(); setDragging(i) }} onClick={(e) => e.stopPropagation()} />
                ))}
                {precise?.clicks.map((c, i) => <circle key={`c${i}`} cx={c[0]} cy={c[1]} r={3 * unit} fill={c[2] ? '#4ade80' : '#f87171'} stroke="#000" strokeWidth={0.6 * unit} />)}
                {!precise && <path d={`M${item.crop_px / 2} ${item.crop_px * 0.44} V${item.crop_px * 0.47} M${item.crop_px / 2} ${item.crop_px * 0.53} V${item.crop_px * 0.56} M${item.crop_px * 0.44} ${item.crop_px / 2} H${item.crop_px * 0.47} M${item.crop_px * 0.53} ${item.crop_px / 2} H${item.crop_px * 0.56}`}
                  stroke="#00e5ff" strokeWidth={0.8 * unit} />}
              </svg>
              {picture === 'loading' && <div className="queue-busy"><Loading inline label="Loading the picture…" /></div>}
              {picture === 'failed' && (
                <div className="queue-busy quick-failed">
                  <span className="small">The picture could not be loaded.</span>
                  <button className="btn small" onClick={() => reload(urls[0])}>Try again</button>
                  <button className="btn small" onClick={() => setItems((have) => have.slice(1))}>Skip it</button>
                </div>
              )}
            </div>
            {precise ? (
              <div className="precise-bar small">
                <b>Precise editing</b> — right-click a spot to include it, left-click one to leave it out, drag the corners.
                {precise.busy ? ' Outlining…' : ''} <span className="kbd">Space</span> saves, <span className="kbd">Esc</span> cancels.
              </div>
            ) : <MaskSwitch count={item.masks.length} index={shownMask} onChange={(i) => { setEdited((e) => { const n = { ...e }; delete n[item.key]; return n }); setMasks((m) => ({ ...m, [item.key]: i })) }} />}
            {edited[item.key] && !precise && <p className="small muted" style={{ margin: 0 }}>Outline edited by you (saved).</p>}
            <ErrorNote error={editError} />
          </div>
          <div className="card queue-choices">
            <div className="ai-call">
              <span className="small secondary">The AI says</span>
              <b className="ai-call-class"><span className={`look-swatch look-${look.look}`} style={{ borderColor: look.colour }} />
                {callText(vocabulary, item.ai_value, item.ai_subtype, typesAsked)}</b>
              <span className="small muted">{item.ai_probability != null ? `score ${fmt(item.ai_probability, 2)} · ` : ''}
                {item.decision === 'accepted' ? 'counted' : item.reason?.startsWith('rule:') ? `left out (${item.reason.slice(5).replace(/_/g, ' ')})` : 'left out'}</span>
              {item.tier != null && TIER_TEXT[item.tier] && <span className="small muted">{TIER_TEXT[item.tier]}</span>}
            </div>
            <fieldset className="plain-fieldset round-actions" disabled={picture !== 'ready' || !!precise}>
              <button className="btn primary big" onClick={() => answer({ kind: 'keep' })}><span className="kbd">Enter</span> Keep as it is</button>
              <div className="row">
                {meta.remove_class && <button className="btn" onClick={() => answer({ kind: 'class', pick: { identity: meta.remove_class, subtype: null } })}
                  title={`Recorded as “${classOf(vocabulary, meta.remove_class)?.name ?? meta.remove_class}”`}><span className="kbd">Del</span> Not an object</button>}
                {unsure && <button className="btn" onClick={() => answer({ kind: 'class', pick: { identity: unsure, subtype: null } })}><span className="kbd">0</span> Unsure</button>}
                <button className="btn" onClick={togglePrecise} title="Fix the outline spot by spot"><span className="kbd">Space</span> Precise edit</button>
              </div>
            </fieldset>
            <h3 className="round-change">Or change the class</h3>
            <fieldset className="plain-fieldset" disabled={picture !== 'ready' || !!precise}>
              <ClassChooser vocabulary={vocabulary} keys={keys} typesAsked={typesAsked} active={null} waiting={classKeys.waiting} waitMs={waitMs} swatch={lookSwatch(vocabulary, meta.remove_class)}
                onPick={classKeys.pick} onPickType={classKeys.pickType} />
            </fieldset>
            <div className="row">
              <button className="btn small" disabled={!history.length} onClick={undo}><span className="kbd">Z</span> Undo last</button>
              {history[0] && <span className="muted small ellipsis">Last: {history[0].text}</span>}
            </div>
          </div>
        </div>
      ) : asked === 'never' || asked === 'asking' || asked === 'more' ? (
        askError ? <Empty><ErrorNote error={askError} /><button className="btn" onClick={() => void ask()}>Try again</button></Empty>
          : <Loading label="Finding the next object…" />
      ) : (
        <Empty><p style={{ marginTop: 0 }}><b>Nothing left for you here right now. Thank you!</b></p>
          <p className="small secondary">Objects other people are looking at come back to the list if they leave them.</p>
          {history.length > 0 && <p className="small secondary">Pressed a wrong key? <button className="btn small" onClick={undo}><span className="kbd">Z</span> Undo last</button></p>}
          <div className="row" style={{ justifyContent: 'center' }}><button className="btn" onClick={() => void ask()}>Check again</button><Link className="btn" to="/annotate">All sessions</Link></div></Empty>
      )}
      <KeyLegend round={1} vocabulary={vocabulary} keys={keys} typesAsked={typesAsked} waitMs={waitMs} sam={meta.segment_anything} />
    </div>
  )
}
