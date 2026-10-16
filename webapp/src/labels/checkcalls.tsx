/* Round 1, "Check the AI's calls": one field (tile) at a time, its candidates one after the other (accepted and
   rejected, most informative first) with the AI's class, type, score and outline. The server hands each person a field
   nobody else is on (held while this page is open); once someone finishes a field it is not handed to anyone else.
   The keys are those of round 2, acting on the object on screen: Enter keeps the AI's call, 1–9 (and the type keys)
   choose the class, 0 unsure, Delete not an object, Tab / Shift+Tab a bigger or smaller outline (the three Segment
   Anything masks come with the object: no waiting), Space precise editing (right-click a spot to include, left-click one
   to leave out, drag the corners; Space again keeps it), Z undoes the last unsaved answer. Answers stay on this page
   ("3 unsaved changes", with a draft in this browser in case of a reload) until Save (stays on the field) or Save and
   continue (finishes the field, every object answered, and brings the next one). */
import { useCallback, useEffect, useRef, useState } from 'react'
import type { MouseEvent as ReactMouseEvent } from 'react'
import { Link } from 'react-router-dom'
import type { RoundAnswer, RoundDetail, RoundItem, RoundProgress, RoundSamOptions, RoundSaved, RoundTile, SessionDetail } from '../api'
import { api, apiUrl, enc } from '../api'
import { Empty, ErrorNote, Loading, fmt } from '../components'
import type { ClassPick } from './classKeys'
import { classOf } from './classKeys'
import { ClassChooser } from './classes'
import { PicturesBeingMade, type Picture, usePreload } from './quickcheck'
import type { RoundEvent } from './rounds'
import { KeyLegend, MaskSwitch, ROUND_TITLE, RoundProgressLine, SaveButtons, SaveState, TIER_TEXT, callText, draftKey, lookOf, lookSwatch, polygonPoints,
  readDraft, stepMask, unsureClass, useCandidateKeys, useLeaveGuard, useLive, useSaveKeys, writeDraft } from './rounds'

const AHEAD = 6  // pictures loaded ahead of the object on screen
const GUARD_MS = 150  // a key pressed sooner after a new object appears does not answer it (a double press)
const pictureUrl = (session: string, o: RoundItem) => apiUrl(`/api/annotate/${session}/object.jpg?tile_id=${enc(o.tile_id)}&x=${o.x}&y=${o.y}`)
type Answer = { kind: 'keep' } | { kind: 'class'; pick: ClassPick }
/** An answer on the page, not saved yet: the call, and the outline chosen with it (a mask, or one edited precisely, crop px). */
interface Local { answer: Answer; mask: number | null; edited: number[][] | null }
/** Precise editing of the outline on screen: the polygon (crop pixels) and the spots clicked (x, y, 1 include / 0 leave out). */
interface Precise { points: number[][]; clicks: number[][]; busy: boolean }

function circle(x: number, y: number, r: number) {
  return Array.from({ length: 16 }, (_, i) => [x + r * Math.cos((i * Math.PI) / 8), y + r * Math.sin((i * Math.PI) / 8)])
}

const origin = (o: RoundItem) => [Math.round(o.x) - Math.floor(o.crop_px / 2), Math.round(o.y) - Math.floor(o.crop_px / 2)]

/** The labels an answer is saved as (the outline edited precisely in field-picture pixels). */
function records(o: RoundItem, local: Local, noObject: (identity: string) => boolean) {
  const { answer, mask, edited } = local
  const call = answer.kind === 'keep' ? { task: 'class', candidate: o.candidate_id, keep: true, mask }
    : { task: 'class', candidate: o.candidate_id, value: answer.pick.identity, subtype: answer.pick.subtype ?? '', mask }
  const [x0, y0] = origin(o)
  const keepsOutline = edited && !noObject(answer.kind === 'keep' ? o.ai_value : answer.pick.identity)
  return keepsOutline ? [call, { task: 'outline', candidate: o.candidate_id, value: 'edited', points: edited.map(([x, y]) => [Math.round(x + x0), Math.round(y + y0)]) }] : [call]
}

/** An answer as the server keeps it once saved. */
function asSaved(o: RoundItem, local: Local): RoundAnswer {
  const pick = local.answer.kind === 'keep' ? { identity: o.ai_value, subtype: o.ai_subtype } : { identity: local.answer.pick.identity, subtype: local.answer.pick.subtype ?? '' }
  return { ...pick, keep: local.answer.kind === 'keep', mask: local.mask, outline: local.edited }
}

/** `onEvent` hears what the person does (the tutorial follows it). */
export function CheckCalls({ detail, onEvent }: { detail: SessionDetail; onEvent?: (event: RoundEvent) => void }) {
  const tell = (event: RoundEvent) => onEvent?.(event)
  const meta = detail.meta as SessionDetail['meta'] & RoundDetail
  const { vocabulary, keys, subtypes: typesAsked } = meta
  const waitMs = Math.round(1000 * vocabulary.subtype_wait_s)
  const unsure = unsureClass(vocabulary)
  const [tile, setTile] = useState<RoundTile | null>(null)
  const [answers, setAnswers] = useState<Record<string, Local>>({})  // unsaved, by candidate
  const [cursor, setCursor] = useState(0)  // the object on screen; -1: every object of the field has an answer
  const [masks, setMasks] = useState<Record<string, number>>({})  // the mask chosen per candidate before answering (else its default)
  const [edited, setEdited] = useState<Record<string, number[][]>>({})  // outlines edited precisely before answering (crop pixels)
  const [precise, setPrecise] = useState<Precise | null>(null)
  const [dragging, setDragging] = useState<number | null>(null)
  const [history, setHistory] = useState<{ key: string; before: Local | undefined; cursor: number }[]>([])
  const [progress, setProgress] = useState<RoundProgress | null>(meta.progress)
  const [openBy, setOpenBy] = useState<string[]>([])
  const [asking, setAsking] = useState(true)
  const [askError, setAskError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [restored, setRestored] = useState(0)
  const shownAt = useRef(0)
  const overlay = useRef<SVGSVGElement>(null)
  const objects = tile?.objects ?? []
  const draft = tile?.tile_id ? draftKey(detail.annotator, meta.name, tile.tile_id) : null
  const unsaved = Object.keys(answers).length
  const mayLeave = useLeaveGuard(unsaved)
  const noObject = (identity: string) => identity === meta.remove_class || identity === unsure

  const answered = (o: RoundItem, now = answers) => !!now[o.key] || !!o.mine
  const firstOpen = (list: RoundItem[], now: Record<string, Local>, from = 0) => {
    for (let i = 0; i < list.length; i++) { const j = (from + i) % list.length; if (!now[list[j].key] && !list[j].mine) return j }
    return -1
  }

  /** The next field for this person (`skip`: put the one on the page aside), with its unsaved draft from this browser. */
  const ask = useCallback(async (skip?: string) => {
    setAsking(true)
    setAskError(null)
    try {
      const next = await api.get<RoundTile>(`/api/annotate/${meta.name}/next${skip ? `?skip=${enc(skip)}` : ''}`)
      const kept = next.tile_id ? readDraft<{ answers: Record<string, Local> }>(draftKey(detail.annotator, meta.name, next.tile_id)) : null
      const known = new Set(next.objects.map((o) => o.key))
      const back = Object.fromEntries(Object.entries(kept?.answers ?? {}).filter(([k]) => known.has(k)))
      setTile(next)
      setAnswers(back)
      setRestored(Object.keys(back).length)
      setCursor(firstOpen(next.objects, back))
      setMasks({}); setEdited({}); setHistory([]); setPrecise(null); setSaveError(null)
      setProgress(next.progress)
      setOpenBy(next.open_by)
    } catch (e) {
      setAskError((e as Error).message)
    } finally {
      setAsking(false)
    }
  }, [meta.name, detail.annotator])  // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { void ask() }, [ask])
  useEffect(() => { if (draft) writeDraft(draft, unsaved ? { answers } : null) }, [draft, answers, unsaved])
  useLive(meta.name, tile?.tile_id ? { tile_id: tile.tile_id } : null, (live) => {
    if (live.progress) setProgress(live.progress)
    if (live.open_by) setOpenBy(live.open_by)
  })

  const item = cursor >= 0 ? objects[cursor] : undefined
  const ahead = cursor >= 0 ? [...objects.slice(cursor), ...objects.slice(0, cursor)].slice(0, AHEAD) : []
  const urls = ahead.map((o) => pictureUrl(meta.name, o))
  const [states, reload] = usePreload(urls)
  const picture: Picture = item ? states[pictureUrl(meta.name, item)] ?? 'loading' : 'loading'
  useEffect(() => { if (picture === 'ready') shownAt.current = performance.now() }, [item?.key, picture])
  const local = item ? answers[item.key] : undefined
  const maskOf = (o: RoundItem) => masks[o.key] ?? answers[o.key]?.mask ?? o.mine?.mask ?? o.mask
  const outlineNow = (o: RoundItem) => edited[o.key] ?? (masks[o.key] == null ? answers[o.key]?.edited ?? o.mine?.outline : null) ?? o.masks[maskOf(o)]
    ?? circle(o.crop_px / 2, o.crop_px / 2, o.radius_px)

  /** Answers the object on screen (on the page only) and shows the next one without an answer. */
  const answer = (what: Answer) => {
    if (!item || picture !== 'ready' || performance.now() - shownAt.current < GUARD_MS) return
    const precisely = edited[item.key] ?? (masks[item.key] == null ? answers[item.key]?.edited ?? item.mine?.outline ?? null : null)  // an edit stands until another size is chosen
    const mask = !precisely && item.masks.length > 1 ? maskOf(item) : null
    const now = { ...answers, [item.key]: { answer: what, mask, edited: precisely } }
    setHistory((h) => [{ key: item.key, before: answers[item.key], cursor }, ...h].slice(0, 200))
    setAnswers(now)
    setCursor(firstOpen(objects, now, cursor + 1))
    setPrecise(null)
  }

  const undo = () => {
    if (precise) { setPrecise(null); return }  // inside precise editing: undo leaves it without keeping it
    const last = history[0]
    if (!last) return
    setAnswers((now) => { const next = { ...now }; if (last.before) next[last.key] = last.before; else delete next[last.key]; return next })
    setHistory((h) => h.slice(1))
    setCursor(last.cursor)
  }

  /** Saves every unsaved answer of the field at once; `finish`: the field is done and the next one comes. */
  const save = async (finish: boolean) => {
    if (!tile?.tile_id || saving || (!finish && !unsaved)) return
    setSaving(true)
    try {
      const sent = Object.entries(answers).flatMap(([key, l]) => { const o = objects.find((i) => i.key === key); return o ? records(o, l, noObject) : [] })
      const saved = await api.post<RoundSaved>(`/api/annotate/${meta.name}/tiles/${enc(tile.tile_id)}/save`, { records: sent, submit: finish })
      setProgress(saved.progress)
      setSaveError(null)
      if (draft) writeDraft(draft, null)
      tell(finish ? 'finish' : 'save')
      if (finish) { await ask(); return }
      setTile({ ...tile, objects: objects.map((o) => (answers[o.key] ? { ...o, mine: asSaved(o, answers[o.key]) } : o)) })
      setAnswers({}); setHistory([]); setRestored(0)
    } catch (e) {
      setSaveError((e as Error).message)  // nothing was saved: every answer is still on the page
    } finally {
      setSaving(false)
    }
  }
  const open = objects.filter((o) => !answered(o)).length
  const canFinish = !!tile?.tile_id && !open && !precise
  useSaveKeys(() => void save(false), () => { if (canFinish) void save(true) })
  const skip = () => { if (tile?.tile_id && mayLeave()) void ask(tile.tile_id) }

  /** Segment Anything again with the spots clicked (the object's centre included). */
  const refine = async (clicks: number[][]) => {
    if (!item) return
    const [x0, y0] = origin(item)
    setPrecise((p) => (p ? { ...p, clicks, busy: true } : p))
    try {
      const points = [[Math.round(item.x), Math.round(item.y), 1], ...clicks.map(([x, y, on]) => [Math.round(x + x0), Math.round(y + y0), on])]
      const found = await api.post<RoundSamOptions>(`/api/annotate/${meta.name}/fields/${enc(item.tile_id)}/segment`, { points })
      const best = found.options[found.chosen]
      setPrecise((p) => (p && best ? { ...p, points: best.points.map(([x, y]) => [x - x0, y - y0]), busy: false } : p && { ...p, busy: false }))
      setSaveError(best ? null : 'Segment Anything found no outline with these spots: drag the corners instead.')
    } catch (e) {
      setPrecise((p) => (p ? { ...p, busy: false } : p))
      setSaveError((e as Error).message)
    }
  }

  const togglePrecise = () => {
    if (!item) return
    if (!precise) { setPrecise({ points: outlineNow(item).map((p) => [...p]), clicks: [], busy: false }); return }
    setEdited((e) => ({ ...e, [item.key]: precise.points }))  // kept on the page; saved with the answer
    setPrecise(null)
  }

  const chooseMask = (i: number) => {
    if (!item || precise) return
    setEdited((e) => { const next = { ...e }; delete next[item.key]; return next })
    setMasks((m) => ({ ...m, [item.key]: i }))
  }

  const classKeys = useCandidateKeys(keys, waitMs, {
    answering: picture === 'ready' && !precise && !!item,
    onClass: (pick) => { tell(pick.subtype ? 'subtype' : 'class'); answer({ kind: 'class', pick }) },
    onUnsure: () => { if (unsure) { tell('unsure'); answer({ kind: 'class', pick: { identity: unsure, subtype: null } }) } },
    onRemove: () => { if (meta.remove_class) { tell('remove'); answer({ kind: 'class', pick: { identity: meta.remove_class, subtype: null } }) } },
    onMask: (bigger) => { if (item) { tell('mask'); chooseMask(stepMask(item.masks, maskOf(item), bigger)) } },
    onPrecise: () => { if (item) { tell(precise ? 'precise-out' : 'precise-in'); togglePrecise() } },
    onUndo: () => { tell('undo'); undo() },
    onEnter: () => { tell('keep'); answer({ kind: 'keep' }) },
    onEscape: () => setPrecise(null),
  })

  /** A spot on the picture in crop pixels. */
  const spot = (e: { clientX: number; clientY: number }) => {
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
      const at = spot(e)
      if (at) setPrecise((p) => (p ? { ...p, points: p.points.map((q, i) => (i === dragging ? at : q)) } : p))
    }
    const up = () => setDragging(null)
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
    return () => { window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up) }
  }, [dragging])  // eslint-disable-line react-hooks/exhaustive-deps

  const look = item ? lookOf(vocabulary, meta.remove_class, item.ai_value) : null
  const outline = item ? (precise ? precise.points : outlineNow(item)) : null
  const unit = item ? item.crop_px / 300 : 1
  const mine = item ? (local ? asSaved(item, local) : item.mine) : null
  const continueTitle = open ? `${open} object${open === 1 ? '' : 's'} on this field still need${open === 1 ? 's' : ''} an answer`
    : 'Save your answers; this field is then done and the next one comes (Ctrl+Enter)'
  return (
    <div className="annotate-page round-page">
      <div className="annotate-top">
        <div className="annotate-heading">
          <Link to="/annotate" className="small">← All sessions</Link>
          <h1>{meta.title}</h1>
          <div className="muted small">{meta.stain} · Round 1 · {ROUND_TITLE[1]}</div>
        </div>
        <div className="annotate-progress">
          <RoundProgressLine progress={progress} me={detail.annotator} />
          {tile?.tile_id && <span className="small muted">{tile.tiles_left.toLocaleString()} field{tile.tiles_left === 1 ? '' : 's'} left for you, this one included</span>}
        </div>
        {tile?.tile_id && <div className="annotate-actions">
          <SaveState unsaved={unsaved} saving={saving} />
          <button className="btn" onClick={skip} disabled={saving} title="Put this field aside and get another one">Skip field</button>
          <SaveButtons unsaved={unsaved} saving={saving} onSave={() => void save(false)} onContinue={() => void save(true)} canContinue={canFinish} continueTitle={continueTitle}
            primary={false} />
        </div>}
      </div>
      {meta.instructions && <div className="instructions">{meta.instructions}</div>}
      <ErrorNote error={saveError && `Not saved: ${saveError.replace(/^\d{3}: /, '')} — your answers are still here.`} />
      {restored > 0 && <div className="found found-warn small round-restored" role="status">{restored} answer{restored === 1 ? '' : 's'} you had not saved came back from this
        browser. They are not saved yet. <button className="link-btn small" onClick={() => { setAnswers({}); setRestored(0); setCursor(firstOpen(objects, {})) }}>Discard them</button></div>}
      {openBy.length > 0 && <div className="found small" role="status">{openBy.join(', ')} {openBy.length === 1 ? 'is' : 'are'} also on this field.</div>}
      {tile?.pictures && tile.pictures.ready < tile.pictures.total && <PicturesBeingMade pictures={tile.pictures} />}
      {asking ? <Loading label="Finding the next field…" />
        : askError ? <Empty><ErrorNote error={askError} /><button className="btn" onClick={() => void ask()}>Try again</button></Empty>
        : !tile?.tile_id ? (
          <Empty><p style={{ marginTop: 0 }}><b>Nothing left for you here right now. Thank you!</b></p>
            <p className="small secondary">Fields other people are working on come back to the list if they leave them unfinished.</p>
            <div className="row" style={{ justifyContent: 'center' }}><button className="btn" onClick={() => void ask()}>Check again</button><Link className="btn" to="/annotate">All sessions</Link></div></Empty>
        ) : <>
          <TileStrip objects={objects} cursor={cursor} answers={answers} onPick={(i) => { setPrecise(null); setCursor(i) }} />
          {item && look ? (
            <div className="queue-layout">
              <div className={`card queue-image${precise ? ' is-precise' : ''}`} data-candidate={item.candidate_id} data-tile={item.tile_id}>
                <div className="queue-crop">
                  <img key={item.key} src={pictureUrl(meta.name, item)} alt="the object to check" />
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
                      <button className="btn small" onClick={() => reload(pictureUrl(meta.name, item))}>Try again</button>
                    </div>
                  )}
                </div>
                {precise ? (
                  <div className="precise-bar small">
                    <b>Precise editing</b> — right-click a spot to include it, left-click one to leave it out, drag the corners.
                    {precise.busy ? ' Outlining…' : ''} <span className="kbd">Space</span> keeps it, <span className="kbd">Esc</span> cancels.
                  </div>
                ) : <MaskSwitch count={item.masks.length} index={maskOf(item)} onChange={chooseMask} />}
                {(edited[item.key] || mine?.outline) && !precise && <p className="small muted" style={{ margin: 0 }}>Outline edited by you{edited[item.key] ? ' — kept with your answer' : ''}.</p>}
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
                {mine && <p className={`small round-answer${local ? ' unsaved' : ''}`}>Your answer: <b>{mine.keep ? `kept (${callText(vocabulary, mine.identity, mine.subtype, typesAsked)})`
                  : callText(vocabulary, mine.identity, mine.subtype, typesAsked)}</b> — {local ? 'not saved yet' : 'saved'}. Answer again to change it.</p>}
                <fieldset className="plain-fieldset round-actions" disabled={picture !== 'ready' || !!precise}>
                  <button className="btn primary big" onClick={() => { tell('keep'); answer({ kind: 'keep' }) }}><span className="kbd">Enter</span> Keep as it is</button>
                  <div className="row">
                    {meta.remove_class && <button className="btn" onClick={() => { tell('remove'); answer({ kind: 'class', pick: { identity: meta.remove_class, subtype: null } }) }}
                      title={`Recorded as “${classOf(vocabulary, meta.remove_class)?.name ?? meta.remove_class}”`}><span className="kbd">Del</span> Not an object</button>}
                    {unsure && <button className="btn" onClick={() => { tell('unsure'); answer({ kind: 'class', pick: { identity: unsure, subtype: null } }) }}><span className="kbd">0</span> Unsure</button>}
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
                  <span className="muted small">Nothing is saved until you press Save.</span>
                </div>
              </div>
            </div>
          ) : (
            <Empty><p style={{ marginTop: 0 }}><b>Every object on this field has an answer.</b></p>
              <p className="small secondary">{unsaved ? `${unsaved} of them ${unsaved === 1 ? 'is' : 'are'} not saved yet. ` : ''}Press <b>Save and continue</b> to finish this field
                and get the next one, or click an object above to look at it again.</p>
              <div className="row" style={{ justifyContent: 'center' }}>
                {history.length > 0 && <button className="btn" onClick={undo}><span className="kbd">Z</span> Undo last</button>}
                <SaveButtons unsaved={unsaved} saving={saving} onSave={() => void save(false)} onContinue={() => void save(true)} canContinue={canFinish} continueTitle={continueTitle} />
              </div></Empty>
          )}
        </>}
      <KeyLegend round={1} vocabulary={vocabulary} keys={keys} typesAsked={typesAsked} waitMs={waitMs} sam={meta.segment_anything} />
    </div>
  )
}

/** The objects of the field, in order: answered and saved, answered on this page only (not saved), still to do, and the
 *  one on screen; click one to look at it again. */
function TileStrip({ objects, cursor, answers, onPick }: { objects: RoundItem[]; cursor: number; answers: Record<string, Local>; onPick: (i: number) => void }) {
  const done = objects.filter((o) => answers[o.key] || o.mine).length
  return (
    <div className="tile-strip" aria-label="Objects on this field">
      <span className="small secondary"><b>{cursor >= 0 ? `Object ${cursor + 1} of ${objects.length}` : `${objects.length} objects`}</b> on this field · {done} answered</span>
      <span className="tile-strip-dots">{objects.map((o, i) => {
        const state = answers[o.key] ? 'unsaved' : o.mine ? 'saved' : 'open'
        const what = state === 'unsaved' ? 'answered, not saved yet' : state === 'saved' ? 'answered and saved' : 'no answer yet'
        return <button key={o.key} className={`tile-dot ${state}${i === cursor ? ' now' : ''}`} onClick={() => onPick(i)} title={`Object ${i + 1}: ${what}`}
          aria-label={`Object ${i + 1}: ${what}`} aria-current={i === cursor} />
      })}</span>
    </div>
  )
}
