/* The quick-check queue: one object at a time, answered with one key (a number for a counted type, a letter for a
   named look-alike, Del the reject, ` unsure: labels/classKeys.ts), without waiting. The page keeps the next
   few objects (the server holds them for this person, so people labelling together get different ones) with their
   pictures already loaded, so an answer shows the next object at once. Answers are saved in the background, in order,
   and tried again until they are; Undo brings the last object back. */
import { useCallback, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import type { QueueItem, QueueNext, QueuePictures, SessionDetail } from '../api'
import { api, apiUrl, enc } from '../api'
import { Empty, ErrorNote, Loading, Progress, fmt } from '../components'
import type { ClassPick } from './classKeys'
import { classText, resolveClass } from './classKeys'
import { ClassChooser, ClassKeyHint, useClassKeys } from './classes'

const ASK = 8  // objects asked for at a time: the one on screen and the next ones, pictures loaded
const REFILL = 4  // ask for more when fewer than this many are left on the page
const GUARD_MS = 150  // a key pressed sooner after a new object appears does not answer it (a double press)
const typing = (e: KeyboardEvent) => ['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement).tagName)
const pictureUrl = (session: string, o: QueueItem) => apiUrl(`/api/annotate/${session}/object.jpg?tile_id=${enc(o.tile_id)}&x=${o.x}&y=${o.y}`)
export type Picture = 'loading' | 'ready' | 'failed'

/** Loads these pictures in the browser ahead of showing them; says where each stands (by address). */
export function usePreload(urls: string[]): [Record<string, Picture>, (url: string) => void] {
  const [states, setStates] = useState<Record<string, Picture>>({})
  const images = useRef(new Map<string, HTMLImageElement>())
  const load = useCallback((url: string) => {
    const image = new Image()
    const settle = (state: Picture) => setStates((s) => ({ ...s, [url]: state }))
    image.onload = () => { image.decode().catch(() => undefined).finally(() => settle('ready')) }
    image.onerror = () => settle('failed')
    settle('loading')
    image.src = url
    images.current.set(url, image)
  }, [])
  const wanted = urls.join('\n')
  useEffect(() => {
    const keep = new Set(wanted.split('\n'))
    keep.forEach((url) => { if (url && !images.current.has(url)) load(url) })
    images.current.forEach((_, url) => { if (!keep.has(url)) images.current.delete(url) })
  }, [wanted, load])
  return [states, load]
}

interface Pending { id: number; record: Record<string, unknown> }

/** Saves answers in the background, in the order given, a few at a time; a failed save is tried again (after 1, 2, 4 … 15
 *  seconds), unless the server refused it (then `retry` tries again). An answer not sent yet can be taken back (`cancel`). */
export function useSaver(session: string) {
  const queue = useRef<Pending[]>([])
  const sending = useRef(new Set<number>())
  const running = useRef(false)
  const lastId = useRef(0)
  const [waiting, setWaiting] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [refused, setRefused] = useState(false)

  const pump = useCallback(async () => {
    if (running.current) return
    running.current = true
    setRefused(false)
    let pause = 1000
    while (queue.current.length) {
      const batch = queue.current.slice(0, 20)
      sending.current = new Set(batch.map((p) => p.id))
      try {
        await api.post(`/api/annotate/${session}/labels`, { records: batch.map((p) => p.record) })
        queue.current = queue.current.filter((p) => !sending.current.has(p.id))
        setError(null)
        pause = 1000
      } catch (e) {
        const message = (e as Error).message
        setError(message)
        if (/^4\d\d:/.test(message)) { setRefused(true); break }
        await new Promise((resolve) => setTimeout(resolve, pause))
        pause = Math.min(15000, 2 * pause)
      } finally {
        sending.current = new Set()
        setWaiting(queue.current.length)
      }
    }
    running.current = false
  }, [session])

  const save = useCallback((record: Record<string, unknown>) => {
    const id = ++lastId.current
    queue.current.push({ id, record })
    setWaiting(queue.current.length)
    void pump()
    return id
  }, [pump])

  const cancel = useCallback((id: number) => {
    if (sending.current.has(id) || !queue.current.some((p) => p.id === id)) return false
    queue.current = queue.current.filter((p) => p.id !== id)
    setWaiting(queue.current.length)
    return true
  }, [])

  useEffect(() => {  // leaving with answers not saved yet: the browser asks first
    if (!waiting) return
    const warn = (e: BeforeUnloadEvent) => { e.preventDefault() }
    window.addEventListener('beforeunload', warn)
    return () => window.removeEventListener('beforeunload', warn)
  }, [waiting])

  return { save, cancel, waiting, error, refused, retry: pump }
}

/** True once `value` has stayed true for `ms` (so a quick save does not flash "Saving…"). */
export function useLasting(value: boolean, ms: number) {
  const [lasting, setLasting] = useState(false)
  useEffect(() => {
    if (!value) { setLasting(false); return }
    const timer = setTimeout(() => setLasting(true), ms)
    return () => clearTimeout(timer)
  }, [value, ms])
  return lasting
}

export function QuickCheck({ detail }: { detail: SessionDetail }) {
  const meta = detail.meta
  const { vocabulary, keys, subtypes: typesAsked } = meta
  const [items, setItems] = useState<QueueItem[]>([])  // the object on screen first, then the next ones
  const [history, setHistory] = useState<{ item: QueueItem; save: number; pick: ClassPick }[]>([])
  const [remaining, setRemaining] = useState(0)
  const [pictures, setPictures] = useState<QueuePictures | null>(null)
  const [asked, setAsked] = useState<'never' | 'asking' | 'more' | 'nothing'>('never')
  const [askError, setAskError] = useState<string | null>(null)
  const answered = useRef(new Set<string>())  // keys answered on this visit, so a reply sent before their save lands skips them
  const asking = useRef(false)
  const shownAt = useRef(0)
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
      const next = await api.get<QueueNext>(`/api/annotate/${meta.name}/next?count=${ASK}`)
      const known = new Set(itemsRef.current.map((i) => i.key))
      const fresh = next.objects.filter((o) => !known.has(o.key) && !answered.current.has(o.key))
      setItems((have) => [...have, ...fresh.filter((o) => !have.some((i) => i.key === o.key))])
      setRemaining(Math.max(0, next.remaining - pendingRef.current))
      setPictures(next.pictures)
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
  useEffect(() => {  // keep the next few objects on the page
    if (items.length < REFILL && asked === 'more') void ask()
  }, [items.length, asked, ask])

  const urls = items.map((o) => pictureUrl(meta.name, o))
  const [states, reload] = usePreload(urls)
  const item = items[0]
  const picture: Picture = item ? states[urls[0]] ?? 'loading' : 'loading'
  useEffect(() => { if (picture === 'ready') shownAt.current = performance.now() }, [item?.key, picture])

  const statesRef = useRef(states)
  statesRef.current = states
  /** Answers the object on screen (several answers of one key press go to one object after the other). */
  const choose = useCallback((pick: ClassPick) => {
    const current = itemsRef.current[0]
    if (!current || statesRef.current[pictureUrl(meta.name, current)] !== 'ready' || performance.now() - shownAt.current < GUARD_MS) return
    const save = saver.save({ task: 'class', tile_id: current.tile_id, x: current.x, y: current.y, value: pick.identity, subtype: pick.subtype ?? '' })
    answered.current.add(current.key)
    itemsRef.current = itemsRef.current.slice(1)
    setHistory((h) => [{ item: current, save, pick }, ...h].slice(0, 50))
    setItems((have) => have.filter((i) => i.key !== current.key))
    setRemaining((r) => Math.max(0, r - 1))
    setAsked((a) => (a === 'nothing' ? 'more' : a))  // answers change what the queue serves: ask again when running low
  }, [meta.name, saver])
  const classKeys = useClassKeys(keys, choose)

  const undo = useCallback(() => {
    const last = history[0]
    if (!last) return
    if (!saver.cancel(last.save)) saver.save({ task: 'class', tile_id: last.item.tile_id, x: last.item.x, y: last.item.y, deleted: true })
    answered.current.delete(last.item.key)
    setHistory((h) => h.slice(1))
    setItems((have) => [last.item, ...have.filter((i) => i.key !== last.item.key)])
    setRemaining((r) => r + 1)
  }, [history, saver])

  const skip = useCallback(() => { setItems((have) => have.slice(1)) }, [])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (typing(e) || e.metaKey || e.ctrlKey || e.altKey || e.repeat) return
      if (e.key === 'Backspace' || e.key.toLowerCase() === 'z') { e.preventDefault(); undo(); return }
      if (picture === 'ready' && classKeys.press(e.key)) e.preventDefault()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [classKeys, picture, undo])

  const saving = useLasting(saver.waiting > 0, 800)
  const done = history.length
  return (
    <div className="annotate-page">
      <div className="annotate-top">
        <div className="annotate-heading">
          <Link to="/annotate" className="small">← All sessions</Link>
          <h1>{meta.title}</h1>
          <div className="muted small">{meta.stain} · one object at a time{meta.ai_view === 'full' ? ' · detection’s call is shown' : ' · decide for yourself'}</div>
        </div>
        <div className="annotate-progress">
          <span className="small secondary">{done ? `${done.toLocaleString()} labelled by you this visit` : 'Press a number key to answer'}
            {asked !== 'never' && asked !== 'asking' ? ` · ${remaining.toLocaleString()} left` : ''}</span>
          {done > 0 && <span className={`small ${saver.error ? 'warning-text' : 'muted'}`}>{saver.error ? 'Not saved yet' : saving ? 'Saving…' : 'All answers saved ✓'}</span>}
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
      {item ? (
        <div className="queue-layout">
          <div className="card queue-image">
            <div className="queue-crop">
              <img key={item.key} src={urls[0]} alt="the object to label" />
              <svg viewBox="0 0 100 100" aria-hidden="true">
                <path d="M50 40 V46 M50 54 V60 M40 50 H46 M54 50 H60" stroke="#00e5ff" strokeWidth={0.9} />
              </svg>
              {picture === 'loading' && <div className="queue-busy"><Loading inline label="Loading the picture…" /></div>}
              {picture === 'failed' && (
                <div className="queue-busy quick-failed">
                  <span className="small">The picture could not be loaded.</span>
                  <button className="btn small" onClick={() => reload(urls[0])}>Try again</button>
                  <button className="btn small" onClick={skip}>Skip it</button>
                </div>
              )}
            </div>
            <p className="muted small">{item.labels_so_far ? `${item.labels_so_far} other ${item.labels_so_far === 1 ? 'person has' : 'people have'} labelled this one` : 'Nobody has labelled this one yet'}</p>
          </div>
          <div className="card queue-choices">
            <h2>What is under the cross?</h2>
            {meta.ai_view === 'full' && <p className="small secondary" title={`probability ${fmt(item.ai_probability, 2)}`}>
              Detection’s call: <b>{classText(vocabulary, resolveClass(vocabulary, item.ai_value), typesAsked)}</b></p>}
            <fieldset className="plain-fieldset" disabled={picture !== 'ready'}>
              <ClassChooser vocabulary={vocabulary} keys={keys} active={null} big onPick={classKeys.pick} />
            </fieldset>
            <div className="row">
              <button className="btn small" disabled={!history.length} onClick={undo}><span className="kbd">Z</span> Undo last</button>
              {history[0] && <span className="muted small ellipsis">Last: {classText(vocabulary, history[0].pick, typesAsked)}</span>}
            </div>
            <p className="muted small" style={{ margin: 0 }}>Keys: <ClassKeyHint vocabulary={vocabulary} keys={keys} />,
              {' '}<span className="kbd">Z</span> undo</p>
          </div>
        </div>
      ) : asked === 'never' || asked === 'asking' || asked === 'more' ? (
        askError ? (
          <Empty><ErrorNote error={askError} /><button className="btn" onClick={() => void ask()}>Try again</button></Empty>
        ) : <Loading label="Finding the next object…" />
      ) : (
        <Empty><p style={{ marginTop: 0 }}><b>Nothing left for you in this queue right now. Thank you!</b></p>
          {history.length > 0 && <p className="small secondary">Pressed a wrong key? <button className="btn small" onClick={undo}><span className="kbd">Z</span> Undo last</button></p>}
          <div className="row" style={{ justifyContent: 'center' }}><button className="btn" onClick={() => void ask()}>Check again</button><Link className="btn" to="/annotate">All sessions</Link></div></Empty>
      )}
    </div>
  )
}

/** "Pictures ready: 1,240 of 1,500" while the queue's pictures are still being made (objects with one are shown first). */
export function PicturesBeingMade({ pictures }: { pictures: QueuePictures }) {
  return (
    <div className="quick-pictures">
      <Progress done={pictures.ready} total={pictures.total} />
      <span className="small secondary">Pictures ready: {pictures.ready.toLocaleString()} of {pictures.total.toLocaleString()} — the rest are being made; objects with a picture come first.</span>
    </div>
  )
}
