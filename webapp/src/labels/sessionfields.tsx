/* The fields of a session for the person labelling: by TMA, then core, each core drawn as its own tiles (each tile at
   its place in the core, the core's tiles outside the session faint), a tile's state by its outline colour, the full
   facts on hover, a click opens the field. "By confidence" puts the cores detection is least sure of first. A field
   opened from either list goes on in it: its page's "Next" follows that order (fieldOrder.ts, labels/picker.py). */
import { useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import type { FieldState, SessionDetail } from '../api'
import { apiUrl, enc } from '../api'
import { useFetch, useStored } from '../hooks'
import type { FieldOrder } from './fieldOrder'
import { fieldPath, orderKey } from './fieldOrder'

type Field = SessionDetail['fields'][number]
/** A field's confidence (GET …/confidence): `mean` its objects' average closeness to the threshold (1 at it, 0 from 0.2
 *  away; None without objects), `near` their sum, `objects` how many. */
type Sure = { near: number; objects: number; mean: number | null }
type Layout = NonNullable<SessionDetail['cores']>[string]
/** A field's state for this person: where it stands for everyone (`sessions.field_states`), and "mine" when they
 *  finished it while others are still asked. */
type Look = FieldState | 'mine' | 'bad'
export const LOOK_TEXT: Record<Look, string> = { not_started: 'To do', in_progress: 'In progress', done: 'Done', preparing: 'Being prepared', mine: 'Finished by you', bad: 'Marked bad' }
const LOOK_ORDER: Look[] = ['not_started', 'in_progress', 'mine', 'done', 'bad', 'preparing']
const LOOK_HINT: Partial<Record<Look, string>> = { in_progress: 'Saved but not finished (Save progress), or open now', done: 'Finished (Finish field)' }
const BADGE: Record<Look, string> = { not_started: 'status-todo', in_progress: 'status-running', done: 'status-done', preparing: 'status-queued', mine: 'status-done', bad: 'status-outdated' }

const stateOf = (f: Field): FieldState => f.state ?? (!f.prepared ? 'preparing' : f.done_by_me ? 'done' : f.open_by_others ? 'in_progress' : 'not_started')
const lookOf = (f: Field): Look => (Object.keys(f.marked_bad ?? {}).length ? 'bad' : f.done_by_me && stateOf(f) !== 'done' ? 'mine' : stateOf(f))
/** The reasons people gave for marking a whole field bad (labels.exclusions types). */
const BAD_REASON: Record<string, string> = { '1': 'fold or tissue damage', '2': 'artifact', '3': 'no tissue', '4': 'out of focus or bad staining' }
const who = (people: string[], me: string | null) => people.map((p) => (p === me ? 'you' : p)).join(', ')
/** Someone else has the field open: only one person works on a field at a time. */
const takenBy = (f: Field, me: string | null) => (f.open_by ?? []).filter((p) => p !== me)
const day = (t: number) => new Date(t * 1000).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })

/** The hover text of a tile: its state, who is on it or has it open now, who finished it and when, how many of the
 *  people asked are done. */
function hoverText(f: Field, i: number, needed: number, me: string | null, sure?: Sure, slides = false): string {
  const done = f.done_by ?? []
  const on = [...new Set([...(f.working ?? []), ...(f.open_by ?? [])])].filter((p) => !done.includes(p))
  return [
    `Field ${i + 1} · ${LOOK_TEXT[lookOf(f)]}`,
    slides ? `slide ${f.core || '?'}` : `${f.tma_name || f.tma || 'TMA ?'} · core ${f.core || '?'}`,
    sure ? (sure.objects && sure.mean != null ? `average ${sure.mean.toFixed(2)} closeness · ${sure.objects} object${sure.objects === 1 ? '' : 's'}` : 'no objects') : '',
    on.length ? `In progress · ${who(on, me)} (saved but not finished, or open now)` : '',
    done.length ? `Finished by ${done.map((p) => `${p === me ? 'you' : p}${f.done_at?.[p] ? ` (${day(f.done_at[p])})` : ''}`).join(', ')}` : '',
    `${Math.min(done.length, needed)} of ${needed} ${needed === 1 ? 'person' : 'people'} done`,
    f.ai_reviewed ? 'AI review labels exist (not shown while labelling)' : '',
    ...(f.elsewhere ?? []).map((e) => `Labelled in another session: ${e.title}${e.finished ? ` (finished by ${e.finished} ${e.finished === 1 ? 'person' : 'people'} there)` : ''} — not shown here`),
    ...Object.entries(f.marked_bad ?? {}).map(([p, k]) => `Marked bad by ${p === me ? 'you' : p}: ${BAD_REASON[k] ?? 'bad'}`),
    takenBy(f, me).length ? `Open now by ${who(takenBy(f, me), me)}: it opens for you once they close it`
      : f.prepared ? 'Click to open' : 'Being prepared: it opens once ready',
  ].filter(Boolean).join('\n')
}

/** A core's state from its fields': done when every field is, being prepared while none is ready, to do while none has
 *  started, else in progress. */
function coreState(list: Field[]): FieldState {
  const states = list.map(stateOf)
  if (states.every((s) => s === 'done')) return 'done'
  if (states.every((s) => s === 'preparing')) return 'preparing'
  if (states.every((s) => s === 'not_started' || s === 'preparing')) return 'not_started'
  return 'in_progress'
}
/** The progress of some fields: how many are done, in progress and to do, and finished by this person. A core is done
 *  when all its session fields are; a TMA when all its cores are. */
function tally(list: Field[]) {
  const n = (s: FieldState) => list.filter((f) => stateOf(f) === s).length
  return { total: list.length, done: n('done'), progress: n('in_progress'), todo: n('not_started') + n('preparing'), mine: list.filter((f) => f.done_by_me).length }
}
/** How a session's units are named: the cores of a TMA study, or the slides of a whole-slide project. */
const UNIT = { core: { one: 'core', title: 'Core' }, slide: { one: 'slide', title: 'Slide' } }
const coresDone = (cores: Core[]) => cores.filter((c) => c.list.every(({ f }) => stateOf(f) === 'done')).length
const progressText = (t: ReturnType<typeof tally>, todo = true) => [
  t.progress && `${t.progress.toLocaleString()} in progress`, todo && t.todo && `${t.todo.toLocaleString()} to do`, t.mine && `${t.mine.toLocaleString()} by you`,
].filter(Boolean).join(' · ')
const of = (done: number, total: number, what: string) => `${done.toLocaleString()}/${total.toLocaleString()} ${what}${total === 1 ? '' : 's'}`
/** A core's line: "3/8 fields done · 1 in progress · 4 to do · 2 by you" */
function coreLine(list: Field[]) {
  const t = tally(list)
  return [`${of(t.done, t.total, 'field')} done`, progressText(t)].filter(Boolean).join(' · ')
}
/** A TMA's line: "2/30 cores · 120/900 fields done · 14 in progress · 40 by you" */
function tmaLine(cores: Core[], unit = 'core') {
  const t = tally(cores.flatMap((c) => c.list.map(({ f }) => f)))
  return [of(coresDone(cores), cores.length, unit), `${of(t.done, t.total, 'field')} done`, progressText(t, false)].filter(Boolean).join(' · ')
}

/** A level of the list that folds open and shut; what it holds is only drawn while it is open (a session of thousands
 *  of fields stays quick: closed TMAs and cores load no pictures). */
function Fold({ className, summary, open: initial, children }: { className: string; summary: ReactNode; open: boolean; children: () => ReactNode }) {
  const [open, setOpen] = useState(initial)
  return (
    <details className={className} open={open} onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)}>
      <summary>{summary}</summary>
      {open && children()}
    </details>
  )
}
/** Above this many fields the TMAs start folded (open one to see its cores). */
const FOLDED_FROM = 300

type Numbered = { f: Field; i: number }
type Core = { id: string; name: string; list: Numbered[] }
const natural = (a: string, b: string) => a.localeCompare(b, undefined, { numeric: true })

export function SessionFieldMap({ detail }: { detail: SessionDetail }) {
  const { meta, fields } = detail
  const name = meta.name
  const slides = detail.unit === 'slide'
  const unit = UNIT[slides ? 'slide' : 'core']
  const [order, setOrder] = useStored<FieldOrder>(orderKey(detail.annotator), 'natural')
  const confidence = useFetch<Record<string, Sure>>(order === 'confidence' ? `/api/annotate/${name}/confidence` : null)
  const tmas = useMemo(() => {
    const byTma = new Map<string, Map<string, Core>>()
    fields.forEach((f, i) => {
      const tma = f.tma_name || f.tma || 'Other fields'
      const coreId = f.core_id || f.core || '—'
      const cores = byTma.get(tma) ?? byTma.set(tma, new Map()).get(tma)!
      const core = cores.get(coreId) ?? cores.set(coreId, { id: coreId, name: f.core || coreId, list: [] }).get(coreId)!
      core.list.push({ f, i })
    })
    const place = (n: Numbered) => (n.f.box ? [0, n.f.box[1], n.f.box[0]] : [1, 0, 0])  // top to bottom, left to right (labels/picker.py)
    const byPlace = (a: Numbered, b: Numbered) => { const p = place(a), q = place(b); return p[0] - q[0] || p[1] - q[1] || p[2] - q[2] || a.i - b.i }
    return [...byTma].sort(([a], [b]) => natural(a, b)).map(([tma, cores]) => ({ tma, cores: [...cores.values()].sort((a, b) => natural(a.name, b.name))
      .map((c) => ({ ...c, list: [...c.list].sort(byPlace) })) }))
  }, [fields])
  const ranked = useMemo(() => {  // By confidence: every field, least confident first (the highest average closeness), no objects last
    if (!confidence.data) return null
    const sure = confidence.data
    const score = (f: Field) => { const s = sure[f.tile_id]; return s && s.objects && s.mean != null ? s.mean : -1 }
    return fields.map((f, i) => ({ f, i })).sort((a, b) => score(b.f) - score(a.f) || a.i - b.i)
  }, [fields, confidence.data])
  const whole = tally(fields)
  const allCores = tmas.flatMap((t) => t.cores)
  const tmasDone = tmas.filter((t) => t.cores.every((c) => c.list.every(({ f }) => stateOf(f) === 'done'))).length
  return (
    <div className="session-fields">
      <p className="session-progress small" aria-label="The session's progress">
        <b>{slides ? '' : `${of(tmasDone, tmas.length, 'TMA')} · `}{of(coresDone(allCores), allCores.length, unit.one)} · {of(whole.done, whole.total, 'field')} done</b>
        {progressText(whole) && <span className="muted"> · {progressText(whole)}</span>}
      </p>
      <div className="row session-fields-bar">
        <ul className="row small tile-legend" aria-label="Tile outlines">
          {LOOK_ORDER.map((look) => <li key={look} title={LOOK_HINT[look]}><span className={`tile-swatch look-${look}`} />{LOOK_TEXT[look]}</li>)}
          {fields.some((f) => f.ai_reviewed) && <li title="AI review labels exist on the field (not shown while labelling)"><span className="ai-dot in-legend" />AI reviewed</li>}
          {fields.some((f) => f.elsewhere?.length) && <li title="People labelled the field in another session. This session is independent: its state here and what you see while labelling come from this session only"><span className="elsewhere-dot in-legend" />Labelled in another session</li>}
        </ul>
        <span className="spacer" />
        <div className="segmented small" role="group" aria-label="How the fields are shown">
          <button className={`btn small${order === 'natural' ? ' chosen' : ''}`} aria-pressed={order === 'natural'} onClick={() => setOrder('natural')}>Natural order</button>
          <button className={`btn small${order === 'confidence' ? ' chosen' : ''}`} aria-pressed={order === 'confidence'} onClick={() => setOrder('confidence')}
            title="Every field in one list, the ones detection is least sure of first: those with the most objects close to the acceptance threshold">By confidence</button>
        </div>
      </div>
      {order === 'confidence' && !confidence.data && <p className="small muted" style={{ margin: 0 }}>{confidence.error ? 'The confidence could not be read; the natural order stands.' : 'Weighing detection’s confidence…'}</p>}
      {order === 'confidence' && ranked ? <RankedTiles name={name} list={ranked} needed={detail.needed ?? 1} me={detail.annotator} sure={confidence.data!} slides={slides} />
        : tmas.map(({ tma, cores }) => {
        return (
          <Fold key={tma} className="annotate-tma" open={fields.length <= FOLDED_FROM}
            summary={<><b>{tma}</b><span className="small muted">{tmaLine(cores, unit.one)}</span></>}>
            {() => <div className={`core-maps${slides ? ' slide-maps' : ''}`}>{cores.map((core) => {
              const list = core.list.map(({ f }) => f)
              const state = coreState(list)
              return (
                <Fold key={core.id} className="core-map-card card" open
                  summary={<><b>{unit.title} {core.name}</b><span className={`badge ${BADGE[state]}`}>{LOOK_TEXT[state]}</span>
                    <span className="small muted">{coreLine(list)}</span></>}>
                  {() => <CoreMap name={name} layout={detail.cores?.[core.id]} list={core.list} needed={detail.needed ?? 1} me={detail.annotator} slides={slides} />}
                </Fold>
              )
            })}</div>}
          </Fold>
        )
      })}
    </div>
  )
}

/** One core as its tiles, each at its place: the session's tiles as pictures outlined in their state's colour (a click
 *  opens the field), the core's other tiles faint. Without a layout (no squares known), the tiles in a row. */
/** One field's tile: its picture outlined in its state's colour, the AI dot, the facts on hover; a click opens it
 *  (not while someone else has it open). */
function Tile({ name, n: { f, i }, needed, me, style, className = '', sure, slides = false, order = 'natural' }: {
  name: string; n: Numbered; needed: number; me: string | null; style?: React.CSSProperties; className?: string; sure?: Sure; slides?: boolean
  /** the list it is in: the field page's "Next" goes on in it */
  order?: FieldOrder
}) {
  const look = lookOf(f)
  const body = <><img src={apiUrl(`/api/annotate/${name}/fields/${enc(f.tile_id)}/thumb.jpg`)} alt="" loading="lazy" decoding="async"
    onError={(e) => { e.currentTarget.style.visibility = 'hidden' }} />{f.ai_reviewed && <span className="ai-dot" aria-hidden="true" />}{f.elsewhere?.length ? <span className={`elsewhere-dot${f.ai_reviewed ? ' second' : ''}`} aria-hidden="true" /> : null}</>
  const taken = takenBy(f, me).length > 0
  const common = { className: `core-tile look-${look}${className}${taken ? ' taken' : ''}`, style, title: hoverText(f, i, needed, me, sure, slides), 'aria-label': `Field ${i + 1}: ${LOOK_TEXT[look]}` }
  return f.prepared && !taken ? <Link to={fieldPath(name, f.tile_id, order)} {...common}>{body}</Link> : <span {...common}>{body}</span>
}

/** Ranked tiles drawn so far, and how many more each time the end of the list comes into view. */
const RANKED_FIRST = 240
const RANKED_MORE = 240

/** By confidence: every field of the session in one wrapping list of tiles, least confident first, with no TMA or core
 *  around them (the hover says where each is from). Drawn a few rows at a time as the list is scrolled, the pictures
 *  loading as they come into view, so thousands of fields stay quick. */
function RankedTiles({ name, list, needed, me, sure, slides = false }: { name: string; list: Numbered[]; needed: number; me: string | null; sure: Record<string, Sure>; slides?: boolean }) {
  const [shown, setShown] = useState(RANKED_FIRST)
  const end = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const target = end.current
    if (!target || shown >= list.length) return
    const seen = new IntersectionObserver((entries) => { if (entries.some((e) => e.isIntersecting)) setShown((n) => Math.min(list.length, n + RANKED_MORE)) },
      { rootMargin: '600px 0px' })
    seen.observe(target)
    return () => seen.disconnect()
  }, [shown, list.length])
  return (
    <div className="ranked">
      <div className="ranked-tiles">{list.slice(0, shown).map((n) => <Tile key={n.f.tile_id} name={name} n={n} needed={needed} me={me} className=" ranked-tile" sure={sure[n.f.tile_id]} slides={slides} order="confidence" />)}</div>
      {shown < list.length && <div ref={end} className="small muted ranked-more">{(list.length - shown).toLocaleString()} more fields below…</div>}
    </div>
  )
}

/** One core (or a whole slide) as its tiles, each at its place. A slide has its picture under its tiles, the tiles
 *  outside the session washed out over it. */
function CoreMap({ name, layout, list, needed, me, slides = false }: { name: string; layout: Layout | undefined; list: Numbered[]; needed: number; me: string | null; slides?: boolean }) {
  const placed = layout && list.every(({ f }) => f.box)
  const pct = (v: number, of: number) => `${(100 * v) / of}%`
  const tile = (n: Numbered, style?: React.CSSProperties) => <Tile key={n.f.tile_id} name={name} n={n} needed={needed} me={me} style={style} className={placed ? '' : ' in-row'} slides={slides} />
  if (!placed) return <div className="core-row">{list.map((n) => tile(n))}</div>
  const { x, y, width, height } = layout
  const under = layout.picture && layout.slide ? layout.slide : null
  return (  // as wide as its column: three cores side by side fill the width (CSS), so the tiles' size follows the window
    <div className={`core-map${under ? ' slide-map' : ''}`} style={{ aspectRatio: `${width} / ${height}` }}>
      {under && <img className="slide-map-picture" src={apiUrl(layout.picture!)} alt="" loading="lazy" decoding="async" draggable={false}
        style={{ left: pct(under[0] - x, width), top: pct(under[1] - y, height), width: pct(under[2], width), height: pct(under[3], height) }}
        onError={(e) => { e.currentTarget.style.visibility = 'hidden' }} />}
      {layout.others.map(([ox, oy, w, h, id], k) => <span key={k} className="core-tile-other" aria-hidden="true"
        style={{ left: pct(ox - x, width), top: pct(oy - y, height), width: pct(w, width), height: pct(h, height) }}>
        {id && <img src={apiUrl(`/api/annotate/${name}/fields/${enc(id)}/thumb.jpg`)} alt="" loading="lazy" decoding="async" draggable={false}
          onError={(e) => { e.currentTarget.style.visibility = 'hidden' }} />}</span>)}
      {list.map((n) => { const [bx, by, w, h] = n.f.box!; return tile(n, { left: pct(bx - x, width), top: pct(by - y, height), width: pct(w, width), height: pct(h, height) }) })}
    </div>
  )
}

