/* The fields of a session for the person labelling: by TMA, then core, each core drawn as its own tiles (each tile at
   its place in the core, the core's tiles outside the session faint), a tile's state by its outline colour, the full
   facts on hover, a click opens the field. "By confidence" puts the cores detection is least sure of first. */
import { useMemo } from 'react'
import { Link } from 'react-router-dom'
import type { FieldState, SessionDetail } from '../api'
import { apiUrl, enc } from '../api'
import { useFetch, useStored } from '../hooks'

type Field = SessionDetail['fields'][number]
type Layout = NonNullable<SessionDetail['cores']>[string]
/** A field's state for this person: where it stands for everyone (`sessions.field_states`), and "mine" when they
 *  finished it while others are still asked. */
type Look = FieldState | 'mine'
export const LOOK_TEXT: Record<Look, string> = { not_started: 'To do', in_progress: 'In progress', done: 'Done', preparing: 'Being prepared', mine: 'Finished by you' }
const LOOK_ORDER: Look[] = ['not_started', 'in_progress', 'mine', 'done', 'preparing']
const BADGE: Record<Look, string> = { not_started: 'status-todo', in_progress: 'status-running', done: 'status-done', preparing: 'status-queued', mine: 'status-done' }

const stateOf = (f: Field): FieldState => f.state ?? (!f.prepared ? 'preparing' : f.done_by_me ? 'done' : f.open_by_others ? 'in_progress' : 'not_started')
const lookOf = (f: Field): Look => (f.done_by_me && stateOf(f) !== 'done' ? 'mine' : stateOf(f))
const who = (people: string[], me: string | null) => people.map((p) => (p === me ? 'you' : p)).join(', ')
/** Someone else has the field open: only one person works on a field at a time. */
const takenBy = (f: Field, me: string | null) => (f.open_by ?? []).filter((p) => p !== me)
const day = (t: number) => new Date(t * 1000).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })

/** The hover text of a tile: its state, who is on it or has it open now, who finished it and when, how many of the
 *  people asked are done. */
function hoverText(f: Field, i: number, needed: number, me: string | null): string {
  const done = f.done_by ?? []
  const on = [...new Set([...(f.working ?? []), ...(f.open_by ?? [])])].filter((p) => !done.includes(p))
  return [
    `Field ${i + 1} · ${LOOK_TEXT[lookOf(f)]}`,
    on.length ? `Working on it now: ${who(on, me)}` : '',
    done.length ? `Finished by ${done.map((p) => `${p === me ? 'you' : p}${f.done_at?.[p] ? ` (${day(f.done_at[p])})` : ''}`).join(', ')}` : '',
    `${Math.min(done.length, needed)} of ${needed} ${needed === 1 ? 'person' : 'people'} done`,
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
/** "3 fields · 1 done · 1 in progress · 1 yours" */
function countText(list: Field[]) {
  const n = (s: FieldState) => list.filter((f) => stateOf(f) === s).length
  const mine = list.filter((f) => f.done_by_me).length
  return [`${list.length} field${list.length === 1 ? '' : 's'}`, n('done') && `${n('done')} done`, n('in_progress') && `${n('in_progress')} in progress`,
    mine && `${mine} by you`].filter(Boolean).join(' · ')
}

type Numbered = { f: Field; i: number }
type Core = { id: string; name: string; list: Numbered[] }
type Order = 'natural' | 'confidence'
const natural = (a: string, b: string) => a.localeCompare(b, undefined, { numeric: true })

export function SessionFieldMap({ detail }: { detail: SessionDetail }) {
  const { meta, fields } = detail
  const name = meta.name
  const [order, setOrder] = useStored<Order>(`stainid.fieldOrder.${detail.annotator ?? ''}`, 'natural')
  const confidence = useFetch<Record<string, { near: number; objects: number }>>(order === 'confidence' ? `/api/annotate/${name}/confidence` : null)
  const tmas = useMemo(() => {
    const byTma = new Map<string, Map<string, Core>>()
    fields.forEach((f, i) => {
      const tma = f.tma_name || f.tma || 'Other fields'
      const coreId = f.core_id || f.core || '—'
      const cores = byTma.get(tma) ?? byTma.set(tma, new Map()).get(tma)!
      const core = cores.get(coreId) ?? cores.set(coreId, { id: coreId, name: f.core || coreId, list: [] }).get(coreId)!
      core.list.push({ f, i })
    })
    return [...byTma].sort(([a], [b]) => natural(a, b)).map(([tma, cores]) => ({ tma, cores: [...cores.values()].sort((a, b) => natural(a.name, b.name)) }))
  }, [fields])
  const near = (core: Core) => core.list.reduce((sum, { f }) => sum + (confidence.data?.[f.tile_id]?.near ?? 0), 0)
  const ordered = order === 'confidence' && confidence.data
    ? tmas.map((t) => ({ ...t, cores: [...t.cores].sort((a, b) => near(b) - near(a) || natural(a.name, b.name)) }))  // least confident first
    : tmas
  return (
    <div className="session-fields">
      <div className="row session-fields-bar">
        <ul className="row small tile-legend" aria-label="Tile outlines">
          {LOOK_ORDER.map((look) => <li key={look}><span className={`tile-swatch look-${look}`} />{LOOK_TEXT[look]}</li>)}
        </ul>
        <span className="spacer" />
        <div className="segmented small" role="group" aria-label="Order of the cores">
          <button className={`btn small${order === 'natural' ? ' chosen' : ''}`} aria-pressed={order === 'natural'} onClick={() => setOrder('natural')}>Natural order</button>
          <button className={`btn small${order === 'confidence' ? ' chosen' : ''}`} aria-pressed={order === 'confidence'} onClick={() => setOrder('confidence')}
            title="Within each TMA, the cores detection is least sure of first: those with the most objects close to the acceptance threshold">By confidence</button>
        </div>
      </div>
      {order === 'confidence' && !confidence.data && <p className="small muted" style={{ margin: 0 }}>{confidence.error ? 'The confidence could not be read; the natural order stands.' : 'Weighing detection’s confidence…'}</p>}
      {ordered.map(({ tma, cores }) => {
        const all = cores.flatMap((c) => c.list.map(({ f }) => f))
        return (
          <details key={tma} className="annotate-tma" open>
            <summary><b>{tma}</b><span className="small muted">{countText(all)}</span></summary>
            <div className="core-maps">{cores.map((core) => {
              const list = core.list.map(({ f }) => f)
              const state = coreState(list)
              const sure = confidence.data && order === 'confidence' ? near(core) : null
              return (
                <details key={core.id} className="core-map-card card" open>
                  <summary><b>Core {core.name}</b><span className={`badge ${BADGE[state]}`}>{LOOK_TEXT[state]}</span>
                    <span className="small muted">{countText(list)}{sure != null ? ` · ${sure.toFixed(1)} near the threshold` : ''}</span></summary>
                  <CoreMap name={name} layout={detail.cores?.[core.id]} list={core.list} needed={detail.needed ?? 1} me={detail.annotator} />
                </details>
              )
            })}</div>
          </details>
        )
      })}
    </div>
  )
}

/** One core as its tiles, each at its place: the session's tiles as pictures outlined in their state's colour (a click
 *  opens the field), the core's other tiles faint. Without a layout (no squares known), the tiles in a row. */
function CoreMap({ name, layout, list, needed, me }: { name: string; layout: Layout | undefined; list: Numbered[]; needed: number; me: string | null }) {
  const placed = layout && list.every(({ f }) => f.box)
  const pct = (v: number, of: number) => `${(100 * v) / of}%`
  const tile = ({ f, i }: Numbered, style?: React.CSSProperties) => {
    const look = lookOf(f)
    const body = <img src={apiUrl(`/api/annotate/${name}/fields/${enc(f.tile_id)}/thumb.jpg`)} alt="" loading="lazy" onError={(e) => { e.currentTarget.style.visibility = 'hidden' }} />
    const taken = takenBy(f, me).length > 0
    const common = { className: `core-tile look-${look}${placed ? '' : ' in-row'}${taken ? ' taken' : ''}`, style, title: hoverText(f, i, needed, me), 'aria-label': `Field ${i + 1}: ${LOOK_TEXT[look]}` }
    return f.prepared && !taken ? <Link key={f.tile_id} to={`/annotate/${name}/${enc(f.tile_id)}`} {...common}>{body}</Link>
      : <span key={f.tile_id} {...common}>{body}</span>
  }
  if (!placed) return <div className="core-row">{list.map((n) => tile(n))}</div>
  const { x, y, width, height } = layout
  return (
    <div className="core-map" style={{ aspectRatio: `${width} / ${height}` }}>
      {layout.others.map(([ox, oy, w, h], k) => <span key={k} className="core-tile-other" aria-hidden="true"
        style={{ left: pct(ox - x, width), top: pct(oy - y, height), width: pct(w, width), height: pct(h, height) }} />)}
      {list.map((n) => { const [bx, by, w, h] = n.f.box!; return tile(n, { left: pct(bx - x, width), top: pct(by - y, height), width: pct(w, width), height: pct(h, height) }) })}
    </div>
  )
}

