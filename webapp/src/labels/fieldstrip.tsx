/* Fix the field's two smaller zoom levels (levelmap.tsx), beside the field: the whole core with the session's fields by
   state, this field highlighted (it only shows: a click on another field opens it, no view box, no dragging), then
   the middle level around this field with the detailed view's extent, where a click or a drag moves the detailed view
   (zooming in when it shows the whole field). GET …/fields/{tile}/core: the core's size, where the field picture lies in it, and its
   pictures. */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { SessionDetail } from '../api'
import { apiUrl, enc } from '../api'
import { useFetch, useStored } from '../hooks'
import { PALETTE } from '../palette'
import type { Rect, Square } from '../levelmap'
import { LevelMap, middleWindow } from '../levelmap'
import type { ViewControl } from '../viewer'

/** The field's core, or (a whole-slide project, `unit` "slide") its slide in slide px, the middle picture covering
 *  `mid_box` (x0, y0, x1, y1) of it; "region": the slide's picture is not made yet, so its region, `shift` from the
 *  slide's px (the fields' squares) to the region's. */
interface FieldCore {
  core_id: string; width: number; height: number; origin: [number, number]; field: [number, number]; thumb: string; mid: string
  unit?: 'slide' | 'region'; mid_box?: [number, number, number, number]; shift?: [number, number]
}
type Field = SessionDetail['fields'][number]

const STATE_COLOUR: Record<string, string> = { done: PALETTE.states.unknown, in_progress: PALETTE.markers.others, not_started: '#000000', preparing: PALETTE.states.inactive }
const FILLED = new Set(['done', 'in_progress'])  // to do: only a thin outline, so the core stays visible where every tile is a field

/** Whether the strip shows (remembered in this browser); a button in the field page's head turns it on and off. */
export function useStripShown() { return useStored<boolean>('stainid.field.levels.v2', true) }  // v2: on again for everyone

/** The link between the detailed view and the strip: the viewer reports what is in view (`onView`) and is moved
 *  through `control`; only the strip draws again as the view moves, not the whole field page. */
export interface Levels { control: { current: ViewControl | null }; onView: (r: Rect) => void; sink: { current: ((r: Rect) => void) | null } }
export function useLevels(): Levels {
  const control = useRef<ViewControl | null>(null)
  const sink = useRef<((r: Rect) => void) | null>(null)
  const onView = useCallback((r: Rect) => sink.current?.(r), [])
  return useMemo(() => ({ control, onView, sink }), [onView])
}

const STATE_WORD: Record<string, string> = { done: 'done', in_progress: 'in progress (saved, not finished, or open now)', not_started: 'to do', preparing: 'being prepared' }

export function FieldStrip({ name, tile, fields, levels, onOpen }: {
  name: string; tile: string; fields: Field[]; levels: Levels; onOpen: (tile: string) => void
}) {
  const [view, setView] = useState<Rect | null>(null)
  useEffect(() => { levels.sink.current = setView; return () => { levels.sink.current = null } }, [levels])
  const onCentre = (x: number, y: number, zoom: number) => levels.control.current?.centre(x, y, zoom)
  const core = useFetch<FieldCore>(`/api/annotate/${name}/fields/${enc(tile)}/core`)
  const c = core.data
  const here = fields.find((f) => f.tile_id === tile)
  const squares = useMemo<Square[]>(() => (c ? fields.filter((f) => f.core_id === c.core_id && f.box).map((f, i) => {
    const [sx, sy] = c.shift ?? [0, 0]
    const [bx, by, w, h] = f.box!
    const x = bx - sx, y = by - sy
    const state = f.done_by_me ? 'done' : f.state ?? 'not_started'
    return { key: f.tile_id, x, y, w, h, colour: STATE_COLOUR[state] ?? '#000000', fill: FILLED.has(state), current: f.tile_id === tile,
      title: `Field ${i + 1} · ${STATE_WORD[state] ?? 'to do'}${f.tile_id === tile ? ' (this one)' : f.prepared ? ' · click to open' : ' · being prepared'}`,
      onOpen: f.tile_id !== tile && f.prepared ? () => onOpen(f.tile_id) : undefined }
  }) : []), [c, fields, tile, onOpen])
  const word = c?.unit === 'slide' ? 'slide' : c?.unit === 'region' ? 'region' : 'core'
  if (!c) return <div className="level-strip" aria-hidden={!core.error}>{core.error ? null : <div className="card level-card level-wait small muted">Loading…</div>}</div>
  const [ox, oy] = c.origin
  const whole: Rect = { x0: 0, y0: 0, x1: c.width, y1: c.height }
  const picture: Rect = { x0: ox, y0: oy, x1: ox + c.field[0], y1: oy + c.field[1] }
  const detail = view ? { x0: view.x0 + ox, y0: view.y0 + oy, x1: view.x1 + ox, y1: view.y1 + oy } : picture
  const fieldSide = here?.box ? Math.max(here.box[2], here.box[3]) : c.field[0]
  const midBox = c.mid_box ? { x0: c.mid_box[0], y0: c.mid_box[1], x1: c.mid_box[2], y1: c.mid_box[3] } : null
  const mid = middleWindow(detail, c.width, c.height, 3.5, fieldSide * 1.8, midBox)
  const zoomed = view ? (view.x1 - view.x0) < c.field[0] * 0.98 : false
  const move = (x: number, y: number) => onCentre(x - ox, y - oy, zoomed ? 1 : 3)
  return (
    <div className="level-strip">
      <div className="card level-card">
        <div className="level-title small"><b>Whole {word}</b><span className="muted"> · fields by state</span></div>
        <LevelMap src={apiUrl(c.thumb)} width={c.width} height={c.height} window={whole} next={null} squares={squares}
          label={`The whole ${word}: this session's fields by state; click a field to open it`} />
        <div className="level-legend small muted"><span><i className="filled" style={{ borderColor: STATE_COLOUR.done, background: STATE_COLOUR.done }} />done</span>
          <span title="Saved but not finished, or open now"><i className="filled" style={{ borderColor: STATE_COLOUR.in_progress, background: STATE_COLOUR.in_progress }} />in progress</span>
          <span><i style={{ borderColor: STATE_COLOUR.not_started }} />to do</span><span><i className="current" />this field</span></div>
        <div className="level-hint small muted">click a field to open it</div>
      </div>
      <div className="card level-card">
        <div className="level-title small"><b>Around this field</b></div>
        <LevelMap src={apiUrl(c.mid)} srcBox={midBox ?? undefined} width={c.width} height={c.height} window={mid} next={detail}
          squares={squares.filter((s) => s.current)} onMove={move} label="Around this field: the part the detailed view shows" />
      </div>
    </div>
  )
}
