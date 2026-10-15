import { useState } from 'react'
import type { ReactNode } from 'react'
import type { MapCell, ScreenState } from '../api'
import { apiUrl } from '../api'
import { Loading } from '../components'

/** Everything technical on a screen (file locations, full tables, step options): folded away, never removed. What is
inside is only built once it is first opened, so anything it loads is not loaded for people who never look. */
export function Advanced({ children, title = 'Advanced', hint }: { children: ReactNode; title?: string; hint?: string }) {
  const [opened, setOpened] = useState(false)
  return (
    <details className="advanced" onToggle={(e) => { if (e.currentTarget.open) setOpened(true) }}>
      <summary><span>{title}</span>{hint && <span className="muted small"> · {hint}</span>}</summary>
      <div className="advanced-body">{opened && children}</div>
    </details>
  )
}

const MARK: Record<ScreenState, string> = { done: '✓', todo: '', attention: '!', warning: '!', optional: '–', running: '…' }

export function ScreenMark({ state, number }: { state: ScreenState; number: number }) {
  return <span className={`screen-mark mark-${state}`}>{MARK[state] || number}</span>
}

/** What stainID worked out by itself, shown before it asks anything. */
export function Found({ children, tone = 'good' }: { children: ReactNode; tone?: 'good' | 'warn' | 'plain' }) {
  return <div className={`found found-${tone}`}>{children}</div>
}

/** Shows `children(data)` once a screen's data is in; a loading line before that, and the error in plain words with
"Try again" if it failed. Later reloads keep the old data on screen. */
export function ScreenData<T>({ state, label, children }: {
  state: { data: T | null; error: string | null; reload: () => void }; label?: ReactNode; children: (data: T) => ReactNode
}) {
  if (state.data !== null && state.data !== undefined) return <>{children(state.data)}</>
  if (state.error) return <Problem text={`This could not be loaded: ${plainError(state.error)}`} onRetry={state.reload} />
  return <Loading label={label} />
}

/** An API error without its status code. */
export const plainError = (error: string) => error.replace(/^\d{3}: /, '')

/** An error in plain words, with a "Try again" button when it can be retried. */
export function Problem({ text, onRetry }: { text: string; onRetry?: () => void }) {
  return (
    <div className="error-note problem-note" role="alert">
      <span>{text}</span>{onRetry && <button className="btn small" onClick={onRetry}>Try again</button>}
    </div>
  )
}

/** Back and Next at the foot of a screen. Next is the filled button unless the screen has its own main action to do
first (`quiet`). */
export function ScreenFoot({ back, next, nextLabel, busy, disabled, extra, quiet = false }: {
  back?: () => void; next?: () => void; nextLabel?: string; busy?: boolean; disabled?: boolean; extra?: ReactNode; quiet?: boolean
}) {
  return (
    <div className="screen-foot">
      {back && <button className="btn" onClick={back}>← Back</button>}
      <span className="spacer" />
      {extra}
      {next && <button className={`btn${quiet ? '' : ' primary'}`} disabled={busy || disabled} onClick={next}>{busy ? 'Saving…' : nextLabel ?? 'Next'} →</button>}
    </div>
  )
}

export function cellTone(cell: MapCell): string {
  if (cell.note) return 'note'
  if (cell.tissue_control) return 'control'
  if (!cell.donor_id) return 'empty'
  return 'donor'
}

function CellReading({ cell }: { cell: MapCell }) {
  if (cell.tissue_control) return <span>control: {cell.tissue_control}</span>
  if (!cell.donor_id) return cell.note ? <span>{cell.note}</span> : <span className="muted">no core</span>
  const pathology = cell.cerad || cell.braak ? ` · C ${cell.cerad || '?'} B ${cell.braak || '?'}` : ''
  return (
    <span title={cell.note}>
      <b>{cell.donor_id}</b> {cell.region || <i>no region</i>}{cell.disease_group ? ` · ${cell.disease_group}` : ''}{pathology}
      {cell.note && <span className="cell-note"> · {cell.note}</span>}
      {cell.sample_region_id && <span className="muted sample-id"> {cell.sample_region_id}</span>}
    </span>
  )
}

/** A TMA map as the array is drawn: one cell per core position with what was read from it. With `onEdit` the cells
can be typed in (and pasted into); with `thumbs` each cell shows the core found at that position. */
export function MapGrid({ grid, cells, onEdit, thumbs, positions }: {
  grid: string[][]; cells: MapCell[]; onEdit?: (row: number, column: number, text: string) => void
  thumbs?: string; positions?: string[]
}) {
  const byLabel = Object.fromEntries(cells.map((c) => [c.label, c]))
  const width = Math.max(1, ...grid.map((r) => r.length))
  const letter = (c: number) => String.fromCharCode(65 + c)
  const onSlide = positions ? new Set(positions) : null
  return (
    <div className="map-grid-wrap">
      <div className="map-grid" style={{ gridTemplateColumns: `28px repeat(${width}, minmax(104px, 1fr))` }}>
        <span />
        {Array.from({ length: width }, (_, c) => <span key={c} className="map-axis">{letter(c)}</span>)}
        {grid.map((row, r) => (
          <MapRow key={r} row={row} r={r} width={width} byLabel={byLabel} onEdit={onEdit} thumbs={thumbs} onSlide={onSlide} />
        ))}
      </div>
    </div>
  )
}

function MapRow({ row, r, width, byLabel, onEdit, thumbs, onSlide }: {
  row: string[]; r: number; width: number; byLabel: Record<string, MapCell>; onEdit?: (row: number, column: number, text: string) => void
  thumbs?: string; onSlide: Set<string> | null
}) {
  return (
    <>
      <span className="map-axis">{r + 1}</span>
      {Array.from({ length: width }, (_, c) => {
        const label = `${String.fromCharCode(65 + c)}-${r + 1}`
        const cell = byLabel[label] ?? { label, raw: '', donor_id: '', region: '', region_code: '', disease_group: '', cerad: '', braak: '', pathology_raw: '', tissue_control: '', note: '' }
        const missing = onSlide && !onSlide.has(label)
        return (
          <div key={c} className={`map-cell tone-${cellTone(cell)}${missing ? ' off-slide' : ''}`} title={missing ? `${label} was not found on the slide` : label}>
            <div className="map-cell-top">
              <span className="map-label">{label}</span>
              {thumbs && !missing && <Thumb src={apiUrl(`${thumbs}/${label}.jpg?size=96`)} />}
            </div>
            {onEdit ? (
              <textarea rows={3} value={row[c] ?? ''} placeholder="donor region group" onChange={(e) => onEdit(r, c, e.target.value)}
                onPaste={(e) => {
                  const text = e.clipboardData.getData('text')
                  if (text.includes('\t')) { e.preventDefault(); pasteBlock(text, r, c, onEdit) }
                }} />
            ) : <div className="map-raw">{row[c] || <span className="muted">—</span>}</div>}
            <div className="map-reading small"><CellReading cell={cell} /></div>
          </div>
        )
      })}
    </>
  )
}

/** Cells copied from a spreadsheet (tab-separated) are spread over the grid from the cell pasted into. */
function pasteBlock(text: string, row: number, column: number, onEdit: (row: number, column: number, text: string) => void) {
  text.replace(/\r/g, '').replace(/\n$/, '').split('\n').forEach((line, i) => line.split('\t').forEach((value, j) => onEdit(row + i, column + j, value.replace(/^"|"$/g, ''))))
}

function Thumb({ src }: { src: string }) {
  const [failed, setFailed] = useState(false)
  return failed ? null : <img className="map-thumb" src={src} alt="" loading="lazy" onError={() => setFailed(true)} />
}

export const plural = (n: number, one: string, many = `${one}s`) => `${n.toLocaleString()} ${n === 1 ? one : many}`

export function MapSummaryLine({ summary }: { summary: { positions: number; cores: number; donors: number; controls: number; empty: number; unreadable: number; groups: string[] } }) {
  return (
    <span className="small secondary">
      {plural(summary.cores, 'donor core')} from {plural(summary.donors, 'donor')}{summary.controls ? `, ${plural(summary.controls, 'control core')}` : ''}{summary.empty ? `, ${plural(summary.empty, 'empty position')}` : ''}
      {summary.groups.length ? ` · groups ${summary.groups.join(', ')}` : ''}
      {summary.unreadable > 0 && <b className="status-outdated"> · {summary.unreadable === 1 ? '1 cell needs' : `${summary.unreadable} cells need`} a look</b>}
    </span>
  )
}
