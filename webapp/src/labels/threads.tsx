/* AT8 threads mode of "Fix the field" (H) and its tutorial: the detected thread network of the field, redrawn live at
   another thread threshold by one slider (labels/thread_threshold.py on the server). Experts never edit single threads:
   they move the slider until the lines match the threads they see and press Enter ("Threads look right"); untouched,
   that says detection's threshold is right. The network is drawn only in this mode. */
import { memo, useMemo } from 'react'
import { fmt } from '../components'
import { PALETTE } from '../palette'

/** GET …/fields/{tile}/threads (and the tutorial's practice-threads/ladder): the network at every step. */
export interface ThreadLadder {
  status: string
  factors: number[]
  /** the step of detection's own threshold (1.0) */
  detection: number
  steps: { factor: number; length_um: number; lines: number[][] }[]
  dab_threshold?: number
  ridge_threshold?: number
  /** the tutorial's practice: its picture's size, its field box and its credit line */
  width?: number; height?: number; box?: { x: number; y: number; width: number; height: number }; credit?: string; tile_id?: string
}

export const THREAD_KEY = 'h'
export const THREAD_COLOUR = PALETTE.overlays.threads

/** The step nearest a factor (the saved one), on the ladder's log scale. */
export function stepOf(factors: number[], factor: number | null | undefined, fallback: number) {
  if (factor == null || !factors.length) return fallback
  let best = fallback
  let gap = Infinity
  factors.forEach((f, i) => { const d = Math.abs(Math.log(f) - Math.log(factor)); if (d < gap) { gap = d; best = i } })
  return best
}

/** The lines of one step as one path (thousands of lines draw at once). */
export const ThreadLines = memo(function ThreadLines({ lines, unit }: { lines: number[][]; unit: number }) {
  const d = useMemo(() => lines.map((l) => {
    let out = `M${l[0]} ${l[1]}`
    for (let i = 2; i < l.length; i += 2) out += `L${l[i]} ${l[i + 1]}`
    return l.length === 2 ? `${out}h0.01` : out
  }).join(''), [lines])
  return (
    <g className="thread-lines" pointerEvents="none">
      <path d={d} fill="none" stroke="#000" strokeOpacity={0.45} strokeWidth={3.6 * unit} strokeLinecap="round" strokeLinejoin="round" />
      <path d={d} fill="none" stroke={THREAD_COLOUR} strokeWidth={1.8 * unit} strokeLinecap="round" strokeLinejoin="round" />
    </g>
  )
})

/** The thread-threshold slider: the ladder's steps, detection's marked on the track. */
export function ThreadSlider({ ladder, index, onChange, disabled }: { ladder: ThreadLadder; index: number; onChange: (index: number) => void; disabled?: boolean }) {
  const last = Math.max(1, ladder.factors.length - 1)
  const at = ladder.detection / last
  const factor = ladder.factors[index] ?? 1
  return (
    <label className="threshold-slider small">
      <span className="row"><span>Thread threshold</span><span className="spacer" /><b>{factor === 1 ? 'detection’s' : `${factor.toFixed(2)}× detection’s`}</b></span>
      <span className="threshold-track">
        <input type="range" min={0} max={last} step={1} value={index} disabled={disabled} aria-label="Thread threshold"
          onChange={(e) => onChange(Number(e.target.value))}
          onPointerUp={(e) => e.currentTarget.blur()} />{/* let go: Enter, H and [ ] go to the page again */}
        <span className="threshold-mark" style={{ left: `calc(${at * 100}% + ${(0.5 - at) * 14}px)` }} title="detection’s thread threshold" />
      </span>
      <span className="row muted"><span>{ladder.factors[0]}× more</span><span className="spacer" /><span>detection’s (marked)</span><span className="spacer" />
        <span>{ladder.factors[last]}× fewer</span></span>
    </label>
  )
}

/** What the slider gives: the thread length in the field against detection's. */
export function threadText(ladder: ThreadLadder, index: number) {
  const now = ladder.steps[index]?.length_um ?? 0
  const detected = ladder.steps[ladder.detection]?.length_um ?? 0
  const lengths = `${fmt(Math.round(now))} µm of threads in the field`
  if (index === ladder.detection) return `${lengths} (detection’s).`
  return `${lengths}; detection’s ${fmt(Math.round(detected))} µm (${detected ? `${Math.round((100 * now) / detected)}%` : '—'}).`
}

/** The side card of threads mode: the slider, what it gives, Enter. */
export function ThreadsCard({ ladder, pending, error, index, onChange, onConfirm, saving, saved, practice }: {
  ladder: ThreadLadder | null; pending?: string | null; error?: string | null; index: number; onChange: (index: number) => void
  onConfirm: () => void; saving?: boolean; saved?: number | null; practice?: boolean
}) {
  return (
    <div className="card side-block start-card threads-card" role="group" aria-label="Threads mode: the thread threshold">
      <h3>Threads</h3>
      <p className="small muted" style={{ margin: '0 0 4px' }}>The thread network detection found, at one thread threshold · <span className="kbd">H</span> or <span className="kbd">Esc</span> to leave</p>
      {error ? <p className="small bad-text">{error}</p>
        : !ladder ? <p className="small muted">{pending ?? 'Drawing the threads of this field…'}</p>
          : ladder.status !== 'done' || !ladder.steps.length ? <p className="small muted">No thread network here: detection has not run on this core yet.</p>
            : <>
              <ThreadSlider ladder={ladder} index={index} onChange={onChange} disabled={saving} />
              <p className="small" style={{ margin: '4px 0' }}>{threadText(ladder, index)}</p>
              <p className="small muted" style={{ margin: '0 0 6px' }}>Move it until the lines match the threads you see (lower: fainter threads join; higher: only the
                darkest stay), then press Enter. Untouched, Enter says detection’s threshold is right. You never edit single threads.
                {practice ? ' In this practice nothing is saved.' : ' It tunes the thread detector (Models page); the analysis keeps detection’s threshold.'}</p>
              <div className="row"><button className="btn primary" onClick={onConfirm} disabled={saving}>Threads look right</button>
                <span className="small muted"><span className="kbd">Enter</span> · <span className="kbd">[</span> <span className="kbd">]</span> one step</span></div>
              {saved != null && <p className="small good-text" style={{ margin: '6px 0 0' }}>✓ Saved: {saved === 1 ? 'detection’s threshold' : `${saved.toFixed(2)}× detection’s`}</p>}
            </>}
    </div>
  )
}
