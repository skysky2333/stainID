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
  not_counted?: NotCounted
  /** "wide": the simple neurite brush's ladder, 0.25x-4x */
  ladder?: string
}

/** GET …/fields/{tile}/threads/areas (the tutorial: in its ladder): where the field is not counted — detection's
 *  automatic areas with people's saved edits, no AI areas — one white PNG placed in field px, and note points. */
export interface NotCounted { status: string; x?: number; y?: number; width?: number; height?: number; png?: string | null; labels?: number[][] }

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

let masks = 0

/** The lines of one step as one path (thousands of lines draw at once). With `areas`, the parts of the field that are
 *  not counted are shaded lighter with a small "not counted" note, and the lines inside them are drawn faded: the
 *  slider is judged on counted tissue only. */
export const ThreadLines = memo(function ThreadLines({ lines, unit, areas }: { lines: number[][]; unit: number; areas?: NotCounted | null }) {
  const d = useMemo(() => lines.map((l) => {
    let out = `M${l[0]} ${l[1]}`
    for (let i = 2; i < l.length; i += 2) out += `L${l[i]} ${l[i + 1]}`
    return l.length === 2 ? `${out}h0.01` : out
  }).join(''), [lines])
  const id = useMemo(() => `threads-counted-${++masks}`, [])
  const shade = areas?.status === 'done' && areas.png ? areas : null
  const place = shade ? { x: shade.x ?? 0, y: shade.y ?? 0, width: shade.width ?? 0, height: shade.height ?? 0 } : null
  const drawn = (opacity: number) => <>
    <path d={d} fill="none" stroke="#000" strokeOpacity={0.45 * opacity} strokeWidth={3.6 * unit} strokeLinecap="round" strokeLinejoin="round" />
    <path d={d} fill="none" stroke={THREAD_COLOUR} strokeOpacity={opacity} strokeWidth={1.8 * unit} strokeLinecap="round" strokeLinejoin="round" />
  </>
  return (
    <g className="thread-lines" pointerEvents="none">
      {shade && place && <>
        <defs>
          <filter id={`${id}-black`}><feColorMatrix type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 1 0" /></filter>
          <mask id={id} maskUnits="userSpaceOnUse" x={-1e5} y={-1e5} width={2e5} height={2e5}>
            <rect x={-1e5} y={-1e5} width={2e5} height={2e5} fill="#fff" />
            <image href={shade.png!} {...place} preserveAspectRatio="none" filter={`url(#${id}-black)`} style={{ imageRendering: 'pixelated' }} />
          </mask>
        </defs>
        <image className="not-counted" href={shade.png!} {...place} preserveAspectRatio="none" opacity={0.4} style={{ imageRendering: 'pixelated' }} />
      </>}
      {shade ? <>
        <g className="thread-faded">{drawn(0.25)}</g>
        <g className="thread-counted" mask={`url(#${id})`}>{drawn(1)}</g>
        {(shade.labels ?? []).map(([x, y], i) => (
          <text key={i} x={x} y={y} textAnchor="middle" dominantBaseline="middle" fontSize={12 * unit} fill="#333" stroke="#fff" strokeWidth={3 * unit}
            paintOrder="stroke" className="not-counted-note">not counted</text>
        ))}
      </> : drawn(1)}
    </g>
  )
})

/** The thread level: one step of the ladder (`index`; its factor on the slide's DAB threshold, both cuts of the thread
 *  filter), detection's marked on the track. Shown the way people expect: further right a lower factor, so more and
 *  fainter lines count as threads (neurite). Detailed threads mode and the simple neurite brush share it. */
export function ThreadSlider({ ladder, index, onChange, disabled }: { ladder: ThreadLadder; index: number; onChange: (index: number) => void; disabled?: boolean }) {
  const last = Math.max(1, ladder.factors.length - 1)
  const at = (last - ladder.detection) / last
  const factor = ladder.factors[index] ?? 1
  return (
    <label className="threshold-slider small">
      <span className="row"><span>Thread level</span><span className="spacer" /><b>{factor === 1 ? 'detection’s' : `${factor.toFixed(2)}× detection’s cut`}</b></span>
      <span className="threshold-track">
        <input type="range" min={0} max={last} step={1} value={last - index} disabled={disabled} aria-label="Thread level"
          aria-valuetext={factor === 1 ? 'detection’s' : `${factor.toFixed(2)} times detection’s cut`}
          onChange={(e) => onChange(last - Number(e.target.value))}
          onPointerUp={(e) => e.currentTarget.blur()} />{/* let go: Enter, H and [ ] go to the page again */}
        <span className="threshold-mark" style={{ left: `calc(${at * 100}% + ${(0.5 - at) * 14}px)` }} title="detection’s thread level" />
      </span>
      <span className="row muted"><span>← fewer</span><span className="spacer" /><span>More counted as neurite →</span></span>
    </label>
  )
}

/** One step of the thread level with [ / ]: `]` to the right (more counted), `[` to the left. */
export function threadKeyStep(ladder: ThreadLadder, index: number, key: string): number {
  return Math.max(0, Math.min(ladder.factors.length - 1, index + (key === ']' ? -1 : 1)))
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
export function ThreadsCard({ ladder, pending, error, index, onChange, onConfirm, saving, saved, practice, notCounted }: {
  ladder: ThreadLadder | null; pending?: string | null; error?: string | null; index: number; onChange: (index: number) => void
  onConfirm: () => void; saving?: boolean; saved?: number | null; practice?: boolean; notCounted?: boolean
}) {
  return (
    <div className="card side-block start-card threads-card" role="group" aria-label="Threads mode: the thread level">
      <h3>Threads</h3>
      <p className="small muted" style={{ margin: '0 0 4px' }}>The thread network detection found, at one thread level · <span className="kbd">H</span> or <span className="kbd">Esc</span> to leave</p>
      {error ? <p className="small bad-text">{error}</p>
        : !ladder ? <p className="small muted">{pending ?? 'Drawing the threads of this field…'}</p>
          : ladder.status !== 'done' || !ladder.steps.length ? <p className="small muted">No thread network here: detection has not run on this core yet.</p>
            : <>
              <ThreadSlider ladder={ladder} index={index} onChange={onChange} disabled={saving} />
              <p className="small" style={{ margin: '4px 0' }}>{threadText(ladder, index)}</p>
              {notCounted && <p className="small muted" style={{ margin: '0 0 4px' }}>Lighter areas are not counted (fold, artifact, no tissue): their threads are drawn faded. Judge the slider on the rest.</p>}
              <p className="small muted" style={{ margin: '0 0 6px' }}>Move it until the lines match the threads you see (to the right more and fainter lines count;
                to the left only the darkest stay), then press Enter. Untouched, Enter says detection’s level is right. You never edit single threads.
                {practice ? ' In this practice nothing is saved.' : ' It tunes the thread detector (Models page); the analysis keeps detection’s threshold.'}</p>
              <div className="row"><button className="btn primary" onClick={onConfirm} disabled={saving}>Threads look right</button>
                <span className="small muted"><span className="kbd">Enter</span> · <span className="kbd">[</span> <span className="kbd">]</span> one step</span></div>
              {saved != null && <p className="small good-text" style={{ margin: '6px 0 0' }}>✓ Saved: {saved === 1 ? 'detection’s thread level' : `${saved.toFixed(2)}× detection’s cut`}</p>}
            </>}
    </div>
  )
}
