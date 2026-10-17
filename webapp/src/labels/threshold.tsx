/* The acceptance-threshold slider: on "Fix the field" step 1 of every field (fieldStart.ts: where objects start, the
   lock), and on the Cores view a preview that is never saved. Its track is marked where detection's own threshold is. */
import { SLIDER_STEP } from './fieldStart'

export function LockIcon() {
  return (
    <svg className="lock-icon" width="12" height="13" viewBox="0 0 12 13" aria-hidden="true">
      <rect x="1.5" y="5.5" width="9" height="7" rx="1.5" fill="currentColor" />
      <path d="M3.5 5.5V4a2.5 2.5 0 0 1 5 0v1.5" fill="none" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  )
}

export function ThresholdSlider({ value, detection, onChange, disabled, label = 'Acceptance threshold' }: {
  value: number; detection: number; onChange: (value: number) => void; disabled?: boolean; label?: string
}) {
  return (
    <label className="threshold-slider small">
      <span className="row"><span>{label}</span><span className="spacer" /><b>{value.toFixed(2)}</b></span>
      <span className="threshold-track">
        <input type="range" min={0} max={1} step={SLIDER_STEP} value={value} disabled={disabled} aria-label={label}
          onChange={(e) => onChange(Number(e.target.value))} />
        <span className="threshold-mark" style={{ left: `calc(${detection * 100}% + ${(0.5 - detection) * 14}px)` }}
          title={`detection’s threshold: ${detection.toFixed(2)}`} />
      </span>
      <span className="row muted"><span>0</span><span className="spacer" /><span>detection: {detection.toFixed(2)} (marked)</span><span className="spacer" /><span>1</span></span>
    </label>
  )
}
