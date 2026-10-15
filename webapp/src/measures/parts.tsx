import type { ReactNode } from 'react'
import { ErrorNote, Loading } from '../components'

/** Like `Loaded`, with a "Try again" button when the request failed. */
export function Waiting<T>({ state, label, children }: {
  state: { data: T | null; error: string | null; retry: () => void }; label?: ReactNode; children: (data: T) => ReactNode
}) {
  if (state.data !== null) return <>{children(state.data)}</>
  if (state.error) {
    return <div className="row"><ErrorNote error={state.error} /><button className="btn small" onClick={state.retry}>Try again</button></div>
  }
  return <Loading label={label} />
}

/** Previous / next buttons for a long list shown a page at a time: "26–50 of 312". */
export function Pager({ offset, size, total, onChange }: { offset: number; size: number; total: number; onChange: (offset: number) => void }) {
  if (total <= size) return null
  return (
    <div className="pager">
      <button className="btn small" disabled={offset === 0} onClick={() => onChange(Math.max(0, offset - size))}>← Previous</button>
      <span className="muted small">{(offset + 1).toLocaleString()}–{Math.min(total, offset + size).toLocaleString()} of {total.toLocaleString()}</span>
      <button className="btn small" disabled={offset + size >= total} onClick={() => onChange(offset + size)}>Next →</button>
    </div>
  )
}

/** A row of buttons where exactly one is chosen (a few plain choices). */
export function Choice<T extends string>({ value, options, onChange, label }: {
  value: T; options: [T, string][]; onChange: (value: T) => void; label: string
}) {
  return (
    <div className="choice" role="radiogroup" aria-label={label}>
      {options.map(([key, text]) => (
        <button key={key} role="radio" aria-checked={value === key} className={value === key ? 'active' : ''} onClick={() => onChange(key)}>{text}</button>
      ))}
    </div>
  )
}
