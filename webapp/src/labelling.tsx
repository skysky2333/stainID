/* Words and small pieces shared by the Labels, Models and annotation pages. */
import { useState } from 'react'
import type { ReactNode } from 'react'
import type { AiView, Metrics } from './api'
import { Modal } from './components'

/** What each stain's model looks for, in the plural ("finds 90% of real neurons"). */
export const STAIN_OBJECTS: Record<string, string> = { NeuN: 'neurons', '6E10': 'plaques', AT8: 'tau+ neurons' }
export const objectsOf = (stain: string) => STAIN_OBJECTS[stain] ?? 'objects'

/** What people who label see of the AI, as the admin chooses it. */
export const AI_VIEW_TEXT: Record<AiView, string> = {
  full: 'AI calls and outlines filled in (people correct them)',
  positions: 'blind: the objects to check are marked, the AI’s calls are hidden',
  none: 'nothing from the AI',
}
/** The same, as the person labelling reads it. */
export const AI_VIEW_FOR_LABELLERS: Record<AiView, string> = {
  full: 'The AI’s guesses are filled in — correct the wrong ones',
  positions: 'The AI’s guesses are hidden — decide for yourself',
  none: 'No AI help — mark what you see',
}
export const TOOL_TEXT: Record<string, string> = {
  class: 'correct the class of each object', point: 'add objects the AI missed', outline: 'fix or draw outlines', tag: 'tag the whole field',
}
export const KIND_TEXT = { fields: 'check whole fields', queue: 'one object at a time' } as const

/** A trained model's cross-validated quality in a word, from how many real objects it finds and how many of its calls are right. */
export function quality(m: Metrics | null | undefined): { word: string; tone: 'good' | 'fair' | 'weak' } | null {
  if (!m || !Number.isFinite(m.precision) || !Number.isFinite(m.recall)) return null
  const f1 = m.precision + m.recall ? (2 * m.precision * m.recall) / (m.precision + m.recall) : 0
  if (f1 >= 0.9) return { word: 'Excellent', tone: 'good' }
  if (f1 >= 0.8) return { word: 'Good', tone: 'good' }
  if (f1 >= 0.65) return { word: 'Fair', tone: 'fair' }
  return { word: 'Weak — label more objects', tone: 'weak' }
}
export const percent = (v: number | null | undefined) => (v == null || !Number.isFinite(v) ? '—' : `${Math.round(100 * v)}%`)

/** "Finds 89% of real neurons · 84% of the objects it counts are real", the plain reading of recall and precision. */
export function QualityLine({ m, stain }: { m: Metrics; stain: string }) {
  return <span>finds <b>{percent(m.recall)}</b> of real {objectsOf(stain)} · <b>{percent(m.precision)}</b> of what it counts is right</span>
}

export function QualityBadge({ m }: { m: Metrics | null | undefined }) {
  const q = quality(m)
  return q ? <span className={`badge quality-${q.tone}`}>{q.word}</span> : <span className="badge">not tested</span>
}

/** Asks before an action and says what will happen; the confirm button shows progress while the action runs. */
export function ConfirmDialog({ title, children, confirm, danger = false, onConfirm, onClose }: {
  title: string; children: ReactNode; confirm: string; danger?: boolean; onConfirm: () => Promise<unknown>; onClose: () => void
}) {
  const [busy, setBusy] = useState(false)
  const run = async () => {
    setBusy(true)
    try { await onConfirm() } finally { setBusy(false) }
    onClose()
  }
  return (
    <Modal title={title} onClose={onClose}>
      <div className="secondary">{children}</div>
      <div className="row"><span className="spacer" />
        <button className="btn" onClick={onClose}>Cancel</button>
        <button className={`btn ${danger ? 'danger' : 'primary'}`} disabled={busy} onClick={run} autoFocus>{busy ? 'Working…' : confirm}</button>
      </div>
    </Modal>
  )
}

/** Copies text to the clipboard, also on plain-http addresses where the clipboard API is missing. */
export async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard && window.isSecureContext) { await navigator.clipboard.writeText(text); return true }
  } catch { /* not allowed here: fall back to selecting the text */ }
  const area = document.createElement('textarea')
  area.value = text
  area.style.position = 'fixed'
  area.style.opacity = '0'
  document.body.appendChild(area)
  area.select()
  const copied = document.execCommand('copy')
  area.remove()
  return copied
}

export function CopyButton({ text, label = 'Copy link', primary = false }: { text: string; label?: string; primary?: boolean }) {
  const [result, setResult] = useState<'copied' | 'failed' | null>(null)
  const copy = async () => {
    setResult((await copyText(text)) ? 'copied' : 'failed')
    setTimeout(() => setResult(null), 2500)
  }
  return <button className={`btn small${primary ? ' primary' : ''}`} onClick={copy} title={text}>
    {result === 'copied' ? 'Copied ✓' : result === 'failed' ? 'Select and copy it by hand' : label}</button>
}
