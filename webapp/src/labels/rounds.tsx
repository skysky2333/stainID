/* Pieces shared by the two annotation rounds (labels/checkcalls.tsx, labels/fixfield.tsx): the words, the live progress
   (polled from the server, so what others decide shows within seconds), and switching between the Segment Anything
   masks of an object (smaller / bigger, no waiting). */
import { useEffect, useRef, useState } from 'react'
import type { RoundLive, RoundProgress } from '../api'
import { api } from '../api'
import { Progress } from '../components'
import type { ClassPick, KeyMap, Vocabulary } from './classKeys'
import { classText } from './classKeys'
import { ClassKeyHint, classColour, useClassKeys } from './classes'

export const ROUND_TITLE: Record<1 | 2, string> = { 1: 'Check the AI’s calls', 2: 'Fix the field' }
export const ROUND_TEXT: Record<1 | 2, string> = {
  1: 'One object at a time with the AI’s call and outline: keep it, change it, or say it is not an object.',
  2: 'Whole fields with every object the AI considered: add what it missed, fix outlines, change any call.',
}
/** What someone does on a round page, as the tutorial follows it. */
export type RoundEvent = 'hover' | 'class' | 'subtype' | 'unsure' | 'remove' | 'undo' | 'mask' | 'add' | 'added' | 'precise-in' | 'precise-edit'
  | 'precise-out' | 'finish' | 'keep'
export const LIVE_MS = 3000  // how often a round page asks what others decided meanwhile
export const TIER_TEXT: Record<number, string> = {
  0: 'Shown because the AI is unsure about it (its score is near the cut-off).',
  1: 'Shown because the AI rejected it although it looks strong.',
  2: 'Shown because the AI accepted it although its score is low.',
}
/** Mask sizes as people read them, small to large. */
export const maskName = (i: number, n: number) => (n === 3 ? ['small', 'medium', 'large'][i] : n === 1 ? 'the outline' : `size ${i + 1}`)

/** Asks the server what changed for this page every few seconds (and once at once): `body` says what the page shows. */
export function useLive(session: string, body: Record<string, unknown> | null, onLive: (live: RoundLive) => void) {
  const latest = useRef({ body, onLive })
  latest.current = { body, onLive }
  const [failed, setFailed] = useState(false)
  useEffect(() => {
    if (!latest.current.body) return
    let stopped = false
    const ask = async () => {
      const wanted = latest.current.body
      if (!wanted || document.hidden) return
      try {
        const live = await api.post<RoundLive>(`/api/annotate/${session}/live`, wanted)
        if (!stopped) { latest.current.onLive(live); setFailed(false) }
      } catch { if (!stopped) setFailed(true) }
    }
    void ask()
    const timer = setInterval(ask, LIVE_MS)
    return () => { stopped = true; clearInterval(timer) }
  }, [session, body === null])
  return failed
}

/** "312 of 500 checked · Ann 120 · Ben 80", with a bar; `unit` names what is counted. */
export function RoundProgressLine({ progress, unit, me }: { progress: RoundProgress | null; unit: string; me?: string | null }) {
  if (!progress) return null
  const others = progress.people.slice(0, 4)
  return (
    <div className="round-progress">
      <span className="small secondary"><b>{progress.decided.toLocaleString()}</b> of {progress.total.toLocaleString()} {unit} done
        {others.length > 0 && <span className="muted"> · {others.map((p) => `${p.person === me ? 'you' : p.person} ${p.decided.toLocaleString()}`).join(' · ')}</span>}</span>
      <Progress done={progress.decided} total={progress.total} />
    </div>
  )
}

/** The bigger / smaller switch between an object's masks: the size shown, and Tab / Shift+Tab. */
export function MaskSwitch({ count, index, onChange, disabled = false }: { count: number; index: number; onChange: (i: number) => void; disabled?: boolean }) {
  if (count < 2) return null
  return (
    <div className="mask-switch" role="group" aria-label="Outline size">
      <div className="mask-switch-head small"><b>Outline size</b> <span aria-live="polite">{maskName(index, count)}</span>
        <span className="muted"> · {index + 1} of {count}</span></div>
      <div className="mask-switch-row">
        <button className="btn small" disabled={disabled || index <= 0} onClick={() => onChange(index - 1)} title="Smaller outline (Shift+Tab or −)">
          <span className="kbd">⇧ Tab</span> Smaller</button>
        <span className="mask-dots">
          {Array.from({ length: count }, (_, i) => <button key={i} className={`mask-dot${i === index ? ' on' : ''}`} onClick={() => onChange(i)}
            title={maskName(i, count)} aria-label={maskName(i, count)} aria-pressed={i === index} disabled={disabled} />)}
        </span>
        <button className="btn small" disabled={disabled || index >= count - 1} onClick={() => onChange(index + 1)} title="Bigger outline (Tab or +)">
          Bigger <span className="kbd">Tab</span></button>
      </div>
    </div>
  )
}

/** The next mask for Tab (bigger) or Shift+Tab (smaller), skipping missing ones; the same when there is none that way. */
export function stepMask(masks: (unknown | null)[], index: number, bigger: boolean): number {
  for (let i = index + (bigger ? 1 : -1); i >= 0 && i < masks.length; i += bigger ? 1 : -1) if (masks[i]) return i
  return index
}

/** "plaque · compact" or "not a plaque", as the pages say a call. */
export function callText(vocabulary: Vocabulary, identity: string, subtype: string | null | undefined, typesAsked: boolean) {
  const pick: ClassPick = { identity, subtype: subtype || null }
  return classText(vocabulary, pick, typesAsked)
}

/** The class meaning "unsure" in this vocabulary (its first shown class with the role skip). */
export function unsureClass(vocabulary: Vocabulary): string | undefined {
  return vocabulary.classes.find((c) => c.role === 'skip' && !c.hidden)?.id
}

export const polygonPoints = (points: number[][]) => points.map((p) => p.join(',')).join(' ')

export function polygonArea(points: number[][]) {
  let sum = 0
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) sum += points[j][0] * points[i][1] - points[i][0] * points[j][1]
  return Math.abs(sum) / 2
}

/* ---------- how a candidate looks, by its call (the same in both rounds) ---------- */

export const GREY = '#59606b'  // rejected or removed: dotted grey (on a light halo), never hidden
export const UNSURE_COLOUR = '#e879f9'  // unsure: dotted in its own colour
export const NEW_COLOUR = '#ffffff'  // added, no class yet
export type Look = 'class' | 'removed' | 'unsure' | 'new'

/** How an object with this call is drawn: solid in its class colour; removed (the stain's "not an object") dotted grey;
 *  unsure dotted in the unsure colour; no class yet (just added) dotted white. */
export function lookOf(vocabulary: Vocabulary, removeClass: string, identity: string | null | undefined): { look: Look; colour: string; dash: boolean } {
  if (!identity) return { look: 'new', colour: NEW_COLOUR, dash: true }
  const role = vocabulary.classes.find((c) => c.id === identity)?.role
  if (role === 'skip') return { look: 'unsure', colour: UNSURE_COLOUR, dash: true }
  if (identity === removeClass) return { look: 'removed', colour: GREY, dash: true }
  return { look: 'class', colour: classColour(vocabulary, identity), dash: false }
}

/** A class swatch drawn as its look (solid colour, or a dotted outline for "not an object" and "unsure"). */
export function lookSwatch(vocabulary: Vocabulary, removeClass: string) {
  return (identity: string) => {
    const look = lookOf(vocabulary, removeClass, identity)
    return look.dash ? { background: 'transparent', border: `2px dotted ${look.colour}` } : { background: look.colour }
  }
}

/* ---------- the keys of both rounds, acting on the object under the cursor (round 2) or on screen (round 1) ---------- */

export interface CandidateActions {
  onClass: (pick: ClassPick) => void  // 1–9 and the type keys: this class
  onUnsure: () => void  // 0
  onRemove: () => void  // Delete / Backspace: not an object (undoable)
  onMask: (bigger: boolean) => void  // Tab / + bigger, Shift+Tab / − smaller
  onPrecise: () => void  // Space: precise editing in and out
  onUndo: () => void  // Z
  onEnter?: () => void  // round 1: keep the AI's call; while drawing by hand: close the outline
  onEscape?: () => void
  /** false while class keys should not answer (precise editing, a picture still loading) */
  answering: boolean
}

/** The key handling shared by both rounds; returns the class keys (which class waits for its type). */
export function useCandidateKeys(keys: KeyMap, waitMs: number, actions: CandidateActions) {
  const latest = useRef(actions)
  latest.current = actions
  const classKeys = useClassKeys(keys, waitMs, (pick) => latest.current.onClass(pick))
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const a = latest.current
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement).tagName) || e.metaKey || e.ctrlKey || e.altKey) return
      if (e.key === 'Tab') { e.preventDefault(); a.onMask(!e.shiftKey); return }
      if (e.key === '+' || e.key === '=') { a.onMask(true); return }
      if (e.key === '-' || e.key === '_') { a.onMask(false); return }
      if (e.repeat) return
      if (e.key === ' ') { e.preventDefault(); classKeys.cancel(); a.onPrecise(); return }
      if (e.key === 'Escape') { classKeys.cancel(); a.onEscape?.(); return }
      if (e.key.toLowerCase() === 'z' && !keys.direct.z) { e.preventDefault(); classKeys.cancel(); a.onUndo(); return }
      if (!a.answering) return
      if (e.key === 'Enter' && !classKeys.waiting) { e.preventDefault(); a.onEnter?.(); return }
      if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); classKeys.cancel(); a.onRemove(); return }
      if (classKeys.press(e.key)) { e.preventDefault(); return }
      if (e.key === '0') { e.preventDefault(); a.onUnsure() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [classKeys, keys])
  return classKeys
}

/** What the colours and lines on the field mean, with each class's key. */
export function ClassLegend({ vocabulary, keys, removeClass }: { vocabulary: Vocabulary; keys: KeyMap; removeClass: string }) {
  const keyOf = (id: string) => Object.entries(keys.direct).find(([, [i, s]]) => i === id && s === null)?.[0]
  return (
    <ul className="class-legend small">
      {vocabulary.classes.filter((c) => !c.hidden && c.role !== 'skip' && c.id !== removeClass).map((c) => (
        <li key={c.id}><span className="look-swatch" style={{ borderColor: classColour(vocabulary, c.id), background: classColour(vocabulary, c.id) }} />
          {keyOf(c.id) && <span className="kbd">{keyOf(c.id)}</span>} {c.name}</li>
      ))}
      <li><span className="look-swatch look-removed" style={{ borderColor: GREY }} /><span className="kbd">Del</span> not an object, or left out by the AI</li>
      <li><span className="look-swatch look-unsure" style={{ borderColor: UNSURE_COLOUR }} /><span className="kbd">0</span> unsure</li>
      <li><span className="look-swatch look-new" style={{ borderColor: '#888' }} /> new, no class yet</li>
    </ul>
  )
}

/** The keys in one compact line, always on screen. */
export function KeyLegend({ round, vocabulary, keys, typesAsked, waitMs, sam }: {
  round: 1 | 2; vocabulary: Vocabulary; keys: KeyMap; typesAsked: boolean; waitMs: number; sam: boolean
}) {
  return (
    <div className="key-help key-legend small secondary" aria-label="Keys">
      {round === 2 && <span>Point at an object: the keys act on it</span>}
      {round === 1 && <span><span className="kbd">Enter</span> keep the AI’s call</span>}
      <ClassKeyHint vocabulary={vocabulary} keys={keys} typesAsked={typesAsked} waitMs={waitMs} />
      <span><span className="kbd">0</span> unsure</span>
      <span><span className="kbd">Del</span> not an object</span>
      <span><span className="kbd">Tab</span> / <span className="kbd">⇧ Tab</span> bigger / smaller outline</span>
      {sam && <span><span className="kbd">Space</span> precise edit</span>}
      <span><span className="kbd">Z</span> undo</span>
      {round === 2 && sam && <span>Click empty tissue: add a missed object</span>}
      <span><span className="kbd">Esc</span> cancel</span>
    </div>
  )
}
