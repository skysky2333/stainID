/* Pieces shared by the two annotation rounds (labels/checkcalls.tsx, labels/fixfield.tsx): the words, the heartbeat
   (polled from the server: the field on the page stays this person's, and the live progress in fields), switching
   between the Segment Anything masks of an object (smaller / bigger, no waiting), and unsaved changes: both pages keep
   every change on the page until the person presses Save (a draft in this browser survives a reload, never sent before),
   ask before leaving with unsaved changes, and save a field's changes all at once. */
import { useCallback, useEffect, useRef, useState } from 'react'
import type { RoundLive, RoundProgress } from '../api'
import { api } from '../api'
import { Progress } from '../components'
import type { ClassPick, KeyMap, Vocabulary } from './classKeys'
import { classText } from './classKeys'
import { ClassKeyHint, useClassKeys } from './classes'
import { CLASS_COLOURS, INACTIVE_GREY, classColour } from '../classPalette'

export const ROUND_TITLE: Record<1 | 2, string> = { 1: 'Check detection’s calls', 2: 'Fix the field' }
export const ROUND_TEXT: Record<1 | 2, string> = {
  1: 'One object at a time, with detection’s call and outline: keep it, change it, or say it is not an object.',
  2: 'Whole fields with every object detection considered: point at one and press a key to change its call, fix its outline, add what it missed.',
}
/** What someone does on a round page, as the tutorial follows it. */
export type RoundEvent = 'hover' | 'class' | 'subtype' | 'unsure' | 'remove' | 'undo' | 'mask' | 'add' | 'added' | 'precise-in' | 'precise-edit'
  | 'precise-out' | 'save' | 'finish' | 'keep'
export const LIVE_MS = 3000  // how often a round page renews its hold on the field and asks for the progress
export const TIER_TEXT: Record<number, string> = {
  0: 'Shown because detection is unsure about it (its score is near the cut-off).',
  1: 'Shown because detection rejected it although it looks strong.',
  2: 'Shown because detection accepted it although its score is low.',
}
/** What each of a candidate's outlines is, in the server's order: Segment Anything's three (small to large), then the
 *  stain-based outline when there are four; one alone is just "the outline". */
export function shapeNames(n: number): string[] {
  if (n === 1) return ['the outline']
  const sam = Math.min(n, 3)
  return Array.from({ length: n }, (_, i) => (i < sam ? `SAM ${i + 1}/${sam}` : 'stain outline'))
}
/** An outline as people read it ("stain outline", "SAM 2/3"). */
export const maskName = (i: number, n: number, names?: string[]) => names?.[i] ?? shapeNames(n)[i] ?? `outline ${i + 1}`

function area(points: number[][]) {
  let sum = 0
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) sum += points[j][0] * points[i][1] - points[i][0] * points[j][1]
  return Math.abs(sum) / 2
}

/** The outlines' indices from the smallest to the largest (the missing ones left out): the order the bigger / smaller
 *  keys and the dots go through, whatever order the server keeps them in. */
export function bySize(masks: (number[][] | null)[]): number[] {
  return masks.map((m, i) => [i, m ? area(m) : -1] as const).filter(([, a]) => a >= 0).sort((a, b) => a[1] - b[1]).map(([i]) => i)
}

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

/** "3 of 12 fields done · 2 in progress · 7 left · you 2 · Ben 1", with a bar: everyone's work, live. */
export function RoundProgressLine({ progress, me }: { progress: RoundProgress | null; me?: string | null }) {
  if (!progress) return null
  const others = progress.people.slice(0, 4)
  const open = progress.open_now ?? 0
  return (
    <div className="round-progress">
      <span className="small secondary"><b>{progress.decided.toLocaleString()}</b> of {progress.total.toLocaleString()} field{progress.total === 1 ? '' : 's'} done
        {open > 0 && <> · {open.toLocaleString()} in progress</>}
        {progress.left != null && progress.left > 0 && <> · {progress.left.toLocaleString()} left</>}
        {others.length > 0 && <span className="muted"> · {others.map((p) => `${p.person === me ? 'you' : p.person} ${p.decided.toLocaleString()}`).join(' · ')}</span>}</span>
      <Progress done={progress.decided} total={progress.total} />
    </div>
  )
}

/* ---------- unsaved changes ---------- */

/** Where a page keeps one person's unsaved changes on one field in this browser (so a reload or a crash loses nothing). */
export const draftKey = (person: string | null | undefined, session: string, tile: string) => `stainid.draft.${person ?? ''}.${session}.${tile}`

export function readDraft<T>(key: string): T | null {
  try {
    const raw = localStorage.getItem(key)
    return raw ? (JSON.parse(raw) as T) : null
  } catch {
    return null
  }
}

/** Keeps (or with null, forgets) a draft; never sent anywhere. */
export function writeDraft(key: string, value: unknown) {
  try {
    if (value == null) localStorage.removeItem(key)
    else localStorage.setItem(key, JSON.stringify(value))
  } catch { /* storage unavailable: the changes stay on the page only */ }
}

export const changesText = (n: number) => `${n.toLocaleString()} unsaved change${n === 1 ? '' : 's'}`
const leaveText = (n: number) => `You have ${changesText(n)} on this field.\n\nLeave without saving? They stay in this browser and come back when you open this field again, but nobody else sees them until you press Save.`

/** Asks before leaving while there are unsaved changes: closing or reloading the tab, or following a link of the app.
 *  Returns `mayLeave()` for leaving by a button (true when there is nothing to lose or the person agrees). */
export function useLeaveGuard(unsaved: number) {
  useEffect(() => {
    if (!unsaved) return
    const unload = (e: BeforeUnloadEvent) => { e.preventDefault(); e.returnValue = '' }
    const click = (e: MouseEvent) => {
      const link = (e.target as HTMLElement | null)?.closest?.('a[href]') as HTMLAnchorElement | null
      if (!link || link.target === '_blank' || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return
      if (!window.confirm(leaveText(unsaved))) { e.preventDefault(); e.stopPropagation() }
    }
    window.addEventListener('beforeunload', unload)
    document.addEventListener('click', click, true)
    return () => { window.removeEventListener('beforeunload', unload); document.removeEventListener('click', click, true) }
  }, [unsaved])
  return useCallback(() => !unsaved || window.confirm(leaveText(unsaved)), [unsaved])
}

/** Ctrl+S (⌘S) saves, Ctrl+Enter (⌘Enter) saves and goes on to the next field. */
export function useSaveKeys(onSave: () => void, onContinue: () => void) {
  const latest = useRef({ onSave, onContinue })
  latest.current = { onSave, onContinue }
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey) || e.altKey || e.repeat) return
      if (e.key.toLowerCase() === 's') { e.preventDefault(); latest.current.onSave() }
      else if (e.key === 'Enter') { e.preventDefault(); latest.current.onContinue() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
}

/** "3 unsaved changes" (marked), "Saving…" or "All changes saved". */
export function SaveState({ unsaved, saving }: { unsaved: number; saving: boolean }) {
  return (
    <span className={`small save-state${unsaved && !saving ? ' unsaved' : ' muted'}`} aria-live="polite" data-unsaved={unsaved}>
      {saving ? 'Saving…' : unsaved ? <><span className="unsaved-dot" aria-hidden="true" />{changesText(unsaved)}</> : 'All changes saved'}
    </span>
  )
}

/** The two save buttons of a field: Save (stays) and Save and continue (finishes the field; the filled button unless
 *  another action is the page's main one). */
export function SaveButtons({ unsaved, saving, onSave, onContinue, canContinue, continueTitle, primary = true }: {
  unsaved: number; saving: boolean; onSave: () => void; onContinue: () => void; canContinue: boolean; continueTitle: string; primary?: boolean
}) {
  return <>
    <button className="btn" onClick={onSave} disabled={saving || !unsaved} title="Save your changes and stay on this field (Ctrl+S)">Save</button>
    <button className={`btn${primary ? ' primary big' : ''}`} onClick={onContinue} disabled={saving || !canContinue} title={continueTitle}>Save and continue →</button>
  </>
}

/** The bigger / smaller switch between an object's masks: the size shown, and Tab / Shift+Tab. */
export function MaskSwitch({ masks, index, onChange, disabled = false, names }: {
  masks: (number[][] | null)[]; index: number; onChange: (i: number) => void; disabled?: boolean; names?: string[]
}) {
  const order = bySize(masks)
  const count = masks.length
  const rank = order.indexOf(index)
  if (order.length < 2) return null
  return (
    <div className="mask-switch" role="group" aria-label="Outline size">
      <div className="mask-switch-head small"><b>Outline</b> <span aria-live="polite">{maskName(index, count, names)}</span>
        <span className="muted"> · {rank + 1} of {order.length} by size</span></div>
      <div className="mask-switch-row">
        <button className="btn small" disabled={disabled || rank <= 0} onClick={() => onChange(order[rank - 1])} title="Smaller outline (Shift+Tab or −)">
          <span className="kbd">⇧ Tab</span> Smaller</button>
        <span className="mask-dots">
          {order.map((i) => <button key={i} className={`mask-dot${i === index ? ' on' : ''}`} onClick={() => onChange(i)}
            title={maskName(i, count, names)} aria-label={maskName(i, count, names)} aria-pressed={i === index} disabled={disabled} />)}
        </span>
        <button className="btn small" disabled={disabled || rank < 0 || rank >= order.length - 1} onClick={() => onChange(order[rank + 1])} title="Bigger outline (Tab or +)">
          Bigger <span className="kbd">Tab</span></button>
      </div>
    </div>
  )
}

/** The next outline for Tab (bigger) or Shift+Tab (smaller), by size (`bySize`); the same when there is none that way. */
export function stepMask(masks: (number[][] | null)[], index: number, bigger: boolean): number {
  const order = bySize(masks)
  const rank = order.indexOf(index)
  if (rank < 0) return order[0] ?? index
  const next = order[rank + (bigger ? 1 : -1)]
  return next ?? index
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

export const GREY = INACTIVE_GREY  // not active (rejected, removed, or new without a class yet): dotted light grey on a dark halo, never hidden
/** A class's colour, on the tissue as on every page (classPalette.ts). */
export const tissueColour = classColour

/** The annotation pages take the window exactly (no page scrollbar, which would fight panning the picture). */
export function useFitScreen() {
  useEffect(() => {
    document.documentElement.classList.add('fit-screen')
    return () => document.documentElement.classList.remove('fit-screen')
  }, [])
}
export type Look = 'class' | 'removed' | 'unsure' | 'new'

/** How an object with this call is drawn. Dotted is grey and means "not active": rejected by detection or removed by a
 *  person (the stain's "not an object"), or just added and waiting for its class. Every chosen category is solid in its
 *  own colour: each class its class colour, unsure (0) its own. */
export function lookOf(vocabulary: Vocabulary, removeClass: string, identity: string | null | undefined): { look: Look; colour: string; dash: boolean } {
  if (!identity) return { look: 'new', colour: GREY, dash: true }
  const role = vocabulary.classes.find((c) => c.id === identity)?.role
  if (role === 'skip') return { look: 'unsure', colour: classColour(vocabulary, identity), dash: false }
  if (identity === removeClass) return { look: 'removed', colour: GREY, dash: true }
  return { look: 'class', colour: tissueColour(vocabulary, identity), dash: false }
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
  onEnter?: () => void  // round 1: keep detection's call; while drawing by hand: close the outline
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

/** The class keys with their colours: 1–9 (each class its own colour, solid), then 0 unsure. While a class with types
 *  waits for its type, the type keys show under it. */
export function ClassLegend({ vocabulary, keys, removeClass, waiting }: { vocabulary: Vocabulary; keys: KeyMap; removeClass: string; waiting?: string | null }) {
  const keyOf = (id: string) => Object.entries(keys.direct).find(([, [i, s]]) => i === id && s === null)?.[0]
  return (
    <ul className="class-legend small">
      {vocabulary.classes.filter((c) => !c.hidden && c.role !== 'skip' && c.id !== removeClass).map((c) => (
        <li key={c.id} className={waiting === c.id ? 'waiting' : ''} title={c.description}>
          <span className="kbd key-slot">{keyOf(c.id) ?? '·'}</span>
          <span className="look-swatch" style={{ borderColor: tissueColour(vocabulary, c.id), background: tissueColour(vocabulary, c.id) }} />
          <span className="ellipsis">{c.name}</span>
          {waiting === c.id && <span className="muted"> — now its type: {Object.entries(keys.then[c.id] ?? {}).map(([k, t]) =>
            `${k} ${c.subtypes.find((s) => s.id === t)?.name ?? t}`).join(' · ')} · 0 none</span>}
        </li>
      ))}
      {vocabulary.classes.filter((c) => c.id === removeClass).map((c) => (
        <li key={c.id} title={c.description}><span className="kbd key-slot">{keyOf(c.id) ?? 'Del'}</span>
          <span className="look-swatch look-removed" style={{ borderColor: GREY }} /><span className="ellipsis">{c.name} — dotted grey{keyOf(c.id) ? ' (or Del)' : ''}</span></li>
      ))}
      {vocabulary.classes.filter((c) => c.role === 'skip' && !c.hidden).slice(0, 1).map((c) => (
        <li key={c.id} title={c.description}><span className="kbd key-slot">0</span>
          <span className="look-swatch look-unsure" style={{ borderColor: classColour(vocabulary, c.id), background: classColour(vocabulary, c.id) }} />{c.name}</li>
      ))}
    </ul>
  )
}

/** What the lines on the field mean, in two lines. */
export function LookLegend() {
  return (
    <ul className="look-legend small secondary">
      <li><span className="look-swatch" style={{ borderColor: CLASS_COLOURS[0], background: CLASS_COLOURS[0] }} />solid, in the class colour: counts as that class</li>
      <li><span className="look-swatch look-removed" style={{ borderColor: GREY }} />dotted grey: rejected by detection or removed — a class key brings it back</li>
      <li><span className="look-swatch look-new" style={{ borderColor: GREY }} /><span className="look-diamond" aria-hidden="true" />dotted grey with a diamond: new, choose its class</li>
    </ul>
  )
}

/** A polygon with at most `most` corners (Douglas–Peucker with a growing tolerance), so every corner can be dragged. */
export function simplifyPolygon(points: number[][], most = 48): number[][] {
  if (points.length <= most) return points
  const keep = (tolerance: number) => {
    const marked = new Array(points.length).fill(false)
    marked[0] = marked[points.length - 1] = true
    const stack: [number, number][] = [[0, points.length - 1]]
    while (stack.length) {
      const [a, b] = stack.pop()!
      const [ax, ay] = points[a], [bx, by] = points[b]
      const length = Math.hypot(bx - ax, by - ay) || 1
      let far = -1, worst = tolerance
      for (let i = a + 1; i < b; i++) {
        const d = Math.abs((bx - ax) * (ay - points[i][1]) - (ax - points[i][0]) * (by - ay)) / length
        if (d > worst) { worst = d; far = i }
      }
      if (far >= 0) { marked[far] = true; stack.push([a, far], [far, b]) }
    }
    return points.filter((_, i) => marked[i])
  }
  let tolerance = 0.5
  let out = keep(tolerance)
  while (out.length > most && tolerance < 1e4) { tolerance *= 1.5; out = keep(tolerance) }
  return out.length >= 3 ? out : points
}

/** The keys in one compact line, always on screen. */
export function KeyLegend({ round, vocabulary, keys, typesAsked, waitMs, sam }: {
  round: 1 | 2; vocabulary: Vocabulary; keys: KeyMap; typesAsked: boolean; waitMs: number; sam: boolean
}) {
  return (
    <div className="key-help key-legend small secondary" aria-label="Keys">
      {round === 2 && <span>Point at an object: the keys act on it</span>}
      {round === 1 && <span><span className="kbd">Enter</span> keep detection’s call</span>}
      <ClassKeyHint vocabulary={vocabulary} keys={keys} typesAsked={typesAsked} waitMs={waitMs} />
      <span><span className="kbd">0</span> unsure</span>
      <span><span className="kbd">Del</span> not an object</span>
      <span><span className="kbd">Tab</span> / <span className="kbd">⇧ Tab</span> bigger / smaller outline</span>
      {sam && <span><span className="kbd">Space</span> precise edit</span>}
      <span><span className="kbd">Z</span> undo (unsaved changes)</span>
      {round === 2 && sam && <span>Click empty tissue: add a missed object</span>}
      <span><span className="kbd">Esc</span> cancel</span>
      <span><span className="kbd">Ctrl S</span> save</span>
      <span><span className="kbd">Ctrl Enter</span> save and continue</span>
    </div>
  )
}
