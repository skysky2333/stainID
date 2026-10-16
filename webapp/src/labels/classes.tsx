/* Choosing a label class on the labelling pages: the stain's classes with their keys, and under the class being chosen
   its types (subtypes). Keys follow labels/classKeys.ts: a class key, then within a moment a number for the type (by
   its place), or a type's own key for both at once. With the mouse: click the class, then its type (or “No type”). */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import type { ClassDef, ClassPick, KeyMap, Vocabulary, Waiting } from './classKeys'
import { expire, keyOf, pressKey } from './classKeys'
import { classExample } from '../api'

/** The class keys of a labelling page: what was answered (`onAnswer`, in order) and which class waits for its type.
 *  `waitMs` 0: a class with types waits until its type (or no type) is chosen. */
export function useClassKeys(keys: KeyMap, waitMs: number, onAnswer: (pick: ClassPick) => void) {
  const [waiting, setWaitingState] = useState<Waiting | null>(null)
  const waitingRef = useRef<Waiting | null>(null)
  const answerRef = useRef(onAnswer)
  answerRef.current = onAnswer
  const setWaiting = (next: Waiting | null) => { waitingRef.current = next; setWaitingState(next) }

  useEffect(() => {  // the time for the type is up: the class stands without one
    if (!waiting || waiting.sticky || waitMs <= 0) return
    const timer = setTimeout(() => {
      const answer = expire(waitingRef.current, performance.now(), waitMs)
      if (answer) { setWaiting(null); answerRef.current(answer) }
    }, Math.max(0, waitMs - (performance.now() - waiting.at)) + 10)
    return () => clearTimeout(timer)
  }, [waiting, waitMs])

  /** A key press; true when it meant something here. */
  const press = useCallback((key: string) => {
    const result = pressKey(keys, waitingRef.current, key, performance.now(), waitMs)
    if (!result.handled) return false
    setWaiting(result.waiting)
    result.answers.forEach((a) => answerRef.current(a))
    return true
  }, [keys, waitMs])

  /** A class clicked: answered at once, or waiting (without a time limit) for its type to be clicked. */
  const pick = useCallback((identity: string) => {
    if (keys.then[identity]) setWaiting({ identity, at: performance.now(), sticky: true })
    else { setWaiting(null); answerRef.current({ identity, subtype: null }) }
  }, [keys])

  /** A type (or no type: null) clicked for the waiting class. */
  const pickType = useCallback((subtype: string | null) => {
    const current = waitingRef.current
    if (!current) return
    setWaiting(null)
    answerRef.current({ identity: current.identity, subtype })
  }, [])

  const cancel = useCallback(() => setWaiting(null), [])
  return useMemo(() => ({ waiting, press, pick, pickType, cancel }), [waiting, press, pick, pickType, cancel])
}

/** One colour per class, from its place among the stain's classes (positive classes first), so it keeps its colour. */
export function classColour(vocabulary: Vocabulary, identity: string | undefined): string {
  const order = [...vocabulary.classes].sort((a, b) => Number(a.role !== 'positive') - Number(b.role !== 'positive'))
  const i = order.findIndex((c) => c.id === identity)
  return i < 0 ? 'var(--text-muted)' : `var(--series-${(i % 8) + 1})`
}

const shownTypes = (c: ClassDef, typesAsked: boolean) => (typesAsked ? c.subtypes.filter((s) => !s.hidden) : [])

/** The classes as buttons with their keys; under the class being chosen (`active`) its types. `counts`: how many objects
 *  of each class (by identity) are on the page. */
export function ClassChooser({ vocabulary, keys, typesAsked, active, waiting, waitMs, counts, big = false, swatch, onPick, onPickType }: {
  vocabulary: Vocabulary; keys: KeyMap; typesAsked: boolean; active: ClassPick | null; waiting: Waiting | null; waitMs: number
  counts?: Record<string, number>; big?: boolean; swatch?: (identity: string) => CSSProperties; onPick: (identity: string) => void; onPickType: (subtype: string | null) => void
}) {
  const open = waiting?.identity ?? active?.identity
  const described = vocabulary.classes.find((c) => c.id === open)
  const type = described?.subtypes.find((s) => s.id === active?.subtype && active?.identity === described.id)
  return (
    <div className="class-chooser">
      <div className="class-list">
        {vocabulary.classes.filter((c) => !c.hidden).map((c) => {
          const types = shownTypes(c, typesAsked)
          const chosen = c.id === open
          return (
            <div key={c.id} className={`class-row${chosen ? ' open' : ''}`}>
              <button className={`btn class-btn${big ? ' big' : ''}${chosen ? ' chosen' : ''}`} aria-pressed={chosen} title={c.description}
                onClick={() => onPick(c.id)}>
                <span className="kbd">{keyOf(keys, c.id) ?? '·'}</span><span className="swatch" style={swatch ? swatch(c.id) : { background: classColour(vocabulary, c.id) }} />
                <span className="class-name">{c.name}</span><span className="spacer" />
                {types.length > 0 && <span className="muted small">{types.length} types</span>}
                {counts && <span className="muted small">{counts[c.id] ?? ''}</span>}
              </button>
              {chosen && types.length > 0 && (
                <div className={`type-list${waiting ? ' asking' : ''}`} role="group" aria-label={`Type of ${c.name}`}>
                  {waiting && !waiting.sticky && waitMs > 0 && <span key={waiting.at} className="type-timer" style={{ animationDuration: `${waitMs}ms` }} />}
                  {types.map((s, i) => {
                    const own = keyOf(keys, c.id, s.id)
                    const on = active?.identity === c.id && active.subtype === s.id && !waiting
                    return (
                      <button key={s.id} className={`btn small type-btn${on ? ' chosen' : ''}`} aria-pressed={on} title={s.description}
                        onClick={() => onPickType(s.id)}>
                        <span className="kbd">{i + 1}</span>{s.name}{own && <span className="kbd muted-kbd" title={`or ${own.toUpperCase()} at once`}>{own.toUpperCase()}</span>}
                      </button>
                    )
                  })}
                  <button className={`btn small type-btn${active?.identity === c.id && !active.subtype && !waiting ? ' chosen' : ''}`} onClick={() => onPickType(null)}
                    title="The type is not clear, or you are not sure">
                    <span className="kbd">0</span>No type / not sure</button>
                </div>
              )}
            </div>
          )
        })}
      </div>
      {described && <ClassHelp vocabulary={vocabulary} identity={described} typeId={type?.id} />}
    </div>
  )
}

/** What the class (and its type) looks like: the plain description and the example picture, if there is one. */
export function ClassHelp({ vocabulary, identity, typeId }: { vocabulary: Vocabulary; identity: ClassDef; typeId?: string }) {
  const type = identity.subtypes.find((s) => s.id === typeId)
  const example = type?.example || identity.example
  return (
    <div className="class-help small">
      {example && <img src={classExample(vocabulary.stain, example)} alt={`Example: ${type?.name ?? identity.name}`} />}
      <span><b>{identity.name}</b>: {identity.description}{type && <><br /><b>{type.name}</b>: {type.description}</>}</span>
    </div>
  )
}

/** The keys in one line, for under the picture: "1–7 class, then 1–3 type (or Q W E at once), 0 no type". */
export function ClassKeyHint({ vocabulary, keys, typesAsked, waitMs }: { vocabulary: Vocabulary; keys: KeyMap; typesAsked: boolean; waitMs: number }) {
  const numbers = vocabulary.classes.filter((c) => !c.hidden).map((c) => keyOf(keys, c.id)).filter((k): k is string => !!k)
  const direct = Object.entries(keys.direct).filter(([, [, s]]) => s !== null).map(([k]) => k.toUpperCase())
  const most = Math.max(0, ...Object.values(keys.then).map((t) => Object.keys(t).length))
  return (
    <span>
      <span className="kbd">{numbers[0]}</span>{numbers.length > 1 && <>–<span className="kbd">{numbers[numbers.length - 1]}</span></>} class
      {typesAsked && most > 0 && <>, then <span className="kbd">1</span>–<span className="kbd">{most}</span> type
        {waitMs > 0 ? ` within ${(waitMs / 1000).toLocaleString()} s` : ''}
        {direct.length > 0 && <> (or {direct.map((k) => <span key={k} className="kbd">{k}</span>)} at once)</>}, <span className="kbd">0</span> no type</>}
    </span>
  )
}
