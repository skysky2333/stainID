/* Choosing a label class on the labelling pages: the stain's keys in their groups (the server's
   `vocabulary.assign_keys`): the counted types (1–4), the named look-alikes (Q W E R), the reject (Del) and "unsure"
   (`). One key, or one click, picks a type at once. */
import { useCallback, useMemo, useRef } from 'react'
import type { CSSProperties } from 'react'
import type { ClassDef, ClassPick, KeyMap, Vocabulary } from './classKeys'
import { keyGroups, keySummary, pressKey } from './classKeys'
import { classExample } from '../api'
import { classColour } from '../classPalette'

export { classColour }

/** The class keys of a labelling page: a key press or a click answers (`onAnswer`) at once. */
export function useClassKeys(keys: KeyMap, onAnswer: (pick: ClassPick) => void) {
  const answerRef = useRef(onAnswer)
  answerRef.current = onAnswer

  /** A key press; true when it meant something here. */
  const press = useCallback((key: string) => {
    const pick = pressKey(keys, key)
    if (!pick) return false
    answerRef.current(pick)
    return true
  }, [keys])

  /** A class (and type) clicked. */
  const pick = useCallback((identity: string, subtype: string | null = null) => answerRef.current({ identity, subtype }), [])
  return useMemo(() => ({ press, pick }), [press, pick])
}

/** A swatch as a pick is drawn: solid in its colour, dotted grey for the reject, dotted amber for "unsure". */
export function pickSwatch(vocabulary: Vocabulary, identity: string, subtype: string | null = null): CSSProperties {
  const colour = classColour(vocabulary, identity, subtype)
  return identity === vocabulary.reject || identity === vocabulary.unsure ? { background: 'transparent', border: `2px dotted ${colour}` } : { background: colour }
}

/** The keys as buttons in their groups ("Plaques 1–4", "Not plaques Q–W", Del, `). `counts`: how many objects of each
 *  class (by identity) are on the page. */
export function ClassChooser({ vocabulary, keys, active, counts, big = false, swatch, onPick }: {
  vocabulary: Vocabulary; keys: KeyMap; active: ClassPick | null; counts?: Record<string, number>; big?: boolean
  swatch?: (identity: string, subtype: string | null) => CSSProperties; onPick: (identity: string, subtype: string | null) => void
}) {
  const described = vocabulary.classes.find((c) => c.id === active?.identity)
  return (
    <div className="class-chooser">
      {keyGroups(vocabulary, keys).map((g) => (
        <div key={g.group} className="class-group" role="group" aria-label={g.title}>
          <div className="class-group-title small muted">{g.title} <span className="kbd-text">{g.keys}</span></div>
          <div className="class-list">
            {g.items.map((item) => {
              const chosen = active?.identity === item.identity && (active.subtype ?? null) === item.subtype
              return (
                <button key={item.key} className={`btn class-btn${big ? ' big' : ''}${chosen ? ' chosen' : ''}`} aria-pressed={chosen} title={item.description}
                  onClick={() => onPick(item.identity, item.subtype)}>
                  <span className="kbd">{item.label}</span>
                  <span className="swatch" style={swatch ? swatch(item.identity, item.subtype) : pickSwatch(vocabulary, item.identity, item.subtype)} />
                  <span className="class-name">{item.name}</span><span className="spacer" />
                  {counts && <span className="muted small">{counts[item.identity] ?? ''}</span>}
                </button>
              )
            })}
          </div>
        </div>
      ))}
      {described && <ClassHelp vocabulary={vocabulary} identity={described} typeId={active?.subtype ?? undefined} />}
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

/** The keys in one line, for under the picture: "Plaques 1–4 · Not plaques Q–W · Del not a plaque · ` unsure · B brush"
 *  (`brush` false on pages without the brush). */
export function ClassKeyHint({ vocabulary, keys, brush = false }: { vocabulary: Vocabulary; keys: KeyMap; brush?: boolean }) {
  return <span className="class-key-hint">{keySummary(vocabulary, keys, brush)}</span>
}
