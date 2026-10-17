/* The label classes on the labelling pages: identity classes with their types (subtypes), how stored labels read in
   them, and the keys. A class key answers at once when its class has no types to ask; otherwise the class waits a
   moment (`subtype_wait_s`) for a number key that picks its type by place (1 = the first type), and stands without
   a type when none follows (0 or Enter: no type at once). A type's own key answers class and type in one press.
   No imports: tests/js/class_keys.test.ts runs this file in Node. */

export type ClassRole = 'positive' | 'negative' | 'skip'
export interface SubtypeDef { id: string; name: string; description: string; key: string; example: string; hidden: boolean; builtin?: boolean }
export interface ClassDef {
  id: string; name: string; role: ClassRole; description: string; key: string; example: string; hidden: boolean; builtin?: boolean
  subtypes: SubtypeDef[]
}
/** A stain's classes, with how the values of labels made before types existed map onto them (value -> [identity, type]). */
export interface Vocabulary { stain: string; subtype_wait_s: number; classes: ClassDef[]; aliases: Record<string, [string, string]>; untyped: string }
/** `direct`: key -> [identity, type or null]; `then`: identity -> {"1": type, …}, the types picked by place after its key. */
export interface KeyMap { direct: Record<string, [string, string | null]>; then: Record<string, Record<string, string>> }
export interface ClassPick { identity: string; subtype: string | null }
/** A class key pressed whose type may follow (`sticky`: chosen with the mouse, it waits until a type is chosen). */
export interface Waiting { identity: string; at: number; sticky?: boolean }
export interface KeyResult { answers: ClassPick[]; waiting: Waiting | null; handled: boolean }

const live = (waiting: Waiting | null, now: number, waitMs: number) =>
  !!waiting && (waiting.sticky || waitMs <= 0 || now - waiting.at <= waitMs)

/** What a key press means, given the class waiting for its type (if any): the answers it gives (a waiting class whose
 *  time is up counts first, without a type) and the class that now waits. Keys that mean nothing here are not handled. */
export function pressKey(keys: KeyMap, waiting: Waiting | null, key: string, now: number, waitMs: number): KeyResult {
  const k = key.length === 1 ? key.toLowerCase() : key
  const answers: ClassPick[] = []
  let still = waiting
  if (waiting && !live(waiting, now, waitMs)) { answers.push({ identity: waiting.identity, subtype: null }); still = null }
  if (still) {
    const subtype = keys.then[still.identity]?.[k]
    if (subtype) return { answers: [{ identity: still.identity, subtype }], waiting: null, handled: true }
    if (k === '0' || k === 'Enter') return { answers: [{ identity: still.identity, subtype: null }], waiting: null, handled: true }
  }
  const direct = keys.direct[k]
  if (!direct) return { answers, waiting: still, handled: answers.length > 0 }
  if (still) answers.push({ identity: still.identity, subtype: null })  // another class key: the waiting class stands without a type
  const [identity, subtype] = direct
  if (subtype === null && keys.then[identity]) return { answers, waiting: { identity, at: now }, handled: true }
  return { answers: [...answers, { identity, subtype }], waiting: null, handled: true }
}

/** The answer of a waiting class whose time is up (none while it may still get its type). */
export function expire(waiting: Waiting | null, now: number, waitMs: number): ClassPick | null {
  return waiting && !live(waiting, now, waitMs) ? { identity: waiting.identity, subtype: null } : null
}

/** A stored label (or a detection call) as identity and type ids: labels made before types existed through the aliases,
 *  "identity / type" by id or name, else as it is. */
export function resolveClass(vocabulary: Vocabulary, value: string | null | undefined, subtype?: string | null): ClassPick {
  const v = (value ?? '').trim()
  if (!v) return { identity: '', subtype: null }
  const byRef = (ref: string) => vocabulary.classes.find((c) => c.id === ref) ?? vocabulary.classes.find((c) => c.name.toLowerCase() === ref.toLowerCase())
  const typeOf = (c: ClassDef, ref: string) => c.subtypes.find((s) => s.id === ref) ?? c.subtypes.find((s) => s.name.toLowerCase() === ref.toLowerCase())
  let found = byRef(v)
  let sub = subtype || null
  if (!found && vocabulary.aliases[v]) {
    const [identity, legacy] = vocabulary.aliases[v]
    return { identity, subtype: sub ?? (legacy || null) }
  }
  if (!found) {
    const parts = v.split(/\s*[/:]\s*/)
    const head = parts.length > 1 ? byRef(parts[0]) : undefined
    if (!head) return { identity: v, subtype: sub }
    found = head
    sub = sub ?? parts.slice(1).join(' / ')
  }
  return { identity: found.id, subtype: sub ? (typeOf(found, sub)?.id ?? sub) : null }
}

export function classOf(vocabulary: Vocabulary, identity: string): ClassDef | undefined {
  return vocabulary.classes.find((c) => c.id === identity)
}

export function roleOf(vocabulary: Vocabulary, identity: string): ClassRole {
  if (!identity) return 'skip'
  return classOf(vocabulary, identity)?.role ?? 'negative'
}

/** "tau+ neuron · pretangle", "plaque (type not recorded)" (a positive class whose type was not given), "not tau". */
export function classText(vocabulary: Vocabulary, pick: ClassPick, typesAsked = true): string {
  const found = classOf(vocabulary, pick.identity)
  if (!found) return pick.identity
  if (pick.subtype && typesAsked) return `${found.name} · ${found.subtypes.find((s) => s.id === pick.subtype)?.name ?? pick.subtype}`
  return found.subtypes.length && found.role === 'positive' && typesAsked ? `${found.name} (${vocabulary.untyped})` : found.name
}

/** The key that answers this class (and type) at once, if there is one. */
export function keyOf(keys: KeyMap, identity: string, subtype: string | null = null): string | undefined {
  return Object.entries(keys.direct).find(([, [i, s]]) => i === identity && s === subtype)?.[0]
}
