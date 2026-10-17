/* The label classes on the labelling pages: identity classes with their types (subtypes), how stored labels read in
   them, and the keys. One key picks a type at once (the server's `vocabulary.assign_keys`, sent as the session's
   `keys`): numbers the counted types (1–4 built in, 5–9 added), letters the named look-alikes (Q W E R, then T Y U …),
   Del the single reject (dotted grey), ` "unsure" (dotted amber; 0 works too). B is the brush on the field pages.
   No imports: tests/js/class_keys.test.ts runs this file in Node. */

export type ClassRole = 'positive' | 'negative' | 'skip'
export interface SubtypeDef { id: string; name: string; description: string; key: string; example: string; hidden: boolean; builtin?: boolean }
export interface ClassDef {
  id: string; name: string; role: ClassRole; description: string; key: string; example: string; hidden: boolean; builtin?: boolean
  subtypes: SubtypeDef[]
}
/** A stain's classes, with how the values of labels made before types existed map onto them (value -> [identity, type]),
 *  the reject and "unsure" class ids and each class's colour slot (by "identity" and "identity/type"). */
export interface Vocabulary {
  stain: string; subtype_wait_s: number; classes: ClassDef[]; aliases: Record<string, [string, string]>; untyped: string
  reject?: string; unsure?: string; slots?: Record<string, number>
}
export type KeyGroupKind = 'counted' | 'lookalike' | 'reject' | 'unsure'
export interface KeyItem { key: string; label: string; identity: string; subtype: string | null; name: string; description: string }
/** One group of the panel: "Plaques" 1–4, "Not plaques" Q–W, "not a plaque" Del, "unsure" `. */
export interface KeyGroup { group: KeyGroupKind; title: string; keys: string; items: KeyItem[] }
/** `direct`: key -> [identity, type or null] (keys in lower case, "delete" for Del); `groups` and `summary` for the panel
 *  and the key lists; `aliases`: quiet second keys ("0" for "`", "backspace" for "delete"); `brush`: the brush key. */
export interface KeyMap {
  direct: Record<string, [string, string | null]>; groups?: KeyGroup[]; summary?: string; aliases?: Record<string, string>; brush?: string
}
export interface ClassPick { identity: string; subtype: string | null }

export const REJECT_KEY = 'delete'
export const UNSURE_KEY = '`'
export const BRUSH_KEY = 'b'

/** A keyboard event's key as the key map names it: lower case, "delete" for Delete and Backspace, "`" for 0. */
export function keyName(keys: KeyMap, key: string): string {
  const k = key === 'Delete' || key === 'Backspace' ? 'backspace' : key.length === 1 ? key.toLowerCase() : key
  if (keys.direct[k]) return k
  const alias = keys.aliases?.[k] ?? (k === 'backspace' ? REJECT_KEY : k === '0' ? UNSURE_KEY : undefined)
  return alias ?? k
}

/** What a key press answers (none for a key that means nothing here). */
export function pressKey(keys: KeyMap, key: string): ClassPick | null {
  const found = keys.direct[keyName(keys, key)]
  return found ? { identity: found[0], subtype: found[1] } : null
}

/** A key as the pages print it: "1", "Q", "Del", "`". */
export function keyLabel(key: string): string {
  return key === REJECT_KEY ? 'Del' : key.toUpperCase()
}

/** The panel's groups: the server's, else made from the key map (a vocabulary built by hand, in a test). */
export function keyGroups(vocabulary: Vocabulary, keys: KeyMap): KeyGroup[] {
  if (keys.groups?.length) return keys.groups
  const kinds: Record<string, KeyItem[]> = { counted: [], lookalike: [], reject: [], unsure: [] }
  for (const [key, [identity, subtype]] of Object.entries(keys.direct)) {
    const c = classOf(vocabulary, identity)
    const s = c?.subtypes.find((t) => t.id === subtype)
    const kind = key === REJECT_KEY ? 'reject' : key === UNSURE_KEY ? 'unsure' : /^\d$/.test(key) ? 'counted' : 'lookalike'
    kinds[kind].push({ key, label: keyLabel(key), identity, subtype, name: s?.name ?? c?.name ?? identity, description: s?.description ?? c?.description ?? '' })
  }
  return (Object.entries(kinds) as [KeyGroupKind, KeyItem[]][]).filter(([, items]) => items.length).map(([group, items]) => ({
    group, items, title: group === 'counted' ? 'Counted' : group === 'lookalike' ? 'Not counted' : items[0].name,
    keys: items.length > 1 ? `${items[0].label}–${items[items.length - 1].label}` : items[0].label,
  }))
}

/** The keys in one line: "Plaques 1–4 · Not plaques Q–W · Del not a plaque · ` unsure · B brush" (`brush` false: without
 *  the brush, for pages that have none). */
export function keySummary(vocabulary: Vocabulary, keys: KeyMap, brush = true): string {
  const line = keys.summary ?? keyGroups(vocabulary, keys).map((g) => (g.group === 'counted' || g.group === 'lookalike' ? `${g.title} ${g.keys}` : `${g.keys} ${g.title}`))
    .join(' · ')
  return brush ? (line.endsWith('B brush') ? line : `${line} · B brush`) : line.replace(/ · B brush$/, '')
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

/** The classes in the order every page lists them: counted, the named look-alikes, the reject, classes without a key,
 *  "unsure" last (the server's `Vocabulary.by_key`). Hidden ones are left out unless `hidden`. */
export function byKey(vocabulary: Vocabulary, keys: KeyMap, hidden = false): ClassDef[] {
  const place = new Map(vocabulary.classes.map((c, i) => [c.id, i]))
  const first = new Map<string, number>()
  keyGroups(vocabulary, keys).flatMap((g) => g.items).forEach((item, i) => { if (!first.has(item.identity)) first.set(item.identity, i) })
  const rank = (c: ClassDef) => [c.role === 'skip' ? 1 : 0, first.has(c.id) ? 0 : 1, first.get(c.id) ?? 0, place.get(c.id) ?? 0] as const
  return vocabulary.classes.filter((c) => hidden || !c.hidden).sort((a, b) => {
    const x = rank(a), y = rank(b)
    for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return x[i] < y[i] ? -1 : 1
    return 0
  })
}

/** The key that answers this class (and type) at once, if there is one. */
export function keyOf(keys: KeyMap, identity: string, subtype: string | null = null): string | undefined {
  return Object.entries(keys.direct).find(([, [i, s]]) => i === identity && s === subtype)?.[0]
}
