/* The order "Next" follows on a field page (labels/picker.py on the server): the picker's order the field was opened
   from — "natural" (TMA, core, the tile's place in the core) or "confidence" (the By-confidence list, least confident
   first). The field's address keeps it (`?order=confidence`), so a refresh or the next field keeps it too. */
import { useSearchParams } from 'react-router-dom'
import { enc } from '../api'

export type FieldOrder = 'natural' | 'confidence'

/** Where the picker keeps the order a person chose (Natural order / By confidence). */
export const orderKey = (annotator: string | null | undefined) => `stainid.fieldOrder.${annotator ?? ''}`

/** The order of the field page open now (from its address). */
export function useFieldOrder(): FieldOrder {
  const [params] = useSearchParams()
  return params.get('order') === 'confidence' ? 'confidence' : 'natural'
}

/** A field's page, keeping the order it is worked in. */
export const fieldPath = (name: string, tile: string, order: FieldOrder) => `/annotate/${name}/${enc(tile)}${order === 'confidence' ? '?order=confidence' : ''}`

/** The next field after `tile` in that order (`skip`: the field is put aside for now; none: the first in the order). */
export const nextFieldUrl = (name: string, tile: string | null, order: FieldOrder, skip = false) =>
  `/api/annotate/${name}/next?order=${order}${tile ? `&${skip ? 'skip' : 'after'}=${enc(tile)}` : ''}`

/** The order the picker shows now (what "Start" / "Continue with the next field" follows). */
export function storedOrder(annotator: string | null | undefined): FieldOrder {
  try { return JSON.parse(localStorage.getItem(orderKey(annotator)) ?? '""') === 'confidence' ? 'confidence' : 'natural' } catch { return 'natural' }
}
