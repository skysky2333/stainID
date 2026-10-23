/* What the Cores view draws of detection's objects (CoreView.tsx): accepted ones at every zoom, rejected candidates
 * only once zoomed in close enough to tell them apart. Plain functions, tested by tests/js/core_layers.test.ts. */

/** Rejected candidates are drawn (and their outlines fetched) once the view spans at most this much across, about two
 *  labelling fields: further out they are too small to tell apart and only crowd the accepted objects. */
export const REJECTED_VIEW_UM = 1100
/** The pixel size assumed when the image has none (µm per pixel). */
export const FALLBACK_PIXEL_UM = 0.5

export interface ViewRect { x0: number; y0: number; x1: number; y1: number }
export interface Shown { cls: string; x: number; y: number; outline?: unknown }
export interface ObjectLayers { objects: boolean; rejected: boolean }

/** How much of the tissue the view spans across (µm): its larger side; Infinity before the picture reports a view. */
export function viewSpanUm(view: ViewRect | null, pixelUm: number | null | undefined): number {
  if (!view) return Infinity
  const um = pixelUm && Number.isFinite(pixelUm) && pixelUm > 0 ? pixelUm : FALLBACK_PIXEL_UM
  return Math.max(view.x1 - view.x0, view.y1 - view.y0) * um
}

export const closeEnough = (spanUm: number) => spanUm <= REJECTED_VIEW_UM

/** The objects drawn: every kept one; the rejected ones with their layer on and the view close enough; with the objects
 *  layer off, only those whose outline is shown. */
export function drawnObjects<T extends Shown>(objects: T[], layers: ObjectLayers, close: boolean, showOutlines: boolean): T[] {
  return objects.filter((o) => (o.cls !== 'rejected' || (layers.rejected && close)) && (layers.objects || (showOutlines && !!o.outline)))
}

const inside = (o: Shown, v: ViewRect) => o.x >= v.x0 && o.x <= v.x1 && o.y >= v.y0 && o.y <= v.y1

/** The legend's line under "also show rejected candidates": zoomed out, that they come when zoomed in; zoomed in,
 *  how many are in view (none here: where they are is elsewhere in the core). Null when nothing is to say. */
export function rejectedNote(objects: Shown[], layers: ObjectLayers, close: boolean, view: ViewRect | null, word = 'core'): string | null {
  if (!layers.objects || !layers.rejected) return null
  const rejected = objects.filter((o) => o.cls === 'rejected')
  if (!rejected.length) return null
  if (!close) return 'Zoom in to see rejected candidates'
  const here = view ? rejected.filter((o) => inside(o, view)).length : rejected.length
  return here ? `${here.toLocaleString()} rejected in view` : `No rejected candidates in this part of the ${word}`
}
