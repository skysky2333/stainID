/* The segmentation check (outline sessions and their scores): what the API sends. */

/** One object a person outlined: `ref` is its id within the field, `points` its polygon in field pixels. */
export interface OutlineObject { ref: string; value: string; points: number[][] }

/** A field to outline (GET /api/annotate/<session>/outline/<field>). */
export interface OutlineField {
  tile_id: string; box: { x: number; y: number; width: number; height: number }; objects: (OutlineObject & { x: number; y: number })[]
  complete: boolean; segment_anything: boolean; pixel_size_um: number
}

/** The outlines Segment Anything offers for a click, smallest first, and the one the stain's outline step would choose. */
export interface SamOptions { options: { points: number[][]; area_px: number; score: number }[]; chosen: number }

/** What the pipeline found in a field, shown only when the person asks: counted objects (circles), their Segment Anything
 *  outlines and every candidate (flat [x0, y0, x1, y1, …] polygons; null until the check has found them). */
export interface OutlineReference { counted: { x: number; y: number; r: number }[]; outlines: number[][]; candidates: number[][] | null }

export type Outcome = 'counted' | 'rejected' | 'missed' | 'merged' | 'split' | 'poor outline'

/** The figures of one group of objects (a stain, a TMA, a density third …). */
export interface CheckFigures {
  objects: number; counted: number
  proposal_recall: number | null; proposal_recall_low: number | null
  detection_recall: number | null; detection_recall_low: number | null
  detection_precision: number | null; detection_precision_low: number | null; f1: number | null
  outcomes: Record<Outcome, number>; proposal_outcomes: Record<string, number>; counted_outcomes: Record<string, number>
  proposal_iou_median: number | null; boundary_iou_median: number | null; outline_iou_median: number | null
}
export interface CheckGroup extends CheckFigures { key: string; fields: number }
export interface SizeBin { key: string; objects: number; proposal_recall: number | null; detection_recall: number | null; outcomes: Record<Outcome, number> }

export interface StainScores extends Partial<CheckFigures> {
  stain: string; sessions: { name: string; title: string }[]; fields_total: number; fields_complete: number; fields_scored: number
  people: string[]; waiting: number; not_found_again: number; objects: number
  by_tma: CheckGroup[]; by_group: CheckGroup[]; by_density: CheckGroup[]; by_size: SizeBin[]
  proposal_iou_histogram: number[]; detection_iou_histogram: number[]
}
/** One rater of a dataset package whose polygons were scored (image-set and whole-slide projects). */
export interface RaterScores extends StainScores { title: string; dataset: string; rater: string }
export interface CheckReport {
  made: number; iou: number; iou_low: number; stains: StainScores[]; imported: RaterScores[]; failed: string[]; rules: Record<string, unknown>
}

/** One outline session of the check, and how far it is. */
export interface CheckSession {
  name: string; title: string; stain: string; open: boolean; created: number; fields: number; complete: number; objects: number
  people: string[]; link: string; tmas: string[]
}
export interface CheckOverview {
  stains: string[]; checks: CheckSession[]; report: CheckReport | null; files: { label: string; path: string; exists: boolean }[]
  segment_anything: boolean; tmas: Record<string, { all: string[]; default: string[] }>; default_fields: number; now: number
  /** a TMA study (else the `tma` column holds image batches); read-only sessions imported from a dataset package */
  tma_study: boolean; imported_sessions: number
}
