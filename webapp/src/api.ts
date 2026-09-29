export type Row = Record<string, string | number | boolean | null>

export interface StageStatus { done: number; total: number; unit: string }
export interface Summary {
  donors: number
  groups: Record<string, number>
  tmas: string[]
  cores: number
  donor_regions: number
  fields: Record<string, number>
  status: Record<string, { detection: StageStatus; nuclei: StageStatus; masks: StageStatus }>
}
export interface ProjectInfo {
  name: string
  root: string
  pixel_size_um: number
  stains: Record<string, string>
  groups: Record<string, string>
  paths: Record<string, Record<string, { path: string; exists: boolean }>>
}
export interface CoreRow {
  core_id: string; tma: string; core_label: string; donor_id: string | null; sample_region_id: string | null
  region: string | null; disease_group: string | null; technical_replicate: string | null; stains: string[]
}
export interface FieldRow { tile_id: string; stain: string; selection_order: number; x_px: number; y_px: number; width_px: number; height_px: number }
export interface CoreDetail extends Omit<CoreRow, 'stains'> {
  images: { stain: string; native_width_px: number; native_height_px: number; tissue_fraction: number; tissue_status: string }[]
  fields: FieldRow[]
}
export interface FieldObject { x: number; y: number; radius_px: number; model_class: string; probability: number | null; area_um2: number | null; [k: string]: unknown }
export interface Outline { label: number; points: [number, number][]; area_um2: number | null; circularity: number | null; seed_source: string }
export interface Job {
  id: string; workflow: string; title: string; options: Record<string, unknown>; argv: string[]; status: string
  created: number; started: number | null; ended: number | null; returncode: number | null
  progress: { done: number; total: number; last_line: string }
}
export interface ReviewSet { name: string; legacy: boolean; items: number; labelled: number; label_counts: Record<string, number>; title: string; stain: string; created: number | null }
export interface ReviewData { meta: { title: string; stain: string; label_options: string[]; instructions: string; fov_um: number }; items: Row[] }
export interface TableInfo { name: string; columns: number; size_kb: number; donor_region: boolean; contrasts: boolean }
export interface ModelInfo { key: string; role: string; path: string; exists: boolean; size_mb?: number; modified?: string; bundle?: Record<string, unknown> }

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  if (!response.ok) {
    const text = await response.text()
    let detail = text
    try { detail = JSON.parse(text).detail ?? text } catch { /* plain text error */ }
    throw new Error(`${response.status}: ${detail}`)
  }
  const type = response.headers.get('content-type') ?? ''
  return (type.includes('application/json') ? response.json() : response.text()) as Promise<T>
}

export const api = {
  get: <T,>(url: string) => request<T>(url),
  post: <T,>(url: string, body: unknown) =>
    request<T>(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),
  del: <T,>(url: string) => request<T>(url, { method: 'DELETE' }),
}

export const enc = encodeURIComponent
export const coreThumb = (coreId: string, stain: string, size = 512) => `/api/images/cores/${enc(coreId)}/${enc(stain)}.jpg?size=${size}`
export const fieldImage = (tileId: string) => `/api/images/fields/${enc(tileId)}.jpg`
export const fieldLayer = (tileId: string, layer: string) => `/api/images/fields/${enc(tileId)}/${layer}.png`
