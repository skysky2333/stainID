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
export interface Group { code: string; label: string; color?: number }
export interface PathInfo { path: string; exists: boolean; help: string }
export interface ProjectConfig {
  name: string; pixel_size_um: number; field_context_px: number; stains: Record<string, string>
  tma: { prefix: string; rows: number; columns: number }; groups: Group[]
  inputs: Record<string, string>; models: Record<string, string>; outputs: Record<string, string>
}
export interface ProjectInfo {
  name: string
  root: string
  configured: boolean
  config: ProjectConfig
  defaults: ProjectConfig
  help: Record<string, string>
  pixel_size_um: number
  stains: Record<string, string>
  tma_prefix: string
  groups: Group[]
  paths: Record<string, Record<string, PathInfo>>
  recent: string[]
}
export interface StepOption { key: string; label: string; kind: 'number' | 'text' | 'select' | 'stains' | 'bool'; default: unknown; help: string; choices: string[]; advanced: boolean }
export interface ResourceInfo {
  id: string; label: string; description: string; path: string; exists: boolean
  made_by: { id: string; title: string } | null; download: string; template: string; setting: string; format: string
}
export interface StepProgress { state: 'done' | 'partial' | 'todo'; done: number; total: number; unit: string; note: string; outdated?: boolean }
export interface Step {
  id: string; stage: string; title: string; summary: string; details: string; needs: ResourceInfo[]; uses: ResourceInfo[]; produces: ResourceInfo[]
  command: string[] | null; options: StepOption[]; heavy: boolean; view: string | null; duration: string; progress: StepProgress; job: Job | null
  after: string[]
}
export interface SlideRow { slide_path: string; tma: string; stain: string }
export interface SlidesInfo { folder: string; folder_exists: boolean; stains: string[]; saved: boolean; rows: SlideRow[] }
export interface FsListing { path: string; parent: string | null; dirs: string[]; files: string[]; shortcuts: { label: string; path: string }[] }
export interface Metrics { n: number; positives: number; auc: number; precision: number; recall: number }
export interface TrainedModel {
  stain: string; created: number; validation: string; threshold: number; sets: string[]; new_model: Metrics; current_model: Metrics | null
  current_model_path: string; path: string; active: boolean
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
  created: number; started: number | null; ended: number | null; returncode: number | null; waiting: string
  progress: { done: number; total: number; last_line: string; error: string }
}
export interface ReviewSet {
  name: string; legacy: boolean; items: number; labelled: number; label_counts: Record<string, number>; title: string; stain: string
  created: number | null; purpose: 'training' | 'check' | 'legacy'
}
export interface ReviewData { meta: { title: string; stain: string; label_options: string[]; instructions: string; fov_um: number }; items: Row[] }
export interface TableInfo { name: string; path: string; columns: number; size_kb: number; donor_region: boolean; contrasts: boolean; description: string; main: boolean }
export interface ModelInfo {
  key: string; role: string; path: string; exists: boolean; source: 'train' | 'download' | 'manual'; stain: string | null
  size_mb?: number; modified?: string; bundle?: Record<string, unknown>
}

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
  put: <T,>(url: string, body: unknown) =>
    request<T>(url, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),
  del: <T,>(url: string) => request<T>(url, { method: 'DELETE' }),
  upload: <T,>(url: string, file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<T>(url, { method: 'POST', body: form })
  },
}

export const runStep = (workflow: string, options: Record<string, unknown>) => api.post<Job>('/api/jobs', { workflow, options })
export const reveal = (path: string) => api.post('/api/reveal', { path })
export const downloadUrl = (path: string) => `/api/download?path=${encodeURIComponent(path)}`

export const enc = encodeURIComponent
export const coreThumb = (coreId: string, stain: string, size = 512) => `/api/images/cores/${enc(coreId)}/${enc(stain)}.jpg?size=${size}`
export const fieldImage = (tileId: string) => `/api/images/fields/${enc(tileId)}.jpg`
export const fieldLayer = (tileId: string, layer: string) => `/api/images/fields/${enc(tileId)}/${layer}.png`
