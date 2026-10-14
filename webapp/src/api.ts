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
  tma: { prefix: string; rows: number; columns: number }; groups: Group[]; analysis: { tiles: string }
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
  stain: string; created: number; validation: string; threshold: number; sets?: string[]; sources?: Record<string, number>; new_model: Metrics; current_model: Metrics | null
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
  source?: 'app' | 'command line'; pid: number | null; log_file?: string
}
export interface TableInfo { name: string; path: string; columns: number; size_kb: number; donor_region: boolean; contrasts: boolean; description: string; main: boolean }
export interface ModelInfo {
  key: string; role: string; path: string; exists: boolean; source: 'train' | 'download' | 'manual'; stain: string | null
  size_mb?: number; modified?: string; bundle?: Record<string, unknown>
}

export interface Session { admin: boolean; annotator: string | null; default_password: boolean; local: boolean; project: string | null; allowed_origins?: string[] }
export type AiView = 'full' | 'positions' | 'none'
export interface StainOverview {
  stain: string; fields: number; fields_prepared: number; fields_reviewed: number; fields_with_labels: number
  objects_labelled: number; objects_added: number; values: Record<string, number>; outlines: number; outlines_edited: number
  people: string[]; agreement: number | null; objects_with_several_labels: number
  ai: { n: number; agreement: number | null; precision: number | null; recall: number | null }; tags: Record<string, number>
}
export interface SessionInfo {
  name: string; title: string; stain: string; kind: 'fields' | 'queue'; tools: string[]; ai_view: AiView; per_field: number; rule: string
  instructions: string; created: number; open: boolean; people: number; labels: number
  fields?: number; fields_covered?: number; fields_prepared?: number
}
export interface LabelsOverview { stains: StainOverview[]; sessions: SessionInfo[] }
export interface OpenSession { name: string; title: string; stain: string; kind: 'fields' | 'queue'; open: boolean; fields: number; done_by_me: number; my_labels: number }
export interface SessionField { tile_id: string; done_by_me: boolean; annotators_done: number; open_by_others: number; prepared: boolean }
export interface SessionDetail {
  meta: SessionInfo & { fields: string[]; classes: string[]; tags: string[]; class_instructions: string }
  fields: SessionField[]; annotator: string | null
}
export interface Label {
  id: string; source: string; stain: string; task: 'class' | 'outline' | 'tag' | 'note' | 'done'; tile_id: string
  x: number | null; y: number | null; value: string; points: number[][] | null; session: string; confirmed: boolean; time: number; deleted?: boolean
}
export interface FieldObjectLite { x: number; y: number; radius_px: number; ai_value?: string; ai_probability?: number }
export interface FieldWork {
  tile_id: string; box: { x: number; y: number; width: number; height: number }; objects: FieldObjectLite[]
  outlines: { x: number; y: number; points: number[][] }[]; layers: string[]; mine: Label[]
}
export interface QueueItem { tile_id: string; x: number; y: number; radius_px: number; ai_value: string; ai_probability: number; labels_so_far: number; remaining: number }

/* Which stainID server this page talks to. Normally the one that served it (same origin: sign-in travels in cookies).
   It can also be another one: an address chosen on the sign-in screen (kept in this browser), else `window.STAINID_SERVER`
   from config.js (set when the web app is served from somewhere else). A server elsewhere only answers pages from the
   addresses it was started with (`stainid serve --allow-origin URL`); sign-in then travels as a token. */
declare global { interface Window { STAINID_SERVER?: string } }

const SERVER_KEY = 'stainid.server'
const readStored = (key: string) => { try { return localStorage.getItem(key) ?? '' } catch { return '' } }
const writeStored = (key: string, value: string) => {
  try { if (value) localStorage.setItem(key, value); else localStorage.removeItem(key) } catch { /* storage unavailable */ }
}
export const cleanServer = (address: string) => {
  const trimmed = address.trim().replace(/\/+$/, '')
  return trimmed && !/^https?:\/\//.test(trimmed) ? `http://${trimmed}` : trimmed
}
export const savedServer = readStored(SERVER_KEY)
export const configuredServer = cleanServer(window.STAINID_SERVER ?? '')
export const apiBase = cleanServer(savedServer) || configuredServer
export const remote = apiBase !== '' && new URL(apiBase, location.href).origin !== location.origin
/** The server's address as people read it. */
export const serverName = apiBase || location.origin
const TOKEN_KEY = `stainid.token ${apiBase}`
let token = remote ? readStored(TOKEN_KEY) : ''

/** Switch to another server ('' = the one that served this page); the page reloads. */
export function chooseServer(address: string) {
  writeStored(SERVER_KEY, cleanServer(address))
  location.reload()
}

/** The full address of an API path, for what the browser loads by itself (images, downloads, links). */
export function apiUrl(path: string) {
  const url = apiBase + path
  return remote && token ? `${url}${url.includes('?') ? '&' : '?'}access_token=${encodeURIComponent(token)}` : url
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const headers = remote && token ? { ...(init?.headers as Record<string, string> | undefined), Authorization: `Bearer ${token}` } : init?.headers
  const response = await fetch(apiBase + url, { ...init, headers })
  if (!response.ok) {
    const text = await response.text()
    let detail = text
    try { detail = JSON.parse(text).detail ?? text } catch { /* plain text error */ }
    throw new Error(`${response.status}: ${detail}`)
  }
  const type = response.headers.get('content-type') ?? ''
  const body = await (type.includes('application/json') ? response.json() : response.text())
  if (remote && url.startsWith('/api/session') && typeof body?.token === 'string') {  // signing in / out changes who this page is
    token = body.token
    writeStored(TOKEN_KEY, token)
  }
  return body as T
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
export const downloadUrl = (path: string) => apiUrl(`/api/download?path=${encodeURIComponent(path)}`)

export const enc = encodeURIComponent
export const coreThumb = (coreId: string, stain: string, size = 512) => apiUrl(`/api/images/cores/${enc(coreId)}/${enc(stain)}.jpg?size=${size}`)
export const fieldImage = (tileId: string) => apiUrl(`/api/images/fields/${enc(tileId)}.jpg`)
export const fieldLayer = (tileId: string, layer: string) => apiUrl(`/api/images/fields/${enc(tileId)}/${layer}.png`)
