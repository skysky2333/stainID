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
  /** the jobs of this step running or waiting now (a step can run in several parts) */
  active?: Job[]
}
export interface SlideRow {
  slide_path: string; tma: string; stain: string; rows?: string; columns?: string; grid_source?: string; orientation?: string; new?: boolean
  /** which values the file name gave clearly (new scans only) */
  from_name?: { tma: boolean; stain: boolean }
  /** what an unclear file name might mean */
  options?: { tma: string[]; stain: string[] }
}
export interface SlidesInfo {
  folder: string; folder_path: string; folder_exists: boolean; stains: string[]; saved: boolean; rows: SlideRow[]; missing: SlideRow[]
  tma_prefix: string; prefix_guess: string; table: string; tmas: string[]
}

/* the projects this server knows (/api/projects) */
export type ProjectState = 'not_set_up' | 'set_up' | 'running' | 'results' | 'missing'
export interface ProjectEntry {
  id: string; name: string; path: string; last_opened: number | null; in_projects_dir: boolean; exists: boolean
  status: { state: ProjectState; text: string; running: string[] }
}
export interface ProjectsInfo {
  projects: ProjectEntry[]; default: string | null; projects_dir: string; projects_dir_from: '' | 'setting' | 'command line'
  copyable: Record<string, string>
}

/* the guided setup (/api/setup) */
export type ScreenState = 'done' | 'todo' | 'attention' | 'warning' | 'optional' | 'running'
export interface SetupScreen { id: string; title: string; state: ScreenState; summary: string }
export interface Estimate { seconds: number; steps: { id: string; seconds: number }[]; device: string; slides: number; cores: number; tiles: number; text: string; slowest: string }
export interface SetupOverview { screens: SetupScreen[]; set_up: boolean; estimate: Estimate; remaining: { id: string; title: string }[] }
export interface GridReport {
  rows?: number; columns?: number; grid_source?: string; score?: number; rmse_preview_px?: number; cores_on_grid?: number; positions?: number
  positions_with_tissue?: number; tried?: { rows: number; columns: number; score: number | null }[]; error?: string; checked?: boolean; time?: number
}
export interface SlideReport {
  slide_path: string; name: string; tma: string; stain: string; rows: string; columns: string; grid_source: string; orientation: string
  image: string; report: GridReport; checked: boolean; found: boolean; error: string
}
export interface CoresInfo { slides: SlideReport[]; rows: number; columns: number; orientations: Record<string, string> }
export interface MapCell {
  label: string; raw: string; donor_id: string; region: string; region_code: string; disease_group: string; cerad: string; braak: string
  pathology_raw: string; tissue_control: string; note: string; sample_region_id?: string
}
export interface SampleIdRule { plain_prefix: string; prefixed_prefix: string; region: 'name' | 'code'; regions: Record<string, string>; source: 'layout' | 'setting'; examples: string[] }
export interface MapSummary { positions: number; cores: number; donors: number; controls: number; empty: number; unreadable: number; groups: string[]; regions: string[] }
export interface MapTma { tma: string; size: [number, number]; slides: string[]; positions: string[]; saved: MapSummary | null; sources: string[] }
export interface MapOverview {
  tmas: MapTma[]; path: string; id_style: { digits?: number; prefixed_digits?: number }; tma_prefix: string
  sample_ids: SampleIdRule; sample_ids_setting: { plain_prefix?: string; prefixed_prefix?: string; region?: string }
}
export interface ParsedMap { cells: MapCell[]; summary: MapSummary }
export interface FoundMap extends ParsedMap { name: string; source: string; tma: string; grid: string[][] }
export interface SavedMap extends ParsedMap { tma: string; size: [number, number]; grid: string[][]; saved: boolean }
export interface DonorColumn { index: number; header: string; field: string; samples: string[] }
export interface DonorPreview {
  header_row: number; columns: DonorColumn[]; records: Record<string, string>[]; total: number; kept: number; notes: string[]
  table_columns: string[]; fields: Record<string, string>
}
export interface DonorTable { path: string; count: number; columns: string[]; fields: Record<string, string>; preview: Record<string, string>[] }
export interface SheetTable { name: string; source: string; rows: string[][] }
export interface Fix { label: string; action: string; args: Record<string, unknown> }
export interface Issue { id: string; level: 'problem' | 'warning' | 'note'; title: string; detail: string; items: { text: string; fixes: Fix[] }[]; more: number; fixes: Fix[] }
export interface FsListing { path: string; parent: string | null; dirs: string[]; files: string[]; shortcuts: { label: string; path: string }[] }
export interface Metrics { n: number; positives: number; auc: number; precision: number; recall: number }
export interface TrainedModel {
  stain: string; created: number; validation: string; threshold: number; sets?: string[]; sources?: Record<string, number>; new_model: Metrics; current_model: Metrics | null
  current_model_path: string; path: string; active: boolean; sessions?: { name: string; title: string; objects: number }[]
}
export interface CoreRow {
  core_id: string; tma: string; core_label: string; donor_id: string | null; sample_region_id: string | null
  region: string | null; disease_group: string | null; technical_replicate: string | null; stains: string[]
}
export interface CoreDetail extends Omit<CoreRow, 'stains'> {
  images: { stain: string; native_width_px: number; native_height_px: number; tissue_fraction: number; tissue_status: string }[]
}
export interface FieldObject { x: number; y: number; radius_px: number; model_class: string; probability: number | null; area_um2: number | null; [k: string]: unknown }
export interface Outline { label: number; points: [number, number][]; area_um2: number | null; circularity: number | null; seed_source: string }
export interface Job {
  id: string; workflow: string; title: string; options: Record<string, unknown>; argv: string[]; status: string
  created: number; started: number | null; ended: number | null; returncode: number | null; waiting: string
  progress: { done: number; total: number; last_line: string; error: string }
  source?: 'app' | 'command line'; pid: number | null; log_file?: string
}
export interface TableInfo {
  name: string; title: string; path: string; columns: number; size_kb: number; modified: number; donor_region: boolean; contrasts: boolean; description: string; main: boolean
}
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
  fields?: number; fields_covered?: number; fields_prepared?: number; labellers: Labeller[]; labelling_now: number
}
export interface Labeller { person: string; fields_done: number; labels: number; changed: number; confirmed: number; outlines: number; tags: number; last_active: number }
export interface LabelsOverview { stains: StainOverview[]; sessions: SessionInfo[]; reference: ReferenceLabels | null }
export interface OpenSession { name: string; title: string; stain: string; kind: 'fields' | 'queue'; open: boolean; instructions: string; fields: number; done_by_me: number; my_labels: number }
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

/* labelling fields (whole-core area): choosing fields, their state in a session, the labels of a session */
export interface Blocker { reason: string; step: string; step_title: string; link: string }
export interface FieldPlace { core_id?: string; core_label?: string; tma?: string; region?: string | null; disease_group?: string | null; x_px?: number; y_px?: number; width_px?: number; height_px?: number }
export interface PickerField { tile_id: string; x_px: number; y_px: number; width_px: number; height_px: number; labelled_by: string[]; sessions: string[]; blocked: Blocker | null }
export interface PickerCore {
  core_id: string; tma: string; core_label: string; region: string | null; disease_group: string | null; donor_id: string | null
  width: number | null; height: number | null; fields: PickerField[]; blocked: Blocker | null
}
export interface PickerPage { cores: PickerCore[]; total_cores: number; total_fields: number; ready_fields: number; facets: { tmas: string[]; regions: string[]; groups: string[] } }
export type FieldState = 'preparing' | 'not_started' | 'in_progress' | 'done'
export interface SessionFieldState extends FieldPlace { tile_id: string; state: FieldState; done_by: string[]; working: string[]; open_by: string[] }
export interface LabelSet { name: string; title: string; reference: boolean; labels: number; objects: number; fields: number; classes: Record<string, number>; people: string[] }
export interface ReferenceLabels { name: string; title: string; labels: number; stains: Record<string, number>; sets: string[]; fields: number }
/* the labelled objects of a session, each as the crop the model sees (labels view) */
export type LabelFrame = 'field' | 'pilot' | 'core'
export type AiVerdict = 'agrees' | 'type' | 'disagrees'
export interface ObjectLabel { value: string; source: string; time: number | null }
export interface CropState { state: 'ready' | 'waiting' | 'preparing' | 'failed' | 'missing'; ahead?: number; error?: string }
export interface LabelledObject {
  tile_id: string; x: number; y: number; frame: LabelFrame | null; value: string; labels: ObjectLabel[]; time: number | null
  ai_value: string | null; ai_probability: number | null; ai: AiVerdict | null; radius_px: number | null
  outline: number[][] | null; outline_by: 'person' | 'AI' | null; crop: CropState
  core_id?: string | null; core_label?: string | null; tma?: string | null; region?: string | null; disease_group?: string | null
  core_x: number | null; core_y: number | null
}
export interface SessionObjects {
  name: string; title: string; reference: boolean; stain: string; stains: string[]; vocabulary: string[]
  classes: Record<string, number>; people: Record<string, number>; ai: Partial<Record<AiVerdict | 'unknown', number>>
  objects: LabelledObject[]; total: number; all: number
  crop: { size_px: number; um: number; pixel_um: number } | null; preparing: { objects: number; cores: number; ahead: number } | null
}

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

/* Which project this page shows. It is part of the page's address, /p/<project>/..., which the router uses as its base
   (main.tsx), so the app's own links need no project. Every API call of the page then goes to /api/projects/<project>/...,
   except the routes that belong to the server as a whole (sign-in, the projects list). This is the only place that adds
   the project; pages write plain /api/... paths. Without a project in the address (the Projects page, and the
   addresses of the previous release) calls stay unscoped, which the server answers for the project opened last. */
const PROJECT_IN_ADDRESS = /^\/p\/([A-Za-z0-9][A-Za-z0-9_.-]*)(?=\/|$)/
const SERVER_WIDE = /^\/api\/(session|projects)(\/|\?|$)/
export const projectId = location.pathname.match(PROJECT_IN_ADDRESS)?.[1] ?? ''
/** The base of this project's pages ('' on the Projects page). */
export const appBase = projectId ? `/p/${projectId}` : ''
/** A page of this project as a plain address, for an <a href> or a link to send someone (router links need no prefix). */
export const appPath = (path: string) => `${appBase}${path}`
/** The address of a project's pages; going there loads the page afresh, for that project. */
export const projectPath = (id: string, page = '/') => `/p/${id}${page}`

function scoped(path: string) {
  return projectId && path.startsWith('/api/') && !SERVER_WIDE.test(path) ? `/api/projects/${projectId}${path.slice(4)}` : path
}

/** The full address of an API path, for what the browser loads by itself (images, downloads, links). */
export function apiUrl(path: string) {
  const url = apiBase + scoped(path)
  return remote && token ? `${url}${url.includes('?') ? '&' : '?'}access_token=${encodeURIComponent(token)}` : url
}

// Requests in flight, for the loading bar at the top of the page (see ActivityBar in components.tsx).
let inFlight = 0
const activityListeners = new Set<(count: number) => void>()
export function onActivity(listener: (count: number) => void): () => void {
  activityListeners.add(listener)
  return () => { activityListeners.delete(listener) }
}
function setInFlight(change: number) {
  inFlight += change
  activityListeners.forEach((listener) => listener(inFlight))
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  setInFlight(1)
  try { return await send<T>(url, init) } finally { setInFlight(-1) }
}

async function send<T>(url: string, init?: RequestInit): Promise<T> {
  const headers = remote && token ? { ...(init?.headers as Record<string, string> | undefined), Authorization: `Bearer ${token}` } : init?.headers
  const response = await fetch(apiBase + scoped(url), { ...init, headers })
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
export const slidePicture = (slidePath: string, size = 320) => apiUrl(`/api/slides/picture?path=${encodeURIComponent(slidePath)}&size=${size}`)
export const reveal = (path: string) => api.post('/api/reveal', { path })
export const downloadUrl = (path: string) => apiUrl(`/api/download?path=${encodeURIComponent(path)}`)

export const enc = encodeURIComponent
export const coreThumb = (coreId: string, stain: string, size = 512) => apiUrl(`/api/images/cores/${enc(coreId)}/${enc(stain)}.jpg?size=${size}`)
export const fieldImage = (tileId: string) => apiUrl(`/api/images/fields/${enc(tileId)}.jpg`)
export const fieldLayer = (tileId: string, layer: string) => apiUrl(`/api/images/fields/${enc(tileId)}/${layer}.png`)
