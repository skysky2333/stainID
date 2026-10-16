/* Labelling fields: what a field is, its small picture, the picker for choosing fields by hand, and where each field of
   a session stands. A field is one whole-core tile shown to people as a picture. */
import { useState } from 'react'
import { Link } from 'react-router-dom'
import type { Blocker, FieldState, PickerCore, PickerField, PickerPage, SessionFieldState, SessionInfo } from '../api'
import { api, apiUrl, coreThumb, enc } from '../api'
import { Empty, ErrorNote, GroupBadge, Loaded, Modal, withError } from '../components'
import { useFetch } from '../hooks'
import { ConfirmDialog } from '../labelling'
import { useProject } from '../project'

export const FIELD_TEXT = 'A field is one square of tissue about 560 µm across (a 2048-pixel tile of a core), shown to people as one picture. '
  + 'A core holds about 15–40 fields.'

export const STATE_TEXT: Record<FieldState, string> = { preparing: 'Being prepared', not_started: 'Not started', in_progress: 'In progress', done: 'Done' }
const STATE_CLASS: Record<FieldState, string> = { preparing: 'status-queued', not_started: 'status-todo', in_progress: 'status-running', done: 'status-done' }

export function FieldStateBadge({ state }: { state: FieldState }) {
  return <span className={`badge ${STATE_CLASS[state]}`}>{STATE_TEXT[state]}</span>
}

/** The small picture of a field (cut from its core's thumbnail). */
export function FieldThumb({ src, alt }: { src: string; alt: string }) {
  const [failed, setFailed] = useState(false)
  return (
    <div className="field-thumb">
      {failed ? <span className="muted small">no picture</span> : <img src={src} alt={alt} loading="lazy" onError={() => setFailed(true)} />}
    </div>
  )
}

/** Why a core or field cannot be labelled yet, and the step that comes first. */
export function BlockedNote({ blocked }: { blocked: Blocker }) {
  return (
    <div className="blocked-note small">
      <span>{blocked.reason}</span>
      <Link className="btn small" to={blocked.link}>{blocked.step ? `Go to “${blocked.step_title}”` : `Open ${blocked.step_title}`}</Link>
    </div>
  )
}

/* ---------- choosing fields by hand ---------- */

const PAGE = 12

/** Cores of one stain with their fields drawn on the picture; click a field to choose it. Fields (or whole cores) that
 *  cannot be labelled yet are greyed out with the reason and a link to the step that comes first; all others stay choosable. */
export function FieldPicker({ stain, chosen, onChange, source = '' }: { stain: string; chosen: string[]; onChange: (ids: string[]) => void; source?: string }) {
  const { tmaName } = useProject()
  const [filters, setFilters] = useState({ tma: '', region: '', group: '', core: '', free: true })
  const [page, setPage] = useState(0)
  const query = new URLSearchParams({ stain, tma: filters.tma, region: filters.region, group: filters.group, core: filters.core,
    free: String(filters.free), offset: String(page * PAGE), limit: String(PAGE), source })
  const picker = useFetch<PickerPage>(`/api/labels/picker?${query}`)
  const set = (change: Partial<typeof filters>) => { setFilters({ ...filters, ...change }); setPage(0) }
  const selected = new Set(chosen)
  const toggle = (ids: string[], on: boolean) => onChange(on ? [...chosen, ...ids.filter((id) => !selected.has(id))] : chosen.filter((id) => !ids.includes(id)))
  const facets = picker.data?.facets
  return (
    <div className="field-picker">
      <div className="row picker-filters">
        <label className="field">TMA<select value={filters.tma} onChange={(e) => set({ tma: e.target.value })}>
          <option value="">all</option>{(facets?.tmas ?? []).map((t) => <option key={t} value={t}>{tmaName(t)}</option>)}</select></label>
        <label className="field">Region<select value={filters.region} onChange={(e) => set({ region: e.target.value })}>
          <option value="">all</option>{(facets?.regions ?? []).map((r) => <option key={r} value={r}>{r}</option>)}</select></label>
        <label className="field">Group<select value={filters.group} onChange={(e) => set({ group: e.target.value })}>
          <option value="">all</option>{(facets?.groups ?? []).map((g) => <option key={g} value={g}>{g}</option>)}</select></label>
        <label className="field">Core<input type="search" value={filters.core} placeholder="e.g. B-2" style={{ width: 110 }} onChange={(e) => set({ core: e.target.value })} /></label>
        <label className="layer-toggle"><input type="checkbox" checked={filters.free} onChange={(e) => set({ free: e.target.checked })} />only fields nobody has labelled</label>
      </div>
      <div className="row small picker-legend">
        <span><span className="swatch outline picker-free" />choosable</span><span><span className="swatch picker-chosen" />chosen</span>
        <span><span className="swatch outline picker-labelled" />has labels or is in a session</span><span><span className="swatch picker-blocked" />not ready yet</span>
        <span className="spacer" />
        {picker.data && <span className="secondary">{picker.data.ready_fields.toLocaleString()} of {picker.data.total_fields.toLocaleString()} fields ready
          · <b>{chosen.length.toLocaleString()} chosen</b></span>}
        {chosen.length > 0 && <button className="link-btn small" onClick={() => onChange([])}>Clear</button>}
      </div>
      <Loaded state={picker} label="Loading the cores…">
        {(data) => data.cores.length ? <>
          <div className="picker-grid">{data.cores.map((c) => <PickerCard key={c.core_id} core={c} stain={stain} selected={selected} onToggle={toggle} />)}</div>
          {data.total_cores > PAGE && (
            <div className="row picker-pages">
              <button className="btn small" disabled={page === 0} onClick={() => setPage(page - 1)}>‹ Previous</button>
              <span className="small secondary">cores {page * PAGE + 1}–{Math.min(data.total_cores, (page + 1) * PAGE)} of {data.total_cores}</span>
              <button className="btn small" disabled={(page + 1) * PAGE >= data.total_cores} onClick={() => setPage(page + 1)}>Next ›</button>
            </div>
          )}
        </> : <Empty>{filters.free || filters.tma || filters.region || filters.group || filters.core
          ? 'No fields match these filters.' : <>There are no {stain} fields yet. Fields are the tiles of each core: run <Link to="/workflow#tiles">Cut cores into tiles</Link> first.</>}</Empty>}
      </Loaded>
    </div>
  )
}

function PickerCard({ core, stain, selected, onToggle }: { core: PickerCore; stain: string; selected: Set<string>; onToggle: (ids: string[], on: boolean) => void }) {
  const { tmaName } = useProject()
  const ready = core.fields.filter((f) => !f.blocked)
  const chosenHere = core.fields.filter((f) => selected.has(f.tile_id)).length
  const width = core.width ?? Math.max(...core.fields.map((f) => f.x_px + f.width_px))
  const height = core.height ?? Math.max(...core.fields.map((f) => f.y_px + f.height_px))
  const tone = (f: PickerField) => (f.blocked ? 'blocked' : selected.has(f.tile_id) ? 'chosen' : f.labelled_by.length || f.sessions.length ? 'labelled' : 'free')
  const title = (f: PickerField) => f.blocked ? f.blocked.reason : [f.labelled_by.length ? `labelled by ${f.labelled_by.join(', ')}` : '',
    f.sessions.length ? `in session ${f.sessions.join(', ')}` : '', selected.has(f.tile_id) ? 'chosen: click to leave out' : 'click to choose'].filter(Boolean).join(' · ')
  return (
    <div className={`card picker-card${core.blocked ? ' is-blocked' : ''}`}>
      <div className="row picker-card-head"><b>{tmaName(core.tma)} · {core.core_label}</b><GroupBadge group={core.disease_group} />
        {core.region && <span className="muted small">{core.region}</span>}</div>
      <div className="picker-picture" style={{ aspectRatio: `${width} / ${height}` }}>
        <img src={coreThumb(core.core_id, stain, 512)} alt={`${core.core_label} ${stain}`} loading="lazy" />
        <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
          {core.fields.map((f) => (
            <rect key={f.tile_id} x={f.x_px} y={f.y_px} width={f.width_px} height={f.height_px} className={`picker-square ${tone(f)}`}
              onClick={() => !f.blocked && onToggle([f.tile_id], !selected.has(f.tile_id))}>
              <title>{title(f)}</title>
            </rect>
          ))}
        </svg>
      </div>
      {core.blocked ? <BlockedNote blocked={core.blocked} /> : (
        <div className="row small">
          <span className="secondary">{chosenHere ? `${chosenHere} of ${core.fields.length} chosen` : `${ready.length} of ${core.fields.length} fields ready`}</span>
          <span className="spacer" />
          {ready.length > 0 && (chosenHere < ready.length
            ? <button className="link-btn small" onClick={() => onToggle(ready.map((f) => f.tile_id), true)}>Choose all</button>
            : <button className="link-btn small" onClick={() => onToggle(ready.map((f) => f.tile_id), false)}>Leave all out</button>)}
        </div>
      )}
      {!core.blocked && core.fields.some((f) => f.blocked) && <BlockedNote blocked={core.fields.find((f) => f.blocked)!.blocked!} />}
    </div>
  )
}

/** How the fields of a session are chosen: stainID picks them, or the admin does, in the picker. */
export function FieldChoice({ stain, mode, n, chosen, onMode, onN, onChosen, autoText }: {
  stain: string; mode: 'auto' | 'pick'; n: number; chosen: string[]
  onMode: (mode: 'auto' | 'pick') => void; onN: (n: number) => void; onChosen: (ids: string[]) => void
  autoText?: string  // how stainID chooses, when not as for reviews
}) {
  return (
    <fieldset className="field field-choice"><legend>Which fields?</legend>
      <p className="small secondary" style={{ margin: '0 0 8px' }}>{FIELD_TEXT}</p>
      <div className="choice-grid">
        <label className={`choice-card${mode === 'auto' ? ' chosen' : ''}`}>
          <input type="radio" name="field-mode" checked={mode === 'auto'} onChange={() => onMode('auto')} />
          <span><b>Let stainID choose (recommended)</b><span className="small secondary">{autoText ?? `Fields nobody has labelled yet, spread over TMAs and groups,
            from cores that are ready for labelling.`}</span>
            {mode === 'auto' && <label className="field inline-number">How many fields
              <input type="number" min={1} max={500} value={n} style={{ width: 90 }} onChange={(e) => onN(Math.max(1, Number(e.target.value)))} /></label>}</span>
        </label>
        <label className={`choice-card${mode === 'pick' ? ' chosen' : ''}`}>
          <input type="radio" name="field-mode" checked={mode === 'pick'} onChange={() => onMode('pick')} />
          <span><b>Choose them myself</b><span className="small secondary">Browse the cores and click the squares to label: for example one region, or cores
            you know are difficult.</span></span>
        </label>
      </div>
      {mode === 'pick' && <FieldPicker stain={stain} chosen={chosen} onChange={onChosen} />}
    </fieldset>
  )
}

/* ---------- the fields of a session ---------- */

/** Every field of a session: its picture, where it stands, who finished it or works on it; fields can be added or taken out. */
export function SessionFields({ s, onChanged }: { s: SessionInfo; onChanged: () => void }) {
  const fields = useFetch<SessionFieldState[]>(`/api/labels/sessions/${s.name}/fields`, 15000)
  const [show, setShow] = useState<FieldState | ''>('')
  const [adding, setAdding] = useState(false)
  const [removing, setRemoving] = useState<SessionFieldState | null>(null)
  const [error, setError] = useState<string | null>(null)
  const { tmaName } = useProject()
  const changed = () => { fields.reload(); onChanged() }
  return (
    <section className="session-fields">
      <Loaded state={fields} label="Loading the fields…">
        {(list) => {
          const count = (state: FieldState) => list.filter((f) => f.state === state).length
          const shown = show ? list.filter((f) => f.state === show) : list
          return <>
            <div className="row">
              <h3 style={{ margin: 0 }}>Fields ({list.length})</h3>
              <div className="segmented">
                <button className={`btn small${show === '' ? ' chosen' : ''}`} onClick={() => setShow('')}>All</button>
                {(['done', 'in_progress', 'not_started', 'preparing'] as FieldState[]).filter(count).map((state) => (
                  <button key={state} className={`btn small${show === state ? ' chosen' : ''}`} onClick={() => setShow(state)}>{STATE_TEXT[state]} {count(state)}</button>
                ))}
              </div>
              <span className="spacer" />
              <button className="btn small" onClick={() => setAdding(true)}>+ Add fields</button>
            </div>
            {list.length ? (
              <div className="field-grid">{shown.map((f) => (
                <div key={f.tile_id} className="field-card">
                  <FieldThumb src={apiUrl(`/api/labels/fields/${enc(f.tile_id)}/thumb.jpg`)} alt={f.tile_id} />
                  <div className="row small"><FieldStateBadge state={f.state} /><span className="spacer" />
                    <button className="link-btn small" onClick={() => setRemoving(f)} title="Take this field out of the session (its labels are kept)">Remove</button></div>
                  <div className="small secondary ellipsis" title={f.tile_id}>{f.tma ? `${tmaName(f.tma)} · ${f.core_label}` : f.tile_id}{f.region ? ` · ${f.region}` : ''}</div>
                  {(f.done_by.length > 0 || f.working.length > 0 || f.open_by.length > 0) && (
                    <div className="small muted">{[f.done_by.length ? `done: ${f.done_by.join(', ')}` : '', f.working.length ? `working: ${f.working.join(', ')}` : '',
                      f.open_by.length ? `open now: ${f.open_by.join(', ')}` : ''].filter(Boolean).join(' · ')}</div>
                  )}
                </div>
              ))}</div>
            ) : <Empty>This session has no fields. <button className="btn small" onClick={() => setAdding(true)}>Add fields</button></Empty>}
          </>
        }}
      </Loaded>
      <ErrorNote error={error} />
      {adding && <AddFields s={s} onClose={() => setAdding(false)} onAdded={changed} />}
      {removing && (
        <ConfirmDialog title="Take this field out of the session?" confirm="Remove field" onClose={() => setRemoving(null)}
          onConfirm={() => withError(async () => { await api.post(`/api/labels/sessions/${s.name}/fields/remove`, { tile_ids: [removing.tile_id] }); changed() }, setError)}>
          <p>People will no longer be shown this field in “{s.title}”. <b>Labels already made on it are kept</b> and still count for training.</p>
        </ConfirmDialog>
      )}
    </section>
  )
}

function AddFields({ s, onClose, onAdded }: { s: SessionInfo; onClose: () => void; onAdded: () => void }) {
  const [mode, setMode] = useState<'auto' | 'pick'>('auto')
  const [n, setN] = useState(10)
  const [chosen, setChosen] = useState<string[]>([])
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const add = async () => {
    setBusy(true)
    const done = await withError(() => api.post(`/api/labels/sessions/${s.name}/fields`, mode === 'auto' ? { n } : { tile_ids: chosen }), setError)
    setBusy(false)
    if (done) { onAdded(); onClose() }
  }
  return (
    <Modal title={`Add ${s.stain} fields to “${s.title}”`} onClose={onClose} wide>
      <div className="new-session">
        <FieldChoice stain={s.stain} mode={mode} n={n} chosen={chosen} onMode={setMode} onN={setN} onChosen={setChosen} />
        <ErrorNote error={error} />
        <div className="row"><span className="spacer" /><button className="btn" onClick={onClose}>Cancel</button>
          <button className="btn primary" disabled={busy || (mode === 'pick' && !chosen.length)} onClick={add}>
            {busy ? 'Adding…' : mode === 'auto' ? `Add ${n} fields` : `Add ${chosen.length} field${chosen.length === 1 ? '' : 's'}`}</button></div>
      </div>
    </Modal>
  )
}
