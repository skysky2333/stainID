/* Looking inside a session's labels (or the reference labels): counts per class and person, and every labelled field
   with its objects drawn on it, filtered by class or field. */
import { useState } from 'react'
import type { LabelledField, SessionLabels } from '../api'
import { apiUrl, enc, fieldImage } from '../api'
import { Empty, Loaded, Modal } from '../components'
import { useFetch } from '../hooks'
import { useProject } from '../project'
import { FieldThumb } from './fields'

const PAGE = 24
/** A colour per class, in the order the classes are listed (series colours of the theme). */
export const classColor = (classes: string[], value: string) => `var(--series-${(Math.max(0, classes.indexOf(value)) % 8) + 1})`

export function SessionLabelsView({ name, title, onClose }: { name: string; title: string; onClose: () => void }) {
  const [stain, setStain] = useState('')
  const [value, setValue] = useState('')
  const [tile, setTile] = useState('')
  const [page, setPage] = useState(0)
  const [open, setOpen] = useState<LabelledField | null>(null)
  const query = new URLSearchParams({ stain, value, tile, offset: String(page * PAGE), limit: String(PAGE) })
  const data = useFetch<SessionLabels>(`/api/labels/sessions/${name}/labels?${query}`)
  const { tmaName } = useProject()
  const filter = (change: () => void) => { change(); setPage(0) }
  return (
    <Modal title={`Labels of “${title}”`} onClose={onClose} wide>
      <Loaded state={data} label="Loading the labels…">
        {(d) => {
          const classes = Object.keys(d.classes)
          return (
            <div className="examine">
              {d.reference && <p className="small secondary" style={{ margin: 0 }}>Labels made before labelling sessions existed, brought in from earlier label sets.
                They cannot be edited here; they count for training like any session.</p>}
              {d.stains.length > 1 && (
                <div className="segmented">{d.stains.map((s) => <button key={s} className={`btn small${s === d.stain ? ' chosen' : ''}`}
                  onClick={() => filter(() => { setStain(s); setValue('') })}>{s}</button>)}</div>
              )}
              <div className="class-chips">
                <button className={`btn small${value === '' ? ' chosen' : ''}`} onClick={() => filter(() => setValue(''))}>All classes</button>
                {classes.map((c) => (
                  <button key={c} className={`btn small${value === c ? ' chosen' : ''}`} onClick={() => filter(() => setValue(c))}>
                    <span className="swatch" style={{ background: classColor(classes, c) }} />{c} <span className="muted">{d.classes[c].toLocaleString()}</span></button>
                ))}
              </div>
              <div className="row small">
                <span className="secondary">Labelled by: {Object.entries(d.people).map(([p, n]) => `${p.replace(/^import: /, '')} (${n.toLocaleString()})`).join(' · ') || 'nobody yet'}</span>
                <span className="spacer" />
                <label className="field">Field<input type="search" value={tile} placeholder="core or field ID" style={{ width: 150 }}
                  onChange={(e) => filter(() => setTile(e.target.value))} /></label>
              </div>
              {d.fields.length ? <>
                <div className="field-grid">{d.fields.map((f) => (
                  <button key={f.tile_id} className="field-card clickable" onClick={() => setOpen(f)} title="Open this field larger">
                    <FieldThumb src={apiUrl(`/api/labels/fields/${enc(f.tile_id)}/thumb.jpg`)} alt={f.tile_id}
                      box={f.offset && f.width_px ? { x: f.offset[0], y: f.offset[1], width: f.width_px, height: f.height_px ?? f.width_px } : undefined}
                      marks={f.objects.map((o) => ({ x: o.x, y: o.y, color: classColor(classes, o.value) }))} />
                    <span className="small secondary ellipsis">{f.tma ? `${tmaName(f.tma)} · ${f.core_label}` : f.tile_id}</span>
                    <span className="small muted">{f.objects.length} object{f.objects.length === 1 ? '' : 's'}</span>
                  </button>
                ))}</div>
                {d.total_fields > PAGE && (
                  <div className="row picker-pages">
                    <button className="btn small" disabled={page === 0} onClick={() => setPage(page - 1)}>‹ Previous</button>
                    <span className="small secondary">fields {page * PAGE + 1}–{Math.min(d.total_fields, (page + 1) * PAGE)} of {d.total_fields.toLocaleString()}</span>
                    <button className="btn small" disabled={(page + 1) * PAGE >= d.total_fields} onClick={() => setPage(page + 1)}>Next ›</button>
                  </div>
                )}
              </> : <Empty>No labelled objects{value ? ` of class “${value}”` : ''}{tile ? ' in fields matching this search' : ''}.</Empty>}
              {open && <FieldLarge field={open} classes={classes} onClose={() => setOpen(null)} />}
            </div>
          )
        }}
      </Loaded>
    </Modal>
  )
}

/** One field at full size with its labelled objects (labels on fields without a picture are listed only). */
function FieldLarge({ field, classes, onClose }: { field: LabelledField; classes: string[]; onClose: () => void }) {
  const [size, setSize] = useState<{ w: number; h: number } | null>(null)
  return (
    <Modal title={field.tile_id} onClose={onClose} wide>
      {field.picture ? (
        <div className="field-large">
          <img src={fieldImage(field.tile_id)} alt={field.tile_id} onLoad={(e) => setSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })} />
          {size && (
            <svg viewBox={`0 0 ${size.w} ${size.h}`} preserveAspectRatio="none" aria-hidden="true">
              {field.objects.map((o, i) => <circle key={i} cx={o.x} cy={o.y} r={18} fill="none" stroke={classColor(classes, o.value)} strokeWidth={4}><title>{o.value} · {o.source}</title></circle>)}
            </svg>
          )}
        </div>
      ) : <p className="muted small">This field is not one of the project’s tiles (it comes from an earlier label set), so there is no picture of it.</p>}
      <div className="class-chips small">{classes.filter((c) => field.objects.some((o) => o.value === c)).map((c) => (
        <span key={c} className="person-chip"><span className="swatch" style={{ background: classColor(classes, c) }} />{c} {field.objects.filter((o) => o.value === c).length}</span>
      ))}</div>
    </Modal>
  )
}
